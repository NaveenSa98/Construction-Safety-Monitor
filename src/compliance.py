"""
compliance.py
-------------
Pure compliance logic for Construction Site PPE detection.
No model inference here — takes structured detections as input and returns
a per-worker compliance report plus a scene-level verdict.

Verdict levels:
    SAFE   — every worker has all critical AND advisory PPE
    ALERT  — every worker has critical PPE, but advisory PPE (gloves/goggles)
             is missing on at least one worker
    UNSAFE — at least one worker is missing critical PPE (helmet/vest/boots)

Usage:
    from src.compliance import check_compliance

    detections = [
        {"class_name": "person",  "bbox": [x1, y1, x2, y2], "confidence": 0.91},
        {"class_name": "helmet",  "bbox": [x1, y1, x2, y2], "confidence": 0.87},
        ...
    ]
    report = check_compliance(detections)
    print(report["scene_verdict"])   # "SAFE" | "ALERT" | "UNSAFE"
"""


# ── PPE Categories ────────────────────────────────────────────────────────────

# Missing any critical PPE → worker is non-compliant → scene is UNSAFE
CRITICAL_PPE = {"helmat", "vest"}

# Missing advisory PPE → per-worker warning only, does not affect scene verdict
ADVISORY_PPE = {"gloves", "goggles", "boot"}

ALL_PPE = CRITICAL_PPE | ADVISORY_PPE


# ── Confidence Thresholds (per class) ────────────────────────────────────────
# Smaller/harder-to-detect items get lower thresholds to reduce false negatives.

CONF_THRESHOLDS = {
    "person":  0.70,
    "helmat":  0.70,
    "vest":    0.45,
    "boot":    0.25,
    "gloves":  0.60,
    "goggles": 0.18,
}


# ── Spatial Association Parameters ───────────────────────────────────────────

# Minimum IoU between a PPE box and a worker box to count as associated
MIN_ASSOCIATION_IOU = 0.04

# Vertical zone each PPE type is expected to occupy within the worker bbox.
# Expressed as (top_ratio, bottom_ratio) relative to worker height.
# Rejects geometrically impossible associations (e.g. helmet at feet).
VERTICAL_ZONES = {
    "helmat":  (0.00, 0.80),   # widened — covers bent-over workers
    "goggles": (0.10, 0.70),   # eye region
    "vest":    (0.10, 1.00),   # torso — relaxed for crouching
    "gloves":  (0.00, 1.30),   # hands — extended down for crouching/reaching
    "boot":    (0.20, 1.60),   # feet — greatly extended to handle partial person detections
}

# How much to expand the worker bbox (in each direction) before association.
# Handles PPE detected slightly outside the person bbox edge (e.g. helmet
# above a bent-forward worker, gloves at the side of the body).
BBOX_EXPAND_RATIO = 0.5


# =============================================================================
# Layer 1 — Geometry
# =============================================================================

def _expand_bbox(bbox: list, ratio: float) -> list:
    """
    Expand a [x1, y1, x2, y2] bbox by ratio in all directions.
    Used to catch PPE that is detected just outside the person bbox
    (e.g. helmet above a bent-forward worker).
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    pad_x = w * ratio
    pad_y = h * ratio
    return [x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y]


def compute_iou(box_a: list, box_b: list) -> float:
    """
    Compute Intersection over Union between two [x1, y1, x2, y2] boxes.
    Returns a float in [0.0, 1.0].
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    if inter_area == 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union_area = area_a + area_b - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


# =============================================================================
# Layer 2 — Filter
# =============================================================================

def _filter_detections(detections: list) -> tuple:
    """
    Split detections into workers and PPE items.
    Drops any detection below its class confidence threshold (R7 gate).

    Returns:
        workers  : list of person detections that passed the confidence gate
        ppe_items: list of PPE detections that passed the confidence gate
    """
    workers = []
    ppe_items = []

    for det in detections:
        cls  = det["class_name"]
        conf = det["confidence"]
        threshold = CONF_THRESHOLDS.get(cls, 0.25)

        if conf < threshold:
            continue  # R7: reject low-confidence detections

        if cls == "person":
            workers.append(det)
        elif cls in ALL_PPE:
            ppe_items.append(det)

    return workers, ppe_items


# =============================================================================
# Layer 2b — Person Deduplication (NMS)
# =============================================================================

PERSON_NMS_IOU = 0.45  # merge person boxes that overlap more than this

def _deduplicate_persons(workers: list) -> list:
    """
    Remove duplicate person detections caused by overlapping bounding boxes.
    Keeps the highest-confidence detection when two person boxes overlap
    more than PERSON_NMS_IOU.
    """
    if len(workers) <= 1:
        return workers

    # Sort by confidence descending so we keep the strongest detections
    sorted_workers = sorted(workers, key=lambda d: d["confidence"], reverse=True)
    keep = []

    for candidate in sorted_workers:
        is_duplicate = False
        for kept in keep:
            iou = compute_iou(candidate["bbox"], kept["bbox"])
            if iou > PERSON_NMS_IOU:
                is_duplicate = True
                break
        if not is_duplicate:
            keep.append(candidate)

    return keep


# =============================================================================
# Layer 3 — Vertical Zone Check
# =============================================================================

def _in_vertical_zone(ppe_class: str, ppe_box: list, worker_box: list) -> bool:
    """
    Return True if the PPE box centre falls within the expected vertical zone
    of the worker bounding box.

    Prevents a helmet near Worker A from being credited to Worker B whose
    bounding box barely overlaps at an incorrect vertical position.
    """
    zone = VERTICAL_ZONES.get(ppe_class)
    if zone is None:
        return True  # no zone defined — accept the association

    _, wy1, _, wy2 = worker_box
    worker_height = wy2 - wy1
    if worker_height <= 0:
        return True

    _, py1, _, py2 = ppe_box
    ppe_centre_y = (py1 + py2) / 2.0

    zone_top    = wy1 + zone[0] * worker_height
    zone_bottom = wy1 + zone[1] * worker_height

    return zone_top <= ppe_centre_y <= zone_bottom


# =============================================================================
# Layer 4 — PPE-to-Worker Association
# =============================================================================

def _associate_ppe(workers: list, ppe_items: list) -> list:
    worker_records = [
        {
            "worker_id":  idx,
            "bbox":       w["bbox"],
            "confidence": w["confidence"],
            "ppe_found":  set(),
        }
        for idx, w in enumerate(workers)
    ]

    for ppe in ppe_items:
        best_score  = -1.0
        best_worker = None

        px = (ppe["bbox"][0] + ppe["bbox"][2]) / 2
        py = (ppe["bbox"][1] + ppe["bbox"][3]) / 2

        for record in worker_records:
            expanded = _expand_bbox(record["bbox"], BBOX_EXPAND_RATIO)
            iou_exp  = compute_iou(ppe["bbox"], expanded)
            # IoU with the original (non-expanded) bbox: > 0 means the PPE
            # physically overlaps the person box, a very strong signal.
            iou_orig = compute_iou(ppe["bbox"], record["bbox"])

            # Skip only when there is zero overlap with both the original and
            # the expanded bbox.  Checking only iou_exp causes close-up workers
            # (huge expanded bbox → tiny IoU) to be skipped while a distant
            # worker with a smaller expanded bbox "steals" their PPE.
            if iou_orig == 0.0 and iou_exp < MIN_ASSOCIATION_IOU:
                continue
            if not _in_vertical_zone(ppe["class_name"], ppe["bbox"], record["bbox"]):
                continue

            wx = (record["bbox"][0] + record["bbox"][2]) / 2
            wy = (record["bbox"][1] + record["bbox"][3]) / 2
            wh = max(record["bbox"][3] - record["bbox"][1], 1)
            norm_dist = ((px - wx)**2 + (py - wy)**2) ** 0.5 / wh

            # Give a 5× bonus when PPE actually overlaps the person box so that
            # a large foreground worker always outscores a background worker
            # whose expanded bbox happens to overlap the same PPE item.
            score = (iou_orig * 5 + iou_exp) / (1.0 + norm_dist)

            if score > best_score:
                best_score  = score
                best_worker = record

        # Centroid fallback when no IoU match at all.
        # Normalise distance by worker height so a large foreground worker
        # (whose body centroid is far from their head in raw pixels) is not
        # beaten by a small background worker with a closer absolute centroid.
        if best_worker is None:
            best_norm_dist = float("inf")
            for record in worker_records:
                wx1, wy1, wx2, wy2 = record["bbox"]
                ww = max(wx2 - wx1, 1)
                wh = max(wy2 - wy1, 1)
                if not (wx1 - ww * 0.5 <= px <= wx2 + ww * 0.5):
                    continue
                wx = (wx1 + wx2) / 2
                wy = (wy1 + wy2) / 2
                norm_dist = ((px - wx)**2 + (py - wy)**2) ** 0.5 / wh
                if norm_dist < best_norm_dist:
                    best_norm_dist = norm_dist
                    best_worker    = record

        if best_worker is not None:
            best_worker["ppe_found"].add(ppe["class_name"])

    return worker_records


# =============================================================================
# Layer 5 — Per-Worker Rule Evaluation (R1–R5, R7)
# =============================================================================

def _evaluate_worker(record: dict) -> dict:
    """
    Apply safety rules to a single worker.

    R1 — helmet present   (critical)
    R2 — vest present     (critical)
    R3 — boots present    (advisory)
    R4 — gloves present   (advisory)
    R5 — goggles present  (advisory)
    R7 — confidence gate already applied in Layer 2 (filter step)

    Returns a result dict with violations, alerts, and compliance flag.
    """
    ppe = record["ppe_found"]

    violations = sorted(CRITICAL_PPE - ppe)   # missing critical PPE
    alerts     = sorted(ADVISORY_PPE - ppe)   # missing advisory PPE

    return {
        "worker_id":    record["worker_id"],
        "bbox":         record["bbox"],
        "ppe_detected": sorted(ppe),
        "violations":   violations,   # helmet / vest / boots missing
        "alerts":       alerts,        # gloves / goggles missing
        "compliant":    len(violations) == 0,
    }


# =============================================================================
# Layer 6 — Scene Verdict
# =============================================================================

def _compute_scene_verdict(worker_results: list) -> str:
    """
    UNSAFE — any worker missing helmet or vest (critical PPE)
    SAFE   — all workers have helmet and vest (advisory PPE warnings are separate)
    """
    if any(not w["compliant"] for w in worker_results):
        return "UNSAFE"
    return "SAFE"


# =============================================================================
# Public API
# =============================================================================

def check_compliance(detections: list) -> dict:
    """
    Main entry point for compliance checking.

    Args:
        detections: list of dicts, each with keys:
                    - class_name  (str)  : detected class label
                    - bbox        (list) : [x1, y1, x2, y2] in pixels
                    - confidence  (float): model confidence score

    Returns:
        dict with keys:
            scene_verdict     : "SAFE" | "ALERT" | "UNSAFE"
            total_workers     : int
            compliant_workers : int
            workers           : list of per-worker result dicts
    """
    # Layer 1 & 2 — split and confidence-gate
    workers, ppe_items = _filter_detections(detections)

    # Layer 2b — remove duplicate person detections
    workers = _deduplicate_persons(workers)

    # Edge case — no workers detected in frame
    if not workers:
        return {
            "scene_verdict":     "",
            "total_workers":     0,
            "compliant_workers": 0,
            "workers":           [],
            "note":              "No workers detected in frame.",
        }

    # Layer 3 & 4 — associate PPE to workers
    worker_records = _associate_ppe(workers, ppe_items)

    # Layer 5 — evaluate rules per worker
    worker_results = [_evaluate_worker(r) for r in worker_records]

    # Layer 6 — scene verdict
    verdict = _compute_scene_verdict(worker_results)

    compliant_count = sum(1 for w in worker_results if w["compliant"])

    return {
        "scene_verdict":     verdict,
        "total_workers":     len(worker_results),
        "compliant_workers": compliant_count,
        "workers":           worker_results,
    }
