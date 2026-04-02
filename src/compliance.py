""" Rule engine for PPE compliance detection based on YOLOv8 outputs """

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum


# Class definitions — must match data/raw/dataset.yaml (the authoritative class ordering)

CLASS_PERSON            = 0
CLASS_HARDHAT           = 1
CLASS_SAFETY_VEST       = 2
CLASS_SAFETY_BOOTS      = 3
CLASS_SAFETY_GLOVES     = 4
CLASS_SAFETY_GOGGLES    = 5
CLASS_NO_HARDHAT        = 6
CLASS_NO_SAFETY_VEST    = 7
CLASS_NO_SAFETY_BOOTS   = 8
CLASS_NO_SAFETY_GLOVES  = 9
CLASS_NO_SAFETY_GOGGLES = 10
CLASS_MASK              = 11


# Constants for rule thresholds

CONFIDENCE_THRESHOLD   = 0.50   # Minimum detection confidence
IOU_OVERLAP_THRESHOLD  = 0.30   # Minimum fraction of PPE box that must fall within the body region

# Dividing the person's bounding box into relative regions that correspond to body parts
# Used to define expected PPE locations relative to the worker box

HEAD_REGION_UPPER      = 0.00   # Top of worker box
HEAD_REGION_LOWER      = 0.25   # Bottom of head region (top 25% of body)
TORSO_REGION_UPPER     = 0.15   # Top of torso region
TORSO_REGION_LOWER     = 0.65   # Bottom of torso region
FOOT_REGION_UPPER      = 0.80   # Top of foot region (bottom 20% of body)
FOOT_REGION_LOWER      = 1.00   # Bottom of worker box
HAND_REGION_UPPER      = 0.30   # Hands can appear across mid-body
HAND_REGION_LOWER      = 0.85


class WorkerStatus(Enum):
    COMPLIANT    = "COMPLIANT"     # All evaluable rules passed
    ALERT        = "ALERT"         # Critical rules passed; high-severity rules failed
    VIOLATION    = "VIOLATION"     # One or more critical rules failed
    UNVERIFIABLE = "UNVERIFIABLE"  # Worker detection below confidence threshold

class SceneVerdict(Enum):
    SAFE         = "SAFE"          # All workers compliant
    ALERT        = "ALERT"         # No critical violations; some high-severity alerts
    UNSAFE       = "UNSAFE"        # At least one worker with a critical violation
    UNVERIFIABLE = "UNVERIFIABLE"  # No violations/alerts but uncertain detections

class RuleResult(Enum):
    PASS          = "PASS"
    VIOLATION     = "VIOLATION"
    NOT_EVALUABLE = "NOT_EVALUABLE"   # Region not visible / data-limited
    LOW_CONF      = "LOW_CONFIDENCE"  # Detection present but below threshold


@dataclass
class Detection:
    """
    Represents a single detection returned by YOLOv8.
    All coordinates are in absolute pixel values.
    """
    class_id   : int
    confidence : float
    x1         : float
    y1         : float
    x2         : float
    y2         : float

    @property
    def box(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def is_confident(self) -> bool:
        return self.confidence >= CONFIDENCE_THRESHOLD


@dataclass
class WorkerViolation:
    """Records a single rule violation for one worker."""
    rule_id     : str
    rule_name   : str
    description : str

@dataclass
class WorkerCompliance:
    """
    Full compliance assessment for a single detected worker.

    Rules per safety_rules.md:
        R1 — Hard Hat Required          (Critical)
        R2 — Hi-Vis Vest Required       (Critical)
        R3 — Safety Boots Required      (High)
        R4 — Protective Gloves Required (High)
        R5 — Safety Goggles Required    (High)
        R6 — Full Basic PPE Compliance  (Critical, derived: R1 + R2)
    """
    worker_id    : int
    box          : tuple[float, float, float, float]
    confidence   : float
    status       : WorkerStatus

    # Per-rule results
    r1_helmet    : RuleResult = RuleResult.NOT_EVALUABLE
    r2_vest      : RuleResult = RuleResult.NOT_EVALUABLE
    r3_boots     : RuleResult = RuleResult.NOT_EVALUABLE
    r4_gloves    : RuleResult = RuleResult.NOT_EVALUABLE
    r5_goggles   : RuleResult = RuleResult.NOT_EVALUABLE
    r6_full_ppe  : RuleResult = RuleResult.NOT_EVALUABLE

    violations   : list[WorkerViolation] = field(default_factory=list)


@dataclass
class SceneReport:
    """
    Complete compliance report for a single image / frame.
    This is the final output of the compliance engine.
    """
    image_name        : str
    scene_verdict     : SceneVerdict
    total_workers     : int
    compliant_count   : int
    alert_count       : int
    violation_count   : int
    unverifiable_count: int
    workers           : list[WorkerCompliance] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialises the report to a JSON-compatible dictionary."""
        return {
            "image"              : self.image_name,
            "scene_verdict"      : self.scene_verdict.value,
            "total_workers"      : self.total_workers,
            "compliant_count"    : self.compliant_count,
            "alert_count"        : self.alert_count,
            "violation_count"    : self.violation_count,
            "unverifiable_count" : self.unverifiable_count,
            "workers": [
                {
                    "worker_id"  : w.worker_id,
                    "box"        : list(w.box),
                    "confidence" : round(w.confidence, 3),
                    "status"     : w.status.value,
                    "rules": {
                        "R1_helmet"   : w.r1_helmet.value,
                        "R2_vest"     : w.r2_vest.value,
                        "R3_boots"    : w.r3_boots.value,
                        "R4_gloves"   : w.r4_gloves.value,
                        "R5_goggles"  : w.r5_goggles.value,
                        "R6_full_ppe" : w.r6_full_ppe.value,
                    },
                    "violations": [
                        {
                            "rule"        : v.rule_id,
                            "name"        : v.rule_name,
                            "description" : v.description,
                        }
                        for v in w.violations
                    ],
                }
                for w in self.workers
            ],
        }


# Utility functions

def compute_iou(box_a: tuple, box_b: tuple) -> float:
    """
    Computes Intersection over Union (IoU) between two bounding boxes.
    Each box is (x1, y1, x2, y2) in pixel coordinates.
    Returns a float in [0.0, 1.0].
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    if inter_area == 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area == 0.0:
        return 0.0

    return inter_area / union_area

def get_anatomical_region(
    worker_box : tuple,
    upper_frac : float,
    lower_frac : float
) -> tuple[float, float, float, float]:
    """
    Returns a sub-region of the worker bounding box corresponding to a
    body part defined by vertical fractions of the box height.

    upper_frac = 0.0 is the top of the worker box.
    lower_frac = 1.0 is the bottom of the worker box.
    """
    x1, y1, x2, y2 = worker_box
    h = y2 - y1
    region_y1 = y1 + upper_frac * h
    region_y2 = y1 + lower_frac * h
    return (x1, region_y1, x2, region_y2)


def ppe_overlaps_region(
    ppe_box    : tuple,
    region_box : tuple,
    threshold  : float = IOU_OVERLAP_THRESHOLD
) -> bool:
    """
    Returns True if enough of the PPE bounding box falls within the body region.
    """
    px1, py1, px2, py2 = ppe_box
    rx1, ry1, rx2, ry2 = region_box

    inter_x1 = max(px1, rx1)
    inter_y1 = max(py1, ry1)
    inter_x2 = min(px2, rx2)
    inter_y2 = min(py2, ry2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    if inter_area == 0.0:
        return False

    ppe_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)
    if ppe_area == 0.0:
        return False

    return (inter_area / ppe_area) >= threshold


# PPE association functions

def worker_is_visible(
    worker_box     : tuple,
    all_detections : list[Detection],
) -> bool:
    """
    Returns True if at least one confident PPE detection overlaps the
    worker's bounding box, confirming the worker is clearly visible.

    Used by R1/R2 to distinguish:
      - visible worker with no helmet/vest  → VIOLATION
      - occluded / distant worker with no signal → NOT_EVALUABLE
    """
    for det in all_detections:
        if not det.is_confident:
            continue
        if ppe_overlaps_region(det.box, worker_box, threshold=0.10):
            return True
    return False



def find_ppe_for_worker(
    worker_box     : tuple,
    all_detections : list[Detection],
    target_classes : list[int],
    region_upper   : float,
    region_lower   : float,
) -> list[Detection]:
    """
    Finds all confident detections of the specified classes that
    overlap the expected body part of a worker.

    Returns a list of matching Detection objects.
    """
    region = get_anatomical_region(worker_box, region_upper, region_lower)
    matches = []
    for det in all_detections:
        if det.class_id not in target_classes:
            continue
        if not det.is_confident:
            continue
        if ppe_overlaps_region(det.box, region):
            matches.append(det)
    return matches


def find_low_confidence_ppe(
    worker_box     : tuple,
    all_detections : list[Detection],
    target_classes : list[int],
    region_upper   : float,
    region_lower   : float,
) -> list[Detection]:
    """
    Finds detections below the confidence threshold.
    Used to flag LOW_CONFIDENCE results rather than treating
    absence as a definitive violation.
    """
    region = get_anatomical_region(worker_box, region_upper, region_lower)
    matches = []
    for det in all_detections:
        if det.class_id not in target_classes:
            continue
        if det.is_confident:
            continue
        if ppe_overlaps_region(det.box, region):
            matches.append(det)
    return matches


# Rule evaluation

def evaluate_r1_helmet(
    worker     : WorkerCompliance,
    detections : list[Detection],
) -> WorkerCompliance:
    """
    R1 — Hard Hat Required (Critical).
    Checks the head region (top 25% of worker box).
    PASS          : Hardhat detected in head region.
    VIOLATION     : NO-Hardhat detected, or no helmet associated with worker.
    LOW_CONF      : Hardhat detected but below confidence threshold.
    """
    helmet_matches = find_ppe_for_worker(
        worker.box, detections,
        target_classes=[CLASS_HARDHAT],
        region_upper=HEAD_REGION_UPPER,
        region_lower=HEAD_REGION_LOWER,
    )
    no_helmet_matches = find_ppe_for_worker(
        worker.box, detections,
        target_classes=[CLASS_NO_HARDHAT],
        region_upper=HEAD_REGION_UPPER,
        region_lower=HEAD_REGION_LOWER,
    )
    low_conf = find_low_confidence_ppe(
        worker.box, detections,
        target_classes=[CLASS_HARDHAT],
        region_upper=HEAD_REGION_UPPER,
        region_lower=HEAD_REGION_LOWER,
    )

    if helmet_matches:
        worker.r1_helmet = RuleResult.PASS
    elif no_helmet_matches:
        worker.r1_helmet = RuleResult.VIOLATION
        worker.violations.append(WorkerViolation(
            rule_id="R1",
            rule_name="Hard Hat Required",
            description="Worker detected without a hard hat.",
        ))
    elif low_conf:
        worker.r1_helmet = RuleResult.LOW_CONF
        worker.violations.append(WorkerViolation(
            rule_id="R1",
            rule_name="Hard Hat Required",
            description="Helmet detected but below confidence threshold — "
                        "treated as potential violation.",
        ))
    else:
        # No direct helmet signal. Use other PPE detections as a visibility
        # proxy: if any PPE item overlaps this worker's box, the worker is
        # clearly in frame — missing helmet is a confirmed VIOLATION.
        # If nothing is detectable (far/occluded), mark NOT_EVALUABLE.
        if worker_is_visible(worker.box, detections):
            worker.r1_helmet = RuleResult.VIOLATION
            worker.violations.append(WorkerViolation(
                rule_id="R1",
                rule_name="Hard Hat Required",
                description="Worker visible (other PPE detected) but no helmet found.",
            ))
        else:
            worker.r1_helmet = RuleResult.NOT_EVALUABLE

    return worker


def evaluate_r2_vest(
    worker     : WorkerCompliance,
    detections : list[Detection]
) -> WorkerCompliance:
    """
    R2 — High-Visibility Vest Required (Critical).
    Checks the torso region (15%–65% of worker box height).
    PASS          : Safety Vest detected in torso region.
    VIOLATION     : NO-Safety Vest detected, or no vest associated with worker.
    LOW_CONF      : Safety Vest detected but below confidence threshold.
    """
    vest_matches = find_ppe_for_worker(
        worker.box, detections,
        target_classes=[CLASS_SAFETY_VEST],
        region_upper=TORSO_REGION_UPPER,
        region_lower=TORSO_REGION_LOWER,
    )
    no_vest_matches = find_ppe_for_worker(
        worker.box, detections,
        target_classes=[CLASS_NO_SAFETY_VEST],
        region_upper=TORSO_REGION_UPPER,
        region_lower=TORSO_REGION_LOWER,
    )
    low_conf = find_low_confidence_ppe(
        worker.box, detections,
        target_classes=[CLASS_SAFETY_VEST],
        region_upper=TORSO_REGION_UPPER,
        region_lower=TORSO_REGION_LOWER,
    )

    if vest_matches:
        worker.r2_vest = RuleResult.PASS
    elif no_vest_matches:
        worker.r2_vest = RuleResult.VIOLATION
        worker.violations.append(WorkerViolation(
            rule_id="R2",
            rule_name="Hi-Vis Vest Required",
            description="Worker detected without a high-visibility vest.",
        ))
    elif low_conf:
        worker.r2_vest = RuleResult.LOW_CONF
        worker.violations.append(WorkerViolation(
            rule_id="R2",
            rule_name="Hi-Vis Vest Required",
            description="Vest detected but below confidence threshold — "
                        "treated as potential violation.",
        ))
    else:
        # No direct vest signal. Use other PPE detections as a visibility
        # proxy: if any PPE item overlaps this worker's box, the worker is
        # clearly in frame — missing vest is a confirmed VIOLATION.
        # If nothing is detectable (far/occluded), mark NOT_EVALUABLE.
        if worker_is_visible(worker.box, detections):
            worker.r2_vest = RuleResult.VIOLATION
            worker.violations.append(WorkerViolation(
                rule_id="R2",
                rule_name="Hi-Vis Vest Required",
                description="Worker visible (other PPE detected) but no vest found.",
            ))
        else:
            worker.r2_vest = RuleResult.NOT_EVALUABLE

    return worker


def evaluate_r3_boots(
    worker     : WorkerCompliance,
    detections : list[Detection]
) -> WorkerCompliance:
    """
    R3 — Safety Boots Required (High).
    Only evaluated when the foot region is visible. Worker box must be
    at least 150px tall to resolve foot-level detail.
    PASS          : Safety Boots detected in foot region.
    VIOLATION     : NO-Safety Boots detected in foot region.
    LOW_CONF      : NO-Safety Boots detected but below confidence threshold.
    NOT_EVALUABLE : Foot region not visible or worker box too small.
    """
    _, y1, _, y2 = worker.box
    if (y2 - y1) < 150:
        worker.r3_boots = RuleResult.NOT_EVALUABLE
        return worker

    boots_matches = find_ppe_for_worker(
        worker.box, detections,
        target_classes=[CLASS_SAFETY_BOOTS],
        region_upper=FOOT_REGION_UPPER,
        region_lower=FOOT_REGION_LOWER,
    )
    no_boots_matches = find_ppe_for_worker(
        worker.box, detections,
        target_classes=[CLASS_NO_SAFETY_BOOTS],
        region_upper=FOOT_REGION_UPPER,
        region_lower=FOOT_REGION_LOWER,
    )
    low_conf = find_low_confidence_ppe(
        worker.box, detections,
        target_classes=[CLASS_NO_SAFETY_BOOTS],
        region_upper=FOOT_REGION_UPPER,
        region_lower=FOOT_REGION_LOWER,
    )

    if boots_matches:
        worker.r3_boots = RuleResult.PASS
    elif no_boots_matches:
        worker.r3_boots = RuleResult.VIOLATION
        worker.violations.append(WorkerViolation(
            rule_id="R3",
            rule_name="Safety Boots Required",
            description="Non-compliant footwear detected — "
                        "trainers, sandals, or bare feet.",
        ))
    elif low_conf:
        worker.r3_boots = RuleResult.LOW_CONF
        worker.violations.append(WorkerViolation(
            rule_id="R3",
            rule_name="Safety Boots Required",
            description="Non-compliant footwear detected but below confidence threshold — "
                        "treated as potential violation.",
        ))
    else:
        worker.r3_boots = RuleResult.NOT_EVALUABLE

    return worker


def evaluate_r4_gloves(
    worker     : WorkerCompliance,
    detections : list[Detection]
) -> WorkerCompliance:
    """
    R4 — Protective Gloves Required (High).
    Only evaluated when the hand region is visible.
    PASS          : Safety Gloves detected in hand region.
    VIOLATION     : NO-Safety Gloves detected in hand region.
    LOW_CONF      : NO-Safety Gloves detected but below confidence threshold.
    NOT_EVALUABLE : Hands not visible in the frame.
    """
    gloves_matches = find_ppe_for_worker(
        worker.box, detections,
        target_classes=[CLASS_SAFETY_GLOVES],
        region_upper=HAND_REGION_UPPER,
        region_lower=HAND_REGION_LOWER,
    )
    no_gloves_matches = find_ppe_for_worker(
        worker.box, detections,
        target_classes=[CLASS_NO_SAFETY_GLOVES],
        region_upper=HAND_REGION_UPPER,
        region_lower=HAND_REGION_LOWER,
    )
    low_conf = find_low_confidence_ppe(
        worker.box, detections,
        target_classes=[CLASS_NO_SAFETY_GLOVES],
        region_upper=HAND_REGION_UPPER,
        region_lower=HAND_REGION_LOWER,
    )

    if gloves_matches:
        worker.r4_gloves = RuleResult.PASS
    elif no_gloves_matches:
        worker.r4_gloves = RuleResult.VIOLATION
        worker.violations.append(WorkerViolation(
            rule_id="R4",
            rule_name="Protective Gloves Required",
            description="Worker hands visible without protective gloves.",
        ))
    elif low_conf:
        worker.r4_gloves = RuleResult.LOW_CONF
        worker.violations.append(WorkerViolation(
            rule_id="R4",
            rule_name="Protective Gloves Required",
            description="Bare hands detected but below confidence threshold — "
                        "treated as potential violation.",
        ))
    else:
        worker.r4_gloves = RuleResult.NOT_EVALUABLE

    return worker


def evaluate_r5_goggles(
    worker     : WorkerCompliance,
    detections : list[Detection]
) -> WorkerCompliance:
    """
    R5 — Safety Goggles Required (High).
    Checks the head/face region for goggles presence or confirmed absence.
    PASS          : Safety Goggles detected in head region.
    VIOLATION     : NO-Safety Goggles detected (confirmed bare eyes).
    LOW_CONF      : NO-Safety Goggles detected but below confidence threshold.
    NOT_EVALUABLE : No signal — face not visible or too distant.
    """
    goggles_matches = find_ppe_for_worker(
        worker.box, detections,
        target_classes=[CLASS_SAFETY_GOGGLES],
        region_upper=HEAD_REGION_UPPER,
        region_lower=HEAD_REGION_LOWER,
    )
    no_goggles_matches = find_ppe_for_worker(
        worker.box, detections,
        target_classes=[CLASS_NO_SAFETY_GOGGLES],
        region_upper=HEAD_REGION_UPPER,
        region_lower=HEAD_REGION_LOWER,
    )
    low_conf = find_low_confidence_ppe(
        worker.box, detections,
        target_classes=[CLASS_NO_SAFETY_GOGGLES],
        region_upper=HEAD_REGION_UPPER,
        region_lower=HEAD_REGION_LOWER,
    )

    if goggles_matches:
        worker.r5_goggles = RuleResult.PASS
    elif no_goggles_matches:
        worker.r5_goggles = RuleResult.VIOLATION
        worker.violations.append(WorkerViolation(
            rule_id="R5",
            rule_name="Safety Goggles Required",
            description="Worker detected without safety goggles or protective eyewear.",
        ))
    elif low_conf:
        worker.r5_goggles = RuleResult.LOW_CONF
        worker.violations.append(WorkerViolation(
            rule_id="R5",
            rule_name="Safety Goggles Required",
            description="No-goggles indicator detected but below confidence threshold — "
                        "treated as potential violation.",
        ))
    else:
        worker.r5_goggles = RuleResult.NOT_EVALUABLE

    return worker


def evaluate_r6_full_ppe(worker: WorkerCompliance) -> WorkerCompliance:
    """
    R6 — Full Basic PPE Compliance (Critical, derived rule).
    Passes only if both R1 (helmet) and R2 (vest) pass.
    This is the baseline check applied to every detected worker on site.
    No new detections needed — uses results already computed.
    """
    if (worker.r1_helmet == RuleResult.PASS and
            worker.r2_vest == RuleResult.PASS):
        worker.r6_full_ppe = RuleResult.PASS
    elif (worker.r1_helmet == RuleResult.NOT_EVALUABLE or
              worker.r2_vest == RuleResult.NOT_EVALUABLE):
        # Can't confirm full PPE when either component is unevaluable.
        worker.r6_full_ppe = RuleResult.NOT_EVALUABLE
    else:
        worker.r6_full_ppe = RuleResult.VIOLATION

    return worker


# Worker status and scene verdict logic

def resolve_worker_status(worker: WorkerCompliance) -> WorkerCompliance:
    """
    Determines the final WorkerStatus from the accumulated rule results.

    Priority:
        1. UNVERIFIABLE — worker detection below confidence threshold
        2. VIOLATION    — any critical rule (R1, R2, R6) failed
        3. ALERT        — critical rules passed; high-severity rule (R3, R4, R5) failed
        4. COMPLIANT    — all evaluable rules passed

    Critical rules (R1, R2, R6): worker must stop work if failed.
    High-severity rules (R3, R4, R5): worker is on site but needs a reminder.
    """
    if worker.confidence < CONFIDENCE_THRESHOLD:
        worker.status = WorkerStatus.UNVERIFIABLE
        return worker

    critical_results = [
        worker.r1_helmet,
        worker.r2_vest,
        worker.r6_full_ppe,
    ]
    high_results = [
        worker.r3_boots,
        worker.r4_gloves,
        worker.r5_goggles,
    ]

    critical_failed = any(
        r in (RuleResult.VIOLATION, RuleResult.LOW_CONF)
        for r in critical_results
    )
    high_failed = any(
        r in (RuleResult.VIOLATION, RuleResult.LOW_CONF)
        for r in high_results
    )

    if critical_failed:
        worker.status = WorkerStatus.VIOLATION
    elif high_failed:
        worker.status = WorkerStatus.ALERT
    else:
        worker.status = WorkerStatus.COMPLIANT

    return worker


def resolve_scene_verdict(workers: list[WorkerCompliance]) -> SceneVerdict:
    """
    Derives the scene-level verdict from all worker statuses.

    UNSAFE       — at least one worker has a critical violation
    ALERT        — no critical violations; at least one worker has a high-severity alert
    UNVERIFIABLE — no violations/alerts but at least one worker is unverifiable
    SAFE         — all workers compliant
    """
    if not workers:
        return SceneVerdict.UNVERIFIABLE

    statuses = [w.status for w in workers]

    if any(s == WorkerStatus.VIOLATION for s in statuses):
        return SceneVerdict.UNSAFE
    if any(s == WorkerStatus.ALERT for s in statuses):
        return SceneVerdict.ALERT
    if any(s == WorkerStatus.UNVERIFIABLE for s in statuses):
        return SceneVerdict.UNVERIFIABLE
    return SceneVerdict.SAFE


# Main evaluation function

def evaluate_scene(
    raw_detections : list[dict],
    image_name     : str = "unknown"
) -> SceneReport:
    """
    Main entry point for the compliance engine.

    Accepts raw YOLOv8 detection results for a single image and
    returns a fully populated SceneReport.

    Parameters
    ----------
    raw_detections : list[dict]
        Each dict must contain:
            "class_id"   : int
            "confidence" : float
            "x1"         : float  (pixel coordinates)
            "y1"         : float
            "x2"         : float
            "y2"         : float

    image_name : str
        Filename or identifier of the source image.

    Returns
    -------
    SceneReport
        Complete per-worker and scene-level compliance assessment.
    """

    # ── Parse raw detections into Detection objects ──
    detections = [
        Detection(
            class_id   = d["class_id"],
            confidence = d["confidence"],
            x1 = d["x1"], y1 = d["y1"],
            x2 = d["x2"], y2 = d["y2"],
        )
        for d in raw_detections
    ]

    # ── Separate person detections from PPE detections ──
    # Include ALL person detections (not just confident ones) so that
    # low-confidence workers are counted as UNVERIFIABLE rather than silently dropped.
    person_detections = [
        d for d in detections
        if d.class_id == CLASS_PERSON
    ]
    ppe_detections = [
        d for d in detections
        if d.class_id != CLASS_PERSON
    ]

    # ── Evaluate each worker individually ──
    worker_results = []

    for idx, person in enumerate(person_detections):
        worker = WorkerCompliance(
            worker_id  = idx + 1,
            box        = person.box,
            confidence = person.confidence,
            status     = WorkerStatus.UNVERIFIABLE,
        )

        # Only run rule evaluation for confident detections.
        # Low-confidence workers skip directly to resolve_worker_status
        # which marks them UNVERIFIABLE.
        if person.is_confident:
            worker = evaluate_r1_helmet(worker, ppe_detections)
            worker = evaluate_r2_vest(worker, ppe_detections)
            worker = evaluate_r3_boots(worker, ppe_detections)
            worker = evaluate_r4_gloves(worker, ppe_detections)
            worker = evaluate_r5_goggles(worker, ppe_detections)
            worker = evaluate_r6_full_ppe(worker)

        worker = resolve_worker_status(worker)

        worker_results.append(worker)

    # ── Resolve scene-level verdict ──
    scene_verdict = resolve_scene_verdict(worker_results)

    # ── Assemble and return scene report ──
    compliant_count    = sum(1 for w in worker_results
                             if w.status == WorkerStatus.COMPLIANT)
    alert_count        = sum(1 for w in worker_results
                             if w.status == WorkerStatus.ALERT)
    violation_count    = sum(1 for w in worker_results
                             if w.status == WorkerStatus.VIOLATION)
    unverifiable_count = sum(1 for w in worker_results
                             if w.status == WorkerStatus.UNVERIFIABLE)

    return SceneReport(
        image_name         = image_name,
        scene_verdict      = scene_verdict,
        total_workers      = len(worker_results),
        compliant_count    = compliant_count,
        alert_count        = alert_count,
        violation_count    = violation_count,
        unverifiable_count = unverifiable_count,
        workers            = worker_results,
    )
