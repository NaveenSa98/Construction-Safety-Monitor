""" Rule engine for PPE compliance detection based on YOLOv8 outputs """

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional



# Class definitions

CLASS_PERSON            = 0
CLASS_HARDHAT           = 1
CLASS_NO_HARDHAT        = 2
CLASS_SAFETY_VEST       = 3
CLASS_NO_SAFETY_VEST    = 4
CLASS_SAFETY_GLOVES     = 5
CLASS_NO_SAFETY_GLOVES  = 6
CLASS_SAFETY_BOOTS      = 7
CLASS_NO_SAFETY_BOOTS   = 8
CLASS_SAFETY_GOGGLES    = 9
CLASS_NO_SAFETY_GOGGLES = 10
CLASS_NO_SAFETY_HARNESS = 11


# Constants for rule thresholds

CONFIDENCE_THRESHOLD   = 0.50   # Minimum detection confidence
IOU_OVERLAP_THRESHOLD  = 0.30   # Minimum fraction of PPE box that must fall within the body region

# Dividing the person’s bounding box into relative regions that correspond to body parts
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
    COMPLIANT     = "COMPLIANT"
    VIOLATION     = "VIOLATION"
    UNVERIFIABLE  = "UNVERIFIABLE"

class SceneVerdict(Enum):
    SAFE          = "SAFE"
    UNSAFE        = "UNSAFE"
    UNVERIFIABLE  = "UNVERIFIABLE"

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
    """
    worker_id      : int
    box            : tuple[float, float, float, float]
    confidence     : float
    status         : WorkerStatus

    # Per-rule results
    r1_helmet      : RuleResult = RuleResult.NOT_EVALUABLE
    r2_vest        : RuleResult = RuleResult.NOT_EVALUABLE
    r3_full_ppe    : RuleResult = RuleResult.NOT_EVALUABLE
    r4_boots       : RuleResult = RuleResult.NOT_EVALUABLE
    r5_harness     : RuleResult = RuleResult.NOT_EVALUABLE
    r6_gloves      : RuleResult = RuleResult.NOT_EVALUABLE
    r7_goggles     : RuleResult = RuleResult.NOT_EVALUABLE
    r8_correct_ppe : RuleResult = RuleResult.NOT_EVALUABLE

    violations     : list[WorkerViolation] = field(default_factory=list)


@dataclass
class SceneReport:
    """
    Complete compliance report for a single image / frame.
    This is the final output of the compliance engine.
    """
    image_name      : str
    scene_verdict   : SceneVerdict
    total_workers   : int
    compliant_count : int
    violation_count : int
    unverifiable_count: int
    workers         : list[WorkerCompliance] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialises the report to a JSON-compatible dictionary."""
        return {
            "image"              : self.image_name,
            "scene_verdict"      : self.scene_verdict.value,
            "total_workers"      : self.total_workers,
            "compliant_count"    : self.compliant_count,
            "violation_count"    : self.violation_count,
            "unverifiable_count" : self.unverifiable_count,
            "workers": [
                {
                    "worker_id"  : w.worker_id,
                    "box"        : list(w.box),
                    "confidence" : round(w.confidence, 3),
                    "status"     : w.status.value,
                    "rules": {
                        "R1_helmet"      : w.r1_helmet.value,
                        "R2_vest"        : w.r2_vest.value,
                        "R3_full_ppe"    : w.r3_full_ppe.value,
                        "R4_boots"       : w.r4_boots.value,
                        "R5_harness"     : w.r5_harness.value,
                        "R6_gloves"      : w.r6_gloves.value,
                        "R7_goggles"     : w.r7_goggles.value,
                        "R8_correct_ppe" : w.r8_correct_ppe.value,
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
    Returns a sub-region of the worker bounding box corresponding to
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

def get_ppe_centroid_y_fraction(
    ppe_box    : tuple,
    worker_box : tuple
) -> Optional[float]:
    """
    Returns the vertical position of the PPE box centroid as a
    fraction of the worker box height (0.0 = top, 1.0 = bottom).
    Returns None if the PPE centroid is outside the worker box.
    """
    _, wy1, _, wy2 = worker_box
    _, py1, _, py2 = ppe_box
    worker_height = wy2 - wy1
    if worker_height == 0:
        return None
    ppe_centroid_y = (py1 + py2) / 2.0
    fraction = (ppe_centroid_y - wy1) / worker_height
    return fraction

# PPE association function

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
    detections : list[Detection]
) -> WorkerCompliance:
    """
    R1 — Hard Hat / Safety Helmet Required.
    Checks the head region (top 25% of worker box).
    """
    head_matches = find_ppe_for_worker(
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

    if head_matches:
        worker.r1_helmet = RuleResult.PASS
    elif low_conf:
        worker.r1_helmet = RuleResult.LOW_CONF
        worker.violations.append(WorkerViolation(
            rule_id="R1",
            rule_name="Hard Hat Required",
            description="Helmet detected but below confidence threshold — "
                        "treated as potential violation.",
        ))
    elif no_helmet_matches:
        worker.r1_helmet = RuleResult.VIOLATION
        worker.violations.append(WorkerViolation(
            rule_id="R1",
            rule_name="Hard Hat Required",
            description="Worker detected without a hard hat.",
        ))
    else:
        # No positive or negative signal — treat conservatively
        worker.r1_helmet = RuleResult.VIOLATION
        worker.violations.append(WorkerViolation(
            rule_id="R1",
            rule_name="Hard Hat Required",
            description="No helmet detected in head region of worker.",
        ))

    return worker


def evaluate_r2_vest(
    worker     : WorkerCompliance,
    detections : list[Detection]
) -> WorkerCompliance:
    """
    R2 — High-Visibility Vest Required.
    Checks the torso region (15%–65% of worker box height).
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
    elif low_conf:
        worker.r2_vest = RuleResult.LOW_CONF
        worker.violations.append(WorkerViolation(
            rule_id="R2",
            rule_name="Hi-Vis Vest Required",
            description="Vest detected but below confidence threshold — "
                        "treated as potential violation.",
        ))
    elif no_vest_matches:
        worker.r2_vest = RuleResult.VIOLATION
        worker.violations.append(WorkerViolation(
            rule_id="R2",
            rule_name="Hi-Vis Vest Required",
            description="Worker detected without a high-visibility vest.",
        ))
    else:
        worker.r2_vest = RuleResult.VIOLATION
        worker.violations.append(WorkerViolation(
            rule_id="R2",
            rule_name="Hi-Vis Vest Required",
            description="No vest detected in torso region of worker.",
        ))

    return worker


def evaluate_r3_full_ppe(worker: WorkerCompliance) -> WorkerCompliance:
    """
    R3 — Full PPE Compliance.
    Derived rule: passes only if both R1 (helmet) and R2 (vest) pass.
    No new detections needed — uses results already computed.
    """
    if (worker.r1_helmet == RuleResult.PASS and
            worker.r2_vest == RuleResult.PASS):
        worker.r3_full_ppe = RuleResult.PASS
    else:
        worker.r3_full_ppe = RuleResult.VIOLATION

    return worker


def evaluate_r4_boots(
    worker     : WorkerCompliance,
    detections : list[Detection]
) -> WorkerCompliance:
    """
    R4 — Appropriate Safety Footwear Required.
    Only evaluated when foot region is visible (bottom 20% of worker box).
    Worker box must be tall enough to resolve foot-level detail.
    """
    _, y1, _, y2 = worker.box
    worker_height = y2 - y1

    # Only evaluate if worker box is tall enough to see feet
    # Threshold: worker box must be at least 150px tall
    if worker_height < 150:
        worker.r4_boots = RuleResult.NOT_EVALUABLE
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
        worker.r4_boots = RuleResult.PASS
    elif no_boots_matches:
        worker.r4_boots = RuleResult.VIOLATION
        worker.violations.append(WorkerViolation(
            rule_id="R4",
            rule_name="Safety Footwear Required",
            description="Non-compliant footwear detected — "
                        "trainers, slippers, or bare feet.",
        ))
    elif low_conf:
        worker.r4_boots = RuleResult.LOW_CONF
        worker.violations.append(WorkerViolation(
            rule_id="R4",
            rule_name="Safety Footwear Required",
            description="Non-compliant footwear detected but below confidence threshold — "
                        "treated as potential violation.",
        ))
    else:
        worker.r4_boots = RuleResult.NOT_EVALUABLE

    return worker


def evaluate_r5_harness(
    worker     : WorkerCompliance,
    detections : list[Detection]
) -> WorkerCompliance:
    """
    R5 — Safety Harness Required at Height.
    A confirmed NO_SAFETY_HARNESS detection triggers a violation.
    Absence of any detection → NOT_EVALUABLE (cannot confirm harness presence).
    """
    no_harness_matches = find_ppe_for_worker(
        worker.box, detections,
        target_classes=[CLASS_NO_SAFETY_HARNESS],
        region_upper=TORSO_REGION_UPPER,
        region_lower=TORSO_REGION_LOWER,
    )
    low_conf = find_low_confidence_ppe(
        worker.box, detections,
        target_classes=[CLASS_NO_SAFETY_HARNESS],
        region_upper=TORSO_REGION_UPPER,
        region_lower=TORSO_REGION_LOWER,
    )

    if no_harness_matches:
        worker.r5_harness = RuleResult.VIOLATION
        worker.violations.append(WorkerViolation(
            rule_id="R5",
            rule_name="Safety Harness Required",
            description="Worker detected without a safety harness.",
        ))
    elif low_conf:
        worker.r5_harness = RuleResult.LOW_CONF
        worker.violations.append(WorkerViolation(
            rule_id="R5",
            rule_name="Safety Harness Required",
            description="No-harness indicator detected but below confidence threshold — "
                        "treated as potential violation.",
        ))
    else:
        # No negative signal detected — cannot confirm presence or absence
        worker.r5_harness = RuleResult.NOT_EVALUABLE

    return worker

def evaluate_r6_gloves(
    worker     : WorkerCompliance,
    detections : list[Detection]
) -> WorkerCompliance:
    """
    R6 — Protective Gloves Required for Manual Handling.
    Only evaluated when hand region is visible.
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
        worker.r6_gloves = RuleResult.PASS
    elif no_gloves_matches:
        worker.r6_gloves = RuleResult.VIOLATION
        worker.violations.append(WorkerViolation(
            rule_id="R6",
            rule_name="Protective Gloves Required",
            description="Worker hands visible without protective gloves.",
        ))
    elif low_conf:
        worker.r6_gloves = RuleResult.LOW_CONF
        worker.violations.append(WorkerViolation(
            rule_id="R6",
            rule_name="Protective Gloves Required",
            description="Bare hands detected but below confidence threshold — "
                        "treated as potential violation.",
        ))
    else:
        worker.r6_gloves = RuleResult.NOT_EVALUABLE

    return worker


def evaluate_r7_goggles(
    worker     : WorkerCompliance,
    detections : list[Detection]
) -> WorkerCompliance:
    """
    R7 — Protective Eyewear Required.
    Checks the head region for goggles presence or confirmed absence.
    - PASS          : CLASS_SAFETY_GOGGLES detected in head region
    - VIOLATION     : CLASS_NO_SAFETY_GOGGLES detected (confirmed bare eyes)
    - LOW_CONF      : No-goggles detection present but below confidence threshold
    - NOT_EVALUABLE : No signal at all — face not visible or too distant
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
        worker.r7_goggles = RuleResult.PASS
    elif low_conf:
        worker.r7_goggles = RuleResult.LOW_CONF
        worker.violations.append(WorkerViolation(
            rule_id="R7",
            rule_name="Protective Eyewear Required",
            description="No-goggles indicator detected but below confidence threshold — "
                        "treated as potential violation.",
        ))
    elif no_goggles_matches:
        worker.r7_goggles = RuleResult.VIOLATION
        worker.violations.append(WorkerViolation(
            rule_id="R7",
            rule_name="Protective Eyewear Required",
            description="Worker detected without safety goggles or protective eyewear.",
        ))
    else:
        # No signal in either direction — face not visible or too distant
        worker.r7_goggles = RuleResult.NOT_EVALUABLE

    return worker


def evaluate_r8_correct_usage(
    worker     : WorkerCompliance,
    detections : list[Detection]
) -> WorkerCompliance:
    """
    R8 — Correct and Proper PPE Usage.
    Checks whether detected PPE items are positioned in anatomically
    correct regions. A helmet centroid outside the head region or a
    vest centroid outside the torso region is flagged.
    """
    anomalies_found = False

    for det in detections:
        if not det.is_confident:
            continue

        # Check helmet positioning
        if det.class_id == CLASS_HARDHAT:
            fraction = get_ppe_centroid_y_fraction(det.box, worker.box)
            if fraction is not None:
                if fraction > HEAD_REGION_LOWER:
                    anomalies_found = True
                    worker.violations.append(WorkerViolation(
                        rule_id="R8",
                        rule_name="Correct PPE Usage",
                        description=f"Helmet centroid at {fraction:.0%} of worker "
                                    f"height — positioned below expected head region.",
                    ))

        # Check vest positioning
        if det.class_id == CLASS_SAFETY_VEST:
            fraction = get_ppe_centroid_y_fraction(det.box, worker.box)
            if fraction is not None:
                if fraction < TORSO_REGION_UPPER or fraction > TORSO_REGION_LOWER:
                    anomalies_found = True
                    worker.violations.append(WorkerViolation(
                        rule_id="R8",
                        rule_name="Correct PPE Usage",
                        description=f"Vest centroid at {fraction:.0%} of worker "
                                    f"height — positioned outside expected torso region.",
                    ))

    if anomalies_found:
        worker.r8_correct_ppe = RuleResult.VIOLATION
    else:
        worker.r8_correct_ppe = RuleResult.PASS

    return worker


# Worker status and scene verdict logic

def resolve_worker_status(worker: WorkerCompliance) -> WorkerCompliance:
    """
    Determines the final WorkerStatus from the accumulated rule results.

    Priority:
        1. UNVERIFIABLE — if worker confidence is below threshold
        2. VIOLATION    — if any rule returned VIOLATION or LOW_CONF
        3. COMPLIANT    — all evaluated rules passed
    """
    if worker.confidence < CONFIDENCE_THRESHOLD:
        worker.status = WorkerStatus.UNVERIFIABLE
        return worker

    all_results = [
        worker.r1_helmet,
        worker.r2_vest,
        worker.r3_full_ppe,
        worker.r4_boots,
        worker.r5_harness,
        worker.r6_gloves,
        worker.r7_goggles,
        worker.r8_correct_ppe,
    ]

    has_violation = any(
        r in (RuleResult.VIOLATION, RuleResult.LOW_CONF)
        for r in all_results
    )

    if has_violation:
        worker.status = WorkerStatus.VIOLATION
    else:
        worker.status = WorkerStatus.COMPLIANT

    return worker


def resolve_scene_verdict(workers: list[WorkerCompliance]) -> SceneVerdict:
    """
    Derives the scene-level verdict from all worker statuses.

    SAFE         — all workers COMPLIANT
    UNSAFE       — at least one worker in VIOLATION
    UNVERIFIABLE — no violations but at least one worker UNVERIFIABLE
    """
    if not workers:
        return SceneVerdict.UNVERIFIABLE

    statuses = [w.status for w in workers]

    if any(s == WorkerStatus.VIOLATION for s in statuses):
        return SceneVerdict.UNSAFE
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
    person_detections = [
        d for d in detections
        if d.class_id == CLASS_PERSON and d.is_confident
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

        # Apply all rules in sequence
        worker = evaluate_r1_helmet(worker, ppe_detections)
        worker = evaluate_r2_vest(worker, ppe_detections)
        worker = evaluate_r3_full_ppe(worker)
        worker = evaluate_r4_boots(worker, ppe_detections)
        worker = evaluate_r5_harness(worker, ppe_detections)
        worker = evaluate_r6_gloves(worker, ppe_detections)
        worker = evaluate_r7_goggles(worker, ppe_detections)
        worker = evaluate_r8_correct_usage(worker, ppe_detections)
        worker = resolve_worker_status(worker)

        worker_results.append(worker)

    # ── Resolve scene-level verdict ──
    scene_verdict = resolve_scene_verdict(worker_results)

    # ── Assemble and return scene report ──
    compliant_count     = sum(1 for w in worker_results
                              if w.status == WorkerStatus.COMPLIANT)
    violation_count     = sum(1 for w in worker_results
                              if w.status == WorkerStatus.VIOLATION)
    unverifiable_count  = sum(1 for w in worker_results
                              if w.status == WorkerStatus.UNVERIFIABLE)

    return SceneReport(
        image_name         = image_name,
        scene_verdict      = scene_verdict,
        total_workers      = len(worker_results),
        compliant_count    = compliant_count,
        violation_count    = violation_count,
        unverifiable_count = unverifiable_count,
        workers            = worker_results,
    )