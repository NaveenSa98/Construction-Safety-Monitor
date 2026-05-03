# Safety Rules — PPE Compliance Definitions

This document defines every safety rule applied by `compliance.py` to each detected worker. Rules are evaluated per worker using IoU-based spatial association between the worker bounding box and detected PPE bounding boxes.

---

## Rule Classification

PPE is split into two tiers based on real-world construction site requirements:

| Tier | PPE Items | Effect if missing |
| --- | --- | --- |
| **Critical** | Helmet, Vest, Boots | Worker marked non-compliant → scene is **UNSAFE** |
| **Advisory** | Gloves, Goggles | Worker flagged with alert → scene is **ALERT** (not UNSAFE) |

---

## Rules

### R1 — Safety Helmet (Critical)

Every worker must have a helmet detection spatially associated with their bounding box.

**Violation:** No helmet detection overlaps the worker's bbox within the expected head region (top 60% of person bbox).

**Why critical:** Head injury is the leading cause of fatal accidents on construction sites. A helmet is mandatory at all times in any active work zone.

---

### R2 — Safety Vest (Critical)

Every worker must have a high-visibility vest detection overlapping their torso region.

**Violation:** No vest detection overlaps the worker's bbox within the torso region (10%–90% of person height).

**Why critical:** Visibility vests are required for workers in all active zones to prevent struck-by incidents from plant and machinery.

---

### R3 — Safety Boots (Critical)

Every worker must have safety boots detected at the lower portion of their bounding box.

**Violation:** No boots detection overlaps the worker's bbox within the foot region (bottom 40% of person height).

**Why critical:** Safety boots protect against crush injuries, puncture wounds, and slips on construction surfaces.

---

### R4 — Safety Gloves (Advisory)

Workers should have gloves detected when handling materials or equipment.

**Violation:** No gloves detection associated with the worker.

**Why advisory:** Gloves are task-dependent — not all construction activities require them (e.g. supervision, inspection). A missing gloves detection triggers an ALERT rather than UNSAFE.

**Data limitation:** Gloves are small objects and frequently occluded. Detection recall for this class is lower than for larger PPE items.

---

### R5 — Safety Goggles (Advisory)

Workers should have goggles or eye protection detected in the face region.

**Violation:** No goggles detection associated with the worker's head area.

**Why advisory:** Eye protection is task-specific (grinding, cutting, chemical handling). Not all workers in a scene require goggles simultaneously.

**Data limitation:** Goggles are small, close to the face, and often partially obscured by helmets. This class has the lowest detection rate in the dataset.

---

### R7 — PPE Confidence Gate

All PPE detections must exceed a minimum confidence threshold before being used in rule evaluation. Low-confidence detections are discarded to prevent partial or ambiguous detections from being counted as valid PPE.

**Per-class thresholds:**

| Class | Min Confidence |
| --- | --- |
| helmet | 0.50 |
| vest | 0.45 |
| boots | 0.25 |
| gloves | 0.20 |
| goggles | 0.20 |
| person | 0.50 |

Thresholds are lower for small/difficult classes (gloves, goggles, boots) to reduce false negatives.

---

## R6 — Belt/Harness (Removed)

A safety harness rule was considered but removed because the `belt` class has very limited representation in the training dataset. Enforcing a rule backed by insufficient training data would produce unreliable results. This rule can be re-enabled once more annotated harness examples are available.

---

## Spatial Association Logic

A PPE item is associated with a worker if:

1. **IoU** between the PPE bbox and the worker bbox (expanded by 15%) exceeds 0.10
2. **Vertical zone** check passes — the PPE centre falls within the expected body region for that PPE type
3. **Exclusive assignment** — each PPE item is assigned to the highest-overlap worker only

---

## Scene Verdict

| Verdict | Condition |
| --- | --- |
| **SAFE** | All workers compliant on all rules (critical + advisory) |
| **ALERT** | All workers have critical PPE — advisory PPE missing on at least one worker |
| **UNSAFE** | At least one worker missing any critical PPE (R1, R2, or R3) |
