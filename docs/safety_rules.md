# Safety Rules Definition

## Overview

This document defines the complete set of safety rules enforced by the Construction Site
PPE Compliance Monitoring System. All rules are applied on a per-worker basis using
bounding box overlap analysis between detected workers and detected PPE items.

Every worker detected in a scene is individually evaluated against each applicable rule.
A scene-level verdict is derived from the aggregate compliance status of all workers present.

---

## Confidence & Detection Thresholds

-- Minimum detection confidence         0.50
-- Bounding box overlap (IoU) threshold 0.30

- Detections **below the confidence threshold** are flagged as **LOW CONFIDENCE** and
  treated conservatively as potential violations.
- Workers that are **partially or fully occluded** are flagged as **UNVERIFIABLE** and
  treated conservatively as potential violations.

---

## Rule R1 — Hard Hat / Safety Helmet Required

- **Requirement**         Every worker must wear an approved hard hat at all times.
- **Applies To**          All detected workers in any zone of the construction site.  
- **Violation Condition** A worker is detected whose upper-body region has no sufficiently overlapping helmet bounding box.
- **Severity**            Critical

**Violation Examples:**

- Worker present on site with no helmet visible.
- Helmet detected in the scene but not on the worker's head (e.g., placed on the ground).

---

## Rule R2 — High-Visibility (Hi-Vis) Vest Required

- **Requirement**         Every worker must wear a high-visibility reflective vest.
- **Applies To**          All detected workers in any active work zone.
- **Violation Condition** A worker is detected whose torso region has no sufficiently overlapping hi-vis vest bounding box.
- **Severity**            Critical

**Violation Examples:**

- Worker wearing plain clothing with no visible hi-vis vest.
- Hi-vis vest present in scene but not worn on the worker's torso.

---

## Rule R3 — Full PPE Compliance (Unauthorised / Unequipped Personnel in Hazard Zone)

- **Requirement**          Both a hard hat (R1) AND a hi-vis vest (R2) must be present and correctly worn simultaneously.Any person present on the construction site
- **Applies To**           All persons detected in the scene, not exclusively workers.
- **Violation Condition**  Either R1 or R2 is individually violated.
- **Severity**             Critical

**Violation Examples:**

- This rule exists as an explicit combined check. A worker compliant with R1 but violating
  R2 is still considered non-compliant under R3, and vice versa.
- A visitor or supervisor walking through an active zone with no helmet or vest.
- An unidentified individual present on site with no visible PPE whatsoever.

---

## Rule R4 — Appropriate Safety Footwear Required

- **Requirement**         Workers must wear identifiable safety boots — high-ankle, closed-toe, and non-casual footwear.
- **Applies To**          All detected workers where the lower body / foot region is visible.
- **Violation Condition** A worker's visible footwear is identified as trainers, sandals, open-toed shoes, or bare feet.
- **Severity**            High

**Violation Examples:**

- Worker wearing athletic trainers or casual shoes on an active site.
- Worker in sandals or no footwear.

**Detection Notes:**

- This rule is applied only when the foot/ankle region is sufficiently visible in the frame
  (bounding box lower 20% of the worker detection region).
- If footwear is not visible due to occlusion or camera angle, the rule is marked
  **NOT EVALUABLE** for that worker and does not contribute to a violation flag.

---

## Rule R5 — Fall Protection / Safety Harness Required at Height

- **Requirement**         Workers operating at elevated positions must wear a visible safety harness
- **Applies To**          Workers detected in elevated zones — scaffolding, ladders, elevated platforms, or rooftops
- **Violation Condition** A worker is detected at an elevated position with no safety harness bounding box present on their torso/shoulder region
- **Severity**            Critical

**Violation Examples:**

- Worker on scaffolding with no harness visible.
- Worker on a raised platform wearing only standard PPE (helmet + vest) but no harness.

**Detection Notes:**

- Elevated position is inferred from contextual cues in the image (scaffolding structures,
  ladders, elevated platform geometry) combined with worker vertical position in frame.
- This rule is only activated when sufficient contextual evidence of elevation is present.
  Workers on ground level are excluded from this rule.

---

## Rule R6 — Protective Gloves Required for Manual Handling Tasks

- **Requirement**         Workers engaged in manual handling, material carrying, or tool operation must wear protective gloves.
- **Applies To**          All detected workers where hands are visible and a tool, material, or object is being handled
- **Violation Condition** A worker's hands are visible and actively engaged in a task with no glove bounding box detected overlapping the hand region.
- **Severity**            High

**Violation Examples:**

- Worker carrying construction materials with bare hands
- Worker operating power tools without visible hand protection.
- Worker handling sharp or abrasive materials with unprotected hands.

**Detection Notes:**

- This rule is only evaluated when the hand region is sufficiently visible in the frame.
- If hands are not visible due to occlusion, camera angle, or distance, the rule is marked **NOT EVALUABLE** for that worker.
- Gloves are detected as a distinct PPE class their bounding box must overlap the detected hand/wrist region of the worker.

---

## Rule R7 — Protective Eyewear / Safety Goggles Required

- **Requirement**         Workers operating in environments with airborne debris, dust, sparks, or chemical exposure must wear approved safety goggles or protective eyewear.
- **Applies To**          All detected workers where the facial region is sufficiently visible.
- **Violation Condition** A worker is detected in a high-risk visual environment with no safety goggles or protective eyewear bounding box overlapping their facial/eye region.
- **Severity**            High

**Violation Examples:**

- Worker operating an angle grinder or cutting tool without eye protection.
- Worker in a dusty or debris-heavy environment with no goggles visible.
- Worker handling chemicals or solvents without protective eyewear.

**Detection Notes:**

- Goggles/eyewear are detected as a distinct PPE class overlapping the upper facial region of the worker bounding box.
- This rule is only evaluated when the face is sufficiently visible and forward-facing in the frame.
- If the face is occluded, turned away, or too distant to resolve, the rule is marked **NOT EVALUABLE** for that worker.

---

## Rule R8 — Correct and Proper PPE Usage

- **Requirement**         All PPE must be worn correctly and in a manner consistent with its protective function
- **Applies To**          All detected workers where PPE items are present but their positioning is anomalous.
- **Violation Condition** A helmet or vest bounding box is detected but its spatial position relative to the worker is inconsistent with correct   usage.
- **Severity**            High

**Violation Examples:**

- Hard hat detected but positioned on the chin, back of head, or elsewhere off the crown.
- Hi-vis vest detected but hanging off one shoulder or worn open with no coverage of the torso.
- PPE item detected in the worker's hand rather than worn on their body.

**Detection Notes:**

- Anomalous positioning is determined by comparing the centroid of the PPE bounding box
  against the expected anatomical region of the corresponding body part.
- A helmet centroid falling significantly below the expected head region mid-point is
  flagged as improperly worn.

---

## Scene-Level Verdict

The overall scene verdict is determined after evaluating every detected worker against all
applicable rules.

- **SAFE**          All detected workers are fully compliant with all applicable rules.
- **UNSAFE**        One or more workers are in violation of any one or more rules.
- **UNVERIFIABLE**  One or more workers are occluded or below the confidence threshold such that compliance cannot be confirmed. Treated conservatively as a potential violation.

---

## Future Extensions (Post-Core Development)

The following rules are planned for implementation after the core detection pipeline is
complete and validated:

- **R9 — Zone-Based PPE Rules:** Stricter or additional PPE requirements enforced within
  designated high-risk spatial zones (e.g., proximity to heavy machinery, excavation zones).
- **R10 — Temporal Behaviour Analysis:** Detection of recurring or sustained violations
  across multiple video frames, enabling pattern-based alerting.

---
