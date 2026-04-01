# Safety Rules Definition

## Overview

This document defines the safety rules enforced by the Construction Site PPE Compliance
Detection System. Rules are evaluated per detected worker using the model's output classes.
The model directly detects both the presence and absence of each PPE item, so compliance
is determined by which classes are detected on or near a worker.

**Detection threshold:** Minimum confidence 0.50  
**Annotation format:** YOLO bounding box per PPE item, per worker

---

## Rule R1 — Hard Hat Required

**Requirement:** Every worker on site must wear a hard hat at all times.

**Compliant:** `Hardhat` class detected overlapping the worker's head region.  
**Violation:** `NO-Hardhat` class detected, or no helmet detection associated with the worker.

**Severity:** Critical

---

## Rule R2 — High-Visibility Vest Required

**Requirement:** Every worker must wear a high-visibility safety vest.

**Compliant:** `Safety Vest` class detected overlapping the worker's torso region.  
**Violation:** `NO-Safety Vest` class detected, or no vest detection associated with the worker.

**Severity:** Critical

---

## Rule R3 — Safety Boots Required

**Requirement:** Workers must wear safety boots. Casual shoes, sandals, or bare feet are not permitted.

**Compliant:** `Safety Boots` class detected in the worker's lower body region.  
**Violation:** `NO-Safety Boots` class detected in the lower body region.  
**Not evaluable:** Lower body / foot region not visible in the frame.

**Severity:** High

---

## Rule R4 — Protective Gloves Required

**Requirement:** Workers handling materials, tools, or equipment must wear protective gloves.

**Compliant:** `Safety Gloves` class detected overlapping the worker's hand region.  
**Violation:** `NO-Safety Gloves` class detected in the hand region.  
**Not evaluable:** Hands not visible in the frame.

**Severity:** High

---

## Rule R5 — Safety Goggles Required

**Requirement:** Workers must wear safety goggles when operating in environments with dust, debris, sparks, or chemical exposure.

**Compliant:** `Safety Goggles` class detected overlapping the worker's facial region.  
**Violation:** `NO-Safety Goggles` class detected in the facial region.  
**Not evaluable:** Face not sufficiently visible or too distant to resolve.

**Severity:** High

---

## Rule R6 — Full Basic PPE Compliance

**Requirement:** Every worker must simultaneously satisfy R1 (hard hat) and R2 (hi-vis vest) at minimum. This is the baseline compliance check applied to every person detected on site.

**Compliant:** Both `Hardhat` and `Safety Vest` detected on the worker.  
**Violation:** Either R1 or R2 is violated.

**Severity:** Critical

---

## Notes on Detection Behaviour

- **Positive classes** (`Hardhat`, `Safety Vest`, `Safety Boots`, `Safety Gloves`, `Safety Goggles`) confirm PPE is worn correctly.
- **Negative classes** (`NO-Hardhat`, `NO-Safety Vest`, `NO-Safety Boots`, `NO-Safety Gloves`, `NO-Safety Goggles`) are explicitly trained signals that a worker is missing that item. A violation is raised when either a negative class is detected or when no corresponding positive class is detected for a visible body region.

---

## Scene-Level Verdict

After evaluating all detected workers:

| **SAFE** | All detected workers are compliant with all applicable rules. |
| **UNSAFE** | One or more workers are in violation of any rule. |
| **UNVERIFIABLE** | One or more workers are partially occluded or below the confidence threshold. Treated as a potential violation. |

## Future Extensions (Post-Core Development)

The following rules are planned for implementation after the core detection pipeline is
complete and validated:

- **R9 — Zone-Based PPE Rules:** Stricter or additional PPE requirements enforced within
  designated high-risk spatial zones (e.g., proximity to heavy machinery, excavation zones).
- **R10 — Temporal Behaviour Analysis:** Detection of recurring or sustained violations
  across multiple video frames, enabling pattern-based alerting.

---
