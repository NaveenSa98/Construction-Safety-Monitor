# Technical Notes — Evaluation, Design & Limitations

> This document covers the analytical and decision-making aspects of the
> Construction Site PPE Compliance Monitor. It is intended to be read
> alongside [`README.md`](README.md).

---

## Table of Contents

1. [Evaluation Results](#1-evaluation-results)
2. [Design Decisions](#2-design-decisions)
3. [Known Limitations & Failure Cases](#3-known-limitations--failure-cases)

---

## 1. Evaluation Results

All metrics were computed on the **held-out test split (10%)** which was
never seen during training or hyperparameter selection.

### 1.1 — Overall Performance

| Metric | Score | Assessment |
|--------|-------|------------|

| **mAP@0.5** | **0.816** | Strong — well above the 0.5 baseline threshold |
| **mAP@0.5:0.95** | **0.639** | Good — indicates reliable detection across IoU thresholds |
| **Precision** | **0.932** | Excellent — very low false positive rate |
| **Recall** | **0.677** | Moderate — some violations are missed (see §3) |

> **On the Precision–Recall trade-off:**
> For a safety monitoring system, **recall is the more safety-critical
> metric**. A missed violation (false negative) is more dangerous than
> a false alarm (false positive). The current recall of 0.677 is the
> primary area identified for improvement in future iterations.
> This is documented honestly rather than obscured by the high precision
> figure.

---

### 1.2 — Per-Class Performance

| Class | Precision | Recall | mAP@0.5 | Notes |
|-------|-----------|--------|---------|-------|

| Person | 0.926 | 0.796 | 0.693 | Strong — core detection class |
| Hardhat | 0.944 | 0.921 | 0.754 | Excellent — R1 well-covered |
| Safety Vest | 0.903 | 0.923 | 0.810 | Excellent — R2 well-covered |
| Safety Boots | 0.895 | 0.832 | 0.717 | Good — R4 functional |
| Safety Gloves | 0.960 | 0.771 | 0.718 | Good — R6 functional |
| Safety Goggles | 0.848 | 0.917 | 0.700 | Good — R7 functional |
| Mask | 0.989 | 0.868 | 0.724 | Good — incidental class |
| NO-Hardhat | 0.923 | 0.462 | 0.512 | ⚠️ Low recall — violations missed |
| NO-Safety Vest | 0.933 | 0.452 | 0.570 | ⚠️ Low recall — violations missed |
| NO-Safety Boots | 1.000 | 0.333 | 0.512 | ⚠️ Very low recall |
| NO-Safety Gloves | 1.000 | 0.588 | 0.594 | Moderate — improving |

---

### 1.3 — Performance Charts

The following charts are generated automatically by `src/evaluate.py`
and saved to `outputs/evaluation/`:

| Chart | File | What It Shows |
|-------|------|---------------|

| Per-class mAP | `per_class_map50.png` | mAP@0.5 bar chart per class |
| Precision vs Recall | `precision_recall_per_class.png` | Side-by-side comparison per class |
| Summary table | `evaluation_summary_table.png` | Full metrics table with colour coding |
| Failure cases | `failure_cases/` | Annotated worst-case predictions |

---

## 2. Design Decisions

### 2.1 — Why Object Detection, Not Classification

The assignment requires per-worker compliance evaluation, which means
the system must localise individual workers and assess each one
independently. Image classification produces a single scene-level label
and cannot attribute violations to specific individuals.

Object detection with bounding boxes was the only viable approach for
per-worker rule evaluation.

---

### 2.2 — Why YOLOv8 Over Alternatives

| Alternative | Reason Not Chosen |
|-------------|-------------------|

| Faster R-CNN | Two-stage detector — slower inference, more complex |
| SSD | Lower accuracy on small objects than YOLOv8 |
| Custom CNN | No time benefit, far lower accuracy without pretraining |
| YOLOv5 | Superseded by YOLOv8; same API, worse metrics |
| YOLOv9 / v10 | Marginal gain insufficient to justify additional complexity |

YOLOv8n was selected specifically for its balance of speed and accuracy.
The nano variant is sufficient for this task given the resolution of
construction site imagery and the relatively large size of most PPE items.

---

### 2.3 — Why a Separate Compliance Logic Layer

A natural alternative would have been to train a classifier that directly
outputs "SAFE" or "UNSAFE" per worker. This was rejected for the following
reasons:

- **Interpretability** — a rule engine produces explicit, human-readable
  violation descriptions. A classifier produces only a label.
- **Rule extensibility** — adding a new safety rule requires adding one
  function to `compliance.py`, not retraining a model.
- **Auditability** — every compliance decision can be traced back to a
  specific bounding box, a confidence score, and a named rule.
- **Conservative failure handling** — the rule engine can distinguish
  between `VIOLATION`, `NOT_EVALUABLE`, and `LOW_CONFIDENCE` states,
  giving operators richer information than a binary output.

---

### 2.4 — Data Strategy: Multi-Source Merging

Merges four independently downloaded YOLO-format datasets into a single unified dataset of 3500 images, normalised to a deduplicated, validated, and split into train/val/test partitions using python script.

---

## 3. Known Limitations & Failure Cases

Five categories of failure were identified through error analysis
on the test set. Annotated examples are saved in
`outputs/evaluation/failure_cases/`.

---

### Failure Case Type 1 — Small / Distant Workers

**Description:** Workers far from the camera appear as small bounding
boxes (under ~80px height). At this scale, PPE items such as helmets
and vests are too small for reliable detection.

**Impact:** False negatives on PPE detection, leading to incorrect
violation flags for compliant workers.

**Mitigation considered:** Increasing inference image size to 1280×1280
improves small object recall at the cost of inference speed.

---

### Failure Case Type 2 — Poor Lighting Conditions

**Description:** Images taken in low light, strong shadow, or
artificial indoor lighting cause the model to miss detections or
produce low-confidence bounding boxes.

**Impact:** PPE items in shadow are frequently missed, generating
false violations.

**Mitigation considered:** Additional augmentation targeting brightness
and contrast variation, or dedicated low-light training examples.

---

### Failure Case Type 3 — Low Resolution / Blurry Images

**Description:** Images with resolution below approximately 480×480
or with motion blur produce unreliable detections across all classes.

**Impact:** Confidence scores drop below the 0.50 threshold,
resulting in `UNVERIFIABLE` workers rather than definitive verdicts.

**Mitigation considered:** Image quality pre-screening at the
ingestion layer — reject or flag frames below a resolution threshold.

---

### Failure Case Type 4 — Worker Occlusion

**Description:** Workers partially obscured by machinery, scaffolding,
or other workers are frequently detected with incomplete bounding boxes.
PPE overlap calculations against partial worker boxes produce
unreliable results.

**Impact:** Both false positives and false negatives depending on
which part of the worker is occluded.

**Mitigation considered:** The compliance engine conservatively marks
heavily occluded workers as `UNVERIFIABLE`, preventing false compliance
verdicts. This is the correct behaviour.

---

### Failure Case Type 5 — Backward-Facing Workers

**Description:** Workers facing away from the camera do not expose
their face, helmet front, or vest front to the detection model.
The model was predominantly trained on forward-facing workers.

**Impact:** Helmet and vest detections drop significantly for
backward-facing workers, generating false violations.

**Mitigation considered:** Augmenting the training set with explicitly
backward-facing worker images would improve robustness to this case.

---

**Struggles during the training**
The model works well for the common and clear classes like helmets and vests, especially after I merged multiple datasets and increased the total size to around 3500 images. When the objects are clearly visible and the annotations are good, the model can detect them reasonably well.

But it struggles a lot with smaller or less visible items like gloves, boots, and goggles. In my first attempt with around 800+ images, some classes even showed 0 instances after training. That was mainly because of poor annotation quality and low image quality. The bounding boxes were not accurate, and some objects were too small or unclear.

Even after improving the dataset by merging and cleaning it, there are still issues. Some classes have very low recall, precision, and mAP50. This is mostly because of class imbalance and inconsistent annotations across datasets. Also, training sometimes took a long time and even failed, which showed that the dataset and setup were not stable enough.

What I learned from this is that data quality is more important than just having more data. Proper annotation (tight and correct bounding boxes) is very important. Also, small objects need more attention, better images, and maybe more focused data collection.

Overall, the model is okay for basic detection, but not reliable yet for all safety equipment, especially the smaller ones.
