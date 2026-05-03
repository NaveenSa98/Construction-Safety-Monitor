# Dataset Documentation

## Overview

| Property | Value |
| --- | --- |
| Total images | 500 |
| Annotation tool | Roboflow |
| Export format | YOLOv8 (images + .txt labels + data.yaml) |
| Number of classes | 7 |
| License | CC BY 4.0 |

---

## Source

Images were sourced and annotated via Roboflow:

- **Workspace:** zukoos-workspace
- **Project:** construction-ppe-fwz4e
- **Version:** 1
- **URL:** <https://universe.roboflow.com/zukoos-workspace/construction-ppe-fwz4e/dataset/1>

---

## Classes

| ID | Class Name | PPE Type | Tier |
| --- | --- | --- | --- |
| 0 | belt | Safety harness/belt | Removed (R6) |
| 1 | boot | Safety boots | Critical |
| 2 | gloves | Safety gloves | Advisory |
| 3 | goggles | Safety goggles | Advisory |
| 4 | helmat | Safety helmet | Critical |
| 5 | person | Worker (person) | — |
| 6 | vest | High-visibility vest | Critical |

> **Note:** The class name `helmat` is a typo in the original Roboflow annotation — it is preserved as-is to match the exported `data.yaml` and trained model weights.

---

## Dataset Split

Splits were created automatically by Roboflow during export.

| Split | Images |
| --- | --- |
| Train | ~400 |
| Validation | ~50 |
| Test | ~50 |

Exact counts can be verified by running:

```bash
python src/data_preparation.py --data data/raw/data.yaml
```

---

## Annotation Approach

- All images were manually annotated using the Roboflow web interface
- Each visible worker and PPE item was labelled with a tight bounding box
- Partially visible objects were annotated if more than 50% of the object was visible
- One image (`image314_jpg.rf...`) contained a segmentation polygon annotation (belt, class 0) which was converted to a bounding box during data preparation

---

## Known Data Limitations

| Class | Limitation |
| --- | --- |
| **goggles** | Underrepresented — goggles are often not worn or not visible, so few positive examples exist |
| **gloves** | Small object, frequently occluded by tools or the worker's body |
| **belt** | Very few annotated examples — rule R6 was removed because of this |
| **boot** | Lower body is often partially out of frame, especially for close-up shots |

These limitations directly affect per-class mAP scores. See `outputs/evaluation_report.json` for quantitative results.

---

## Data Preparation

The `data_preparation.py` script validates the dataset before training:

- Confirms train/valid/test directory structure exists
- Checks image-label parity (no orphan files)
- Validates label format (5 values per line: class cx cy w h)
- Reports class distribution per split
- Flags any malformed annotations

Run it before training to ensure the dataset is clean:

```bash
python src/data_preparation.py --data data/raw/data.yaml
```
