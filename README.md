# Construction Site PPE Compliance Monitoring System

A computer vision system that detects construction workers in images or video and determines whether each worker is wearing the required Personal Protective Equipment (PPE). The system returns a per-worker compliance status and a scene-level verdict of **SAFE**, **ALERT**, or **UNSAFE**.

---

## Project Structure

```text
Construction-Safety-Monitor/
├── data/
│   └── raw/                  # Roboflow-exported dataset (YOLOv8 format)
│       ├── train/
│       ├── valid/
│       ├── test/
│       └── data.yaml
├── models/
│   └── best.pt               # Trained weights (copied from Colab)
├── notebooks/
│   └── training.ipynb        # Colab training notebook
├── outputs/                  # Evaluation reports, annotated results
├── src/
│   ├── data_preparation.py   # Dataset validation
│   ├── train.py              # Model training (Colab)
│   ├── compliance.py         # PPE rule logic
│   ├── inference.py          # End-to-end inference pipeline
│   └── evaluate.py           # Formal model evaluation
└── docs/
    ├── safety_rules.md       # Rule definitions (R1–R5, R7)
    └── dataset.md            # Dataset provenance and class distribution
```

---

## Quick Start — Local Inference

### 1. Install dependencies

```bash
pip install ultralytics opencv-python pyyaml
```

### 2. Download trained weights

Copy `best.pt` from your Google Drive into the `models/` folder:

```text
models/best.pt
```

### 3. Run on a single image

```bash
python src/inference.py --weights models/best.pt --source data/raw/test/images/your_image.jpg
```

### 4. Save annotated result

```bash
python src/inference.py --weights models/best.pt --source data/raw/test/images/your_image.jpg --output outputs/result.jpg
```

### 5. Run on a video file

```bash
python src/inference.py --weights models/best.pt --source path/to/video.mp4 --output outputs/result.mp4
```

### 6. Live webcam

```bash
python src/inference.py --weights models/best.pt --source 0
```

---

## Training in Google Colab

### 1. Upload project to Google Drive

Upload the entire project folder to:

```text
MyDrive/Construction-Safety-Monitor/
```

### 2. Open the notebook in VS Code

Open `notebooks/training.ipynb` and connect to a Colab runtime with **T4 GPU**.

### 3. Run all cells

The notebook will:

1. Verify GPU
2. Mount Google Drive
3. Install dependencies
4. Download dataset from Roboflow
5. Fix known label issues
6. Validate dataset
7. Train for 25 epochs
8. Display training curves

Trained weights are saved automatically to:

```text
runs/train/ppe_yolov8s/weights/best.pt
```

---

## Validate Dataset

```bash
python src/data_preparation.py --data data/raw/data.yaml
```

---

## Evaluate Model

```bash
python src/evaluate.py --weights models/best.pt --data data/raw/data.yaml
```

Results are saved to `outputs/evaluation_report.json`.

---

## Scene Verdict Logic

| Verdict | Condition |
| --- | --- |
| **SAFE** | Every worker has helmet + vest + boots + gloves + goggles |
| **ALERT** | Every worker has helmet + vest + boots, but gloves or goggles missing |
| **UNSAFE** | Any worker is missing helmet, vest, or boots |

---

## Architecture & Design Decisions

### Model — YOLOv8s

YOLOv8s (small) was chosen over larger variants because:

- The dataset is only 500 images — larger models would overfit
- Colab free tier (T4) handles YOLOv8s at batch=16 comfortably
- Inference speed is suitable for real-time video

### compliance.py is decoupled from inference.py

Detection (what objects are in the frame) and compliance (are the rules satisfied) are kept in separate modules. This means the compliance logic can be tested independently and the detection model can be swapped without changing any rule logic.

### IoU-based PPE association with expanded bbox

Each PPE item is assigned to the worker whose bounding box has the highest IoU with the PPE box. The worker bbox is expanded by 15% before IoU computation to handle PPE detected at the edges (e.g. helmet above a bent-forward worker).

### Vertical zone heuristics

Each PPE type is only accepted within a plausible vertical region of the worker's bbox (e.g. helmets in the top 60%, boots in the bottom 40%). This prevents a helmet from a nearby worker being credited to the wrong person.

### Two-tier PPE classification

- **Critical PPE** (helmet, vest, boots) — missing any → UNSAFE
- **Advisory PPE** (gloves, goggles) — missing → ALERT only

This reflects real construction site practice where helmets, vests and boots are mandatory at all times, while gloves and goggles depend on the task.

---

## Known Limitations

| Limitation | Detail |
| --- | --- |
| Small dataset | 500 images is limited. Rare classes (goggles, gloves) have less training data and lower mAP |
| Occlusion | Heavily occluded PPE (e.g. gloves behind the body) will not be detected |
| Bent/crouching workers | Person bbox may not fully contain the head — bbox expansion partially mitigates this |
| No tracking | Each frame is evaluated independently — no temporal smoothing across video frames |
| Single camera angle | Model trained on a specific dataset — performance may vary with different camera angles or lighting |

---

## Per-Class Evaluation Results

Run `evaluate.py` to generate up-to-date results. Expected weak classes based on dataset size:

- **goggles** — underrepresented in training data
- **gloves** — small, frequently occluded
- **boots** — lower body often partially out of frame

See `outputs/evaluation_report.json` for full results after evaluation.
