# 🏗️ Construction Site PPE Compliance Monitor

> **Protective Equipment detection and safety compliance
> verification for construction site environments using YOLOv8.**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Prerequisites](#2-prerequisites)
3. [Setup Instructions](#3-setup-instructions)
4. [Repository Structure](#4-repository-structure)
5. [Dataset Documentation](#5-dataset-documentation)
6. [Model Architecture](#6-model-architecture)

---

## 1. Project Overview

### What This System Does

This system takes a single image, a directory of images, or a video file
from a construction site and automatically determines whether each visible
worker is wearing the required Personal Protective Equipment (PPE).

For every input frame it produces:

- **Annotated output image** — colour-coded bounding boxes per worker
- **Scene-level verdict** — `SAFE`, `VIOLATION`,`ALERT`, and  `UNVERIFIABLE`
- **Structured JSON report** — per-worker rule results and violation descriptions

### Safety Rules Enforced

| Rule | Requirement | Severity |
|------|-------------|----------|

| R1 | Hard Hat Required | Critical |
| R2 | High-Visibility Vest Required | Critical |
| R3 | Safety Boots Required| High |
| R4 | Protective Gloves Required | High |
| R5 | Safety Goggles Required| High |
| R6 | Full Basic PPE Compliance | Critical |

> Full rule definitions with violation examples:
> [`docs/safety_rules.md`](docs/safety_rules.md)

### Approach Summary

```text
COCO Pretrained YOLOv8n Weights
            ↓
  Fine-tuned on merged PPE dataset
  (4 sources, 11 classes, 3500 images)
            ↓
  Post-processing Compliance Engine
  (Rule evaluation via bounding box
   overlap against anatomical regions)
            ↓
  Annotated Output + JSON Report
```

---

The system follows a strict separation of concerns across five layers:

---

Data Ingestion → Dataset Pipeline → Model Detection → Compliance Engine → Output & Reporting

---

## 2. Prerequisites

### System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|

| Python | 3.10 | 3.11 |
| RAM | 8 GB | 16 GB |
| GPU (for training) | — | Colab GPU T4 free tire |
| GPU (for inference) | — | Optional — CPU works |

---

## 3. Setup Instructions

### 3.1 — Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/construction-safety-monitor.git
cd construction-safety-monitor
```

### 3.2 — Create and Activate the Conda Environment

```bash
conda create -n ppe-monitor python=3.11 -y
conda activate ppe-monitor
```

### 3.3 — Install Dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt` contents:**

ultralytics>=8.0.0
opencv-python>=4.8.0
roboflow>=1.1.0
matplotlib>=3.7.0
Pillow>=10.0.0
numpy>=1.24.0
pandas>=2.0.0
PyYAML>=6.0
imagehash>=4.3.0

### 3.4 — Download Model Weights

Pre-trained weights are available in the `models/` directory of this
repository. If they are not present, download them from the Colab
training notebook output or re-train using the instructions below.

Verify the weights file is in place:

```bash
# Expected location
models/best.pt
```

---

### 3.5 — Run Inference

#### On a Single Image

```bash
python src/inference.py --source path/to/image.jpg
```

#### On a Directory of Images

```bash
python src/inference.py --source path/to/images/
```

#### On a Video File

```bash
python src/inference.py --source path/to/video.mp4
```

#### With a Custom Weights File

```bash
python src/inference.py --source path/to/image.jpg --weights models/best.pt
```

#### Display Output Window During Inference

```bash
python src/inference.py --source path/to/image.jpg --show
```

**Output locations:**

| Output Type | Location |
|-------------|----------|

| Annotated images / video | `outputs/sample_predictions/` |
| JSON violation reports | `outputs/reports/` |

---

### 3.6 — Run Evaluation on Test Set

```bash
python src/evaluate.py --weights models/best.pt --split test
```

**Evaluation outputs:**

| Output | Location |
|--------|----------|

| Per-class mAP chart | `outputs/evaluation/per_class_map50.png` |
| Precision/Recall chart | `outputs/evaluation/precision_recall_per_class.png` |
| Summary table | `outputs/evaluation/evaluation_summary_table.png` |
| Full JSON report | `outputs/evaluation/evaluation_report.json` |
| Failure case images | `outputs/evaluation/failure_cases/` |

---

### 3.7 — Re-train the Model (Google Colab)

1. Open the training notebook:
   [`notebooks/construction_safety_training.ipynb`](notebooks/construction_safety_training.ipynb)
2. Set the Colab runtime to **T4 GPU**:
   `Runtime → Change runtime type → T4 GPU`
3. Run all cells sequentially
4. Training takes approximately **60–90 minutes** on a T4 GPU
5. Best weights are saved automatically to `models/best.pt`

---

## 4. Repository Structure

```text
construction-safety-monitor/
│
├── data/
│   ├── raw/                                            
│   └── processed/                  
│       ├── images/
│       │   ├── train/              
│       │   ├── val/                
│       │   └── test/               
│       ├── labels/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── dataset.yaml            
│
├── src/
│   ├── data_preparation.py         
│   ├── train.py                   
│   ├── compliance.py              
│   ├── inference.py                
│   └── evaluate.py                
│
├── notebooks/
│   └── training.ipynb   
│
├── models/
│   └── best.pt                     
│
├── outputs/
│   ├── sample_predictions/        
│   ├── reports/                    
│   └── evaluation/                 
│       └── failure_cases/          
│
├── docs/
│   ├── safety_rules.md             
│   └── dataset.md                  
│
├── requirements.txt
└── README.md
|__ TECHNICAL_NOTES.md
```

---

## 5. Dataset Documentation

> Full dataset documentation including class distribution, annotation
> [`docs/dataset.md`](docs/dataset.md)

---

## 6. Model Architecture

### Model

| Property | Value |
|----------|-------|

| Architecture | YOLOv8n (nano) |
| Framework | Ultralytics YOLOv8 |
| Pretrained Weights | COCO (80-class general detection) |
| Parameters | ~3.2M |
| Input Resolution | 640 × 640 |
| Task | Multi-class object detection |

### Why YOLOv8

YOLOv8 was selected for the following reasons:

- **Single-pass detection** — detects all objects in one forward pass,
  making it suitable for real-time or near-real-time site monitoring
- **COCO pretraining** — the `person` class is already well-learned,
  providing a strong starting point for worker detection
- **Ultralytics API** — clean, well-documented Python interface that
  separates training, validation, and inference cleanly

### Training Configuration

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|

| Epochs | 100 (early stopping) | Sufficient for convergence |
| Image size | 640 × 640 | Standard YOLO input |
| Batch size | 16 | Optimal for Colab T4 GPU |
| Optimizer | AdamW | Better convergence than SGD for fine-tuning |
| Initial LR | 0.001 | Conservative for pretrained weights |
| Early stopping patience | 20 epochs | Prevents overfitting |
| Mosaic augmentation | Enabled | Improves small object detection |
| Horizontal flip | 0.5 | Natural for construction site images |
| Vertical flip | Disabled | Unnatural for site images |
