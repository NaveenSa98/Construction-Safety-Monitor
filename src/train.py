"""
train.py
--------
YOLOv8s transfer-learning training script.
Designed to run in Google Colab (free tier, T4 GPU).

Usage (in Colab):
    !python src/train.py
"""

from pathlib import Path
import sys

from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Configuration — edit these values before running
# ---------------------------------------------------------------------------

DATA_YAML   = "data/raw/data.yaml"   # path to Roboflow-exported data.yaml
WEIGHTS     = "yolov8s.pt"           # COCO-pretrained; auto-downloaded on first run
EPOCHS      = 25                     # 20–30 is sufficient for this dataset size
BATCH       = 16                     # safe default for T4 (15 GB) + YOLOv8s @ 640 px
IMGSZ       = 640                    # standard YOLOv8 input resolution
RUN_NAME    = "ppe_yolov8s"          # weights saved to runs/train/<RUN_NAME>/weights/

# ---------------------------------------------------------------------------


def main():
    data_yaml = Path(DATA_YAML).resolve()
    if not data_yaml.exists():
        print(f"[ERROR] data.yaml not found: {data_yaml}")
        sys.exit(1)

    print("\n" + "=" * 55)
    print("  PPE DETECTION — YOLOv8s TRAINING")
    print("=" * 55)
    print(f"  data     : {data_yaml}")
    print(f"  weights  : {WEIGHTS}")
    print(f"  epochs   : {EPOCHS}")
    print(f"  batch    : {BATCH}")
    print(f"  imgsz    : {IMGSZ}")
    print(f"  run name : {RUN_NAME}")
    print("=" * 55 + "\n")

    # Load COCO-pretrained YOLOv8s — Ultralytics auto-downloads on first run
    model = YOLO(WEIGHTS)

    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMGSZ,
        name=RUN_NAME,
        # Reproducibility
        seed=42,
        deterministic=True,
        # Colab T4 settings
        device=0,       # GPU
        workers=2,      # safe for Colab free tier
        cache=False,    # avoid RAM pressure
        # Saving
        save=True,
        save_period=-1, # save only best + last (no mid-run checkpoints)
        # Logging
        plots=True,     # training curves saved to run directory
        verbose=True,
    )

    best = Path("runs") / "train" / RUN_NAME / "weights" / "best.pt"
    print("\n" + "=" * 55)
    print(f"  Training complete.")
    print(f"  Best weights : {best if best.exists() else f'runs/train/{RUN_NAME}/weights/best.pt'}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
