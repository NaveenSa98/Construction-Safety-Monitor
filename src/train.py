"""
YOLOv8 fine-tuning script for PPE compliance detection

"""
from pathlib import Path
from ultralytics import YOLO
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_YAML = PROJECT_ROOT / "data" / "processed" / "dataset.yaml"

PRETRAINED_WEIGHTS = "yolov8m.pt" # Medium model for better performance on small dataset
PROJECT_DIR = PROJECT_ROOT / "outputs" / "training_runs"
RUN_NAME = "ppe_detection_v2" 

# Training configuration

TRAINING_CONFIG = {
    "data"         : str(DATASET_YAML),
    "epochs"       : 100,       #  00 epochs ensures full convergence
    "imgsz"        : 640,       # Image resolution
    "batch"        : 32,      
    "patience"     : 20,        # Looser early stopping to avoid cutting off late gains
    "pretrained"   : True,
    "optimizer"    : "AdamW",
    "lr0"          : 0.001,     # Initial learning rate
    "lrf"          : 0.01,      # Final learning rate (fraction of lr0)
    "weight_decay" : 0.0005,    # Prevents overfitting by penalizing large weights
    "warmup_epochs": 3,
    "mosaic"       : 1.0,       # Combines multiple images into one
    "close_mosaic" : 10,        # Disable mosaic in final 10 epochs for clean refinement
    "copy_paste"   : 0.2,       # Augments underrepresented classes (Goggles, Boots, Gloves)
    "flipud"       : 0.0,       # No vertical flip (unnatural for site images)
    "fliplr"       : 0.5,       # Horizontal flip
    "hsv_h"        : 0.015,     # Hue augmentation
    "hsv_s"        : 0.7,       # Saturation augmentation
    "hsv_v"        : 0.4,       # Value/brightness augmentation
    "degrees"      : 5.0,       # Slight rotation augmentation
    "translate"    : 0.1,       # Translation augmentation
    "scale"        : 0.5,       # Scale augmentation
    "amp"          : True,      
    "cache"        : "disk",    # Cache decoded images to disk — eliminates per-epoch I/O cost
    "project"      : str(PROJECT_DIR),
    "name"         : RUN_NAME,
    "exist_ok"     : True,
    "verbose"      : True,
    "device"       : 0,         # GPU device 0 — Colab T4
    "workers"      : 4,         # Parallel data loading threads
    "val"          : True,
    "save"         : True,
    "plots"        : True,      # Auto-generate training curve plots
}

# Training pipeline

def run_training_pipeline() -> None:

    print("[INFO] Initialising YOLOv8 model...")
    model = YOLO(PRETRAINED_WEIGHTS)

    print(f"[INFO] Pretrained weights loaded : {PRETRAINED_WEIGHTS}")
    print(f"[INFO] Dataset                   : {DATASET_YAML}")
    print(f"[INFO] Epochs                    : {TRAINING_CONFIG['epochs']}")
    print(f"[INFO] Image size                : {TRAINING_CONFIG['imgsz']}")
    print(f"[INFO] Batch size                : {TRAINING_CONFIG['batch']}")
    print(f"[INFO] Output directory          : {PROJECT_DIR / RUN_NAME}")
    print("[INFO] Starting training...\n")

    results = model.train(**TRAINING_CONFIG)

    print("\n[INFO] Training complete.")
    print(f"[INFO] Best weights saved to : "
          f"{PROJECT_DIR / RUN_NAME / 'weights' / 'best.pt'}")

    # Copy best weights to models
    best_weights_src  = PROJECT_DIR / RUN_NAME / "weights" / "best.pt"
    best_weights_dest = PROJECT_ROOT / "models" / "best.pt"
    best_weights_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights_src, best_weights_dest)
    print(f"[INFO] Best weights copied to  : {best_weights_dest}")

    return results

if __name__ == "__main__":
    run_training_pipeline()


