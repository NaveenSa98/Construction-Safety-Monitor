"""
YOLOv8 fine-tuning script for PPE compliance detection

"""
from pathlib import Path
from ultralytics import YOLO
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_YAML = PROJECT_ROOT / "data" / "processed" / "dataset.yaml"

PRETRAINED_WEIGHTS = "yolov8m.pt" # Medium model 
PROJECT_DIR = PROJECT_ROOT / "outputs" / "training_runs"
RUN_NAME = "ppe_detection_v2"

# Training configuration

TRAINING_CONFIG = {
    "data"            : str(DATASET_YAML),
    "epochs"          : 100,
    "imgsz"           : 640,        
    "batch"           : -1,         # Auto-select batch
    "patience"        : 30,         # Allow more room before early stopping fires
    "pretrained"      : True,
    "optimizer"       : "AdamW",
    "lr0"             : 0.001,     
    "lrf"             : 0.1,        # Final LR = lr0 × lrf → 0.0001; gentler decay than 0.01
    "cos_lr"          : True,       # Cosine LR schedule — smoother decay, better final mAP
    "weight_decay"    : 0.0005,
    "warmup_epochs"   : 3,
    "momentum"        : 0.937,      # Adam beta1
    "mosaic"          : 1.0,        # Mosaic augmentation — critical for small-object detection
    "close_mosaic"    : 10,         # Disable mosaic in final 10 epochs for clean refinement
    "flipud"          : 0.0,        # No vertical flip — unnatural for site images
    "fliplr"          : 0.5,        # Horizontal flip
    "hsv_h"           : 0.015,      # Hue shift — handles varied lighting on site
    "hsv_s"           : 0.7,        # Saturation shift
    "hsv_v"           : 0.4,        # Brightness shift — important for indoor/outdoor variation
    "degrees"         : 5.0,        # Slight rotation — workers lean/tilt slightly
    "translate"       : 0.1,        # Translation augmentation
    "scale"           : 0.5,        # Scale variation
    "amp"             : True,    
    "cache"           : True,   
    "save_period"     : 10,        
    "seed"            : 42,       
    "project"         : str(PROJECT_DIR),
    "name"            : RUN_NAME,
    "exist_ok"        : True,
    "verbose"         : True,
    "device"          : 0,         
    "workers"         : 2,         
    "val"             : True,
    "save"            : True,
    "plots"           : True,       # Auto-generate training curve plots
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


