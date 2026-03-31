"""Merges the base Roboflow dataset and the custom annotated dataset into a
single unified training-ready structure for YOLOv8."""

import os
import shutil
import random
import yaml
from pathlib import Path

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent

BASE_DATASET_PATH = PROJECT_ROOT/ "data"/"raw"/"base_dataset"
GOGGLES_DATASET_PATH  = PROJECT_ROOT / "data" / "raw" / "goggles_dataset"
FOOTWEAR_DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "footwear_dataset"
PPE_DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "ppe_dataset"
GLOVES_DATASET_PATH   = PROJECT_ROOT / "data" / "raw" / "gloves_dataset"

OUTPUT_PATH = PROJECT_ROOT/ "data"/"processed"

# Define the split ratios for training, validation, and testing
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.20
TEST_SPLIT = 0.10

RANDOM_SEED = 42

# Name class list

CLASS_NAMES = [
    "Person", 
    "Hardhat",
    "NO-Hardhat",
    "Safety Vest",
    "NO-Safety Vest",
    "Safety Gloves",
    "NO-Safety Gloves",
    "Safety Boots",
    "NO-Safety Boots",
    "Safety Goggles",
    "NO-Safety Goggles",
    "Safety Harness",
    "NO-Safety Harness",
]

# Class remapping for YOLOv8 (0-indexed)

BASE_CLASS_MAPPING = {
    5:0,  # Person
    0:1,  # Hardhat
    2:2,  # NO-Hardhat
    7:3,  # Safety Vest
    4:4,  # NO-Safety Vest

}

GOGGLES_CLASS_MAPPING = {
    0:9,    # Goggles 
   1:10,    # NO-Goggles 
}

SAFETY_PPE_MAPPING = {
    1: 5,   # No_Glove            
    2: 10,   # No_Goggles            
    3: 12,   # No_Harness           
    4: 2,    # No_Helmet             
    5: 8,    # No_Shoe              
    6: 0,    # Person                   
}

GLOVES_CLASS_MAPPING = {
    0: 5,    #  Safety Gloves   
    1: 6,   # NO-Gloves 
}

FOOTWEAR_CLASS_REMAP = {
    1: 7,    # Safety Boots
   
}


    
def collect_image_label_pairs(dataset_path: Path) -> list[tuple[Path, Path]]:
    """
    Recursively collects all (image, label) path pairs from a dataset folder.
    Supports both flat and split (train/val/test) folder structures.
    Returns a list of (image_path, label_path) tuples.

    """
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".avif"}
    pairs = []

    images_dirs = list(dataset_path.rglob("images"))
    if not images_dirs:
        # Flat structure — images at root
        images_dirs = [dataset_path]

    for images_dir in images_dirs:
        labels_dir = Path(str(images_dir).replace("images", "labels"))
        if not labels_dir.exists():
            continue
        for img_path in images_dir.iterdir():
            if img_path.suffix.lower() in image_extensions:
                label_path = labels_dir / (img_path.stem + ".txt")
                if label_path.exists():
                    pairs.append((img_path, label_path))

    return pairs

# Label remapping function
def remap_label_file(
    label_path: Path,
    class_remap: dict[int, int]
) -> list[str]:
    """
    Reads a YOLO label file and remaps class indices using the provided mapping.
    Returns the remapped lines as a list of strings.
    Skips lines with class indices not present in the remap table.
    """
    remapped_lines = []
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            original_class = int(parts[0])
            if original_class not in class_remap:
                continue
            new_class = class_remap[original_class]
            remapped_lines.append(f"{new_class} {' '.join(parts[1:])}")
    return remapped_lines

# Validation function for annotation lines
def validate_annotation_line(line: str) -> bool:
    """
    Validates a single YOLO annotation line.
    Checks:
        - Correct number of fields (5: class x y w h)
        - Bounding box values are within [0.0, 1.0]
        - Width and height are greater than zero
    Returns True if valid, False otherwise.
    """
    parts = line.strip().split()
    if len(parts) != 5:
        return False
    try:
        _, x_center, y_center, width, height = map(float, parts)
    except ValueError:
        return False
    if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
        return False
    if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
        return False 
    return True

# Function to copy image and write remapped label
def copy_pair_to_split(
    img_path: Path,
    label_lines: list[str],
    split: str,
    output_path: Path,
    index: int
) -> None:
    """
    Copies an image and writes its remapped label to the correct split folder.
    Renames files using a zero-padded index to avoid name collisions.
    """
    img_dest_dir   = output_path / "images" / split
    label_dest_dir = output_path / "labels" / split

    img_dest_dir.mkdir(parents=True, exist_ok=True)
    label_dest_dir.mkdir(parents=True, exist_ok=True)

    new_stem = f"img_{index:06d}"
    img_dest   = img_dest_dir   / (new_stem + img_path.suffix.lower())
    label_dest = label_dest_dir / (new_stem + ".txt")

    shutil.copy2(img_path, img_dest)

    with open(label_dest, "w") as f:
        f.write("\n".join(label_lines))


def generate_dataset_yaml(output_path: Path, class_names: list[str]) -> None:
    """
    Generates the dataset.yaml configuration file required by YOLOv8.
    """
    yaml_content = {
        "path"  : str(output_path.resolve()),
        "train" : "images/train",
        "val"   : "images/val",
        "test"  : "images/test",
        "nc"    : len(class_names),
        "names" : class_names,
    }
    yaml_path = output_path / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
    print(f"[INFO] dataset.yaml written to: {yaml_path}")

#Main Pipeline
def run_preparation_pipeline() -> None:

    print("[INFO] Starting data preparation pipeline...")
    random.seed(RANDOM_SEED)

    # Step 1: Collect all pairs from both sources
    print("[INFO] Collecting pairs from base dataset...")
    base_pairs = collect_image_label_pairs(BASE_DATASET_PATH)
    print(f"[INFO] Base dataset pairs       : {len(base_pairs)}")

    print("[INFO] Collecting pairs from goggles dataset...")
    goggles_pairs = collect_image_label_pairs(GOGGLES_DATASET_PATH)
    print(f"[INFO] Goggles dataset pairs    : {len(goggles_pairs)}")

    print("[INFO] Collecting pairs from footwear dataset...")
    footwear_pairs = collect_image_label_pairs(FOOTWEAR_DATASET_PATH)
    print(f"[INFO] Footwear dataset pairs   : {len(footwear_pairs)}")

    print("[INFO] Collecting pairs from ppe dataset...")
    ppe_pairs = collect_image_label_pairs(PPE_DATASET_PATH)
    print(f"[INFO] PPE dataset pairs   : {len(ppe_pairs)}")

    print("[INFO] Collecting pairs from gloves dataset...")
    gloves_pairs = collect_image_label_pairs(GLOVES_DATASET_PATH)
    print(f"[INFO] Gloves dataset pairs     : {len(gloves_pairs)}")

# Step 2: Remap, validate, and collect all entries 
    all_entries = []

    # Base dataset
    for img_path, label_path in base_pairs:
        remapped = remap_label_file(label_path, BASE_CLASS_MAPPING)
        valid    = [ln for ln in remapped if validate_annotation_line(ln)]
        if valid:
            all_entries.append((img_path, valid))

    # Goggles dataset (custom source 1)
    for img_path, label_path in goggles_pairs:
        remapped = remap_label_file(label_path, GOGGLES_CLASS_MAPPING)
        valid    = [ln for ln in remapped if validate_annotation_line(ln)]
        if valid:
            all_entries.append((img_path, valid))

    # Footwear dataset (custom source 2)
    for img_path, label_path in footwear_pairs:
        remapped = remap_label_file(label_path, SAFETY_PPE_MAPPING)
        valid    = [ln for ln in remapped if validate_annotation_line(ln)]
        if valid:
            all_entries.append((img_path, valid))

    # PPE dataset (custom source 2)
    for img_path, label_path in ppe_pairs:
        remapped = remap_label_file(label_path, SAFETY_PPE_MAPPING)
        valid    = [ln for ln in remapped if validate_annotation_line(ln)]
        if valid:
            all_entries.append((img_path, valid))

    # Gloves dataset (custom source 3)
    for img_path, label_path in gloves_pairs:
        remapped = remap_label_file(label_path, GLOVES_CLASS_MAPPING)
        valid    = [ln for ln in remapped if validate_annotation_line(ln)]
        if valid:
            all_entries.append((img_path, valid))

    print(f"[INFO] Total valid entries after remapping: {len(all_entries)}")

    # Step 3: Shuffle and split
    random.shuffle(all_entries)
    total      = len(all_entries)
    train_end  = int(total * TRAIN_SPLIT)
    val_end    = train_end + int(total * VAL_SPLIT)

    splits = {
        "train" : all_entries[:train_end],
        "val"   : all_entries[train_end:val_end],
        "test"  : all_entries[val_end:],
    }

    for split_name, entries in splits.items():
        print(f"[INFO] {split_name:5s} split: {len(entries)} images")

    # Step 4: Write to output folders 
    global_index = 0
    for split_name, entries in splits.items():
        for img_path, label_lines in entries:
            copy_pair_to_split(
                img_path, label_lines,
                split_name, OUTPUT_PATH,
                global_index
            )
            global_index += 1

    print(f"[INFO] All {global_index} image-label pairs written to {OUTPUT_PATH}")

    # Step 5: Generate dataset.yaml
    generate_dataset_yaml(OUTPUT_PATH, CLASS_NAMES)

    print("[INFO] Data preparation pipeline complete.")


if __name__ == "__main__":
    run_preparation_pipeline()