import shutil
import random
import yaml
from pathlib import Path

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATASET_PATH = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH      = PROJECT_ROOT / "data" / "processed"

# Define the split ratios for training, validation, and testing
TRAIN_SPLIT = 0.70
VAL_SPLIT   = 0.20
TEST_SPLIT  = 0.10

RANDOM_SEED = 42

# 12-class unified scheme (indices match data/raw/dataset.yaml)
CLASS_NAMES = [
    "Person",           # 0
    "Hardhat",          # 1
    "Safety Vest",      # 2
    "Safety Boots",     # 3
    "Safety Gloves",    # 4
    "Safety Goggles",   # 5
    "NO-Hardhat",       # 6
    "NO-Safety Vest",   # 7
    "NO-Safety Boots",  # 8
    "NO-Safety Gloves", # 9
    "NO-Safety Goggles",# 10
    "Mask",             # 11
]

NUM_CLASSES = len(CLASS_NAMES)  # 11


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

def read_label_file(label_path: Path) -> list[str]:
    """
    Reads a YOLO label file and returns its non-empty lines.
    Class indices are already in the unified 12-class scheme (no remapping needed).
    """
    lines = []
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
    return lines

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
    output_path.mkdir(parents=True, exist_ok=True)
    yaml_path = output_path / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
    print(f"[INFO] dataset.yaml written to: {yaml_path}")

#Main Pipeline
def run_preparation_pipeline() -> None:

    print("[INFO] Starting data preparation pipeline...")
    random.seed(RANDOM_SEED)

    # Step 1: Collect all pairs from the pre-merged raw dataset
    print("[INFO] Collecting pairs from raw dataset...")
    raw_pairs = collect_image_label_pairs(RAW_DATASET_PATH)
    print(f"[INFO] Raw dataset pairs: {len(raw_pairs)}")

    # Step 2: Validate and collect entries (classes already in unified scheme)
    all_entries = []
    for img_path, label_path in raw_pairs:
        lines = read_label_file(label_path)
        valid = [ln for ln in lines if validate_annotation_line(ln)]
        if valid:
            all_entries.append((img_path, valid))

    print(f"[INFO] Total valid entries: {len(all_entries)}")

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