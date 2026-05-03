"""
data_preparation.py
-------------------
Validates the Roboflow-exported YOLOv8 dataset. Checks directory structure,
label integrity, image/label parity, and class distribution. Read-only — does
not modify any files.

Usage:
    python src/data_preparation.py --data data/raw/data.yaml
    python src/data_preparation.py --data data/raw/data.yaml --report outputs/data_report.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def split_dirs(data_yaml_path: Path, split_image_path: str) -> tuple[Path, Path]:
    """Resolve images/ and labels/ dirs relative to the data.yaml location."""
    base = data_yaml_path.parent  # e.g. data/raw/
    images_dir = (base / split_image_path).resolve()
    labels_dir = Path(str(images_dir).replace("images", "labels"))
    return images_dir, labels_dir


def collect_files(directory: Path, extensions: tuple) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(f for f in directory.iterdir() if f.suffix.lower() in extensions)


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ---------------------------------------------------------------------------
# Validation steps
# ---------------------------------------------------------------------------

def check_yaml_keys(cfg: dict) -> list[str]:
    """Return a list of error strings for missing/invalid yaml keys."""
    errors = []
    required = ["train", "val", "nc", "names"]
    for key in required:
        if key not in cfg:
            errors.append(f"  [MISSING KEY] '{key}' not found in data.yaml")
    if "nc" in cfg and "names" in cfg:
        if cfg["nc"] != len(cfg["names"]):
            errors.append(
                f"  [MISMATCH] nc={cfg['nc']} but len(names)={len(cfg['names'])}"
            )
    return errors


def validate_split(
    split_name: str,
    images_dir: Path,
    labels_dir: Path,
    class_names: list[str],
    num_classes: int,
) -> tuple[list[str], Counter]:
    """
    Validate one split. Returns (list_of_error_strings, class_counter).
    """
    errors = []
    class_counts: Counter = Counter()

    # --- Directory existence ---
    if not images_dir.exists():
        errors.append(f"  [{split_name}] images dir not found: {images_dir}")
    if not labels_dir.exists():
        errors.append(f"  [{split_name}] labels dir not found: {labels_dir}")
    if errors:
        return errors, class_counts

    images = collect_files(images_dir, IMAGE_EXTS)
    labels = collect_files(labels_dir, (".txt",))

    img_stems = {f.stem for f in images}
    lbl_stems = {f.stem for f in labels}

    # --- Parity check ---
    orphan_imgs = img_stems - lbl_stems
    orphan_lbls = lbl_stems - img_stems
    if orphan_imgs:
        errors.append(
            f"  [{split_name}] {len(orphan_imgs)} image(s) have no label file"
        )
    if orphan_lbls:
        errors.append(
            f"  [{split_name}] {len(orphan_lbls)} label file(s) have no matching image"
        )

    # --- Label integrity ---
    bad_lines = 0
    for lbl_path in labels:
        with open(lbl_path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                bad_lines += 1
                continue
            try:
                cls_idx = int(parts[0])
                coords = [float(v) for v in parts[1:]]
            except ValueError:
                bad_lines += 1
                continue

            # Class index in range
            if cls_idx < 0 or cls_idx >= num_classes:
                errors.append(
                    f"  [{split_name}] Invalid class index {cls_idx} in {lbl_path.name}"
                )
                continue

            # Coordinates in [0, 1]
            if not all(0.0 <= v <= 1.0 for v in coords):
                errors.append(
                    f"  [{split_name}] Coordinates out of [0,1] in {lbl_path.name}: {coords}"
                )
                continue

            class_counts[class_names[cls_idx]] += 1

    if bad_lines:
        errors.append(
            f"  [{split_name}] {bad_lines} malformed label line(s) (expected 5 values per line)"
        )

    return errors, class_counts


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(
    split_stats: dict,
    all_errors: list[str],
    class_names: list[str],
) -> None:
    print("\n" + "=" * 60)
    print("  DATASET VALIDATION SUMMARY")
    print("=" * 60)

    # Per-split counts
    print(f"\n{'Split':<10} {'Images':>8} {'Labels':>8}")
    print("-" * 30)
    for split, stats in split_stats.items():
        print(f"{split:<10} {stats['images']:>8} {stats['labels']:>8}")

    # Class distribution
    print(f"\n{'Class':<12} ", end="")
    for split in split_stats:
        print(f"{split:>10}", end="")
    print(f"{'Total':>10}")
    print("-" * (12 + 10 * (len(split_stats) + 1)))

    totals: Counter = Counter()
    for cls in class_names:
        print(f"{cls:<12} ", end="")
        row_total = 0
        for split, stats in split_stats.items():
            count = stats["class_counts"].get(cls, 0)
            print(f"{count:>10}", end="")
            row_total += count
            totals[cls] += count
        print(f"{row_total:>10}")
    print("-" * (12 + 10 * (len(split_stats) + 1)))
    grand = sum(totals.values())
    print(f"{'TOTAL':<12} {'':>{10 * len(split_stats)}}{grand:>10}")

    # Errors / warnings
    print()
    if all_errors:
        print(f"  ISSUES FOUND ({len(all_errors)}):")
        for e in all_errors:
            print(e)
        print("\n  Status: FAILED -- fix the issues above before training.")
    else:
        print("  Status: PASSED -- dataset looks clean.")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Validate YOLOv8 dataset from Roboflow export")
    parser.add_argument(
        "--data",
        type=str,
        default="data/raw/data.yaml",
        help="Path to data.yaml (default: data/raw/data.yaml)",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Optional path to save a JSON validation report",
    )
    args = parser.parse_args()

    data_yaml_path = Path(args.data).resolve()
    if not data_yaml_path.exists():
        print(f"[ERROR] data.yaml not found at: {data_yaml_path}")
        sys.exit(1)

    cfg = load_yaml(data_yaml_path)
    print(f"\nLoaded: {data_yaml_path}")
    print(f"Classes ({cfg.get('nc', '?')}): {cfg.get('names', [])}")

    all_errors = check_yaml_keys(cfg)
    if all_errors:
        for e in all_errors:
            print(e)
        sys.exit(1)

    class_names: list[str] = cfg["names"]
    num_classes: int = cfg["nc"]
    splits = {
        "train": cfg.get("train", ""),
        "val":   cfg.get("val", ""),
        "test":  cfg.get("test", ""),
    }

    split_stats = {}
    for split_name, rel_path in splits.items():
        if not rel_path:
            print(f"  [SKIP] '{split_name}' not defined in data.yaml")
            continue

        images_dir, labels_dir = split_dirs(data_yaml_path, rel_path)
        errors, class_counts = validate_split(
            split_name, images_dir, labels_dir, class_names, num_classes
        )
        all_errors.extend(errors)

        images = collect_files(images_dir, IMAGE_EXTS)
        labels = collect_files(labels_dir, (".txt",))
        split_stats[split_name] = {
            "images": len(images),
            "labels": len(labels),
            "images_dir": str(images_dir),
            "labels_dir": str(labels_dir),
            "class_counts": dict(class_counts),
        }

    print_summary(split_stats, all_errors, class_names)

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "data_yaml": str(data_yaml_path),
            "classes": class_names,
            "splits": split_stats,
            "errors": all_errors,
            "passed": len(all_errors) == 0,
        }
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to: {report_path}")

    sys.exit(0 if not all_errors else 1)


if __name__ == "__main__":
    main()
