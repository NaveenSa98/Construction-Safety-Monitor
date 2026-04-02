"""
evaluate.py

Evaluation pipeline for the PPE compliance detection model.

Responsibilities:
    - Run YOLOv8 validation on the held-out test set
    - Report per-class and overall precision, recall, mAP@0.5, mAP@0.5:0.95
    - Generate and save confusion matrix
    - Generate per-class performance bar charts
    - Save a structured JSON evaluation report
    - Identify and save worst-performing images for error analysis

Usage:
    python src/evaluate.py
    python src/evaluate.py --weights models/best.pt --split test
"""

import argparse
import json
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

sys.path.append(str(Path(__file__).resolve().parent))

from compliance import evaluate_scene
from inference import annotate_frame, YOLO_CONF_THRESHOLD


# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

DEFAULT_WEIGHTS  = Path("models/best.pt")
DATASET_YAML     = Path("data/processed/dataset.yaml")
OUTPUT_DIR       = Path("outputs/evaluation")
REPORT_PATH      = OUTPUT_DIR / "evaluation_report.json"


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

# Classes with known data limitations — flagged in report
# Mask (11): limited representation in the training dataset.
DATA_LIMITED_CLASSES = {
    11: "Mask",
}


# ──────────────────────────────────────────────
# METRIC EXTRACTION
# ──────────────────────────────────────────────

def extract_metrics(results) -> dict:
    """
    Extracts per-class and overall metrics from Ultralytics
    validation results object.

    Returns a structured dictionary of all metrics.
    """
    metrics = results.results_dict

    # Overall metrics
    overall = {
        "mAP50"     : round(float(metrics.get("metrics/mAP50(B)",    0)), 4),
        "mAP50_95"  : round(float(metrics.get("metrics/mAP50-95(B)", 0)), 4),
        "precision" : round(float(metrics.get("metrics/precision(B)", 0)), 4),
        "recall"    : round(float(metrics.get("metrics/recall(B)",    0)), 4),
    }

    # Per-class metrics from the results box object
    per_class = []
    if hasattr(results, "box") and hasattr(results.box, "ap_class_index"):
        ap_class_index = results.box.ap_class_index.tolist()
        maps           = results.box.maps.tolist()
        p_per_class    = results.box.p.tolist()
        r_per_class    = results.box.r.tolist()

        for i, cls_idx in enumerate(ap_class_index):
            cls_name = (CLASS_NAMES[cls_idx]
                        if cls_idx < len(CLASS_NAMES)
                        else f"class_{cls_idx}")
            per_class.append({
                "class_id"         : cls_idx,
                "class_name"       : cls_name,
                "mAP50"            : round(float(maps[i]), 4),
                "precision"        : round(float(p_per_class[i]), 4),
                "recall"           : round(float(r_per_class[i]), 4),
                "data_limited"     : cls_idx in DATA_LIMITED_CLASSES,
            })

    return {"overall": overall, "per_class": per_class}


# ──────────────────────────────────────────────
# VISUALISATION
# ──────────────────────────────────────────────

def plot_per_class_map(
    per_class_metrics : list[dict],
    output_dir        : Path,
) -> Path:
    """
    Generates a horizontal bar chart of per-class mAP@0.5.
    Bars for data-limited classes are rendered in a distinct colour.
    Saves and returns the output path.
    """
    if not per_class_metrics:
        return None

    names   = [m["class_name"] for m in per_class_metrics]
    maps    = [m["mAP50"]      for m in per_class_metrics]
    limited = [m["data_limited"] for m in per_class_metrics]

    colours = ["#ef4444" if lim else "#3b82f6" for lim in limited]

    fig, ax = plt.subplots(figsize=(10, max(5, len(names) * 0.55)))
    bars = ax.barh(names, maps, color=colours, edgecolor="white",
                   linewidth=0.5, height=0.65)

    # Add value labels on bars
    for bar, val in zip(bars, maps):
        ax.text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", ha="left", fontsize=9
        )

    ax.set_xlim(0, 1.15)
    ax.set_xlabel("mAP@0.5", fontsize=11)
    ax.set_title("Per-Class Detection Performance (mAP@0.5)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.axvline(x=0.5, color="#94a3b8", linestyle="--",
               linewidth=0.8, label="0.5 threshold")

    legend_patches = [
        mpatches.Patch(color="#3b82f6", label="Sufficient data"),
        mpatches.Patch(color="#ef4444", label="Data-limited class"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()
    out_path = output_dir / "per_class_map50.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Per-class mAP chart saved → {out_path}")
    return out_path


def plot_precision_recall(
    per_class_metrics : list[dict],
    output_dir        : Path,
) -> Path:
    """
    Generates a grouped bar chart comparing precision and recall
    per class. Saves and returns the output path.
    """
    if not per_class_metrics:
        return None

    names      = [m["class_name"] for m in per_class_metrics]
    precisions = [m["precision"]  for m in per_class_metrics]
    recalls    = [m["recall"]     for m in per_class_metrics]

    x      = np.arange(len(names))
    width  = 0.38

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.1), 6))
    ax.bar(x - width/2, precisions, width, label="Precision",
           color="#3b82f6", alpha=0.85)
    ax.bar(x + width/2, recalls,    width, label="Recall",
           color="#10b981", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Precision vs Recall per Class",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10)
    ax.axhline(y=0.5, color="#94a3b8", linestyle="--",
               linewidth=0.8, alpha=0.6)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / "precision_recall_per_class.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Precision/recall chart saved → {out_path}")
    return out_path


def plot_performance_summary_table(
    overall_metrics   : dict,
    per_class_metrics : list[dict],
    output_dir        : Path,
) -> Path:
    """
    Renders a clean summary table as an image showing all
    per-class metrics alongside the overall scores.
    Saves and returns the output path.
    """
    rows = []
    for m in per_class_metrics:
        flag = " *" if m["data_limited"] else ""
        rows.append([
            m["class_name"] + flag,
            f"{m['precision']:.3f}",
            f"{m['recall']:.3f}",
            f"{m['mAP50']:.3f}",
        ])

    # Append overall row
    rows.append([
        "OVERALL",
        f"{overall_metrics['precision']:.3f}",
        f"{overall_metrics['recall']:.3f}",
        f"{overall_metrics['mAP50']:.3f}",
    ])

    col_labels  = ["Class", "Precision", "Recall", "mAP@0.5"]
    n_rows      = len(rows)
    fig_height  = max(4, n_rows * 0.45 + 1.5)

    fig, ax = plt.subplots(figsize=(9, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText    = rows,
        colLabels   = col_labels,
        cellLoc     = "center",
        loc         = "center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Style header row
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#1e3a5f")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Style overall row (last)
    for j in range(len(col_labels)):
        table[n_rows, j].set_facecolor("#dbeafe")
        table[n_rows, j].set_text_props(fontweight="bold")

    # Colour mAP column by performance
    for i in range(1, n_rows):
        val = float(rows[i - 1][3])
        if val >= 0.7:
            colour = "#dcfce7"
        elif val >= 0.5:
            colour = "#fef9c3"
        else:
            colour = "#fee2e2"
        table[i, 3].set_facecolor(colour)

    ax.set_title(
        "Evaluation Results — PPE Compliance Detection\n"
        "* = data-limited class",
        fontsize=12, fontweight="bold", pad=16
    )

    plt.tight_layout()
    out_path = output_dir / "evaluation_summary_table.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Summary table saved → {out_path}")
    return out_path


# ──────────────────────────────────────────────
# ERROR ANALYSIS
# ──────────────────────────────────────────────

def identify_failure_cases(
    model      : YOLO,
    test_dir   : Path,
    output_dir : Path,
    n_cases    : int = 5,
) -> list[dict]:
    """
    Runs inference on the test set and identifies images where the
    model produced the lowest aggregate confidence scores.

    These are saved as labelled failure case samples for documentation.
    Returns a list of failure case metadata dicts.
    """
    image_paths = [
        p for p in sorted(test_dir.iterdir())
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    if not image_paths:
        print(f"[WARN] No test images found in {test_dir}")
        return []

    scored_images = []

    for img_path in image_paths:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        results    = model(frame, conf=YOLO_CONF_THRESHOLD, verbose=False)[0]
        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                detections.append({
                    "class_id"   : int(box.cls[0].item()),
                    "confidence" : float(box.conf[0].item()),
                    "x1": float(box.xyxy[0][0].item()),
                    "y1": float(box.xyxy[0][1].item()),
                    "x2": float(box.xyxy[0][2].item()),
                    "y2": float(box.xyxy[0][3].item()),
                })

        report = evaluate_scene(detections, image_name=img_path.name)

        # Score = mean confidence of all detections (lower = harder image)
        if detections:
            mean_conf = sum(d["confidence"] for d in detections) / len(detections)
        else:
            mean_conf = 0.0

        scored_images.append({
            "path"       : img_path,
            "mean_conf"  : mean_conf,
            "n_workers"  : report.total_workers,
            "violations" : report.violation_count,
            "verdict"    : report.scene_verdict.value,
            "detections" : detections,
            "report"     : report,
        })

    # Sort by ascending mean confidence — hardest images first
    scored_images.sort(key=lambda x: x["mean_conf"])
    failure_cases = scored_images[:n_cases]

    # Save annotated failure case images
    failure_dir = output_dir / "failure_cases"
    failure_dir.mkdir(parents=True, exist_ok=True)

    failure_metadata = []
    for i, case in enumerate(failure_cases):
        frame     = cv2.imread(str(case["path"]))
        annotated = annotate_frame(frame.copy(), case["detections"],
                                   case["report"])

        # Add failure case label overlay
        cv2.putText(
            annotated,
            f"FAILURE CASE {i+1} — mean conf: {case['mean_conf']:.2f}",
            (10, annotated.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 0), 1, cv2.LINE_AA
        )

        out_path = failure_dir / f"failure_case_{i+1:02d}_{case['path'].name}"
        cv2.imwrite(str(out_path), annotated)

        failure_metadata.append({
            "case_number"  : i + 1,
            "image"        : case["path"].name,
            "mean_conf"    : round(case["mean_conf"], 3),
            "scene_verdict": case["verdict"],
            "n_workers"    : case["n_workers"],
            "violations"   : case["violations"],
            "saved_to"     : str(out_path),
        })
        print(f"[INFO] Failure case {i+1} saved → {out_path.name}")

    return failure_metadata


# ──────────────────────────────────────────────
# MAIN EVALUATION PIPELINE
# ──────────────────────────────────────────────

def run_evaluation(weights: Path, split: str = "test") -> None:

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"[INFO] Loading model: {weights}")
    model = YOLO(str(weights))

    # ── Step 1: Run YOLOv8 validation on the specified split ──
    print(f"[INFO] Running validation on '{split}' split...")
    results = model.val(
        data   = str(DATASET_YAML),
        split  = split,
        conf   = 0.50,
        iou    = 0.50,
        plots  = True,
        save_json = True,
        project= str(OUTPUT_DIR),
        name   = "val_run",
        exist_ok = True,
        verbose  = False,
    )

    # ── Step 2: Extract structured metrics ──
    metrics = extract_metrics(results)
    overall = metrics["overall"]
    per_cls = metrics["per_class"]

    # ── Step 3: Print console summary ──
    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Overall mAP@0.5      : {overall['mAP50']:.4f}")
    print(f"  Overall mAP@0.5:0.95 : {overall['mAP50_95']:.4f}")
    print(f"  Overall Precision    : {overall['precision']:.4f}")
    print(f"  Overall Recall       : {overall['recall']:.4f}")
    print("-" * 60)
    print(f"  {'Class':<22} {'P':>7} {'R':>7} {'mAP50':>8}")
    print("-" * 60)
    for m in per_cls:
        flag = " [DATA-LIMITED]" if m["data_limited"] else ""
        print(
            f"  {m['class_name']:<22} "
            f"{m['precision']:>7.3f} "
            f"{m['recall']:>7.3f} "
            f"{m['mAP50']:>8.3f}"
            f"{flag}"
        )
    print("=" * 60 + "\n")

    # ── Step 4: Generate visualisation charts ──
    plot_per_class_map(per_cls, OUTPUT_DIR)
    plot_precision_recall(per_cls, OUTPUT_DIR)
    plot_performance_summary_table(overall, per_cls, OUTPUT_DIR)

    # ── Step 5: Identify failure cases ──
    test_images_dir = Path("data/processed/images") / split
    failure_cases   = []
    if test_images_dir.exists():
        print(f"[INFO] Identifying failure cases from {test_images_dir}...")
        failure_cases = identify_failure_cases(
            model, test_images_dir, OUTPUT_DIR, n_cases=5
        )

    # ── Step 6: Save structured evaluation report ──
    report = {
        "timestamp"        : timestamp,
        "model_weights"    : str(weights),
        "dataset_yaml"     : str(DATASET_YAML),
        "evaluation_split" : split,
        "overall_metrics"  : overall,
        "per_class_metrics": per_cls,
        "data_limited_note": (
            "Classes marked data_limited=true were trained with "
            "insufficient data. Their metrics do not reflect model "
            "architecture capability and should not be compared "
            "against well-represented classes."
        ),
        "failure_cases"    : failure_cases,
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[INFO] Full evaluation report saved → {REPORT_PATH}")
    print(f"[INFO] All evaluation outputs saved → {OUTPUT_DIR}/")


# ──────────────────────────────────────────────
# ARGUMENT PARSER & ENTRY POINT
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate PPE Compliance Detection Model"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=str(DEFAULT_WEIGHTS),
        help=f"Path to model weights. Default: {DEFAULT_WEIGHTS}"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on. Default: test"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(
        weights = Path(args.weights),
        split   = args.split,
    )

