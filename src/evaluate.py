"""
evaluate.py
-----------
Formal evaluation script for the trained PPE detection model.
Runs the model against the test set using Ultralytics .val(),
prints a per-class metrics table, flags underperforming classes,
and saves a JSON report.

Usage:
    python src/evaluate.py --weights models/best.pt --data data/raw/data.yaml
    python src/evaluate.py --weights models/best.pt --data data/raw/data.yaml --output outputs/report.json
"""

import argparse
import json
import sys
from pathlib import Path

from ultralytics import YOLO


# Any class with mAP@0.5 below this is flagged as underperforming
WEAK_MAP_THRESHOLD = 0.50


# =============================================================================
# Argument Parser
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PPE detection model on test set")

    parser.add_argument(
        "--weights", type=str, required=True,
        help="Path to trained weights (best.pt)"
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to data.yaml"
    )
    parser.add_argument(
        "--output", type=str,
        default="outputs/evaluation_report.json",
        help="Path to save JSON report (default: outputs/evaluation_report.json)"
    )

    return parser.parse_args()


# =============================================================================
# Run Evaluation
# =============================================================================

def run_evaluation(weights: str, data_yaml: str) -> tuple:
    """
    Load model and run .val() on the test split.
    Returns (metrics object, list of class names).
    """
    model   = YOLO(weights)
    metrics = model.val(data=data_yaml, split="test", verbose=False)
    return metrics, list(model.names.values())


# =============================================================================
# Print Results Table
# =============================================================================

def print_table(class_names: list, metrics) -> list:
    """
    Print a per-class precision / recall / mAP table to console.
    Flags any class with mAP@0.5 < WEAK_MAP_THRESHOLD as WEAK.
    Returns a list of per-class result dicts for JSON export.
    """
    # Ultralytics stores per-class metrics in metrics.box
    per_class_p    = metrics.box.p       # precision per class
    per_class_r    = metrics.box.r       # recall per class
    per_class_map50 = metrics.box.ap50   # mAP@0.5 per class
    per_class_map   = metrics.box.ap     # mAP@0.5:0.95 per class

    col = "{:<12}  {:>10}  {:>8}  {:>9}  {:>12}  {}"
    sep = "-" * 62

    print("\n" + "=" * 62)
    print("  PPE MODEL EVALUATION — TEST SET")
    print("=" * 62)
    print(col.format("Class", "Precision", "Recall", "mAP@0.5", "mAP@0.5:95", ""))
    print(sep)

    results = []
    for i, cls in enumerate(class_names):
        p    = float(per_class_p[i])
        r    = float(per_class_r[i])
        m50  = float(per_class_map50[i])
        m595 = float(per_class_map[i])
        flag = "  <- WEAK" if m50 < WEAK_MAP_THRESHOLD else ""

        print(col.format(cls, f"{p:.3f}", f"{r:.3f}", f"{m50:.3f}", f"{m595:.3f}", flag))

        results.append({
            "class":        cls,
            "precision":    round(p,    4),
            "recall":       round(r,    4),
            "map50":        round(m50,  4),
            "map50_95":     round(m595, 4),
            "weak":         m50 < WEAK_MAP_THRESHOLD,
        })

    print(sep)

    # Overall row from Ultralytics mean metrics
    mp    = float(metrics.box.mp)
    mr    = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map   = float(metrics.box.map)
    print(col.format("ALL", f"{mp:.3f}", f"{mr:.3f}", f"{map50:.3f}", f"{map:.3f}", ""))
    print("=" * 62 + "\n")

    # Highlight weak classes separately
    weak = [r["class"] for r in results if r["weak"]]
    if weak:
        print(f"  Underperforming classes (mAP@0.5 < {WEAK_MAP_THRESHOLD}):")
        for cls in weak:
            print(f"    - {cls}")
        print()

    return results, {"precision": round(mp, 4), "recall": round(mr, 4),
                     "map50": round(map50, 4), "map50_95": round(map, 4)}


# =============================================================================
# Save JSON Report
# =============================================================================

def save_report(output_path: str, class_results: list, overall: dict,
                weights: str, data_yaml: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    report = {
        "weights":      weights,
        "data_yaml":    data_yaml,
        "split":        "test",
        "overall":      overall,
        "per_class":    class_results,
        "weak_threshold": WEAK_MAP_THRESHOLD,
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"  Report saved: {output_path}\n")


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    # Validate paths
    if not Path(args.weights).exists():
        print(f"[ERROR] Weights not found: {args.weights}")
        sys.exit(1)
    if not Path(args.data).exists():
        print(f"[ERROR] data.yaml not found: {args.data}")
        sys.exit(1)

    print(f"\n  Weights  : {args.weights}")
    print(f"  Data     : {args.data}")
    print(f"  Split    : test\n")

    metrics, class_names = run_evaluation(args.weights, args.data)
    class_results, overall = print_table(class_names, metrics)
    save_report(args.output, class_results, overall, args.weights, args.data)


if __name__ == "__main__":
    main()
