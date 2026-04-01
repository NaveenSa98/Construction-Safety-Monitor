"""
End-to-end inference pipeline for PPE compliance detection.

Accepts a single image, a directory of images, or a video file.
For each input frame:
    1. Runs YOLOv8 object detection
    2. Passes detections to the compliance engine
    3. Renders annotated output with colour-coded worker boxes
    4. Saves a structured JSON violation report

"""

import argparse
import json
import time
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime

import sys

sys.path.append(str(Path(__file__).resolve().parent))

from compliance import (
    evaluate_scene,
    SceneVerdict,
    WorkerStatus,
    RuleResult,
)


# Configuration

DEFAULT_WEIGHTS     = Path("models/best.pt")
OUTPUT_IMAGES_DIR   = Path("outputs/sample_predictions")
OUTPUT_REPORTS_DIR  = Path("outputs/reports")

# Confidence threshold passed to YOLOv8 at inference time
YOLO_CONF_THRESHOLD = 0.25    # Lower than compliance threshold to capture
                               

# Supported image and video extensions
IMAGE_EXTENSIONS    = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".avif"}
VIDEO_EXTENSIONS    = {".mp4", ".avi", ".mov", ".mkv"}

 # Color palette for rendering worker boxes based on compliance status

COLOUR_COMPLIANT     = (34,  197,  94)   # Green
COLOUR_VIOLATION     = (239,  68,  68)   # Red
COLOUR_UNVERIFIABLE  = (249, 115,  22)   # Orange
COLOUR_PPE_BOX       = (148, 163, 184)   # Slate grey — PPE item boxes
COLOUR_SAFE_BG       = (34,  197,  94)   # Green — scene verdict background
COLOUR_UNSAFE_BG     = (239,  68,  68)   # Red
COLOUR_UNVERIF_BG    = (249, 115,  22)   # Orange
COLOUR_TEXT          = (255, 255, 255)   # White text
COLOUR_TEXT_DARK     = (15,   23,  42)   # Dark text for light backgrounds

CLASS_NAMES = {
    0 : "Person",
    1 : "Hardhat",
    2 : "NO-Hardhat",
    3 : "Safety Vest",
    4 : "NO-Safety Vest",
    5 : "Safety Gloves",
    6 : "NO-Safety Gloves",
    7 : "Safety Boots",
    8 : "NO-Safety Boots",
    9 : "Safety Goggles",
    10 : "NO-Safety Goggles",
    11 : "NO-Safety Harness",
}

# Detection parsing

def parse_yolo_results(results) -> list[dict]:
    """
    Parses raw Ultralytics YOLOv8 results into a flat list of
    detection dictionaries consumable by the compliance engine.

    Parameters
    ----------
    results : ultralytics Results object (single image)

    Returns
    -------
    list[dict] — each dict contains:
        class_id, confidence, x1, y1, x2, y2
    """
    detections = []
    if results.boxes is None:
        return detections

    for box in results.boxes:
        detections.append({
            "class_id"   : int(box.cls[0].item()),
            "confidence" : float(box.conf[0].item()),
            "x1"         : float(box.xyxy[0][0].item()),
            "y1"         : float(box.xyxy[0][1].item()),
            "x2"         : float(box.xyxy[0][2].item()),
            "y2"         : float(box.xyxy[0][3].item()),
        })
    return detections

# Rendering utilities

def draw_rounded_rect(
    frame  : np.ndarray,
    pt1    : tuple,
    pt2    : tuple,
    colour : tuple,
    radius : int = 8,
    thickness: int = 2
) -> np.ndarray:
    """
    Draws a rectangle with slightly rounded corners on the frame.
    Falls back to a standard rectangle if radius is too large.
    """
    x1, y1 = int(pt1[0]), int(pt1[1])
    x2, y2 = int(pt2[0]), int(pt2[1])
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)

    if r <= 0:
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness)
        return frame

    cv2.line(frame, (x1 + r, y1), (x2 - r, y1), colour, thickness)
    cv2.line(frame, (x1 + r, y2), (x2 - r, y2), colour, thickness)
    cv2.line(frame, (x1, y1 + r), (x1, y2 - r), colour, thickness)
    cv2.line(frame, (x2, y1 + r), (x2, y2 - r), colour, thickness)
    cv2.ellipse(frame, (x1+r, y1+r), (r,r), 180,  0,  90, colour, thickness)
    cv2.ellipse(frame, (x2-r, y1+r), (r,r), 270,  0,  90, colour, thickness)
    cv2.ellipse(frame, (x1+r, y2-r), (r,r),  90,  0,  90, colour, thickness)
    cv2.ellipse(frame, (x2-r, y2-r), (r,r),   0,  0,  90, colour, thickness)
    return frame


def draw_label_pill(
    frame      : np.ndarray,
    text       : str,
    position   : tuple,
    bg_colour  : tuple,
    font_scale : float = 0.55,
    thickness  : int   = 1,
    padding    : int   = 5,
) -> np.ndarray:
    """
    Draws a filled pill-shaped label with text at the given position.
    position is the top-left corner of the label.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = int(position[0]), int(position[1])
    cv2.rectangle(
        frame,
        (x, y),
        (x + tw + padding * 2, y + th + padding * 2 + baseline),
        bg_colour,
        -1
    )
    cv2.putText(
        frame, text,
        (x + padding, y + th + padding),
        font, font_scale,
        COLOUR_TEXT, thickness, cv2.LINE_AA
    )
    return frame


def render_worker_annotations(
    frame   : np.ndarray,
    report,
) -> np.ndarray:
    """
    Renders per-worker bounding boxes and status labels onto the frame.
    """
    for worker in report.workers:
        x1, y1, x2, y2 = [int(v) for v in worker.box]

        # Select colour based on worker status
        if worker.status == WorkerStatus.COMPLIANT:
            colour = COLOUR_COMPLIANT
            status_text = "COMPLIANT"
        elif worker.status == WorkerStatus.VIOLATION:
            colour = COLOUR_VIOLATION
            status_text = "VIOLATION"
        else:
            colour = COLOUR_UNVERIFIABLE
            status_text = "UNVERIFIABLE"

        # Draw worker bounding box
        draw_rounded_rect(frame, (x1, y1), (x2, y2), colour,
                          radius=6, thickness=2)

        # Draw worker ID + status label above the box
        label = f"W{worker.worker_id} {status_text} {worker.confidence:.0%}"
        label_y = max(y1 - 8, 20)
        draw_label_pill(frame, label, (x1, label_y - 22),
                        bg_colour=colour, font_scale=0.50)

        # Draw individual violation tags below the worker box
        tag_y = y2 + 6
        for violation in worker.violations:
            tag_text = f"! {violation.rule_id}: {violation.rule_name}"
            draw_label_pill(frame, tag_text, (x1, tag_y),
                            bg_colour=COLOUR_VIOLATION,
                            font_scale=0.42)
            tag_y += 22

    return frame


def render_ppe_detections(
    frame      : np.ndarray,
    detections : list[dict],
) -> np.ndarray:
    """
    Renders all non-person detection boxes as thin grey labels.
    Provides visual context for PPE items detected in the scene.
    """
    for det in detections:
        if det["class_id"] == 0:   # Skip person boxes — handled separately
            continue
        if det["confidence"] < 0.25:
            continue
        x1, y1, x2, y2 = [int(det[k]) for k in ("x1", "y1", "x2", "y2")]
        class_name = CLASS_NAMES.get(det["class_id"], str(det["class_id"]))
        conf = det["confidence"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOUR_PPE_BOX, 1)
        draw_label_pill(
            frame,
            f"{class_name} {conf:.0%}",
            (x1, y1),
            bg_colour=COLOUR_PPE_BOX,
            font_scale=0.38,
        )
    return frame


def render_scene_verdict_banner(
    frame  : np.ndarray,
    report,
) -> np.ndarray:
    """
    Renders a scene-level verdict banner at the top of the frame.
    Includes verdict, worker counts, and timestamp.
    """
    h, w = frame.shape[:2]
    banner_height = 52

    # Select banner colour
    if report.scene_verdict == SceneVerdict.SAFE:
        bg_colour   = COLOUR_SAFE_BG
        verdict_str = "SCENE: SAFE"
    elif report.scene_verdict == SceneVerdict.UNSAFE:
        bg_colour   = COLOUR_UNSAFE_BG
        verdict_str = "SCENE: UNSAFE"
    else:
        bg_colour   = COLOUR_UNVERIF_BG
        verdict_str = "SCENE: UNVERIFIABLE"

    # Draw banner background
    cv2.rectangle(frame, (0, 0), (w, banner_height), bg_colour, -1)

    # Verdict text
    cv2.putText(
        frame, verdict_str,
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX, 0.85,
        COLOUR_TEXT, 2, cv2.LINE_AA
    )

    # Worker summary (right-aligned)
    summary = (
        f"Workers: {report.total_workers}  "
        f"OK: {report.compliant_count}  "
        f"Violations: {report.violation_count}  "
        f"Unverifiable: {report.unverifiable_count}"
    )
    (tw, _), _ = cv2.getTextSize(
        summary, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1
    )
    cv2.putText(
        frame, summary,
        (w - tw - 12, 32),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48,
        COLOUR_TEXT, 1, cv2.LINE_AA
    )

    return frame


def annotate_frame(
    frame      : np.ndarray,
    detections : list[dict],
    report,
) -> np.ndarray:
    """
    Full annotation pipeline for a single frame.
    Applies PPE boxes, worker boxes, and scene verdict banner.
    """
    frame = render_ppe_detections(frame, detections)
    frame = render_worker_annotations(frame, report)
    frame = render_scene_verdict_banner(frame, report)
    return frame


# Report saving

def save_json_report(report, output_dir: Path) -> Path:
    """
    Saves the SceneReport as a JSON file to the reports directory.
    Returns the path to the saved file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(report.image_name).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"{stem}_{timestamp}_report.json"
    with open(report_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    return report_path


# Main inference pipeline

def run_on_image(
    model      : YOLO,
    image_path : Path,
    output_dir : Path,
    report_dir : Path,
    show       : bool = False,
) -> dict:
    """
    Runs the full inference + compliance pipeline on a single image.
    Returns a summary dictionary.
    """
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"[WARN] Could not read image: {image_path}")
        return {}

    # Run YOLOv8 detection
    results      = model(frame, conf=YOLO_CONF_THRESHOLD, verbose=False)[0]
    detections   = parse_yolo_results(results)

    # Run compliance engine
    report       = evaluate_scene(detections, image_name=image_path.name)

    # Annotate frame
    annotated    = annotate_frame(frame.copy(), detections, report)

    # Save annotated image
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"pred_{image_path.name}"
    cv2.imwrite(str(out_path), annotated)

    # Save JSON report
    report_path  = save_json_report(report, report_dir)

    # Console summary
    print(
        f"[{report.scene_verdict.value:12s}] {image_path.name} | "
        f"Workers: {report.total_workers} | "
        f"Violations: {report.violation_count} | "
        f"Saved → {out_path.name}"
    )

    if show:
        cv2.imshow("PPE Compliance Monitor", annotated)
        cv2.waitKey(0)

    return report.to_dict()


def run_on_directory(
    model      : YOLO,
    source_dir : Path,
    output_dir : Path,
    report_dir : Path,
    show       : bool = False,
) -> list[dict]:
    """
    Runs inference on all images in a directory.
    Returns a list of report dictionaries.
    """
    image_paths = [
        p for p in sorted(source_dir.iterdir())
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not image_paths:
        print(f"[WARN] No images found in {source_dir}")
        return []

    print(f"[INFO] Processing {len(image_paths)} images from {source_dir}")
    all_reports = []
    for img_path in image_paths:
        report = run_on_image(model, img_path, output_dir, report_dir, show)
        if report:
            all_reports.append(report)

    # Save combined summary report
    summary_path = report_dir / "batch_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_reports, f, indent=2)
    print(f"\n[INFO] Batch complete. Summary saved → {summary_path}")

    total_violations = sum(r.get("violation_count", 0) for r in all_reports)
    print(f"[INFO] Total scenes processed : {len(all_reports)}")
    print(f"[INFO] Total violations found : {total_violations}")

    return all_reports


def run_on_video(
    model      : YOLO,
    video_path : Path,
    output_dir : Path,
    report_dir : Path,
    show       : bool = False,
) -> None:
    """
    Runs inference on a video file frame by frame.
    Saves annotated output video and per-frame reports.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_dir.mkdir(parents=True, exist_ok=True)
    out_video_path = output_dir / f"pred_{video_path.name}"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

    print(f"[INFO] Processing video: {video_path.name} "
          f"({total} frames @ {fps:.1f} fps)")

    frame_idx   = 0
    all_reports = []
    start_time  = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results    = model(frame, conf=YOLO_CONF_THRESHOLD, verbose=False)[0]
        detections = parse_yolo_results(results)
        report     = evaluate_scene(
            detections,
            image_name=f"{video_path.stem}_frame_{frame_idx:05d}"
        )
        annotated  = annotate_frame(frame.copy(), detections, report)
        writer.write(annotated)
        all_reports.append(report.to_dict())

        if show:
            cv2.imshow("PPE Compliance Monitor", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1
        if frame_idx % 50 == 0:
            elapsed = time.time() - start_time
            print(f"[INFO] Frame {frame_idx}/{total} "
                  f"({frame_idx/elapsed:.1f} fps processed)")

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    # Save combined video report
    report_path = report_dir / f"{video_path.stem}_video_report.json"
    with open(report_path, "w") as f:
        json.dump(all_reports, f, indent=2)

    print(f"[INFO] Video processing complete.")
    print(f"[INFO] Annotated video saved → {out_video_path}")
    print(f"[INFO] Video report saved    → {report_path}")



def run_on_webcam(model: YOLO, show: bool = True) -> None:
    """
    Runs live inference on webcam feed (device 0).
    Press Q to quit.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    print("[INFO] Webcam active. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results    = model(frame, conf=YOLO_CONF_THRESHOLD, verbose=False)[0]
        detections = parse_yolo_results(results)
        report     = evaluate_scene(detections, image_name="webcam_frame")
        annotated  = annotate_frame(frame.copy(), detections, report)

        cv2.imshow("PPE Compliance Monitor — Live", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# Entry point

def parse_args():
    parser = argparse.ArgumentParser(
        description="PPE Compliance Inference — Construction Safety Monitor"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Input source: image path, directory, video file, or '0' for webcam."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=str(DEFAULT_WEIGHTS),
        help=f"Path to YOLOv8 model weights. Default: {DEFAULT_WEIGHTS}"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_IMAGES_DIR),
        help=f"Directory for annotated output images/video. "
             f"Default: {OUTPUT_IMAGES_DIR}"
    )
    parser.add_argument(
        "--reports",
        type=str,
        default=str(OUTPUT_REPORTS_DIR),
        help=f"Directory for JSON violation reports. "
             f"Default: {OUTPUT_REPORTS_DIR}"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display output frames in a window during inference."
    )
    return parser.parse_args()


def main():
    args        = parse_args()
    source      = args.source
    weights     = Path(args.weights)
    output_dir  = Path(args.output)
    report_dir  = Path(args.reports)

    # Load model
    print(f"[INFO] Loading model weights from: {weights}")
    model = YOLO(str(weights))
    print(f"[INFO] Model loaded successfully.")

    # Route to appropriate runner
    if source == "0":
        run_on_webcam(model, show=True)

    else:
        source_path = Path(source)

        if not source_path.exists():
            print(f"[ERROR] Source path does not exist: {source_path}")
            return

        if source_path.is_dir():
            run_on_directory(model, source_path, output_dir,
                             report_dir, args.show)

        elif source_path.suffix.lower() in VIDEO_EXTENSIONS:
            run_on_video(model, source_path, output_dir,
                         report_dir, args.show)

        elif source_path.suffix.lower() in IMAGE_EXTENSIONS:
            run_on_image(model, source_path, output_dir,
                         report_dir, args.show)

        else:
            print(f"[ERROR] Unsupported file type: {source_path.suffix}")


if __name__ == "__main__":
    main()