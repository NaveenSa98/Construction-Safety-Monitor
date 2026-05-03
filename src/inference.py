"""
inference.py
------------
End-to-end PPE compliance inference pipeline.
Loads trained YOLOv8s weights, runs detection on an image / video / webcam,
passes detections to compliance.py, and renders annotated output.

Usage:
    # Single image
    python src/inference.py --weights models/best.pt --source image.jpg

    # Video file
    python src/inference.py --weights models/best.pt --source video.mp4 --output out.mp4

    # Webcam
    python src/inference.py --weights models/best.pt --source 0

    # Save annotated image
    python src/inference.py --weights models/best.pt --source image.jpg --output result.jpg
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from compliance import check_compliance, CONF_THRESHOLDS




# =============================================================================
# Visual Style
# =============================================================================

FONT   = cv2.FONT_HERSHEY_SIMPLEX
LINE_AA = cv2.LINE_AA
 
VERDICT_COLORS = {
    "SAFE":   (34,  139,  34),   # green  (BGR)
    "ALERT":  (0,   165, 255),   # orange
    "UNSAFE": (0,     0, 220),   # red
}
 
# Per-class PPE bounding-box colours (BGR)
PPE_COLORS = {
    "person":  (255, 255, 255),
    "helmat":  (0,   255, 255),
    "vest":    (0,   165, 255),
    "boot":    (255, 144,  30),
    "gloves":  (147,  20, 255),
    "goggles": (0,   255,   0)
}
 
# Human-readable display names (model class → label)
DISPLAY_NAMES = {
    "helmat":  "Helmet",
    "vest":    "Vest",
    "boot":    "Boots",
    "gloves":  "Gloves",
    "goggles": "Goggles",
}
 
 
# PPE Status box layout constants
LINE_H  = 16     # pixels per text line
BOX_W   = 240    # fixed width of PPE status box
MAX_VIS = 420    # max visible height before scrollbar appears


# =============================================================================
# Layer 1 — Argument Parser
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="PPE Compliance Inference")

    parser.add_argument(
        "--weights", type=str, required=True,
        help="Path to trained YOLOv8s weights (best.pt)"
    )
    parser.add_argument(
        "--source", type=str, required=True,
        help="Input: image path, video path, or 0 for webcam"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save path for annotated output (optional)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Detection confidence threshold (default: 0.25)"
    )

    return parser.parse_args()


# =============================================================================
# Layer 2 — Model Loader
# =============================================================================

def load_model(weights_path: str) -> YOLO:
    path = Path(weights_path)
    if not path.exists():
        print(f"[ERROR] Weights not found: {path}")
        sys.exit(1)

    model = YOLO(str(path))
    print(f"Model loaded: {path}")
    return model


# =============================================================================
# Layer 3 — Detection Converter
# =============================================================================

def results_to_detections(results, class_names: list) -> list:
    """
    Convert raw Ultralytics results into the flat detection dict list
    that compliance.py expects.

    Returns:
        [{"class_name": str, "bbox": [x1,y1,x2,y2], "confidence": float}, ...]
    """
    detections = []
    for box in results.boxes:
        cls_idx    = int(box.cls[0])
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append({
            "class_name": class_names[cls_idx],
            "bbox":       [x1, y1, x2, y2],
            "confidence": confidence,
        })
    return detections

# =============================================================================
# Drawing utilities
# =============================================================================

def _put(img, text, x, y, scale, color, thickness=1):
    cv2.putText(img, text, (x, y), FONT, scale, color, thickness, LINE_AA)
 
 
def _rounded_fill(img, x1, y1, x2, y2, fill, border, alpha=0.82, r=8):
    """
    Draw a semi-transparent filled rounded rectangle on img in-place.
    Approximated with rectangles + corner circles.
    """
    overlay = img.copy()
    cv2.rectangle(overlay, (x1 + r, y1),     (x2 - r, y2),     fill, -1)
    cv2.rectangle(overlay, (x1,     y1 + r), (x2,     y2 - r), fill, -1)
    for cx, cy in [(x1+r, y1+r), (x2-r, y1+r),
                   (x1+r, y2-r), (x2-r, y2-r)]:
        cv2.circle(overlay, (cx, cy), r, fill, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x1, y1), (x2, y2), border, 1)
 
 
def _wrap(text: str, width: int) -> list:
    """Word-wrap text to at most `width` characters per line."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= width:
            cur = (cur + " " + w).strip()
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [text]
 

# =============================================================================
# Layer 4a — PPE detection boxes on the frame
# =============================================================================

def _draw_ppe_boxes(frame: np.ndarray, detections: list) -> None:
    for det in detections:
        cls = det["class_name"]
        if cls == "person":
            continue
        if det["confidence"] < CONF_THRESHOLDS.get(cls, 0.25):
            continue
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        color = PPE_COLORS.get(cls, (200, 200, 200))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, DISPLAY_NAMES.get(cls, cls),
                    (x1 + 3, y1 - 5), FONT, 0.40, color, 1, LINE_AA)
 
 
def _draw_worker_borders(frame: np.ndarray, report: dict) -> None:
    """
    For each detected worker draw:
      • A 3-pixel coloured border (green/orange/red based on compliance)
      • A small filled worker-ID badge (W0, W1 …) at the top-left of the bbox,
        colour-coded per worker so the viewer can match the panel to the person.
    """
    for worker in report["workers"]:
        wid  = worker["worker_id"]
        x1, y1, x2, y2 = [int(v) for v in worker["bbox"]]
 
        status = ("SAFE"  if worker["compliant"] and not worker["alerts"] else
                  "ALERT" if worker["compliant"] else "UNSAFE")
        v_col  = VERDICT_COLORS[status]

        # Compliance-coloured border
        cv2.rectangle(frame, (x1, y1), (x2, y2), v_col, 3)

        # Worker ID badge — same colour as the compliance border (no per-worker colours)
        badge = f" W{wid} "
        (bw, bh), _ = cv2.getTextSize(badge, FONT, 0.50, 1)
        bx1 = x1
        by1 = max(0, y1 - bh - 6)
        bx2 = x1 + bw + 4
        by2 = y1
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), v_col, -1)
        cv2.putText(frame, badge, (bx1 + 2, by2 - 3),
                    FONT, 0.50, (0, 0, 0), 1, LINE_AA)

# =============================================================================
# Layer 4b — Verdict overlay box
# =============================================================================
 
def _draw_verdict_box(img: np.ndarray, report: dict, x: int, y: int):
    """
    Draw the verdict box at (x, y).
    Fixed size: 210 × 88 px.
    Returns (x1, y1, x2, y2).
    """
    verdict = report["scene_verdict"]
    if not verdict:
        return (x, y, x + 210, y + 88)
 
    v_col = VERDICT_COLORS.get(verdict, (100, 100, 100))
    W, H  = 210, 88
 
    _rounded_fill(img, x, y, x + W, y + H, (18, 18, 18), v_col)
 
    # Coloured square indicator
    sq = 14
    cv2.rectangle(img, (x + 10, y + 10), (x + 10 + sq, y + 10 + sq), v_col, -1)
 
    # Verdict text
    _put(img, verdict, x + 32, y + 22, 0.65, v_col, 2)
 
    # Divider
    cv2.line(img, (x + 8, y + 32), (x + W - 8, y + 32), (70, 70, 70), 1)
 
    # Counts
    _put(img, f"Workers  : {report['total_workers']}",
         x + 10, y + 50, 0.42, (210, 210, 210))
    _put(img, f"Compliant: {report['compliant_workers']}",
         x + 10, y + 68, 0.42, (210, 210, 210))
 
    return (x, y, x + W, y + H)
 
 


# =============================================================================
# Layer 4c — PPE Status overlay box  (scrollable)
# =============================================================================
 
def _build_ppe_lines(report: dict) -> list:
    """
    Build the list of (text, color, bold) tuples that make up the PPE panel.
    One block per worker: header → OK items → violations → alerts → spacer.
    """
    RED  = (80,  80,  220)
    ORG  = (0,  165,  255)
    GRAY = (170, 170, 170)
 
    lines = []
    for worker in report["workers"]:
        wid    = worker["worker_id"]
        status = ("SAFE"  if worker["compliant"] and not worker["alerts"] else
                  "ALERT" if worker["compliant"] else "UNSAFE")

        # Worker header — coloured by compliance status, not by worker index
        lines.append((f"W{wid}  {status}", VERDICT_COLORS.get(status, (230, 230, 230)), True))
 
        # Detected OK
        detected = ", ".join(
            DISPLAY_NAMES.get(p, p) for p in worker["ppe_detected"]
        ) or "none"
        for chunk in _wrap(f"OK: {detected}", 28):
            lines.append((chunk, GRAY, False))
 
        # Violations (critical)
        for item in worker["violations"]:
            lines.append((f"X  {DISPLAY_NAMES.get(item, item)} missing",
                          RED, False))
 
        # Alerts (advisory)
        for item in worker["alerts"]:
            lines.append((f"!  {DISPLAY_NAMES.get(item, item)} missing",
                          ORG, False))
 
        lines.append(("", (50, 50, 50), False))   # spacer row
 
    return lines
 
 
def _draw_ppe_status_box(img: np.ndarray, report: dict, x: int, y: int):
    """
    Draw the PPE Status box at (x, y).
    Width is fixed at BOX_W. Height is capped at MAX_VIS; overflow is
    indicated by a "+N more" label — no scrollbar.
    Returns (x1, y1, x2, y2).
    """
    all_lines = _build_ppe_lines(report)
    total_h   = 28 + len(all_lines) * LINE_H + 10
    visible_h = min(total_h, MAX_VIS)
    W         = BOX_W

    _rounded_fill(img, x, y, x + W, y + visible_h, (18, 18, 18), (90, 90, 90))

    # Header
    _put(img, "PPE Status", x + 10, y + 16, 0.48, (230, 230, 230), 1)
    cv2.line(img, (x + 6, y + 22), (x + W - 6, y + 22), (70, 70, 70), 1)

    cy    = y + 28
    limit = y + visible_h - 4

    for i, (text, color, bold) in enumerate(all_lines):
        if cy + LINE_H > limit:
            remaining = len(all_lines) - i
            _put(img, f"... +{remaining} more",
                 x + 10, limit, 0.34, (120, 120, 120))
            break

        if text == "":
            cv2.line(img, (x + 8, cy + 4), (x + W - 8, cy + 4),
                     (50, 50, 50), 1)
            cy += LINE_H // 2
            continue

        if bold:
            sq = 9
            cv2.rectangle(img,
                          (x + 10, cy - 8),
                          (x + 10 + sq, cy - 8 + sq),
                          color, -1)
            _put(img, text, x + 24, cy, 0.40, color, 2)
        else:
            _put(img, text, x + 12, cy, 0.37, color, 1)

        cy += LINE_H

    return (x, y, x + W, y + visible_h)


# =============================================================================
# Layer 4d — Full frame annotation
# =============================================================================
 
def annotate_frame(frame: np.ndarray, detections: list, report: dict, draw_overlays: bool = True) -> np.ndarray:
    """
    Compose the full annotated frame:
      1. PPE bounding boxes
      2. Worker borders + ID badges
      3. Verdict overlay box  (position from _state)
      4. PPE Status overlay box  (position + scroll from _state)
    """
    out = frame.copy()
    _draw_ppe_boxes(out, detections)
    _draw_worker_borders(out, report)
    if draw_overlays:
        _draw_verdict_box(out, report, *_state["verdict_pos"])
        _draw_ppe_status_box(out, report, *_state["ppe_pos"])
    return out


_state = {
    "verdict_pos": [10, 10],    # [x, y] top-left of verdict box
    "ppe_pos":     [10, 108],   # [x, y] top-left of PPE status box
    "drag":        None,        # "verdict" | "ppe" | None
    "ox": 0, "oy": 0,           # drag offset (mouse → box origin)
    "img_shape":   (480, 640),  # (H, W) of the display frame
}
 
 
def _mouse_cb(event, mx, my, _flags, param):
    """
    OpenCV mouse callback — handles click-drag for both overlay boxes.
    """
    report = param["report"]
    s      = _state

    vx, vy = s["verdict_pos"]
    px, py = s["ppe_pos"]

    v_rect = (vx, vy, vx + 210, vy + 88)

    all_lines = _build_ppe_lines(report)
    ph = min(28 + len(all_lines) * LINE_H + 10, MAX_VIS)
    p_rect = (px, py, px + BOX_W, py + ph)

    def _inside(rect, x, y):
        return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]

    img_h, img_w = s["img_shape"]

    if event == cv2.EVENT_LBUTTONDOWN:
        if _inside(v_rect, mx, my):
            s["drag"] = "verdict"
            s["ox"]   = mx - vx
            s["oy"]   = my - vy
        elif _inside(p_rect, mx, my):
            s["drag"] = "ppe"
            s["ox"]   = mx - px
            s["oy"]   = my - py

    elif event == cv2.EVENT_MOUSEMOVE and s["drag"]:
        key = s["drag"]
        bw  = 210  if key == "verdict" else BOX_W
        bh  = 88   if key == "verdict" else ph
        nx  = max(0, min(mx - s["ox"], img_w - bw))
        ny  = max(0, min(my - s["oy"], img_h - bh))
        s[f"{key}_pos"] = [nx, ny]

    elif event == cv2.EVENT_LBUTTONUP:
        s["drag"] = None
 


# =============================================================================
# Layer 5 — Single Image Pipeline
# =============================================================================

def run_on_image(model: YOLO, source: str, output: str, conf: float) -> None:
    frame = cv2.imread(source)
    if frame is None:
        print(f"[ERROR] Cannot read image: {source}")
        sys.exit(1)
 
    results    = model.predict(source, conf=conf, verbose=False)[0]
    detections = results_to_detections(results, model.names)
    report     = check_compliance(detections)
 
    _print_report(report)
 
    if output:
        # Non-interactive — render once at default box positions and save
        annotated = annotate_frame(frame, detections, report)
        cv2.imwrite(output, annotated)
        print(f"Saved: {output}")
    else:
        # Interactive window with draggable boxes
        win = "PPE Compliance"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

 
        h, w = frame.shape[:2]
        MAX_H, MAX_W = 900, 1400
        if h > MAX_H or w > MAX_W:
            scale = min(MAX_H / h, MAX_W / w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            for det in detections:
                det["bbox"] = [v * scale for v in det["bbox"]]
            report = check_compliance(detections)

        _state["img_shape"] = frame.shape[:2]
        cv2.resizeWindow(win, frame.shape[1], frame.shape[0])
        cv2.setMouseCallback(win, _mouse_cb, {"report": report})

        print("Drag boxes with mouse  |  Hover PPE box + scroll wheel to scroll  |  Q or ESC to quit")
        while True:
            annotated = annotate_frame(frame, detections, report)
            cv2.imshow(win, annotated)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                break
 
        cv2.destroyAllWindows()


# =============================================================================
# Layer 6 — Video / Webcam Pipeline
# =============================================================================

def run_on_video(model: YOLO, source: str, output: str, conf: float) -> None:
    cap_source = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(cap_source)
 
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        sys.exit(1)
 
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Compute display scale once — only applied to the live window, not the saved file
    MAX_DISP_H, MAX_DISP_W = 900, 1400
    display_scale = 1.0
    disp_w, disp_h = w, h
    if not output and (h > MAX_DISP_H or w > MAX_DISP_W):
        display_scale = min(MAX_DISP_H / h, MAX_DISP_W / w)
        disp_w = int(w * display_scale)
        disp_h = int(h * display_scale)
    _state["img_shape"] = (disp_h, disp_w)

    writer = None
    if output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output, fourcc, fps, (w, h))

    win = "PPE Compliance"
    if not output:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, disp_w, disp_h)

    live_report: dict = {
        "scene_verdict": "",
        "total_workers": 0,
        "compliant_workers": 0,
        "workers": [],
    }

    first_frame = True
    print("Running — press Q or ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results    = model.predict(frame, conf=conf, verbose=False)[0]
        detections = results_to_detections(results, model.names)
        report     = check_compliance(detections)

        if writer:
            # Save full-resolution annotated frame to file
            writer.write(annotate_frame(frame, detections, report))
        else:
            # Scale frame and bboxes for the display window
            if display_scale != 1.0:
                disp_frame = cv2.resize(frame, (disp_w, disp_h))
                disp_dets  = [dict(d, bbox=[v * display_scale for v in d["bbox"]]) for d in detections]
                disp_rep   = check_compliance(disp_dets)
            else:
                disp_frame, disp_dets, disp_rep = frame, detections, report

            live_report.update(disp_rep)

            if first_frame:
                cv2.setMouseCallback(win, _mouse_cb, {"report": live_report})
                first_frame = False
                print("Drag boxes with mouse  |  Scroll wheel on PPE box  |  Q or ESC to quit")

            cv2.imshow(win, annotate_frame(disp_frame, disp_dets, disp_rep))
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
 
    cap.release()
    if writer:
        writer.release()
        print(f"Saved: {output}")
    cv2.destroyAllWindows()

# =============================================================================
# Utility — Console Report
# =============================================================================

def _print_report(report: dict) -> None:
    print("\n" + "=" * 50)
    print(f"  Scene Verdict : {report['scene_verdict']}")
    print(f"  Workers       : {report['total_workers']}")
    print(f"  Compliant     : {report['compliant_workers']}")
    for w in report["workers"]:
        status = "SAFE" if w["compliant"] and not w["alerts"] else \
                 "ALERT" if w["compliant"] else "UNSAFE"
        print(f"\n  Worker {w['worker_id']} — {status}")
        print(f"    PPE detected : {', '.join(w['ppe_detected']) or 'none'}")
        if w["violations"]:
            print(f"    Violations   : {', '.join(w['violations'])}")
        if w["alerts"]:
            print(f"    Alerts       : {', '.join(w['alerts'])}")
    print("=" * 50 + "\n")


# =============================================================================
# Layer 7 — Main
# =============================================================================

def main():
    args = parse_args()
    model = load_model(args.weights)

    # Route to image or video pipeline based on source type
    source = args.source
    if source.isdigit():
        run_on_video(model, source, args.output, args.conf)
    else:
        ext = Path(source).suffix.lower()
        if ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            run_on_image(model, source, args.output, args.conf)
        elif ext in {".mp4", ".avi", ".mov", ".mkv"}:
            run_on_video(model, source, args.output, args.conf)
        else:
            print(f"[ERROR] Unsupported source format: {ext}")
            sys.exit(1)


if __name__ == "__main__":
    main()
