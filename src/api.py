from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import sys
import os
import cv2
import numpy as np
import base64
from PIL import Image
import io
import tempfile
import asyncio

# Add the src directory to sys.path so modules can find each other
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from compliance import check_compliance
from inference import results_to_detections, annotate_frame

app = FastAPI(title="PPE Detection API")

# Setup CORS so the frontend can communicate with it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model globally
model_path = os.path.join("models", "best.pt")
model = YOLO(model_path)


def _decode_image(contents: bytes) -> np.ndarray:
    """
    Decode uploaded image bytes into a BGR numpy array.
    Tries OpenCV first, falls back to PIL for formats OpenCV can't handle.
    """
    # Attempt 1: OpenCV
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is not None:
        return frame

    # Attempt 2: PIL fallback (handles more formats like WebP, AVIF, etc.)
    try:
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        frame = np.array(pil_img)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame
    except Exception:
        return None


@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    # Read the file contents and decode the image
    contents = await file.read()
    frame = _decode_image(contents)

    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode the uploaded image. Please use JPG or PNG.")

    # Run inference
    results = model.predict(frame, conf=0.30, iou=0.45, verbose=False)[0]
    detections = results_to_detections(results, model.names)
    
    # Run compliance check
    report = check_compliance(detections)

    # Annotate frame
    annotated_frame = annotate_frame(frame, detections, report, draw_overlays=False)
    
    # Encode annotated frame to base64
    _, buffer = cv2.imencode('.jpg', annotated_frame)
    base64_img = base64.b64encode(buffer).decode('utf-8')

    return {
        "report": report,
        "annotated_image": f"data:image/jpeg;base64,{base64_img}"
    }

import threading
import queue


def _process_video_to_frames(in_path: str):
    """
    Process video frame-by-frame and return sampled annotated frames as base64 images.
    Each frame includes its own compliance report for dynamic dashboard updates.
    """
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise ValueError("Failed to open video file")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Target ~8 output FPS for smooth dashboard playback without huge payloads
    target_fps = min(8, src_fps)
    skip = max(1, int(src_fps / target_fps))

    print(f"[VIDEO] Source FPS: {src_fps}, Sampling every {skip} frame(s), Target FPS: {target_fps}")

    frames = []
    reports = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only process sampled frames
        if frame_idx % skip == 0:
            results = model.predict(frame, conf=0.30, iou=0.45, verbose=False)[0]
            detections = results_to_detections(results, model.names)
            report = check_compliance(detections)
            annotated = annotate_frame(frame, detections, report, draw_overlays=False)

            _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 65])
            b64 = base64.b64encode(buf).decode('utf-8')
            frames.append(f"data:image/jpeg;base64,{b64}")
            reports.append(report)

        frame_idx += 1

    cap.release()
    print(f"[VIDEO] Processed {frame_idx} source frames -> {len(frames)} output frames")

    return {
        "frames": frames,
        "reports": reports,
        "fps": target_fps,
        "total_source_frames": frame_idx,
        "processed_frames": len(frames),
    }


@app.post("/analyze_video")
async def analyze_video(file: UploadFile = File(...)):
    fd_in, in_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd_in)

    contents = await file.read()
    with open(in_path, "wb") as f:
        f.write(contents)

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _process_video_to_frames, in_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[VIDEO ERROR] {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Video processing failed: {e}")
    finally:
        # Clean up input temp file
        try:
            os.unlink(in_path)
        except OSError:
            pass

    return result


def _stream_worker(source, frame_queue: queue.Queue, stop_event: threading.Event):
    """
    Runs in a dedicated thread. Captures frames, runs inference, puts results in a queue.
    """
    print(f"[STREAM THREAD] Opening source: {source}")

    if isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        frame_queue.put({"error": f"Cannot open video source: {source}"})
        return

    print(f"[STREAM THREAD] Source opened successfully")

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                frame_queue.put({"error": "End of video stream"})
                break

            results = model.predict(frame, conf=0.30, iou=0.45, verbose=False)[0]
            detections = results_to_detections(results, model.names)
            report = check_compliance(detections)
            annotated_frame = annotate_frame(frame, detections, report, draw_overlays=False)

            _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            base64_img = base64.b64encode(buffer).decode('utf-8')

            # Put result in queue, drop old frames if queue is full
            try:
                frame_queue.put_nowait({
                    "report": report,
                    "annotated_image": f"data:image/jpeg;base64,{base64_img}"
                })
            except queue.Full:
                # Drop oldest frame and put new one
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
                frame_queue.put_nowait({
                    "report": report,
                    "annotated_image": f"data:image/jpeg;base64,{base64_img}"
                })
    except Exception as e:
        print(f"[STREAM THREAD ERROR] {e}")
        frame_queue.put({"error": str(e)})
    finally:
        cap.release()
        print("[STREAM THREAD] Camera released")


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()

    try:
        config = await websocket.receive_json()
        source = config.get("source", 0)

        if isinstance(source, str) and source.isdigit():
            source = int(source)

        print(f"[STREAM] Starting stream for source: {source}")

        # Create a queue and stop event for thread communication
        frame_queue = queue.Queue(maxsize=2)
        stop_event = threading.Event()

        # Start the capture/inference thread
        worker = threading.Thread(
            target=_stream_worker,
            args=(source, frame_queue, stop_event),
            daemon=True
        )
        worker.start()

        while True:
            try:
                # Wait for a frame from the worker thread (non-blocking with timeout)
                data = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: frame_queue.get(timeout=5.0)
                )
            except Exception:
                # Timeout — no frames for 5 seconds
                print("[STREAM] No frames received for 5s, closing")
                break

            if "error" in data:
                await websocket.send_json(data)
                break

            await websocket.send_json(data)

    except WebSocketDisconnect:
        print("[STREAM] Client disconnected")
    except Exception as e:
        print(f"[STREAM ERROR] {e}")
    finally:
        stop_event.set()
        print("[STREAM] Stop event set, waiting for worker...")
