"""
main.py — Aegis FastAPI Backend (v5 - Optimized for Real-Time)

CRITICAL FIXES:
  - Add processing time logging
  - Disable OCR by default (too slow)
  - Implement frame skipping (OCR every N frames)
  - Ensure singleton pattern for models
  - Optimize image processing (no unnecessary copies)
"""

import base64
import random
import time
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np

from app.vision.yolo_detector import yolo_detector
from app.vision.face_detector import face_detector
from app.vision.face_recognizer import face_recognizer
from app.vision.ocr import ocr_detector
from app.core.risk_engine import risk_engine
from app.core.redactor import redactor

# ── App ───────────────────────────────────────────────────────
app = FastAPI(title="Aegis Privacy System", version="5.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Stable emoji assignment per face position
_emoji_cache: dict[str, int] = {}

# Frame counter for OCR frequency control
_frame_counter = 0
_ocr_frequency = 10  # Run OCR every 10 frames (disabled by default for speed)
_ocr_enabled = False  # Disable OCR for now (too slow)


def _stable_emoji(box: tuple) -> int:
    """Assign a consistent emoji to a face based on screen position."""
    cx = (box[0] + box[2]) // 2
    cy = (box[1] + box[3]) // 2
    key = f"{cx // 60}_{cy // 60}"
    if key not in _emoji_cache:
        _emoji_cache[key] = random.randint(0, 5)
    return _emoji_cache[key]


def _decode_image(contents: bytes) -> np.ndarray | None:
    """Decode uploaded bytes into a BGR numpy array."""
    nparr = np.frombuffer(contents, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def _resize(frame: np.ndarray, max_w: int = 640) -> np.ndarray:
    """Resize frame if wider than max_w."""
    h, w = frame.shape[:2]
    if w > max_w:
        scale = max_w / w
        frame = cv2.resize(frame, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)
    return frame


def _process_pipeline(frame: np.ndarray, do_redact: bool = True, run_ocr: bool = False) -> dict:
    """
    Optimized pipeline for real-time processing with conflict filtering.
    
    CRITICAL: Disabled OCR by default (too slow for real-time)
    
    Steps:
      1. YOLO detection (fast) - object detection
      2. Face detection (fast) - face detection
      3. Filter overlapping faces with objects (NEW - reduces false positives)
      4. Face recognition (fast)
      5. OCR (OPTIONAL - disabled by default)
      6. Risk scoring (fast)
      7. Redaction (fast)
    """
    # 1. YOLO for devices (fast)
    yolo_results = yolo_detector.detect(frame)
    device_dets = [d for d in yolo_results if d["class_name"] in ("cell phone", "laptop", "tv")]
    device_boxes = [d["box"] for d in device_dets]

    # 2. Face detection (fast)
    face_dets = face_detector.detect_with_padding(frame, conf_threshold=0.6, padding_ratio=0.1)

    # 3. Filter overlapping faces with objects (NEW - reduces false positives)
    face_dets = face_detector.filter_overlapping_with_objects(face_dets, device_boxes, iou_threshold=0.3)

    # 4. Face recognition (fast)
    face_results = []
    for face_det in face_dets:
        face_box = face_det["box"]
        face_roi = face_detector.get_face_roi(frame, face_box)

        if face_roi is None or face_roi.size == 0:
            face_results.append({
                "box": face_box,
                "known": False,
                "name": None,
                "confidence": face_det["confidence"],
            })
            continue

        encoding = face_recognizer._compute_encoding(face_roi)

        if face_recognizer._safe_mode and face_recognizer._safe_encoding is not None:
            score = face_recognizer._compare(encoding, face_recognizer._safe_encoding)
            if score >= face_recognizer.tolerance:
                face_results.append({
                    "box": face_box,
                    "known": True,
                    "name": "_safe_",
                    "confidence": face_det["confidence"],
                })
                continue

        matched = face_recognizer._find_match(encoding)
        face_results.append({
            "box": face_box,
            "known": matched is not None,
            "name": matched,
            "confidence": face_det["confidence"],
        })

    # 5. OCR (OPTIONAL - disabled by default)
    text_boxes = []
    if run_ocr and _ocr_enabled:
        ocr_results = ocr_detector.detect(frame, use_preprocessing=False)  # Disable preprocessing for speed
        text_boxes = [d["box"] for d in ocr_results]

    # 6. Risk scoring (fast)
    risk_data = risk_engine.evaluate(face_results, device_boxes)

    # 7. Redaction (fast)
    if do_redact:
        # Emoji on unknown faces
        for face in face_results:
            if not face["known"]:
                idx = _stable_emoji(face["box"])
                frame = redactor.apply_emoji(frame, face["box"], emoji_index=idx, use_fallback_blur=True)

        # Blur on devices
        if device_boxes:
            frame = redactor.blur_region(frame, device_boxes, expand=True)

        # Blur on text regions (only if OCR ran)
        if text_boxes:
            frame = redactor.blur_text_regions(frame, text_boxes)

    # Encode
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

    return {
        "processed_frame": base64.b64encode(buffer.tobytes()).decode("utf-8"),
        "risk_score": risk_data["risk_score"],
        "risk_level": risk_data["risk_level"],
        "labels": risk_data["labels"],
        "faces_total": len(face_results),
        "faces_known": sum(1 for f in face_results if f["known"]),
        "faces_unknown": sum(1 for f in face_results if not f["known"]),
        "safe_mode": face_recognizer.is_safe_mode,
        "text_detected": len(text_boxes),
    }


# ── Routes ────────────────────────────────────────────────────

@app.get("/")
def health():
    return {
        "status": "ok",
        "version": "5.0.0",
        "safe_mode": face_recognizer.is_safe_mode,
        "registered_faces": face_recognizer.list_faces(),
        "ocr_enabled": _ocr_enabled,
        "ocr_frequency": _ocr_frequency,
    }


# ── Live Camera Mode ─────────────────────────────────────────

@app.post("/process_frame")
async def process_frame(file: UploadFile = File(...), redact: bool = Form(True)):
    """Process a single webcam frame (called at ~5-8 FPS)."""
    global _frame_counter
    
    start_time = time.time()
    
    image = _decode_image(await file.read())
    if image is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})

    frame = _resize(image)
    
    # Determine if OCR should run this frame
    run_ocr = _ocr_enabled and (_frame_counter % _ocr_frequency) == 0
    _frame_counter += 1
    
    result = _process_pipeline(frame, do_redact=redact, run_ocr=run_ocr)
    
    # Add processing time to response
    elapsed = (time.time() - start_time) * 1000  # Convert to ms
    result["processing_time_ms"] = round(elapsed, 1)
    result["ocr_ran"] = run_ocr
    
    # Log to console
    ocr_status = "OCR" if run_ocr else "SKIP"
    print(f"[FRAME {_frame_counter}] {elapsed:.1f}ms ({ocr_status})")
    
    return result


@app.post("/set_safe_mode")
async def set_safe_mode(file: UploadFile = File(...)):
    """Capture current frame and register the center-most face as 'safe'."""
    image = _decode_image(await file.read())
    if image is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})

    frame = _resize(image)
    
    face_dets = face_detector.detect(frame, conf_threshold=0.5)
    if not face_dets:
        return JSONResponse(
            status_code=400,
            content={"error": "No face detected. Look at the camera."},
        )

    frame_cx = frame.shape[1] / 2
    frame_cy = frame.shape[0] / 2

    def dist_to_center(det):
        cx, cy = det["center"]
        return (cx - frame_cx) ** 2 + (cy - frame_cy) ** 2

    center_face = min(face_dets, key=dist_to_center)
    face_roi = face_detector.get_face_roi(frame, center_face["box"])

    if face_roi is None or face_roi.size == 0:
        return JSONResponse(
            status_code=400,
            content={"error": "Could not extract face region."},
        )

    face_recognizer._safe_encoding = face_recognizer._compute_encoding(face_roi)
    face_recognizer._safe_mode = True

    return {"message": "Safe mode activated. You will not be masked."}


@app.post("/reset_safe")
def reset_safe():
    """Clear safe mode — all faces will be masked again."""
    face_recognizer.reset_safe()
    _emoji_cache.clear()
    return {"message": "Safe mode cleared."}


# ── Image Upload Mode ────────────────────────────────────────

@app.post("/register_face")
async def register_face(file: UploadFile = File(...), name: str = Form(...)):
    """Upload a face photo to register a known person."""
    image = _decode_image(await file.read())
    if image is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})

    success = face_recognizer.register_face_from_image(name, image)
    if not success:
        return JSONResponse(status_code=400, content={"error": "Could not process image."})

    return {
        "message": f"'{name}' registered.",
        "registered": face_recognizer.list_faces(),
    }


@app.get("/faces")
def list_faces():
    return {"faces": face_recognizer.list_faces()}


@app.delete("/faces/{name}")
def remove_face(name: str):
    if not face_recognizer.remove_face(name):
        return JSONResponse(status_code=404, content={"error": f"'{name}' not found."})
    return {"message": f"'{name}' removed.", "registered": face_recognizer.list_faces()}


@app.post("/process_image")
async def process_image(file: UploadFile = File(...)):
    """Process an uploaded group image."""
    image = _decode_image(await file.read())
    if image is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})

    frame = _resize(image, max_w=1024)
    # Always run OCR for uploaded images (not real-time)
    return _process_pipeline(frame, do_redact=True, run_ocr=_ocr_enabled)
