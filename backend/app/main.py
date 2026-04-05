"""
main.py — Aegis FastAPI Backend (v3)

Two modes:
  1. Live Camera:  POST /process_frame  (webcam frames at ~6-8 FPS)
  2. Image Upload: POST /process_image  (single image processing)

Supporting endpoints:
  POST   /register_face   — upload a face photo to register
  GET    /faces            — list registered faces
  DELETE /faces/{name}     — remove a registered face
  POST   /set_safe_mode    — auto-register center-most person as safe
  POST   /reset_safe       — clear safe mode
"""

import base64
import random
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np

from app.vision.yolo_detector import yolo_detector
from app.vision.face_recognizer import face_recognizer
from app.core.risk_engine import risk_engine
from app.core.redactor import redactor

# ── App ───────────────────────────────────────────────────────
app = FastAPI(title="Aegis Privacy System", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Stable emoji assignment per face position (prevents flickering)
_emoji_cache: dict[str, int] = {}


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


def _process_pipeline(frame: np.ndarray, do_redact: bool = True) -> dict:
    """
    Shared pipeline for both live camera and image upload modes.

    Steps:
      1. YOLO detection → persons + devices
      2. Face recognition → classify known/unknown
      3. Risk scoring
      4. Redaction: emoji on unknown faces, blur on devices
    """
    # 1. YOLO
    yolo_results = yolo_detector.detect(frame)

    # Split into persons and devices
    person_dets = [d for d in yolo_results if d["class_name"] == "person"]
    device_dets = [d for d in yolo_results if d["class_name"] in ("cell phone", "laptop", "tv")]

    # 2. Face recognition (uses top 40% of person boxes)
    face_results = face_recognizer.classify_persons(frame, person_dets)

    # 3. Risk scoring
    device_boxes = [d["box"] for d in device_dets]
    risk_data = risk_engine.evaluate(face_results, device_boxes)

    # 4. Redaction
    if do_redact:
        for face in face_results:
            if not face["known"]:
                idx = _stable_emoji(face["face_box"])
                frame = redactor.apply_emoji(frame, face["face_box"], emoji_index=idx)

        if device_boxes:
            frame = redactor.blur_region(frame, device_boxes)

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
    }


# ── Routes ────────────────────────────────────────────────────

@app.get("/")
def health():
    return {
        "status": "ok",
        "version": "3.0.0",
        "safe_mode": face_recognizer.is_safe_mode,
        "registered_faces": face_recognizer.list_faces(),
    }


# ── Live Camera Mode ─────────────────────────────────────────

@app.post("/process_frame")
async def process_frame(file: UploadFile = File(...), redact: bool = Form(True)):
    """Process a single webcam frame (called at ~6-8 FPS)."""
    image = _decode_image(await file.read())
    if image is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})

    frame = _resize(image)
    return _process_pipeline(frame, do_redact=redact)


@app.post("/set_safe_mode")
async def set_safe_mode(file: UploadFile = File(...)):
    """
    Capture current frame and register the center-most person as 'safe'.
    That person will not be masked in subsequent frames.
    """
    image = _decode_image(await file.read())
    if image is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})

    frame = _resize(image)
    yolo_results = yolo_detector.detect(frame)

    success = face_recognizer.set_safe_from_frame(frame, yolo_results)
    if not success:
        return JSONResponse(
            status_code=400,
            content={"error": "No person detected. Stand in front of the camera."},
        )

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
    """
    Process an uploaded group image. Uses registered faces for matching.
    Returns processed image with emojis on unknown faces and blur on devices.
    """
    image = _decode_image(await file.read())
    if image is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})

    # Don't resize too aggressively for uploaded images (better quality)
    frame = _resize(image, max_w=1024)
    return _process_pipeline(frame, do_redact=True)
