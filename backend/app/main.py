"""
main.py — FastAPI Application Entry Point

Exposes three endpoints:
    GET  /               → health check
    POST /process_frame  → accepts an image, returns JSON with
                           base64-encoded processed frame + risk metadata
    GET  /stream         → placeholder for future WebRTC/streaming

All heavy models (YOLO, EasyOCR) are loaded once at import time
via their respective singleton instances.
"""

import base64
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np

from app.vision.yolo_detector import yolo_detector
from app.vision.ocr import ocr_detector
from app.core.risk_engine import risk_engine
from app.core.redactor import redactor

# ── App Initialisation ────────────────────────────────────────
app = FastAPI(
    title="Aegis Privacy System",
    description="Context-Aware Multi-Modal Privacy Detection & Redaction",
    version="1.0.0",
)

# Allow cross-origin requests from the React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────

@app.get("/")
def health_check():
    """Simple liveness probe."""
    return {"status": "ok", "message": "Aegis backend is running"}


@app.post("/process_frame")
async def process_frame(
    file: UploadFile = File(...),
    redact: bool = Form(True),
):
    """
    Accept an image frame, run the full detection + redaction
    pipeline, and return a JSON response containing:
        - processed_frame : base64-encoded JPEG string
        - risk_score      : int (0–100)
        - risk_level      : "LOW" | "MEDIUM" | "HIGH"
        - labels          : list of triggered detection labels
    """
    # ── 1. Decode the uploaded image ──────────────────────────
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid image — could not decode."},
        )

    # ── 2. Resize for performance (max width 640px) ───────────
    h, w = frame.shape[:2]
    max_width = 640
    if w > max_width:
        scale = max_width / w
        frame = cv2.resize(
            frame,
            (max_width, int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )

    # ── 3. Run YOLO object detection ─────────────────────────
    yolo_results = yolo_detector.detect(frame)

    # ── 4. Run OCR + PII regex scanning ──────────────────────
    ocr_results = ocr_detector.detect(frame)

    # ── 5. Compute risk score ────────────────────────────────
    risk_data = risk_engine.evaluate(yolo_results, ocr_results)

    # ── 6. Apply redaction if requested ──────────────────────
    if redact:
        boxes = [d["box"] for d in yolo_results] + [d["box"] for d in ocr_results]
        frame = redactor.redact(frame, boxes)

    # ── 7. Encode processed frame to base64 JPEG ─────────────
    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to encode output frame."},
        )

    frame_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")

    # ── 8. Return structured JSON response ───────────────────
    return {
        "processed_frame": frame_b64,
        "risk_score": risk_data["risk_score"],
        "risk_level": risk_data["risk_level"],
        "labels": risk_data["labels"],
    }


@app.get("/stream")
def stream_placeholder():
    """Placeholder for future WebRTC / MJPEG streaming."""
    return {"message": "Streaming endpoint — not yet implemented."}
