# Aegis Architecture Diagram

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         AEGIS SYSTEM                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    FRONTEND (React)                          │  │
│  ├──────────────────────────────────────────────────────────────┤  │
│  │                                                              │  │
│  │  ┌─────────────────┐  ┌──────────────┐  ┌──────────────┐   │  │
│  │  │  VideoStream    │  │  Controls    │  │  Dashboard   │   │  │
│  │  │  (Webcam)       │  │  (Toggle)    │  │  (Risk)      │   │  │
│  │  └────────┬────────┘  └──────────────┘  └──────────────┘   │  │
│  │           │                                                  │  │
│  │  ┌────────┴──────────────────────────────────────────────┐  │  │
│  │  │         ImageUpload (Group Photos)                   │  │  │
│  │  └────────┬──────────────────────────────────────────────┘  │  │
│  │           │                                                  │  │
│  └───────────┼──────────────────────────────────────────────────┘  │
│              │                                                      │
│              │ JPEG Frames (FormData)                              │
│              │ 6-8 FPS                                             │
│              ▼                                                      │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                  BACKEND (FastAPI)                          │  │
│  ├─────────────────────────────────────────────────────────────┤  │
│  │                                                             │  │
│  │  ┌──────────────────────────────────────────────────────┐  │  │
│  │  │  /process_frame  /process_image  /register_face     │  │  │
│  │  │  /set_safe_mode  /reset_safe     /faces             │  │  │
│  │  └──────────────────────────────────────────────────────┘  │  │
│  │                                                             │  │
│  │  ┌──────────────────────────────────────────────────────┐  │  │
│  │  │           PROCESSING PIPELINE                        │  │  │
│  │  ├──────────────────────────────────────────────────────┤  │  │
│  │  │                                                      │  │  │
│  │  │  1. Decode JPEG → BGR Array                         │  │  │
│  │  │     └─ Input validation                             │  │  │
│  │  │                                                      │  │  │
│  │  │  2. Resize Frame (max 640px)                        │  │  │
│  │  │     └─ Performance optimization                     │  │  │
│  │  │                                                      │  │  │
│  │  │  3. YOLO Detection (100ms)                          │  │  │
│  │  │     ├─ Detect: person, laptop, phone, tv           │  │  │
│  │  │     └─ Output: bounding boxes + confidence         │  │  │
│  │  │                                                      │  │  │
│  │  │  4. Face Recognition (20ms)                         │  │  │
│  │  │     ├─ Extract face regions (top 40%)              │  │  │
│  │  │     ├─ HSV histogram encoding                       │  │  │
│  │  │     └─ Match: known/unknown/safe                    │  │  │
│  │  │                                                      │  │  │
│  │  │  5. Risk Scoring (1ms)                              │  │  │
│  │  │     ├─ Unknown face: +30                            │  │  │
│  │  │     ├─ Device: +20                                  │  │  │
│  │  │     └─ Cap at 100                                   │  │  │
│  │  │                                                      │  │  │
│  │  │  6. Redaction (25ms)                                │  │  │
│  │  │     ├─ Emoji overlay (unknown faces)                │  │  │
│  │  │     └─ Gaussian blur (devices)                      │  │  │
│  │  │                                                      │  │  │
│  │  │  7. Encode → Base64                                 │  │  │
│  │  │     └─ JPEG quality: 80                             │  │  │
│  │  │                                                      │  │  │
│  │  └──────────────────────────────────────────────────────┘  │  │
│  │                                                             │  │
│  │  ┌──────────────────────────────────────────────────────┐  │  │
│  │  │           VISION MODULES (Singletons)               │  │  │
│  │  ├──────────────────────────────────────────────────────┤  │  │
│  │  │                                                      │  │  │
│  │  │  ┌─────────────────────────────────────────────┐   │  │  │
│  │  │  │ yolo_detector.py                           │   │  │  │
│  │  │  │ ├─ YOLOv8-nano model                       │   │  │  │
│  │  │  │ ├─ Detects: person, laptop, phone, tv     │   │  │  │
│  │  │  │ └─ Returns: boxes, confidence, class_name │   │  │  │
│  │  │  └─────────────────────────────────────────────┘   │  │  │
│  │  │                                                      │  │  │
│  │  │  ┌─────────────────────────────────────────────┐   │  │  │
│  │  │  │ face_recognizer.py                         │   │  │  │
│  │  │  │ ├─ HSV histogram encoding                  │   │  │  │
│  │  │  │ ├─ Safe mode (auto-register)               │   │  │  │
│  │  │  │ ├─ Registry (manual registration)          │   │  │  │
│  │  │  │ └─ Returns: known/unknown classification   │   │  │  │
│  │  │  └─────────────────────────────────────────────┘   │  │  │
│  │  │                                                      │  │  │
│  │  │  ┌─────────────────────────────────────────────┐   │  │  │
│  │  │  │ ocr.py                                      │   │  │  │
│  │  │  │ ├─ EasyOCR (text extraction)                │   │  │  │
│  │  │  │ ├─ Regex PII detection                      │   │  │  │
│  │  │  │ └─ Returns: text, confidence, pii_types    │   │  │  │
│  │  │  └─────────────────────────────────────────────┘   │  │  │
│  │  │                                                      │  │  │
│  │  └──────────────────────────────────────────────────────┘  │  │
│  │                                                             │  │
│  │  ┌──────────────────────────────────────────────────────┐  │  │
│  │  │           CORE MODULES (Singletons)                 │  │  │
│  │  ├──────────────────────────────────────────────────────┤  │  │
│  │  │                                                      │  │  │
│  │  │  ┌─────────────────────────────────────────────┐   │  │  │
│  │  │  │ risk_engine.py                             │   │  │  │
│  │  │  │ ├─ Scoring logic                           │   │  │  │
│  │  │  │ ├─ Risk levels: LOW/MEDIUM/HIGH            │   │  │  │
│  │  │  │ └─ Returns: score, level, labels           │   │  │  │
│  │  │  └─────────────────────────────────────────────┘   │  │  │
│  │  │                                                      │  │  │
│  │  │  ┌─────────────────────────────────────────────┐   │  │  │
│  │  │  │ redactor.py                                │   │  │  │
│  │  │  │ ├─ Emoji overlay (green-screen removal)    │   │  │  │
│  │  │  │ ├─ Gaussian blur (devices)                 │   │  │  │
│  │  │  │ └─ Returns: redacted frame                 │   │  │  │
│  │  │  └─────────────────────────────────────────────┘   │  │  │
│  │  │                                                      │  │  │
│  │  └──────────────────────────────────────────────────────┘  │  │
│  │                                                             │  │
│  └─────────────────────────────────────────────────────────────┘  │
│              │                                                      │
│              │ JSON Response (Base64 + Risk Data)                  │
│              │                                                      │
│              ▼                                                      │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                  FRONTEND DISPLAY                           │  │
│  ├─────────────────────────────────────────────────────────────┤  │
│  │                                                             │  │
│  │  ┌──────────────────────────────────────────────────────┐  │  │
│  │  │  Processed Frame (with emoji/blur)                  │  │  │
│  │  └──────────────────────────────────────────────────────┘  │  │
│  │                                                             │  │
│  │  Risk Score: 45/100                                        │  │
│  │  Risk Level: MEDIUM                                        │  │
│  │  Labels: 1 Unknown Face(s), 1 Device(s)                    │  │
│  │                                                             │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Diagram

```
┌─────────────┐
│ Webcam      │
│ Frame       │
└──────┬──────┘
       │ JPEG (6-8 FPS)
       ▼
┌─────────────────────────────────────┐
│ Frontend: VideoStream Component     │
│ - Capture from canvas               │
│ - Convert to FormData               │
└──────┬──────────────────────────────┘
       │ POST /process_frame
       │ Content-Type: multipart/form-data
       ▼
┌─────────────────────────────────────┐
│ Backend: FastAPI                    │
│ - Receive JPEG                      │
│ - Decode to BGR array               │
│ - Resize if needed                  │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ YOLO Detection                      │
│ - Detect objects                    │
│ - Return bounding boxes             │
└──────┬──────────────────────────────┘
       │
       ├─────────────────────────────────────┐
       │                                     │
       ▼                                     ▼
┌──────────────────────┐        ┌──────────────────────┐
│ Person Boxes         │        │ Device Boxes         │
│ (for face recog)     │        │ (for blur)           │
└──────┬───────────────┘        └──────┬───────────────┘
       │                               │
       ▼                               │
┌──────────────────────┐               │
│ Face Recognition     │               │
│ - Extract face ROI   │               │
│ - HSV histogram      │               │
│ - Match known/safe   │               │
└──────┬───────────────┘               │
       │                               │
       ├───────────────────────────────┤
       │                               │
       ▼                               ▼
┌──────────────────────────────────────────────┐
│ Risk Engine                                  │
│ - Unknown face: +30                          │
│ - Device: +20                                │
│ - Calculate risk_score, risk_level, labels   │
└──────┬───────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────┐
│ Redactor                                     │
│ - Apply emoji on unknown faces               │
│ - Apply blur on devices                      │
│ - Return redacted frame                      │
└──────┬───────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────┐
│ Encode to Base64                             │
│ - JPEG quality: 80                           │
│ - Return JSON response                       │
└──────┬───────────────────────────────────────┘
       │
       │ JSON Response:
       │ {
       │   "processed_frame": "base64...",
       │   "risk_score": 45,
       │   "risk_level": "MEDIUM",
       │   "labels": ["1 Unknown Face(s)"],
       │   ...
       │ }
       │
       ▼
┌──────────────────────────────────────────────┐
│ Frontend: Display                            │
│ - Decode base64 to image                     │
│ - Display processed frame                    │
│ - Show risk dashboard                        │
└──────────────────────────────────────────────┘
```

---

## Module Dependency Graph

```
┌─────────────────────────────────────────────────────────────┐
│                    main.py (FastAPI)                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Imports:                                                   │
│  ├─ yolo_detector (from vision/)                           │
│  ├─ face_recognizer (from vision/)                         │
│  ├─ risk_engine (from core/)                               │
│  └─ redactor (from core/)                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
    ┌─────────────┐    ┌──────────────┐    ┌──────────────┐
    │ yolo_        │    │ face_        │    │ risk_        │
    │ detector.py  │    │ recognizer.py│    │ engine.py    │
    ├─────────────┤    ├──────────────┤    ├──────────────┤
    │ YoloDetector│    │FaceRecognizer│    │ RiskEngine   │
    │ (singleton) │    │ (singleton)  │    │ (singleton)  │
    │             │    │              │    │              │
    │ Depends on: │    │ Depends on:  │    │ Depends on:  │
    │ - ultralytics│   │ - cv2        │    │ - None       │
    │ - cv2       │    │ - numpy      │    │              │
    │ - numpy     │    │              │    │              │
    └─────────────┘    └──────────────┘    └──────────────┘
                              │
                              ▼
                        ┌──────────────┐
                        │ redactor.py  │
                        ├──────────────┤
                        │ Redactor     │
                        │ (singleton)  │
                        │              │
                        │ Depends on:  │
                        │ - cv2        │
                        │ - numpy      │
                        │ - PIL        │
                        └──────────────┘

    ┌──────────────┐
    │ ocr.py       │
    ├──────────────┤
    │ OCRDetector  │
    │ (singleton)  │
    │              │
    │ Depends on:  │
    │ - easyocr    │
    │ - cv2        │
    │ - re         │
    └──────────────┘
```

---

## Risk Scoring Logic

```
┌─────────────────────────────────────────────────────────┐
│              RISK SCORING ALGORITHM                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Input: face_results, device_boxes                     │
│                                                         │
│  score = 0                                              │
│                                                         │
│  For each face in face_results:                        │
│    if face.known == False:                             │
│      score += 30                                        │
│      labels.append("Unknown Face(s)")                  │
│    else:                                                │
│      labels.append("Safe Face(s)")                     │
│                                                         │
│  For each device in device_boxes:                      │
│    score += 20                                          │
│    labels.append("Device(s)")                          │
│                                                         │
│  score = min(score, 100)  # Cap at 100                 │
│                                                         │
│  if score >= 60:                                        │
│    risk_level = "HIGH"                                 │
│  elif score >= 30:                                      │
│    risk_level = "MEDIUM"                               │
│  else:                                                  │
│    risk_level = "LOW"                                  │
│                                                         │
│  Output: {                                              │
│    "risk_score": score,                                │
│    "risk_level": risk_level,                           │
│    "labels": labels                                    │
│  }                                                      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Performance Timeline

```
Frame Received
    │
    ├─ Decode JPEG: 5ms
    │
    ├─ Resize: 5ms
    │
    ├─ YOLO Detection: 100ms ████████████████████████████████████████
    │
    ├─ Face Recognition: 20ms ████████
    │
    ├─ Risk Scoring: 1ms
    │
    ├─ Redaction: 25ms ██████████
    │
    ├─ Encode Base64: 10ms ████
    │
    └─ Total: ~146ms (6-8 FPS)

    GPU Acceleration (estimated):
    ├─ YOLO Detection: 10ms ████
    ├─ Face Recognition: 5ms ██
    ├─ Redaction: 5ms ██
    └─ Total: ~30ms (30-50 FPS)
```

---

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  PRODUCTION SETUP                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Load Balancer (Nginx)                            │  │
│  │ - Route requests                                 │  │
│  │ - SSL/TLS termination                            │  │
│  └──────────────────────────────────────────────────┘  │
│                    │                                    │
│    ┌───────────────┼───────────────┐                   │
│    │               │               │                   │
│    ▼               ▼               ▼                   │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐              │
│  │ Backend │   │ Backend │   │ Backend │              │
│  │ Instance│   │ Instance│   │ Instance│              │
│  │ (GPU)   │   │ (GPU)   │   │ (GPU)   │              │
│  └─────────┘   └─────────┘   └─────────┘              │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Database (Face Registry)                         │  │
│  │ - Store face encodings                           │  │
│  │ - Store user preferences                         │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Cache (Redis)                                    │  │
│  │ - Cache YOLO results                             │  │
│  │ - Cache face encodings                           │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Monitoring (Prometheus + Grafana)                │  │
│  │ - Track FPS, latency, errors                     │  │
│  │ - Alert on anomalies                             │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## File Structure

```
aegis/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py (FastAPI app)
│   │   ├── vision/
│   │   │   ├── __init__.py
│   │   │   ├── yolo_detector.py
│   │   │   ├── face_recognizer.py
│   │   │   └── ocr.py
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── risk_engine.py
│   │   │   └── redactor.py
│   │   └── assets/
│   │       └── emojis/ (PNG files)
│   ├── requirements.txt
│   └── run.sh
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   ├── index.css
│   │   └── components/
│   │       ├── VideoStream.jsx
│   │       ├── Controls.jsx
│   │       ├── Dashboard.jsx
│   │       └── ImageUpload.jsx
│   ├── package.json
│   └── vite.config.js
│
└── docs/
    ├── SETUP.md
    ├── DEVELOPER_GUIDE.md
    ├── PHASE_1_SUMMARY.md
    ├── PHASE_1_CHECKLIST.md
    ├── README_PHASE_1.md
    ├── QUICK_REFERENCE.md
    ├── PHASE_1_COMPLETE.md
    └── ARCHITECTURE.md (this file)
```

---

**Version**: 3.0.0
**Last Updated**: April 6, 2026
