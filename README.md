# Aegis: Real-Time Privacy Detection & Redaction

A real-time computer vision system that detects and redacts sensitive information from live video streams. Uses YOLOv8 for object detection, EasyOCR for text extraction, and regex-based PII scanning.

## Features

- **Live Video Processing** — Real-time webcam stream analysis at 20 FPS
- **Object Detection** — YOLOv8-nano detects persons, devices (laptops, phones, monitors)
- **Face Recognition** — HSV histogram-based known/unknown face classification
- **Text Extraction** — EasyOCR reads visible text; regex patterns detect PII
- **Risk Scoring** — Rule-based engine calculates LOW/MEDIUM/HIGH risk levels
- **Smart Redaction** — Gaussian blur for unknown faces and devices
- **Safe Mode** — Auto-register known faces from reference image
- **Dark UI** — React dashboard with live risk metrics and detection labels

## Architecture

```
Frontend (React)                Backend (FastAPI)
    │                               │
    ├─ VideoStream.jsx             ├─ main.py
    ├─ Controls.jsx                ├─ vision/
    ├─ Dashboard.jsx               │  ├─ yolo_detector.py
    └─ ImageUpload.jsx             │  ├─ face_detector.py
                                   │  └─ ocr.py
                                   └─ core/
                                      ├─ risk_engine.py
                                      └─ redactor.py
```

## Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

Backend runs on `http://localhost:8000`

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs on `http://localhost:5173`

## Tech Stack

| Component | Technology |
|---|---|
| Object Detection | YOLOv8-nano |
| Face Recognition | OpenCV + HSV histograms |
| Text Recognition | EasyOCR |
| PII Detection | Python regex |
| Backend | FastAPI + Uvicorn |
| Image Processing | OpenCV |
| Frontend | React 18 + Vite |

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/process_frame` | Process webcam frame (returns base64 + risk data) |
| `POST` | `/process_image` | Process uploaded image |
| `POST` | `/register_face` | Register a known face |
| `GET` | `/faces` | List registered faces |
| `DELETE` | `/faces/{name}` | Remove registered face |
| `POST` | `/set_safe_mode` | Auto-register faces from image |
| `POST` | `/reset_safe` | Clear safe mode |

## Demo

1. Start both backend and frontend
2. Click "Start Scanning" to begin live video processing
3. Unknown faces are blurred; devices are blurred
4. Risk score updates in real-time based on detections
5. Use "Safe Mode" to register known faces
6. Upload group photos with "Image Upload" for batch processing

## Project Structure

```
aegis/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── vision/
│   │   │   ├── yolo_detector.py
│   │   │   ├── face_detector.py
│   │   │   ├── face_recognizer.py
│   │   │   └── ocr.py
│   │   └── core/
│   │       ├── risk_engine.py
│   │       └── redactor.py
│   ├── requirements.txt
│   └── run.sh
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
├── README.md
├── ARCHITECTURE.md
└── .gitignore
```

## Performance

- **Latency**: ~150ms per frame (CPU), ~30ms (GPU)
- **FPS**: 20 FPS (CPU), 30-50 FPS (GPU)
- **Memory**: ~500MB (models + runtime)

## License

MIT
