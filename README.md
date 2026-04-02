# Aegis: Context-Aware Multi-Modal Privacy Detection & Redaction System

A real-time computer vision system that detects and redacts sensitive information from live video streams using **YOLOv8** object detection, **EasyOCR** text extraction, and **regex-based PII scanning**.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        BROWSER (React)                          │
│                                                                 │
│   Webcam ──► Canvas (resize 640px) ──► JPEG blob @ ~10 FPS      │
│                          │                                      │
│                          ▼                                      │
│              POST /process_frame (FormData)                     │
│                          │                                      │
│              ◄── JSON { base64 frame, risk } ──►                │
│                          │                                      │
│   Display ◄── <img src="data:..."> ──► Dashboard (score/labels) │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     BACKEND (FastAPI)                            │
│                                                                 │
│   main.py ─────┬──► yolo_detector.py  (YOLOv8n — persons,      │
│                │                        laptops, phones, TVs)   │
│                ├──► ocr.py            (EasyOCR + Regex PII)     │
│                ├──► risk_engine.py    (Rule-based scoring)       │
│                └──► redactor.py       (OpenCV Gaussian blur)     │
│                                                                 │
│   Models loaded ONCE at startup (singleton pattern)             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Features

| Feature | Detail |
|---|---|
| **Object Detection** | YOLOv8-nano detects persons, laptops, phones, monitors |
| **Text Extraction** | EasyOCR reads visible text from frames |
| **PII Detection** | Regex patterns flag phone numbers, ID numbers, emails, Aadhaar |
| **Risk Scoring** | Rule-based engine: Person +10, Text +20, PII +40 (capped 100) |
| **Redaction** | Gaussian blur applied to all detected regions |
| **Real-Time UI** | Dark-themed React dashboard with live score & label display |

---

## How It Works

1. **Capture** — The browser captures webcam frames at ~10 FPS via Canvas API.
2. **Send** — Each frame is resized to 640px max width and sent as JPEG to the backend.
3. **Detect** — YOLOv8 identifies objects; EasyOCR extracts text; regex scans for PII.
4. **Score** — The risk engine sums points per detection type → LOW / MEDIUM / HIGH.
5. **Redact** — Sensitive regions are Gaussian-blurred using OpenCV.
6. **Return** — The processed frame (base64) + risk metadata is returned as JSON.
7. **Display** — The frontend renders the redacted frame and updates the dashboard.

---

## Folder Structure

```
aegis/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI routes & pipeline
│   │   ├── vision/
│   │   │   ├── yolo_detector.py    # YOLOv8 wrapper
│   │   │   └── ocr.py             # EasyOCR + regex PII
│   │   └── core/
│   │       ├── risk_engine.py      # Rule-based scoring
│   │       └── redactor.py         # OpenCV blur/pixelate
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx                 # Root layout
│   │   ├── index.css               # Global styles
│   │   └── components/
│   │       ├── VideoStream.jsx     # Webcam + frame loop
│   │       ├── Controls.jsx        # Toggle + start/stop
│   │       └── Dashboard.jsx       # Risk score + labels
│   └── package.json
│
├── README.md
└── .gitignore
```

---

## Setup & Run

### Prerequisites
- Python 3.9+
- Node.js 18+
- [`uv`](https://github.com/astral-sh/uv) (fast Python package manager)

### Backend

```bash
cd backend
uv venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
uvicorn app.main:app --reload
```

> **Note:** First run downloads YOLOv8n (~6 MB) and EasyOCR models (~100 MB).

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open the URL shown in the terminal (usually `http://localhost:5173`).

---

## Quick Start for Collaborators

```bash
git clone <your-repo-url>

# Terminal 1 — Backend
cd backend && uv venv && source .venv/bin/activate && uv pip install -r requirements.txt
uvicorn app.main:app --reload

# Terminal 2 — Frontend
cd frontend && npm install && npm run dev
```

---

## Technologies Used

| Layer | Technology |
|---|---|
| Object Detection | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) |
| Text Recognition | [EasyOCR](https://github.com/JaidedAI/EasyOCR) |
| PII Detection | Python `re` (regex) |
| Backend Framework | [FastAPI](https://fastapi.tiangolo.com/) |
| Image Processing | [OpenCV](https://opencv.org/) |
| Frontend | [React](https://react.dev/) + [Vite](https://vite.dev/) |
| Package Manager | [uv](https://github.com/astral-sh/uv) |

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/process_frame` | Process a frame → returns JSON with `processed_frame` (base64), `risk_score`, `risk_level`, `labels` |
| `GET` | `/stream` | Placeholder for future streaming |

---

## License

MIT
