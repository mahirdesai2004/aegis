# Aegis Developer Guide

## Quick Start

### Backend
```bash
cd aegis/backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
./run.sh
```

### Frontend
```bash
cd aegis/frontend
npm install
npm run dev
```

---

## Architecture

### Backend Stack
- **Framework**: FastAPI (async, fast)
- **Vision**: YOLOv8-nano, EasyOCR, OpenCV
- **Face Recognition**: HSV histogram matching
- **Deployment**: Uvicorn

### Frontend Stack
- **Framework**: React 18 + Vite
- **Styling**: CSS (glass-morphism design)
- **API**: Fetch API (FormData for images)

---

## Module Reference

### Backend Modules

#### `yolo_detector.py`
```python
from app.vision.yolo_detector import yolo_detector

# Detect objects in frame
detections = yolo_detector.detect(frame)
# Returns: [{"box": (x1, y1, x2, y2), "confidence": 0.95, "class_name": "person"}, ...]
```

#### `face_recognizer.py`
```python
from app.vision.face_recognizer import face_recognizer

# Classify persons as known/unknown
results = face_recognizer.classify_persons(frame, person_boxes)
# Returns: [{"box": (...), "face_box": (...), "known": True, "name": "John"}, ...]

# Register a face
face_recognizer.register_face_from_image("John", face_image)

# Set safe mode
face_recognizer.set_safe_from_frame(frame, yolo_results)
```

#### `risk_engine.py`
```python
from app.core.risk_engine import risk_engine

# Calculate risk
risk_data = risk_engine.evaluate(face_results, device_boxes)
# Returns: {"risk_score": 45, "risk_level": "MEDIUM", "labels": [...]}
```

#### `redactor.py`
```python
from app.core.redactor import redactor

# Apply emoji overlay
frame = redactor.apply_emoji(frame, face_box, emoji_index=0)

# Apply blur
frame = redactor.blur_region(frame, device_boxes)
```

#### `ocr.py`
```python
from app.vision.ocr import ocr_detector

# Extract text and detect PII
detections = ocr_detector.detect(frame)
# Returns: [{"box": (...), "text": "...", "confidence": 0.92, "pii_types": ["phone_number"]}, ...]
```

---

## API Endpoints

### Health & Status
```
GET /
Response: {
  "status": "ok",
  "version": "3.0.0",
  "safe_mode": false,
  "registered_faces": ["John", "Jane"]
}
```

### Live Camera
```
POST /process_frame
Content-Type: multipart/form-data
Body: {
  "file": <JPEG image>,
  "redact": true
}
Response: {
  "processed_frame": "base64_encoded_image",
  "risk_score": 45,
  "risk_level": "MEDIUM",
  "labels": ["1 Unknown Face(s)", "1 Device(s)"],
  "faces_total": 2,
  "faces_known": 1,
  "faces_unknown": 1,
  "safe_mode": false
}
```

### Safe Mode
```
POST /set_safe_mode
Content-Type: multipart/form-data
Body: { "file": <JPEG image> }
Response: { "message": "Safe mode activated..." }

POST /reset_safe
Response: { "message": "Safe mode cleared." }
```

### Face Registration
```
POST /register_face
Content-Type: multipart/form-data
Body: {
  "file": <JPEG image>,
  "name": "John"
}
Response: {
  "message": "'John' registered.",
  "registered": ["John"]
}

GET /faces
Response: { "faces": ["John", "Jane"] }

DELETE /faces/{name}
Response: {
  "message": "'John' removed.",
  "registered": ["Jane"]
}
```

### Image Upload
```
POST /process_image
Content-Type: multipart/form-data
Body: { "file": <JPEG image> }
Response: {
  "processed_frame": "base64_encoded_image",
  "risk_score": 60,
  "risk_level": "HIGH",
  "labels": ["2 Unknown Face(s)", "1 Device(s)"],
  ...
}
```

---

## Frontend Components

### `VideoStream.jsx`
Captures webcam frames and sends to backend.

```jsx
<VideoStream
  redact={true}
  streaming={true}
  onRiskUpdate={(data) => console.log(data)}
/>
```

### `Controls.jsx`
Toggle redaction, streaming, safe mode.

```jsx
<Controls
  redact={redact}
  setRedact={setRedact}
  streaming={streaming}
  setStreaming={setStreaming}
  safeMode={safeMode}
  setSafeMode={setSafeMode}
/>
```

### `Dashboard.jsx`
Display risk score, level, and labels.

```jsx
<Dashboard riskData={{
  score: 45,
  level: "MEDIUM",
  labels: ["1 Unknown Face(s)"]
}} />
```

### `ImageUpload.jsx`
Upload and process group photos.

```jsx
<ImageUpload onRiskUpdate={(data) => console.log(data)} />
```

---

## Common Tasks

### Add a New Object Class to YOLO
1. Edit `yolo_detector.py`
2. Add class name to `TARGET_CLASSES`
3. Restart backend

### Adjust Risk Scoring
1. Edit `risk_engine.py`
2. Modify `evaluate()` method
3. Restart backend

### Change Redaction Method
1. Edit `redactor.py`
2. Modify `blur_region()` or `apply_emoji()`
3. Restart backend

### Add PII Pattern
1. Edit `ocr.py`
2. Add regex to `PII_PATTERNS`
3. Restart backend

---

## Performance Tuning

### Reduce Latency
```python
# In yolo_detector.py
self.model = YOLO(model_path, device=0)  # Use GPU

# In main.py
frame = _resize(image, max_w=480)  # Smaller resolution
```

### Increase Accuracy
```python
# In face_recognizer.py
self.tolerance = 0.60  # Stricter matching

# In yolo_detector.py
results = self.model(frame, conf=0.5)  # Higher confidence threshold
```

### Reduce Memory
```python
# Use quantized model
self.model = YOLO("yolov8n-int8.pt")
```

---

## Debugging

### Enable Verbose Logging
```python
# In main.py
results = self.model(frame, verbose=True)
```

### Test Individual Modules
```python
import cv2
from app.vision.yolo_detector import yolo_detector

frame = cv2.imread("test.jpg")
detections = yolo_detector.detect(frame)
print(detections)
```

### Check Model Loading
```bash
python -c "from app.vision.yolo_detector import yolo_detector; print('OK')"
```

---

## Deployment

### Docker
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app/ app/
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables
```bash
YOLO_MODEL=yolov8n.pt
FACE_TOLERANCE=0.50
BLUR_KERNEL=51
```

---

## Testing

### Unit Test Example
```python
import pytest
from app.vision.yolo_detector import yolo_detector

def test_yolo_detection():
    frame = cv2.imread("test_image.jpg")
    detections = yolo_detector.detect(frame)
    assert len(detections) > 0
    assert "box" in detections[0]
```

### Integration Test
```bash
curl -X POST http://localhost:8000/process_frame \
  -F "file=@test.jpg" \
  -F "redact=true"
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| YOLO model not downloading | Check internet, run `python -m ultralytics.cli predict model=yolov8n.pt` |
| Slow inference | Use GPU, reduce resolution, use smaller model |
| Face not recognized | Adjust `tolerance` in `face_recognizer.py` |
| Emoji not showing | Check `assets/emojis/` directory exists |
| CORS errors | Verify `CORSMiddleware` in `main.py` |

---

## Resources

- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [YOLOv8 Docs](https://docs.ultralytics.com/)
- [OpenCV Docs](https://docs.opencv.org/)
- [EasyOCR Docs](https://github.com/JaidedAI/EasyOCR)
- [React Docs](https://react.dev/)

---

## Contributing

1. Create a feature branch: `git checkout -b feature/my-feature`
2. Make changes and test
3. Commit: `git commit -m "Add my feature"`
4. Push: `git push origin feature/my-feature`
5. Create Pull Request

---

## License

See LICENSE file.
