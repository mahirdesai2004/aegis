"""
yolo_detector.py — YOLOv8 Object Detection Module

Uses the Ultralytics YOLOv8-nano pretrained model to detect
privacy-relevant objects in video frames. The model is loaded
ONCE at module import to avoid per-request overhead.

Detected classes: person, laptop, tv, cell phone
"""

from ultralytics import YOLO


class YoloDetector:
    """Wrapper around YOLOv8 for privacy-relevant object detection."""

    # COCO class names relevant to privacy/security scenarios
    TARGET_CLASSES = {"person", "laptop", "tv", "cell phone"}

    def __init__(self, model_path: str = "yolov8n.pt"):
        # Load YOLOv8 nano — smallest model, optimised for CPU inference
        self.model = YOLO(model_path)

    def detect(self, frame):
        """
        Run YOLOv8 inference on a single BGR frame.

        Args:
            frame: numpy array (H, W, 3) in BGR colour space.

        Returns:
            List of detection dicts, each containing:
                - box:        (x1, y1, x2, y2) pixel coordinates
                - confidence: float 0-1
                - class_name: str from COCO label set
        """
        results = self.model(frame, verbose=False)
        detections = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]

                # Only keep classes we care about
                if cls_name not in self.TARGET_CLASSES:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    "box": (x1, y1, x2, y2),
                    "confidence": round(float(box.conf[0]), 3),
                    "class_name": cls_name,
                })

        return detections


# ── Singleton ─────────────────────────────────────────────────
# Instantiated once at startup; reused across all requests.
yolo_detector = YoloDetector()
