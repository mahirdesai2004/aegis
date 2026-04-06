"""
ocr.py — OCR + Text Region Detection Module (v3 - Enhanced)

Uses both:
1. EasyOCR for text recognition
2. OpenCV edge detection for text region localization

Key Improvements:
  - Canny edge detection for text regions
  - Dilation to connect text strokes
  - Contour detection to find text bounding boxes
  - Blur text regions even if OCR fails
  - Preprocessing: grayscale, contrast enhancement, thresholding
  - Image resizing (2x) before OCR for small text
  - Confidence filtering (>0.3)
  - Bounding box padding for better coverage
"""

import re
import easyocr
import cv2
import numpy as np


# ── Regex patterns for PII detection ─────────────────────────
PII_PATTERNS = [
    ("phone_number", re.compile(
        r"(\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,5}[-.\s]?\d{4}"
    )),
    ("id_number", re.compile(
        r"\b[A-Z]{2,5}[-\s]?\d{5,10}\b"
    )),
    ("email", re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    )),
    ("aadhaar_number", re.compile(
        r"\b\d{4}\s\d{4}\s\d{4}\b"
    )),
]


class OCRDetector:
    """Extracts text from frames using OCR + edge detection."""

    def __init__(self, languages=None, use_gpu: bool = False, conf_threshold: float = 0.3):
        """
        Initialize OCR detector.
        
        Args:
            languages: List of language codes (default: ["en"])
            use_gpu: Use GPU acceleration (default: False)
            conf_threshold: Minimum confidence to keep detection (default: 0.3)
        """
        self.reader = easyocr.Reader(languages or ["en"], gpu=use_gpu)
        self.conf_threshold = conf_threshold
        self.resize_scale = 2  # Resize image 2x before OCR for small text
        
        # Edge detection parameters
        self.canny_low = 50
        self.canny_high = 150
        self.dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        self.min_text_area = 100  # Minimum area for text region

    def _detect_text_regions_opencv(self, frame: np.ndarray) -> list[tuple]:
        """
        Detect text regions using OpenCV edge detection.
        
        Steps:
          1. Convert to grayscale
          2. Apply Canny edge detection
          3. Dilate to connect text strokes
          4. Find contours
          5. Filter by area and aspect ratio
        
        Args:
            frame: BGR image
        
        Returns:
            List of (x1, y1, x2, y2) bounding boxes
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Canny edge detection
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        
        # 2. Dilate to connect text strokes
        dilated = cv2.dilate(edges, self.dilation_kernel, iterations=2)
        
        # 3. Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_boxes = []
        for contour in contours:
            # Get bounding rectangle
            x, y, cw, ch = cv2.boundingRect(contour)
            area = cw * ch
            
            # Filter by area (too small = noise, too large = not text)
            if area < self.min_text_area or area > (h * w * 0.5):
                continue
            
            # Filter by aspect ratio (text is usually wider than tall)
            aspect_ratio = cw / (ch + 1e-6)
            if aspect_ratio < 0.3 or aspect_ratio > 10:  # Too narrow or too wide
                continue
            
            # Add padding
            pad = 5
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w, x + cw + pad)
            y2 = min(h, y + ch + pad)
            
            text_boxes.append((x1, y1, x2, y2))
        
        return text_boxes

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Preprocess frame for better OCR accuracy.
        
        Steps:
          1. Convert to grayscale
          2. Enhance contrast (CLAHE)
          3. Apply Gaussian blur (denoise)
          4. Thresholding
          5. Resize 2x for small text
        
        Returns:
            (preprocessed_frame, scale_factor)
        """
        # 1. Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Contrast enhancement (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 3. Gaussian blur (denoise)
        denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # 4. Thresholding (Otsu's method)
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 5. Resize 2x for small text
        h, w = thresh.shape[:2]
        resized = cv2.resize(thresh, (w * self.resize_scale, h * self.resize_scale),
                            interpolation=cv2.INTER_CUBIC)
        
        return resized, self.resize_scale

    def _scale_box_back(self, box: tuple, scale: float, original_h: int, original_w: int) -> tuple:
        """Scale bounding box from preprocessed image back to original frame."""
        x1, y1, x2, y2 = box
        
        # Scale back
        x1 = int(x1 / scale)
        y1 = int(y1 / scale)
        x2 = int(x2 / scale)
        y2 = int(y2 / scale)
        
        # Clamp to frame boundaries
        x1 = max(0, min(x1, original_w - 1))
        y1 = max(0, min(y1, original_h - 1))
        x2 = max(x1 + 1, min(x2, original_w))
        y2 = max(y1 + 1, min(y2, original_h))
        
        return (x1, y1, x2, y2)

    def _expand_box(self, box: tuple, padding_ratio: float = 0.15) -> tuple:
        """Expand bounding box with padding for better coverage."""
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        
        pad_x = int(w * padding_ratio)
        pad_y = int(h * padding_ratio)
        
        return (x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y)

    def detect(self, frame: np.ndarray, use_preprocessing: bool = True) -> list[dict]:
        """
        Detect text regions using both OCR and edge detection.

        Args:
            frame: numpy array (H, W, 3) in BGR colour space
            use_preprocessing: Apply preprocessing for better accuracy

        Returns:
            List of dicts with:
                - box: (x1, y1, x2, y2) in original frame coordinates
                - text: recognized string (or "edge_detected" if only edge detected)
                - confidence: float 0-1
                - pii_types: list of matched PII labels
                - source: "ocr" or "edge_detection"
        """
        h, w = frame.shape[:2]
        detected_boxes = set()  # Track detected boxes to avoid duplicates
        detections = []
        
        # 1. OCR-based detection
        if use_preprocessing:
            processed, scale = self._preprocess(frame)
        else:
            processed = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            scale = 1.0
        
        # Run OCR
        results = self.reader.readtext(processed)
        
        for bbox, text, prob in results:
            # Filter by confidence
            if prob < self.conf_threshold:
                continue
            
            # EasyOCR bbox = [top-left, top-right, bottom-right, bottom-left]
            tl, tr, br, bl = bbox
            x1 = int(min(tl[0], bl[0]))
            y1 = int(min(tl[1], tr[1]))
            x2 = int(max(tr[0], br[0]))
            y2 = int(max(bl[1], br[1]))
            
            # Scale back to original frame
            if scale > 1.0:
                x1, y1, x2, y2 = self._scale_box_back((x1, y1, x2, y2), scale, h, w)
            
            # Expand box with padding
            x1_pad, y1_pad, x2_pad, y2_pad = self._expand_box((x1, y1, x2, y2))
            x1_pad = max(0, x1_pad)
            y1_pad = max(0, y1_pad)
            x2_pad = min(w, x2_pad)
            y2_pad = min(h, y2_pad)
            
            box_key = (x1_pad, y1_pad, x2_pad, y2_pad)
            detected_boxes.add(box_key)
            
            # Scan for PII
            pii_types = self._scan_pii(text)
            
            detections.append({
                "box": (x1_pad, y1_pad, x2_pad, y2_pad),
                "text": text,
                "confidence": round(float(prob), 3),
                "pii_types": pii_types,
                "source": "ocr",
            })
        
        # 2. Edge-based detection (fallback for missed text)
        edge_boxes = self._detect_text_regions_opencv(frame)
        for box in edge_boxes:
            x1, y1, x2, y2 = box
            box_key = (x1, y1, x2, y2)
            
            # Skip if already detected by OCR
            if box_key in detected_boxes:
                continue
            
            # Check for overlap with existing detections
            overlaps = False
            for det in detections:
                dx1, dy1, dx2, dy2 = det["box"]
                # Calculate IoU
                inter_x1 = max(x1, dx1)
                inter_y1 = max(y1, dy1)
                inter_x2 = min(x2, dx2)
                inter_y2 = min(y2, dy2)
                
                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    box_area = (x2 - x1) * (y2 - y1)
                    if inter_area / (box_area + 1e-6) > 0.3:  # 30% overlap threshold
                        overlaps = True
                        break
            
            if not overlaps:
                detections.append({
                    "box": (x1, y1, x2, y2),
                    "text": "[text_region]",
                    "confidence": 0.5,  # Lower confidence for edge-detected regions
                    "pii_types": [],
                    "source": "edge_detection",
                })
        
        return detections

    @staticmethod
    def _scan_pii(text: str) -> list[str]:
        """Check a text string against all PII regex patterns."""
        matched = []
        for label, pattern in PII_PATTERNS:
            if pattern.search(text):
                matched.append(label)
        return matched


# ── Singleton ─────────────────────────────────────────────────
ocr_detector = OCRDetector()
