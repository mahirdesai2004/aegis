"""
ocr.py — OCR + Regex PII Detection Module

Uses EasyOCR to extract visible text from frames, then applies
regex-based pattern matching to flag Personally Identifiable
Information (PII) such as phone numbers and ID-like strings.

The EasyOCR reader is loaded ONCE at module import.
"""

import re
import easyocr
import cv2


# ── Regex patterns for PII detection ─────────────────────────
# Each tuple: (label, compiled regex)
PII_PATTERNS = [
    # International phone formats: +91-XXXXX-XXXXX, (123) 456-7890, etc.
    ("phone_number", re.compile(
        r"(\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,5}[-.\s]?\d{4}"
    )),
    # Generic ID-like patterns: ABC-1234567, 12-34-5678
    ("id_number", re.compile(
        r"\b[A-Z]{2,5}[-\s]?\d{5,10}\b"
    )),
    # Email addresses
    ("email", re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    )),
    # Aadhaar-style: 1234 5678 9012
    ("aadhaar_number", re.compile(
        r"\b\d{4}\s\d{4}\s\d{4}\b"
    )),
]


class OCRDetector:
    """Extracts text from frames and flags PII using regex patterns."""

    def __init__(self, languages=None, use_gpu: bool = False):
        # gpu=False ensures CPU-only execution for portability
        self.reader = easyocr.Reader(languages or ["en"], gpu=use_gpu)

    def detect(self, frame):
        """
        Run OCR on a BGR frame and return text detections.

        Args:
            frame: numpy array (H, W, 3) in BGR colour space.

        Returns:
            List of dicts, each containing:
                - box:        (x1, y1, x2, y2) pixel coordinates
                - text:       the recognised string
                - confidence: float 0-1
                - pii_types:  list of matched PII labels (may be empty)
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.reader.readtext(rgb_frame)

        detections = []
        for bbox, text, prob in results:
            # EasyOCR bbox = [top-left, top-right, bottom-right, bottom-left]
            tl, tr, br, bl = bbox
            x1 = int(min(tl[0], bl[0]))
            y1 = int(min(tl[1], tr[1]))
            x2 = int(max(tr[0], br[0]))
            y2 = int(max(bl[1], br[1]))

            # Run regex PII scan on the recognised text
            pii_types = self._scan_pii(text)

            detections.append({
                "box": (x1, y1, x2, y2),
                "text": text,
                "confidence": round(float(prob), 3),
                "pii_types": pii_types,
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
