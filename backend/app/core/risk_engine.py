"""
risk_engine.py — Rule-Based Risk Scoring Engine

Evaluates YOLO + OCR detection results and computes a simple,
explainable risk score (0–100) with human-readable labels.

Scoring rules (additive, capped at 100):
    ┌───────────────────────┬────────┐
    │ Condition             │ Points │
    ├───────────────────────┼────────┤
    │ Person detected       │  +10   │
    │ Text detected (OCR)   │  +20   │
    │ Sensitive PII pattern │  +40   │
    │ Laptop detected       │   +5   │
    │ TV / Monitor detected │   +5   │
    │ Cell phone detected   │  +10   │
    └───────────────────────┴────────┘
"""


class RiskEngine:
    """Compute a risk score from detection results."""

    # Points awarded per YOLO class
    YOLO_SCORES = {
        "person":     10,
        "laptop":      5,
        "tv":          5,
        "cell phone": 10,
    }

    # Points for OCR text presence (applied once)
    TEXT_SCORE = 20

    # Points if ANY regex PII pattern matched (applied once)
    PII_SCORE = 40

    def evaluate(self, yolo_detections: list, ocr_detections: list) -> dict:
        """
        Evaluate combined risk from YOLO objects and OCR text.

        Args:
            yolo_detections: list of dicts from YoloDetector.detect()
            ocr_detections:  list of dicts from OCRDetector.detect()

        Returns:
            dict with keys:
                - risk_score:  int   (0–100)
                - risk_level:  str   ("LOW" | "MEDIUM" | "HIGH")
                - labels:      list  (human-readable trigger labels)
        """
        score = 0
        labels = set()

        # ── Score YOLO detections ─────────────────────────────
        seen_classes = set()
        for det in yolo_detections:
            cls = det.get("class_name")
            if cls in self.YOLO_SCORES and cls not in seen_classes:
                score += self.YOLO_SCORES[cls]
                labels.add(f"{cls}_detected")
                seen_classes.add(cls)

        # ── Score OCR detections ──────────────────────────────
        if ocr_detections:
            score += self.TEXT_SCORE
            labels.add("text_detected")

            # Check if any detection flagged PII
            pii_found = any(
                det.get("pii_types") for det in ocr_detections
            )
            if pii_found:
                score += self.PII_SCORE
                # Collect specific PII type labels
                for det in ocr_detections:
                    for pii_type in det.get("pii_types", []):
                        labels.add(f"pii_{pii_type}")

        # ── Clamp & classify ─────────────────────────────────
        score = min(score, 100)

        if score >= 70:
            level = "HIGH"
        elif score >= 30:
            level = "MEDIUM"
        else:
            level = "LOW"

        return {
            "risk_score": score,
            "risk_level": level,
            "labels": sorted(labels),
        }


# ── Singleton ─────────────────────────────────────────────────
risk_engine = RiskEngine()
