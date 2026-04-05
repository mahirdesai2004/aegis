"""
risk_engine.py — Simple Risk Scoring

Scoring:
    Unknown face detected  → +30
    Device detected         → +20
    Known/safe face         → +0
    Cap at 100.
"""


class RiskEngine:
    """Evaluate privacy risk from detection results."""

    def evaluate(self, face_results: list[dict], device_boxes: list) -> dict:
        """
        Args:
            face_results: output of face_recognizer.classify_persons()
            device_boxes: list of device bounding boxes

        Returns:
            { risk_score, risk_level, labels }
        """
        score = 0
        labels = []

        # Face-based scoring
        unknown_count = sum(1 for f in face_results if not f["known"])
        known_count = sum(1 for f in face_results if f["known"])

        if unknown_count > 0:
            score += unknown_count * 30
            labels.append(f"{unknown_count} Unknown Face(s)")

        if known_count > 0:
            labels.append(f"{known_count} Safe Face(s)")

        # Device scoring
        if device_boxes:
            score += len(device_boxes) * 20
            labels.append(f"{len(device_boxes)} Device(s)")

        score = min(score, 100)

        if score >= 60:
            level = "HIGH"
        elif score >= 30:
            level = "MEDIUM"
        else:
            level = "LOW"

        return {
            "risk_score": score,
            "risk_level": level,
            "labels": labels,
        }


# ── Singleton ─────────────────────────────────────────────────
risk_engine = RiskEngine()
