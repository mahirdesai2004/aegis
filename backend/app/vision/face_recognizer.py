"""
face_recognizer.py — Face Extraction & Matching

Strategy: Uses YOLO person bounding boxes and extracts the top 40%
as the "face region". No separate face detector needed.

Matching uses HSV colour histograms — lightweight, CPU-friendly,
and good enough for demo-level recognition.

Two modes:
  1. Safe Mode (Live): center-most person is auto-registered as safe
  2. Register Mode (Image Upload): user uploads a face photo to register
"""

import cv2
import numpy as np


class FaceRecognizer:
    """Extract face regions from YOLO person boxes and match identities."""

    def __init__(self, tolerance: float = 0.50):
        self.tolerance = tolerance

        # Registered faces: { name: [encoding, ...] }
        self._registry: dict[str, list[np.ndarray]] = {}

        # Safe mode: auto-captured face from live feed
        self._safe_encoding: np.ndarray | None = None
        self._safe_mode: bool = False

    # ── Registration ──────────────────────────────────────────

    def register_face_from_image(self, name: str, image: np.ndarray) -> bool:
        """
        Register a face from a standalone face photo.
        The entire image is treated as the face region.
        """
        if image is None or image.size == 0:
            return False

        encoding = self._compute_encoding(image)

        if name not in self._registry:
            self._registry[name] = []
        self._registry[name].append(encoding)
        return True

    def remove_face(self, name: str) -> bool:
        if name in self._registry:
            del self._registry[name]
            return True
        return False

    def list_faces(self) -> list[str]:
        return list(self._registry.keys())

    # ── Safe Mode ─────────────────────────────────────────────

    def set_safe_from_frame(self, frame: np.ndarray, person_boxes: list[dict]) -> bool:
        """
        Auto-register the center-most person in the frame as 'safe'.

        Args:
            frame: BGR image.
            person_boxes: list of YOLO detections with 'box' and 'class_name'.

        Returns:
            True if a person was found and registered.
        """
        persons = [d for d in person_boxes if d["class_name"] == "person"]
        if not persons:
            return False

        # Find center-most person
        frame_cx = frame.shape[1] / 2
        frame_cy = frame.shape[0] / 2

        def dist_to_center(det):
            x1, y1, x2, y2 = det["box"]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            return (cx - frame_cx) ** 2 + (cy - frame_cy) ** 2

        center_person = min(persons, key=dist_to_center)
        face_roi = self._extract_face_region(frame, center_person["box"])

        if face_roi is None or face_roi.size == 0:
            return False

        self._safe_encoding = self._compute_encoding(face_roi)
        self._safe_mode = True
        return True

    def reset_safe(self):
        """Clear safe mode."""
        self._safe_encoding = None
        self._safe_mode = False

    @property
    def is_safe_mode(self) -> bool:
        return self._safe_mode

    # ── Recognition ───────────────────────────────────────────

    def classify_persons(
        self, frame: np.ndarray, person_boxes: list[dict]
    ) -> list[dict]:
        """
        For each person detected by YOLO, extract top-40% face region,
        compare against safe/registered faces, and classify.

        Returns list of:
            { box, face_box, known, name }
        """
        results = []

        for det in person_boxes:
            if det["class_name"] != "person":
                continue

            face_roi = self._extract_face_region(frame, det["box"])
            if face_roi is None or face_roi.size == 0:
                results.append({
                    "box": det["box"],
                    "face_box": det["box"],
                    "known": False,
                    "name": None,
                })
                continue

            x1, y1, x2, y2 = det["box"]
            face_h = int((y2 - y1) * 0.40)
            face_box = (x1, y1, x2, y1 + face_h)

            encoding = self._compute_encoding(face_roi)

            # Check safe mode first
            if self._safe_mode and self._safe_encoding is not None:
                score = self._compare(encoding, self._safe_encoding)
                if score >= self.tolerance:
                    results.append({
                        "box": det["box"],
                        "face_box": face_box,
                        "known": True,
                        "name": "_safe_",
                    })
                    continue

            # Check registered faces
            matched = self._find_match(encoding)
            results.append({
                "box": det["box"],
                "face_box": face_box,
                "known": matched is not None,
                "name": matched,
            })

        return results

    # ── For Image Upload mode ─────────────────────────────────

    def classify_faces_in_image(
        self, frame: np.ndarray, person_boxes: list[dict]
    ) -> list[dict]:
        """Same as classify_persons but for uploaded images."""
        return self.classify_persons(frame, person_boxes)

    # ── Internals ─────────────────────────────────────────────

    def _extract_face_region(self, frame: np.ndarray, box: tuple) -> np.ndarray | None:
        """Extract top 40% of a person bounding box as the face region."""
        x1, y1, x2, y2 = box
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        person_h = y2 - y1
        face_h = int(person_h * 0.40)

        face_roi = frame[y1 : y1 + face_h, x1:x2]
        if face_roi.size == 0:
            return None
        return face_roi

    def _compute_encoding(self, face_roi: np.ndarray) -> np.ndarray:
        """HSV histogram-based face encoding. Lightweight and effective."""
        face = cv2.resize(face_roi, (128, 128))
        hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)

        hist_h = cv2.calcHist([hsv], [0], None, [50], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [60], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [60], [0, 256])

        encoding = np.concatenate([hist_h, hist_s, hist_v]).flatten()
        norm = np.linalg.norm(encoding)
        if norm > 0:
            encoding = encoding / norm
        return encoding

    def _compare(self, enc_a: np.ndarray, enc_b: np.ndarray) -> float:
        """Histogram correlation score. 1.0 = identical."""
        return cv2.compareHist(
            enc_a.reshape(-1, 1).astype(np.float32),
            enc_b.reshape(-1, 1).astype(np.float32),
            cv2.HISTCMP_CORREL,
        )

    def _find_match(self, encoding: np.ndarray) -> str | None:
        """Compare against all registered faces."""
        best_name = None
        best_score = 0.0

        for name, enc_list in self._registry.items():
            for reg_enc in enc_list:
                score = self._compare(encoding, reg_enc)
                if score > best_score:
                    best_score = score
                    best_name = name

        return best_name if best_score >= self.tolerance else None


# ── Singleton ─────────────────────────────────────────────────
face_recognizer = FaceRecognizer()
