"""
redactor.py — Frame Redaction Module

Applies visual redaction (Gaussian blur or pixelation) to
specified bounding-box regions of an image using OpenCV.
"""

import cv2
import numpy as np


class Redactor:
    """Apply blur or pixelation to sensitive regions of an image."""

    def __init__(self, method: str = "blur", blur_ksize: tuple = (51, 51)):
        """
        Args:
            method:    "blur" for Gaussian blur, "pixelate" for mosaic effect.
            blur_ksize: kernel size for GaussianBlur (must be odd numbers).
        """
        self.method = method
        self.blur_ksize = blur_ksize

    def redact(self, image: np.ndarray, boxes: list) -> np.ndarray:
        """
        Redact regions in-place and return the modified image.

        Args:
            image: numpy array (H, W, 3) BGR.
            boxes: list of (x1, y1, x2, y2) tuples.

        Returns:
            The same numpy array with redacted regions.
        """
        h, w = image.shape[:2]

        for box in boxes:
            x1, y1, x2, y2 = box

            # Clamp coordinates to image boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Skip degenerate boxes
            if x1 >= x2 or y1 >= y2:
                continue

            roi = image[y1:y2, x1:x2]

            if self.method == "blur":
                roi = cv2.GaussianBlur(roi, self.blur_ksize, 0)
            elif self.method == "pixelate":
                # Shrink then enlarge to create a mosaic effect
                small = cv2.resize(roi, (10, 10), interpolation=cv2.INTER_LINEAR)
                roi = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)

            image[y1:y2, x1:x2] = roi

        return image


# ── Singleton ─────────────────────────────────────────────────
redactor = Redactor(method="blur")
