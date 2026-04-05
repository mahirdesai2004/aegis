"""
redactor.py — Visual Redaction Module

Two redaction types:
  1. Emoji overlay — for unknown faces (green-screen chroma key removal)
  2. Gaussian blur — for devices (phones, laptops, TVs)
"""

import os
import random
import cv2
import numpy as np

# ── Load emoji PNGs at startup ───────────────────────────────
EMOJI_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "emojis")
EMOJI_IMAGES: list[np.ndarray] = []


def _load_emojis():
    """Load all emoji PNGs from assets directory."""
    global EMOJI_IMAGES
    if not os.path.isdir(EMOJI_DIR):
        print(f"[WARN] Emoji dir not found: {EMOJI_DIR}")
        return
    for fname in sorted(os.listdir(EMOJI_DIR)):
        if fname.endswith(".png"):
            img = cv2.imread(os.path.join(EMOJI_DIR, fname), cv2.IMREAD_COLOR)
            if img is not None:
                EMOJI_IMAGES.append(img)
    print(f"[INFO] Loaded {len(EMOJI_IMAGES)} emoji images")


_load_emojis()


class Redactor:
    """Apply emoji overlays and blur to frame regions."""

    def __init__(self, blur_ksize: tuple = (51, 51)):
        self.blur_ksize = blur_ksize

    def blur_region(self, image: np.ndarray, boxes: list) -> np.ndarray:
        """Apply Gaussian blur to bounding boxes."""
        h, w = image.shape[:2]
        for (x1, y1, x2, y2) in boxes:
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                roi = image[y1:y2, x1:x2]
                image[y1:y2, x1:x2] = cv2.GaussianBlur(roi, self.blur_ksize, 0)
        return image

    def apply_emoji(self, image: np.ndarray, box: tuple, emoji_index: int = -1) -> np.ndarray:
        """
        Overlay an emoji PNG onto a face region with green-screen removal.

        Args:
            image:       BGR frame (modified in-place).
            box:         (x1, y1, x2, y2) face bounding box.
            emoji_index: specific emoji, or -1 for random.
        """
        if not EMOJI_IMAGES:
            return self.blur_region(image, [box])

        if emoji_index < 0 or emoji_index >= len(EMOJI_IMAGES):
            emoji_index = random.randint(0, len(EMOJI_IMAGES) - 1)

        x1, y1, x2, y2 = box
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        rw, rh = x2 - x1, y2 - y1
        if rw <= 0 or rh <= 0:
            return image

        # Resize emoji to fit region
        emoji = cv2.resize(EMOJI_IMAGES[emoji_index], (rw, rh), interpolation=cv2.INTER_AREA)

        # Chroma key: remove green background
        hsv = cv2.cvtColor(emoji, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, np.array([35, 80, 80]), np.array([85, 255, 255]))
        alpha = cv2.bitwise_not(green_mask).astype(np.float32) / 255.0
        alpha = cv2.GaussianBlur(alpha, (3, 3), 0)  # smooth edges

        # Blend
        roi = image[y1:y2, x1:x2].astype(np.float32)
        for c in range(3):
            roi[:, :, c] = alpha * emoji[:, :, c].astype(np.float32) + (1.0 - alpha) * roi[:, :, c]
        image[y1:y2, x1:x2] = roi.astype(np.uint8)

        return image


# ── Singleton ─────────────────────────────────────────────────
redactor = Redactor()
