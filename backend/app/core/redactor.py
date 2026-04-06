"""
redactor.py — Visual Redaction Module (v2 - Improved)

Two redaction types:
  1. Emoji overlay — for unknown faces (green-screen chroma key removal)
  2. Gaussian blur — for devices and text regions

Improvements:
  - Expanded bounding boxes before blurring
  - Stronger blur kernel for better visibility
  - Fallback blur if emoji fails
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

    def __init__(self, blur_ksize: tuple = (51, 51), text_blur_ksize: tuple = (31, 31)):
        """
        Initialize redactor.
        
        Args:
            blur_ksize: Kernel size for device blur (larger = stronger)
            text_blur_ksize: Kernel size for text blur (smaller, faster)
        """
        self.blur_ksize = blur_ksize
        self.text_blur_ksize = text_blur_ksize

    def _expand_box(self, box: tuple, padding_ratio: float = 0.2, frame_h: int = 480, frame_w: int = 640) -> tuple:
        """
        Expand bounding box with padding.
        
        Args:
            box: (x1, y1, x2, y2)
            padding_ratio: Padding as ratio of box size
            frame_h: Frame height for boundary clamping
            frame_w: Frame width for boundary clamping
        
        Returns:
            (x1_padded, y1_padded, x2_padded, y2_padded)
        """
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        
        pad_x = int(w * padding_ratio)
        pad_y = int(h * padding_ratio)
        
        x1_pad = max(0, x1 - pad_x)
        y1_pad = max(0, y1 - pad_y)
        x2_pad = min(frame_w, x2 + pad_x)
        y2_pad = min(frame_h, y2 + pad_y)
        
        return (x1_pad, y1_pad, x2_pad, y2_pad)

    def blur_region(self, image: np.ndarray, boxes: list, expand: bool = True, ksize: tuple = None) -> np.ndarray:
        """
        Apply Gaussian blur to bounding boxes.
        
        Args:
            image: BGR frame
            boxes: List of (x1, y1, x2, y2) boxes
            expand: Expand boxes before blurring
            ksize: Kernel size (uses self.blur_ksize if None)
        
        Returns:
            Blurred frame
        """
        if ksize is None:
            ksize = self.blur_ksize
        
        h, w = image.shape[:2]
        
        for box in boxes:
            # Expand box if requested
            if expand:
                x1, y1, x2, y2 = self._expand_box(box, padding_ratio=0.2, frame_h=h, frame_w=w)
            else:
                x1, y1, x2, y2 = box
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
            
            # Apply blur
            if x2 > x1 and y2 > y1:
                roi = image[y1:y2, x1:x2]
                image[y1:y2, x1:x2] = cv2.GaussianBlur(roi, ksize, 0)
        
        return image

    def blur_text_regions(self, image: np.ndarray, boxes: list) -> np.ndarray:
        """
        Apply blur to text regions (lighter blur, faster).
        
        Args:
            image: BGR frame
            boxes: List of text bounding boxes
        
        Returns:
            Blurred frame
        """
        return self.blur_region(image, boxes, expand=True, ksize=self.text_blur_ksize)

    def apply_emoji(self, image: np.ndarray, box: tuple, emoji_index: int = -1, use_fallback_blur: bool = True) -> np.ndarray:
        """
        Overlay an emoji PNG onto a face region with green-screen removal.

        Args:
            image: BGR frame (modified in-place)
            box: (x1, y1, x2, y2) face bounding box
            emoji_index: specific emoji, or -1 for random
            use_fallback_blur: Use blur if emoji fails or unavailable
        
        Returns:
            Frame with emoji or blur applied
        """
        if not EMOJI_IMAGES:
            if use_fallback_blur:
                return self.blur_region(image, [box], expand=True)
            return image

        if emoji_index < 0 or emoji_index >= len(EMOJI_IMAGES):
            emoji_index = random.randint(0, len(EMOJI_IMAGES) - 1)

        x1, y1, x2, y2 = box
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        rw, rh = x2 - x1, y2 - y1
        
        if rw <= 0 or rh <= 0:
            return image

        try:
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
        except Exception as e:
            # Fallback to blur if emoji fails
            if use_fallback_blur:
                print(f"[WARN] Emoji overlay failed: {e}, using blur fallback")
                return self.blur_region(image, [box], expand=True)
        
        return image


# ── Singleton ─────────────────────────────────────────────────
redactor = Redactor()
