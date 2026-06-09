"""Mask visualization helpers for the SAM2 interactive editor."""
from __future__ import annotations

import cv2
import numpy as np

# RGB colours cycled per object index
OBJECT_COLORS: list[tuple[int, int, int]] = [
    (0,   220,  80),   # green
    (0,   140, 255),   # blue-orange
    (255, 140,   0),   # orange
    (180,   0, 255),   # purple
    (0,   220, 220),   # cyan
    (255, 200,   0),   # yellow
    (255,   0, 180),   # pink
    (220, 220, 220),   # white
]


def apply_mask_overlay(
    image_np: np.ndarray,
    mask_bool: np.ndarray,
    color_rgb: tuple[int, int, int] = (0, 220, 80),
    alpha: float = 0.45,
) -> np.ndarray:
    """Return a copy of *image_np* with a semi-transparent colour overlay where *mask_bool* is True."""
    result = image_np.astype(np.float32)
    mask = mask_bool.astype(bool)
    for c, val in enumerate(color_rgb):
        ch = result[:, :, c]
        ch[mask] = ch[mask] * (1.0 - alpha) + val * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def draw_interaction_points(
    image_np: np.ndarray,
    points: list,
    labels: list[int],
    radius: int = 8,
) -> np.ndarray:
    """Draw positive (green) and negative (red) circles at each point."""
    result = image_np.copy()
    h, w = result.shape[:2]
    for (x, y), label in zip(points, labels):
        cx, cy = int(round(x)), int(round(y))
        if not (0 <= cx < w and 0 <= cy < h):
            continue
        fill   = (0, 180, 0)   if label == 1 else (180, 0, 0)
        border = (0, 255, 0)   if label == 1 else (255, 0, 0)
        cv2.circle(result, (cx, cy), radius,     border, thickness=2,  lineType=cv2.LINE_AA)
        cv2.circle(result, (cx, cy), radius - 2, fill,   thickness=-1, lineType=cv2.LINE_AA)
        # White centre dot for contrast
        cv2.circle(result, (cx, cy), 2, (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
    return result


def erode_dilate_mask(
    mask_bool: np.ndarray,
    erode_px: int = 0,
    dilate_px: int = 0,
) -> np.ndarray:
    """Morphologically erode then dilate a boolean mask.  Returns a boolean array."""
    if erode_px <= 0 and dilate_px <= 0:
        return mask_bool.astype(bool)

    mask_u8 = (mask_bool.astype(np.uint8)) * 255
    if erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
        mask_u8 = cv2.erode(mask_u8, k, iterations=1)
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        mask_u8 = cv2.dilate(mask_u8, k, iterations=1)
    return mask_u8 > 127
