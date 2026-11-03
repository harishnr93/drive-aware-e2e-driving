"""Deterministic image corruptions for robustness tests (uint8 RGB [..., H, W, 3]).

Each corruption takes a severity level and a seed, so the same corrupted frames can be
reproduced in every evaluation. Used as a preview in Phase 7 and as the sensor-shift suite in
Phase 9.
"""

from __future__ import annotations

import cv2
import numpy as np


def gaussian_noise(images: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = images.astype(np.int16) + rng.normal(0.0, sigma, images.shape).astype(np.int16)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def motion_blur(images: np.ndarray, kernel: int) -> np.ndarray:
    """Horizontal motion blur with a ``kernel``-pixel box kernel."""
    k = np.zeros((kernel, kernel), np.float32)
    k[kernel // 2, :] = 1.0 / kernel
    flat = images.reshape(-1, *images.shape[-3:])
    out = np.stack([cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REFLECT) for img in flat])
    return out.reshape(images.shape)


def occlusion(images: np.ndarray, fraction: float, seed: int = 0) -> np.ndarray:
    """Black rectangle covering ``fraction`` of the image area at a random position per image."""
    rng = np.random.default_rng(seed)
    flat = images.reshape(-1, *images.shape[-3:]).copy()
    h, w = flat.shape[1:3]
    side_h = int(round(h * np.sqrt(fraction)))
    side_w = int(round(w * np.sqrt(fraction)))
    for img in flat:
        y = rng.integers(0, h - side_h + 1)
        x = rng.integers(0, w - side_w + 1)
        img[y : y + side_h, x : x + side_w] = 0
    return flat.reshape(images.shape)


CORRUPTIONS = {
    "noise": lambda imgs, seed: gaussian_noise(imgs, 20.0, seed),
    "blur": lambda imgs, seed: motion_blur(imgs, 9),
    "occlusion": lambda imgs, seed: occlusion(imgs, 0.25, seed),
}
