"""
Thermal contrast normalization — shared by training-frame extraction AND inference.

The raw FLIR road-transect frames come off the camera with a crushed dynamic range
(measured: pixel std ~9.6, p1..p99 spanning only ~44 of 255 gray levels). Warm deer
sit just a few levels above the background, so a detector sees near-flat gray and
cannot learn — the observed ~0.3 mAP ceiling across every training recipe.

`enhance_contrast` stretches that range so the deer become clearly visible warm blobs.
It MUST be applied identically at train time (frame extraction) and at inference
(count_deer.py), or the model sees a different distribution than it trained on.

CLAHE (default) is the thermal-imagery standard: local, adaptive, and it lifts the
deer without blowing out noise the way a global stretch does. `stretch` (global
1-99 percentile) is offered for ablation.
"""
from __future__ import annotations
import cv2
import numpy as np

_CLAHE = {}


def _clahe(clip: float, tile: int):
    key = (clip, tile)
    if key not in _CLAHE:
        _CLAHE[key] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    return _CLAHE[key]


def enhance_contrast(img: np.ndarray, method: str = "clahe",
                     clip: float = 2.0, tile: int = 8) -> np.ndarray:
    """Return a 3-channel BGR uint8 image with thermal contrast normalized.

    method: 'clahe' (default), 'stretch' (global p1-p99), or 'none' (passthrough).
    Accepts gray or BGR; always returns BGR so downstream (YOLO) gets 3 channels.
    """
    if method == "none":
        return img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if method == "clahe":
        out = _clahe(clip, tile).apply(gray)
    elif method == "stretch":
        lo, hi = np.percentile(gray, 1), np.percentile(gray, 99)
        out = np.clip((gray.astype(np.float32) - lo) * 255.0 / max(1.0, hi - lo),
                      0, 255).astype(np.uint8)
    else:
        raise ValueError(f"unknown contrast method: {method}")
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
