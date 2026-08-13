"""Per-letter recognition via onnxruntime (26 lowercase classes).

Loads models/recognizer.onnx once. Input: normalized 28x28 bitmap (0-1,
ink = 1). Output: dict {letter: prob} for the top-K letters.
"""

import os
from collections import OrderedDict

import numpy as np
import onnxruntime as ort

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "models", "recog", "model.onnx"
)

# EMNIST ByClass labels 36-61 -> 'a'-'z' (ASCII 97-122).
LOWER = "abcdefghijklmnopqrstuvwxyz"
TOP_K = 5

_session = None


def _get_session():
    global _session
    if _session is None:
        path = os.path.abspath(MODEL_PATH)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"recognizer.onnx missing: {path} (train + export in ml/)"
            )
        _session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return _session


def normalize_strokes_to_bitmap(strokes, size: int = 28) -> np.ndarray:
    """Rasterize a letter's strokes into a 28x28 ink bitmap (0=bg, 1=ink).

    Each stroke is either a Stroke (has .points) or a plain list of (x,y,t).
    """
    img = np.zeros((size, size), dtype=np.float32)
    xs, ys = [], []
    for s in strokes:
        pts = s.points if hasattr(s, "points") else s
        for (x, y, _t) in pts:
            xs.append(x)
            ys.append(y)
    if not xs:
        return img

    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    w, h = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
    scale = (size - 4) / max(w, h)
    ox = (size - w * scale) / 2 - x0 * scale
    oy = (size - h * scale) / 2 - y0 * scale

    # draw each stroke as a polyline
    for s in strokes:
        pts = s.points if hasattr(s, "points") else s
        pts = [(int((x * scale + ox)), int((y * scale + oy))) for (x, y, _t) in pts]
        for (ax, ay), (bx, by) in zip(pts[:-1], pts[1:]):
            _draw_line(img, ax, ay, bx, by)
    return img


def _draw_line(img, x0, y0, x1, y1):
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    steps = max(dx, dy, 1)
    for i in range(steps + 1):
        t = i / steps
        x = int(round(x0 + (x1 - x0) * t))
        y = int(round(y0 + (y1 - y0) * t))
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            img[y, x] = 1.0


def recognize(bitmap: np.ndarray, top_k: int = TOP_K) -> OrderedDict:
    sess = _get_session()
    inp_name = sess.get_inputs()[0].name
    # Model was trained with EMNIST transform rot90(k=1)+flipH, which renders
    # letters upside-down relative to real writing. Feeding an upright bitmap
    # rotated 180 deg compensates (R_-1*FH == R2*R1*FH), so upright letters
    # classify correctly without retraining. Remove if retrained w/ k=-1.
    bitmap = np.rot90(bitmap, 2)
    x = bitmap.reshape(1, 1, 28, 28).astype(np.float32)
    logits = sess.run(None, {inp_name: x})[0][0]
    order = np.argsort(logits)[::-1][:top_k]
    out: OrderedDict = OrderedDict()
    for idx in order:
        out[LOWER[idx]] = float(np.exp(logits[idx]) / np.exp(logits).sum())
    return out