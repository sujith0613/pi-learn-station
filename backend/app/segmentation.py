"""Stroke -> letter segmentation, image-based (ScriboGenie-style).

Input: a list of strokes, each stroke = [(x, y, t), ...] (pointer events).
Output: ordered letters (left-to-right), each with a working-space bbox and the
crop of its ink component.

Strategy (mirrors the proven ScriboGenie pipeline instead of pen-lift timing):
1. Rasterize ALL strokes onto a working bitmap (ink = 1) with a stroke
   thickness that keeps a multi-stroke letter connected (a kid lifting the pen
   mid-letter no longer splits the letter into two).
2. Connected components on that bitmap -> each ink blob is a letter candidate.
3. Filter tiny noise blobs, then sort by x (reading order).

Word grouping is a separate, purely spatial x-gap step applied on the component
boxes (see segment_words), so the sentence/LM offsets survive.
"""

from collections import deque
from dataclasses import dataclass, field

import numpy as np

# Working raster size: ink is scaled so the tallest ink extent is TARGET_H px.
TARGET_H = 96
STROKE_THICKNESS = 4          # rendered line width (dilation radius), working px
MIN_COMPONENT_AREA = 12       # drop specks/noise (working px^2)

# Word grouping: an x-gap between consecutive letters is a word boundary when
# it exceeds a multiple of the median letter width (scale-robust).
WORD_GAP_FACTOR = 1.2
MIN_WORD_GAP_PX = 8.0


@dataclass
class Stroke:
    points: list = field(default_factory=list)  # [(x, y, t)]


@dataclass
class Letter:
    x0: float
    x1: float
    y0: float
    y1: float
    crop: np.ndarray  # 2D ink component (0/1) in working px, ready to normalize


# ---------------------------------------------------------------------------
# Rasterization
# ---------------------------------------------------------------------------

def _bbox(strokes: list[Stroke]) -> tuple[float, float, float, float]:
    xs = [p[0] for s in strokes for p in s.points]
    ys = [p[1] for s in strokes for p in s.points]
    if not xs:
        return (0.0, 0.0, 0.0, 0.0)
    return (min(xs), min(ys), max(xs), max(ys))


def _draw_polyline(img: np.ndarray, pts: list[tuple[int, int]]) -> None:
    """Draw a 1px polyline of (x, y) working px (Bresenham-ish)."""
    for (ax, ay), (bx, by) in zip(pts[:-1], pts[1:]):
        _draw_line(img, ax, ay, bx, by)
    for (cx, cy) in pts:
        if 0 <= cx < img.shape[1] and 0 <= cy < img.shape[0]:
            img[cy, cx] = True


def _draw_line(img: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> None:
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    H, W = img.shape
    while True:
        if 0 <= x0 < W and 0 <= y0 < H:
            img[y0, x0] = True
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def _dilate(img: np.ndarray, radius: int) -> np.ndarray:
    """Binary dilation by a square of side 2*radius+1 (cheap, bounded)."""
    out = img.copy()
    for _ in range(radius):
        padded = np.pad(out, 1, mode="edge")
        out = (
            padded[1:-1, 1:-1]
            | padded[:-2, 1:-1] | padded[2:, 1:-1]
            | padded[1:-1, :-2] | padded[1:-1, 2:]
            | padded[:-2, :-2] | padded[:-2, 2:]
            | padded[2:, :-2] | padded[2:, 2:]
        )
    return out


def _rasterize(strokes: list[Stroke]) -> np.ndarray:
    """Rasterize all strokes into a working boolean bitmap (ink=1)."""
    x0, y0, x1, y1 = _bbox(strokes)
    if x1 <= x0 and y1 <= y0:
        return np.zeros((1, 1), dtype=bool)
    w = max(x1 - x0, 1e-3)
    h = max(y1 - y0, 1e-3)
    scale = TARGET_H / max(h, 1e-6)
    W = max(int(round(w * scale)) + 1, 1)
    H = max(int(round(h * scale)) + 1, 1)

    img = np.zeros((H, W), dtype=bool)
    for s in strokes:
        pts = s.points if hasattr(s, "points") else s
        if not pts:
            continue
        px = [(int(round((p[0] - x0) * scale)), int(round((p[1] - y0) * scale)))
              for p in pts]
        _draw_polyline(img, px)
    if STROKE_THICKNESS > 1:
        img = _dilate(img, STROKE_THICKNESS // 2)
    return img


# ---------------------------------------------------------------------------
# Connected components
# ---------------------------------------------------------------------------

def _connected_components(img: np.ndarray) -> list[np.ndarray]:
    """Return the list of ink components as boolean masks (8-connectivity)."""
    H, W = img.shape
    seen = np.zeros_like(img, dtype=bool)
    comps: list[np.ndarray] = []
    for y in range(H):
        for x in range(W):
            if not img[y, x] or seen[y, x]:
                continue
            mask = np.zeros_like(img, dtype=bool)
            stack = deque([(y, x)])
            seen[y, x] = True
            while stack:
                cy, cx = stack.popleft()
                mask[cy, cx] = True
                for ny in (cy - 1, cy, cy + 1):
                    for nx in (cx - 1, cx, cx + 1):
                        if (0 <= ny < H and 0 <= nx < W
                                and img[ny, nx] and not seen[ny, nx]):
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            comps.append(mask)
    return comps


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def segment(strokes: list[Stroke]) -> list[Letter]:
    """Segment strokes into ordered letters (left-to-right, noise filtered)."""
    img = _rasterize(strokes)
    letters: list[Letter] = []
    for mask in _connected_components(img):
        ys, xs = np.where(mask)
        if len(xs) < MIN_COMPONENT_AREA:
            continue
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        crop = mask[y0:y1 + 1, x0:x1 + 1].astype(np.float32)
        letters.append(Letter(x0=x0, x1=x1, y0=y0, y1=y1, crop=crop))
    letters.sort(key=lambda l: l.x0)
    return letters


def segment_words(letters: list[Letter]) -> list[list[Letter]]:
    """Group ordered letters into words by horizontal x-gap."""
    if not letters:
        return []
    widths = sorted(l.x1 - l.x0 for l in letters)
    median_w = widths[len(widths) // 2] if widths else 1.0
    threshold = max(MIN_WORD_GAP_PX, WORD_GAP_FACTOR * median_w)

    words: list[list[Letter]] = []
    cur = [letters[0]]
    for g in letters[1:]:
        gap = g.x0 - cur[-1].x1
        if gap > threshold:
            words.append(cur)
            cur = [g]
        else:
            cur.append(g)
    words.append(cur)
    return words