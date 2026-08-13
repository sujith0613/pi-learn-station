"""Stroke → letter segmentation.

Input: a list of strokes, each stroke = [(x, y, t), ...] (pointer events).
Output: list of letter-groups, each = list of strokes.

Strategy:
1. Cluster consecutive strokes by pen-lift time gap + x-range proximity
   (kids lift the pen between letters).
2. If a single stroke is much wider than the median (>=1.5x), split it by
   x-projection valley.
(Tuned against real finger input on the Pi in Phase 4.)
"""

from dataclasses import dataclass, field

PEN_LIFT_GAP_MS = 400.0
WIDTH_RATIO_SPLIT = 1.5
WORD_GAP_FACTOR = 1.0       # inter-word x-gap threshold as multiple of median letter width
MIN_WORD_GAP_PX = 60.0


@dataclass
class Stroke:
    points: list = field(default_factory=list)  # [(x, y, t)]


def _x_range(strokes: list[Stroke]) -> tuple[float, float]:
    xs = [p[0] for s in strokes for p in s.points]
    return (min(xs), max(xs)) if xs else (0.0, 0.0)


def _last_time(strokes: list[Stroke]) -> float:
    return max((p[2] for s in strokes for p in s.points),
               default=0.0)


def _first_time(strokes: list[Stroke]) -> float:
    return min((p[2] for s in strokes for p in s.points),
               default=0.0)


def group_strokes(strokes: list[Stroke]) -> list[list[Stroke]]:
    groups: list[list[Stroke]] = []

    for stroke in strokes:
        if not groups:
            groups.append([stroke])
            continue

        prev = groups[-1]
        px_lo, px_hi = _x_range(prev)
        gap = _first_time([stroke]) - _last_time(prev)

        sx_lo, sx_hi = _x_range([stroke])
        x_overlap = px_lo <= sx_hi and sx_lo <= px_hi

        if gap <= PEN_LIFT_GAP_MS or x_overlap:
            prev.append(stroke)
        else:
            groups.append([stroke])

    # split over-wide groups by x-projection valley
    split = []
    for g in groups:
        split.extend(_split_wide(g))
    return split


def _split_wide(group: list[Stroke]) -> list[list[Stroke]]:
    # Expanded in Phase 3 with projection analysis; conservative for now.
    return [group]


def _bbox_x(strokes: list[Stroke]) -> tuple[float, float]:
    xs = [p[0] for s in strokes for p in s.points]
    return (min(xs), max(xs)) if xs else (0.0, 0.0)


def segment_words(letter_groups: list[list[Stroke]]) -> list[list[list[Stroke]]]:
    """Group letter-groups into words.

    A word boundary is a horizontal gap between consecutive letters that
    exceeds a dynamic threshold (multiple of the median letter width), so it
    works across canvas sizes and handwriting scales.
    """
    if not letter_groups:
        return []

    widths = [hi - lo for lo, hi in (_bbox_x(g) for g in letter_groups)]
    median_w = sorted(widths)[len(widths) // 2] if widths else 1.0
    threshold = max(MIN_WORD_GAP_PX, WORD_GAP_FACTOR * median_w)

    words: list[list[list[Stroke]]] = []
    cur = [letter_groups[0]]
    for g in letter_groups[1:]:
        gap = _bbox_x(g)[0] - _bbox_x(cur[-1])[1]
        if gap > threshold:
            words.append(cur)
            cur = [g]
        else:
            cur.append(g)
    words.append(cur)
    return words