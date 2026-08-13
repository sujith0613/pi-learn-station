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