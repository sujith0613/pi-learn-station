"""Unit tests for the backend pipeline (segmentation, recognition utils,
disambiguation scoring). Uses the app package; model files must exist for
recognition/disambiguation tests (skipped otherwise).
"""

import os

import numpy as np
import pytest

from app import confusions, segmentation
from app.recognition import normalize_strokes_to_bitmap


def test_segmentation_splits_on_pen_lift_gap():
    s1 = segmentation.Stroke([(0, 0, 0.0), (5, 0, 50.0)])
    s2 = segmentation.Stroke([(50, 0, 1000.0), (55, 0, 1050.0)])  # gap 950ms > 400ms
    groups = segmentation.group_strokes([s1, s2])
    assert len(groups) == 2


def test_segmentation_keeps_touching_strokes():
    s1 = segmentation.Stroke([(0, 0, 0.0), (5, 0, 50.0)])
    s2 = segmentation.Stroke([(4, 0, 100.0), (8, 0, 150.0)])  # x-overlap
    groups = segmentation.group_strokes([s1, s2])
    assert len(groups) == 1


def test_confusions_config():
    assert ("b", "d") in confusions.all_pairs()
    assert "d" in confusions.neighbours("b")
    assert confusions.CONFUSION_LETTERS == sorted(
        {l for p in confusions.all_pairs() for l in p})


def test_bitmap_normalization_shape():
    strokes = [segmentation.Stroke([(1, 1, 0.0), (2, 3, 0.1), (5, 2, 0.2)])]
    img = normalize_strokes_to_bitmap(strokes, size=28)
    assert img.shape == (28, 28)
    assert img.max() == 1.0
    assert img.min() == 0.0


def test_disambiguate_import():
    from app import disambiguate
    assert hasattr(disambiguate, "score_candidate")
    assert hasattr(disambiguate, "disambiguate")


def _stroke(pts):
    return segmentation.Stroke([(x, y, t) for (x, y, t) in pts])


def test_segment_words_splits_on_gap():
    # letter A: two x-overlapping strokes (t=0..50) -> one letter group
    a1 = _stroke([(0, 0, 0.0), (10, 0, 10.0)])
    a2 = _stroke([(4, 0, 40.0), (8, 0, 50.0)])
    # letter B: pen-lift 450ms > 400ms -> new letter, close in x
    b = _stroke([(20, 0, 500.0), (30, 0, 510.0)])
    # letter C: far in x from B -> new word
    c = _stroke([(300, 0, 1000.0), (310, 0, 1010.0)])
    groups = segmentation.group_strokes([a1, a2, b, c])
    assert len(groups) == 3  # A, B, C
    words = segmentation.segment_words(groups)
    assert len(words) == 2
    assert len(words[0]) == 2  # A+B in one word
    assert len(words[1]) == 1  # C alone
