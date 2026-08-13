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
