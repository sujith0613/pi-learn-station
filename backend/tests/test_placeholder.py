"""Unit tests for the backend pipeline (image-based segmentation, recognition
utils, disambiguation scoring). Uses the app package; model files must exist
for recognition/disambiguation tests (skipped otherwise).
"""

import numpy as np
import pytest

from app import confusions, segmentation
from app.recognition import normalize_component_to_bitmap


def _stroke(pts):
    return segmentation.Stroke([(x, y, t) for (x, y, t) in pts])


def test_segmentation_splits_separate_blobs():
    # two distinct ink blobs far apart -> two letters, sorted left-to-right
    s1 = _stroke([(0, 5, 0.0), (10, 5, 10.0)])          # left blob
    s2 = _stroke([(80, 5, 20.0), (95, 5, 30.0)])        # right blob
    letters = segmentation.segment([s1, s2])
    assert len(letters) == 2
    assert letters[0].x0 < letters[1].x0


def test_segmentation_merges_touching_strokes():
    # two strokes that touch -> one letter (multi-stroke letter stays together)
    s1 = _stroke([(0, 0, 0.0), (20, 20, 10.0)])  # diagonal
    s2 = _stroke([(20, 20, 20.0), (20, 5, 30.0)])  # shares endpoint
    letters = segmentation.segment([s1, s2])
    assert len(letters) == 1


def test_segmentation_filters_noise():
    # a real blob plus a tiny speck -> only the real blob survives
    big = _stroke([(0, 0, 0.0), (0, 30, 5.0), (30, 30, 10.0), (30, 0, 15.0)])
    speck = _stroke([(100, 100, 20.0), (100.5, 100.5, 21.0)])
    letters = segmentation.segment([big, speck])
    assert len(letters) == 1


def test_segment_words_splits_on_gap():
    # two letters close (one word) + one letter far (second word)
    a = _stroke([(0, 0, 0.0), (5, 5, 5.0), (10, 0, 10.0)])
    b = _stroke([(20, 0, 20.0), (28, 5, 25.0)])
    c = _stroke([(200, 0, 30.0), (210, 5, 35.0)])
    letters = segmentation.segment([a, b, c])
    words = segmentation.segment_words(letters)
    assert len(words) == 2
    assert len(words[0]) == 2  # a + b
    assert len(words[1]) == 1  # c


def test_bitmap_normalization_shape():
    crop = np.zeros((40, 20), dtype=np.float32)
    crop[10:30, 5:15] = 1.0
    img = normalize_component_to_bitmap(crop)
    assert img.shape == (28, 28)
    assert img.max() == 1.0
    assert img.min() == 0.0


def test_bitmap_normalization_centers_ink():
    crop = np.zeros((100, 40), dtype=np.float32)
    crop[20:80, 10:30] = 1.0
    img = normalize_component_to_bitmap(crop)
    assert img.shape == (28, 28)
    # ink should occupy a centered 20x20-ish region, not touch the border
    assert img[0].sum() == 0
    assert img[:, 0].sum() == 0
    assert img[27].sum() == 0
    assert img[:, 27].sum() == 0


def test_confusions_config():
    assert ("b", "d") in confusions.all_pairs()
    assert "d" in confusions.neighbours("b")
    assert confusions.CONFUSION_LETTERS == sorted(
        {l for p in confusions.all_pairs() for l in p})


def test_disambiguate_import():
    from app import disambiguate
    assert hasattr(disambiguate, "score_candidate")
    assert hasattr(disambiguate, "disambiguate")