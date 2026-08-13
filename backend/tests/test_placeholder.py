"""Placeholder unit tests. Expanded in Phase 3 (segmentation, disambiguation)."""


def test_segmentation_placeholder():
    from app import segmentation
    assert segmentation.PEN_LIFT_GAP_MS > 0


def test_confusions_config():
    from ml import confusions  # noqa: F401  (shared config, may move)
    assert ("b", "d") in confusions.TIER_A