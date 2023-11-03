from spearmint import vis


def test_colors():
    assert hasattr(vis.COLORS, "blue")
    assert hasattr(vis, "DEFAULT_COLOR")
    assert hasattr(vis, "VARIATION_COLOR")
    assert hasattr(vis, "CONTROL_COLOR")
    assert hasattr(vis, "DELTA_COLOR")
    assert hasattr(vis, "POSITIVE_COLOR")
    assert hasattr(vis, "NEGATIVE_COLOR")
    assert hasattr(vis, "NEUTRAL_COLOR")
