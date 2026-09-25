"""Unit tests for the safety fields."""
from core.safety import SafetyFields


def far_from_map(x, y):
    return 5.0   # every point is off the map


def on_map(x, y):
    return 0.0   # every point is a mapped surface


def test_protective_field_stops_the_robot():
    limit, gap = SafetyFields().speed_limit(0.0, 0.0, 0.30, [(0.6, 0.0)])
    assert limit == 0.0
    assert abs(gap - 0.30) < 1e-9


def test_warning_field_caps_the_speed():
    fields = SafetyFields()
    limit, _ = fields.speed_limit(0.0, 0.0, 0.30, [(1.2, 0.0)])
    assert limit == fields.warning_speed_m_s


def test_mapped_surfaces_do_not_trigger_the_fields():
    # A wall 0.4 m away is on the map. If it counted, the robot would stop in
    # every doorway it drove through.
    fields = SafetyFields()
    points = [(0.4, 0.0), (0.4, 0.1)]
    assert fields.unmapped_points(points, on_map) == []
    assert fields.unmapped_points(points, far_from_map) == points
