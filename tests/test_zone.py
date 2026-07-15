from src.zone.zone import Zone, ZoneResidencyTracker

SQUARE = [(0, 0), (10, 0), (10, 10), (0, 10)]


def test_zone_contains_point_inside():
    zone = Zone(SQUARE)
    assert zone.contains((5, 5)) is True


def test_zone_does_not_contain_point_outside():
    zone = Zone(SQUARE)
    assert zone.contains((20, 20)) is False


def test_residency_requires_consecutive_frames():
    zone = Zone(SQUARE)
    tracker = ZoneResidencyTracker(zone, required_frames=3)

    assert tracker.update("p1", (5, 5)) is False
    assert tracker.update("p1", (5, 5)) is False
    assert tracker.update("p1", (5, 5)) is True


def test_residency_resets_when_leaving_zone():
    zone = Zone(SQUARE)
    tracker = ZoneResidencyTracker(zone, required_frames=2)

    tracker.update("p1", (5, 5))
    tracker.update("p1", (20, 20))  # zone 이탈 -> 카운트 리셋
    assert tracker.update("p1", (5, 5)) is False
