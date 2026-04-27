from app.services import drift


def setup_function(_):
    # Test 시작마다 latest report reset
    drift._latest = None


def test_initial_report_is_none():
    assert drift.latest_report() is None


def test_set_latest_report_round_trip():
    report = drift.DriftReport(
        measured_at=1.0,
        samples=30,
        avg_cos=0.999,
        min_cos=0.995,
        violations=0,
        threshold=0.99,
        available=True,
    )
    drift._set_latest(report)
    out = drift.latest_report()
    assert out is not None
    assert out["samples"] == 30
    assert out["violations"] == 0
    assert out["available"] is True


def test_unavailable_report_carries_error():
    report = drift.DriftReport(
        measured_at=1.0,
        samples=0,
        avg_cos=0.0,
        min_cos=0.0,
        violations=0,
        threshold=0.99,
        available=False,
        error="sentence-transformers not installed",
    )
    drift._set_latest(report)
    out = drift.latest_report()
    assert out["available"] is False
    assert "sentence-transformers" in out["error"]


def test_threshold_violation_metric():
    report = drift.DriftReport(
        measured_at=1.0,
        samples=30,
        avg_cos=0.85,
        min_cos=0.50,
        violations=12,
        threshold=0.99,
        available=True,
    )
    drift._set_latest(report)
    out = drift.latest_report()
    assert out["violations"] == 12
    assert out["min_cos"] < out["threshold"]
