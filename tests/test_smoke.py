"""Smoke tests — verify imports and basic logic without downloading models."""

from __future__ import annotations


def test_imports():
    """All core modules import without error."""
    import analyze
    import data
    import evaluate
    import features
    import model
    from methods import METHODS

    assert callable(model.load_model)
    assert callable(data.load_task)
    assert callable(evaluate.evaluate_all)
    assert callable(features.collect_activations)
    assert callable(analyze.compute_drift)
    assert len(METHODS) > 0


def test_method_registry():
    """All expected methods are registered."""
    from methods import METHODS

    expected = {"naive", "replay", "distill", "hidden_distill", "orthogonal", "sfp",
                "sfp_grad", "sfp_random_basis", "sfp_random_selection",
                "forgetting_curve", "forgetting_curve_dist",
                "forgetting_curve_sharp", "forgetting_curve_replay",
                "forgetting_curve_soft"}
    assert expected == set(METHODS.keys())


def test_primary_metrics():
    """PRIMARY_METRIC covers all tasks."""
    from evaluate import PRIMARY_METRIC

    assert set(PRIMARY_METRIC.keys()) == {"math", "code", "ifeval", "safety", "domain"}


def test_leaderboard_score():
    from evaluate import leaderboard_score

    assert leaderboard_score(1.0, 1.0) == 1.0
    assert leaderboard_score(0.0, 0.0) == 0.0
    assert abs(leaderboard_score(0.5, 0.5) - 0.5) < 1e-9


def test_compute_forgetting_empty():
    from evaluate import compute_forgetting

    assert compute_forgetting([]) == {}


def test_compute_forgetting_basic():
    from evaluate import compute_forgetting

    history = [
        {"math": {"math_accuracy": 0.8}},
        {"math": {"math_accuracy": 0.5}, "code": {"code_pass_at_1": 0.6}},
    ]
    f = compute_forgetting(history)
    assert abs(f["math"] - 0.3) < 1e-9


def test_memory_buffer():
    from data import create_memory_buffer

    dataset = [{"input": f"q{i}", "output": f"a{i}", "task_id": "test"} for i in range(100)]
    buf = create_memory_buffer(dataset, size=10)
    assert len(buf) == 10
    assert all("input" in ex and "output" in ex for ex in buf)
