import torch

from america_ai.vectors import (
    bootstrap_stability,
    build_vector,
    orthogonalize_vectors,
    projection_stats,
)


def test_vector_normalization_and_paired_difference():
    positive = torch.tensor([[2.0, 0.0], [4.0, 0.0]])
    negative = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    built = build_vector(positive, negative)
    assert torch.allclose(
        built["paired_deltas"], torch.tensor([[1.0, 0.0], [3.0, 0.0]])
    )
    assert torch.allclose(built["mean_delta"], torch.tensor([2.0, 0.0]))
    assert torch.allclose(built["unit"], torch.tensor([1.0, 0.0]))
    assert abs(float(built["unit"].norm()) - 1.0) < 1e-6


def test_bootstrap_is_deterministic_with_seed():
    deltas = torch.tensor([[1.0, 0.0], [0.9, 0.1], [1.1, -0.1], [1.0, 0.2]])
    assert bootstrap_stability(deltas, samples=10, seed=123) == bootstrap_stability(
        deltas, samples=10, seed=123
    )


def test_projection_stats_reports_success_fraction():
    deltas = torch.tensor([[1.0, 0.0], [2.0, 0.0], [-1.0, 0.0]])
    stats = projection_stats(deltas, torch.tensor([1.0, 0.0]))
    assert abs(stats["fraction_above_zero"] - (2 / 3)) < 1e-6


def test_orthogonalization_preserves_ordered_unit_vectors():
    vectors = {"a": torch.tensor([1.0, 0.0]), "b": torch.tensor([1.0, 1.0])}
    out = orthogonalize_vectors(vectors, ["a", "b"])
    assert torch.allclose(out["a"], torch.tensor([1.0, 0.0]))
    assert torch.allclose(out["b"], torch.tensor([0.0, 1.0]), atol=1e-6)
