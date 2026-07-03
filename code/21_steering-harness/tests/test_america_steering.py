import torch

from america_ai.steering import SteeringVector, apply_injection, build_hooks


def test_hook_construction_uses_correct_layer_and_strength():
    vector = SteeringVector("americana", 21, torch.tensor([1.0, 0.0]), 50.0, 0.04)
    hooks, descriptions = build_hooks([vector])
    assert hooks[0][0] == "blocks.21.hook_resid_post"
    assert descriptions[0]["injection_norm"] == 2.0


def test_negative_steering_reverses_injection_direction():
    pos = SteeringVector("x", 1, torch.tensor([1.0, 0.0]), 10.0, 0.1).injection(
        "cpu", torch.float32
    )
    neg = SteeringVector("x", 1, torch.tensor([1.0, 0.0]), 10.0, -0.1).injection(
        "cpu", torch.float32
    )
    assert torch.allclose(pos, -neg)


def test_generation_only_modifies_only_final_position():
    resid = torch.zeros(1, 3, 2)
    out = apply_injection(resid, torch.tensor([1.0, 2.0]), "generation_only")
    assert torch.allclose(out[:, 0, :], torch.zeros(1, 2))
    assert torch.allclose(out[:, 1, :], torch.zeros(1, 2))
    assert torch.allclose(out[:, 2, :], torch.tensor([[1.0, 2.0]]))


def test_all_positions_modifies_every_position():
    resid = torch.zeros(1, 2, 2)
    out = apply_injection(resid, torch.tensor([1.0, 2.0]), "all_positions")
    assert torch.allclose(out, torch.tensor([[[1.0, 2.0], [1.0, 2.0]]]))
