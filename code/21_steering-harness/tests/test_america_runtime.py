from __future__ import annotations

import pytest
import torch
from torch import nn

from america_ai.runtime import (
    Bundle,
    SteeringState,
    attach_hooks,
    load_bundle,
    save_bundle,
)


def unit(d_model: int, index: int) -> torch.Tensor:
    vec = torch.zeros(d_model)
    vec[index] = 1.0
    return vec


def make_bundle(hook_mode: str = "all_positions") -> Bundle:
    d = 8
    return Bundle(
        schema_version=1,
        source_model="test/source",
        target_model="test/target",
        d_model=d,
        concepts={
            "americana": {
                "unit_vector": unit(d, 0),
                "layer": 1,
                "typical_norm": 10.0,
                "base_strength": 0.1,
            },
            "patriotic_pride": {
                "unit_vector": unit(d, 1),
                "layer": 3,
                "typical_norm": 20.0,
                "base_strength": 0.05,
            },
            # shares layer 3 to exercise per-layer summation
            "star_spangled_bombast": {
                "unit_vector": unit(d, 2),
                "layer": 3,
                "typical_norm": 4.0,
                "base_strength": 0.5,
            },
        },
        presets={
            "off": {
                "americana": 0.0,
                "patriotic_pride": 0.0,
                "star_spangled_bombast": 0.0,
            },
            "america_ai": {
                "americana": 1.0,
                "patriotic_pride": 1.0,
                "star_spangled_bombast": 1.0,
            },
            "anti_mode": {
                "americana": -1.0,
                "patriotic_pride": -1.0,
                "star_spangled_bombast": -1.0,
            },
        },
        hook_mode=hook_mode,
    )


class TestBundleRoundTrip:
    def test_save_load_preserves_schema(self, tmp_path):
        path = tmp_path / "bundle.pt"
        save_bundle(make_bundle(), path)
        loaded = load_bundle(path)
        assert loaded.schema_version == 1
        assert loaded.target_model == "test/target"
        assert loaded.hook_mode == "all_positions"
        assert set(loaded.concepts) == {
            "americana",
            "patriotic_pride",
            "star_spangled_bombast",
        }
        for concept in loaded.concepts.values():
            assert set(concept) >= {
                "unit_vector",
                "layer",
                "typical_norm",
                "base_strength",
            }
            assert abs(float(concept["unit_vector"].norm()) - 1.0) < 1e-5

    def test_rejects_non_unit_vector(self, tmp_path):
        bundle = make_bundle()
        bundle.concepts["americana"]["unit_vector"] = 2.0 * unit(8, 0)
        with pytest.raises(ValueError, match="unit norm"):
            save_bundle(bundle, tmp_path / "bad.pt")

    def test_rejects_unknown_hook_mode(self, tmp_path):
        bundle = make_bundle()
        bundle.hook_mode = "sometimes"
        with pytest.raises(ValueError, match="hook mode"):
            save_bundle(bundle, tmp_path / "bad.pt")


class TestSteeringState:
    def test_preset_resolves_to_expected_magnitudes(self):
        state = SteeringState(make_bundle())
        state.set_preset("america_ai")
        inj1 = state.injection(1)
        assert torch.allclose(inj1, 1.0 * 0.1 * 10.0 * unit(8, 0))
        inj3 = state.injection(3)
        expected = 1.0 * 0.05 * 20.0 * unit(8, 1) + 1.0 * 0.5 * 4.0 * unit(8, 2)
        assert torch.allclose(inj3, expected)

    def test_off_preset_yields_inactive_injection(self):
        state = SteeringState(make_bundle())
        state.set_preset("off")
        assert state.injection(1) is None
        assert state.injection(3) is None
        hidden = torch.randn(2, 5, 8)
        assert state.apply(hidden, 1) is hidden

    def test_negative_multiplier_reverses_direction(self):
        state = SteeringState(make_bundle())
        state.set_preset("america_ai")
        positive = state.injection(1).clone()
        state.set_preset("anti_mode")
        assert torch.allclose(state.injection(1), -positive)

    def test_unknown_concept_rejected(self):
        state = SteeringState(make_bundle())
        with pytest.raises(KeyError):
            state.set_strengths({"freedom_fries": 1.0})

    def test_all_positions_modifies_every_position(self):
        state = SteeringState(make_bundle("all_positions"))
        state.set_strengths({"americana": 1.0})
        hidden = torch.zeros(2, 5, 8)
        out = state.apply(hidden, 1)
        assert torch.all(out[:, :, 0] == 1.0)  # 1.0 * 0.1 * 10.0
        assert torch.all(out[:, :, 1:] == 0.0)
        assert torch.all(hidden == 0.0)  # original untouched

    def test_generation_only_prompt_pass_modifies_final_position_only(self):
        state = SteeringState(make_bundle("generation_only"))
        state.set_strengths({"americana": 1.0})
        hidden = torch.zeros(2, 5, 8)
        out = state.apply(hidden, 1)
        assert torch.all(out[:, -1, 0] == 1.0)
        assert torch.all(out[:, :-1, :] == 0.0)

    def test_generation_only_decode_step_modifies_single_token(self):
        state = SteeringState(make_bundle("generation_only"))
        state.set_strengths({"americana": 1.0})
        hidden = torch.zeros(2, 1, 8)
        out = state.apply(hidden, 1)
        assert torch.all(out[:, 0, 0] == 1.0)

    def test_apply_preserves_dtype(self):
        state = SteeringState(make_bundle())
        state.set_preset("america_ai")
        hidden = torch.zeros(1, 3, 8, dtype=torch.bfloat16)
        out = state.apply(hidden, 1)
        assert out.dtype == torch.bfloat16


class FakeBlock(nn.Module):
    """Mimics a HF decoder layer: returns a tuple whose first element is hidden states."""

    def forward(self, hidden):
        return (hidden, "extra")


class FakeCausalLM(nn.Module):
    def __init__(self, num_layers: int):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(FakeBlock() for _ in range(num_layers))

    def forward(self, hidden):
        for block in self.model.layers:
            hidden = block(hidden)[0]
        return hidden


class TestAttachHooks:
    def test_hooks_apply_at_correct_layers_with_correct_scale(self):
        state = SteeringState(make_bundle("all_positions"))
        state.set_preset("america_ai")
        model = FakeCausalLM(num_layers=5)
        handles = attach_hooks(model, state)
        assert len(handles) == 2  # layers 1 and 3

        out = model(torch.zeros(1, 4, 8))
        assert torch.all(out[:, :, 0] == 1.0)  # americana at layer 1
        assert torch.all(out[:, :, 1] == 1.0)  # patriotic_pride at layer 3
        assert torch.all(out[:, :, 2] == 2.0)  # bombast at layer 3
        assert torch.all(out[:, :, 3:] == 0.0)

    def test_hooks_are_noops_when_off(self):
        state = SteeringState(make_bundle("all_positions"))
        state.set_preset("off")
        model = FakeCausalLM(num_layers=5)
        attach_hooks(model, state)
        out = model(torch.zeros(1, 4, 8))
        assert torch.all(out == 0.0)

    def test_set_strengths_updates_without_reregistering(self):
        state = SteeringState(make_bundle("all_positions"))
        model = FakeCausalLM(num_layers=5)
        attach_hooks(model, state)
        state.set_strengths({"americana": 1.0})
        assert torch.all(model(torch.zeros(1, 2, 8))[:, :, 0] == 1.0)
        state.set_strengths({"americana": -0.5})
        assert torch.all(model(torch.zeros(1, 2, 8))[:, :, 0] == -0.5)
