"""Pure-transformers steering runtime for the America AI deployment.

Consumes ``steering_bundle.pt`` (schema_version 1) produced by
``america_export.py``. No TransformerLens dependency: forward hooks are
registered on ``model.model.layers[L]`` and reproduce the harness semantics

    resid += multiplier * base_strength * typical_norm * unit_vector

at each concept's layer. This module is intentionally self-contained (no
imports from the ``america_ai`` package) so it can be copied verbatim into
the Hugging Face Space repo as a flat ``runtime.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

SCHEMA_VERSION = 1
HOOK_MODES = {"all_positions", "generation_only"}
CONCEPT_KEYS = {"unit_vector", "layer", "typical_norm", "base_strength"}


@dataclass
class Bundle:
    schema_version: int
    source_model: str
    target_model: str
    d_model: int
    concepts: dict[str, dict]
    presets: dict[str, dict[str, float]]
    hook_mode: str
    provenance: dict = field(default_factory=dict)


def save_bundle(bundle: Bundle, path: Path | str) -> None:
    validate_bundle(bundle)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": bundle.schema_version,
            "source_model": bundle.source_model,
            "target_model": bundle.target_model,
            "d_model": bundle.d_model,
            "concepts": bundle.concepts,
            "presets": bundle.presets,
            "hook_mode": bundle.hook_mode,
            "provenance": bundle.provenance,
        },
        path,
    )


def load_bundle(path: Path | str) -> Bundle:
    raw = torch.load(Path(path), map_location="cpu", weights_only=False)
    bundle = Bundle(
        schema_version=int(raw["schema_version"]),
        source_model=raw["source_model"],
        target_model=raw["target_model"],
        d_model=int(raw["d_model"]),
        concepts=raw["concepts"],
        presets=raw["presets"],
        hook_mode=raw["hook_mode"],
        provenance=raw.get("provenance", {}),
    )
    validate_bundle(bundle)
    return bundle


def validate_bundle(bundle: Bundle) -> None:
    if bundle.schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {bundle.schema_version}")
    if bundle.hook_mode not in HOOK_MODES:
        raise ValueError(f"unknown hook mode: {bundle.hook_mode}")
    if not bundle.concepts:
        raise ValueError("bundle has no concepts")
    for name, concept in bundle.concepts.items():
        missing = CONCEPT_KEYS - concept.keys()
        if missing:
            raise ValueError(f"concept {name} missing keys: {sorted(missing)}")
        vec = concept["unit_vector"]
        if not isinstance(vec, torch.Tensor) or vec.shape != (bundle.d_model,):
            raise ValueError(
                f"concept {name} unit_vector must be shape ({bundle.d_model},)"
            )
        if abs(float(vec.float().norm()) - 1.0) > 1e-3:
            raise ValueError(f"concept {name} unit_vector is not unit norm")
    for preset, multipliers in bundle.presets.items():
        unknown = set(multipliers) - set(bundle.concepts)
        if unknown:
            raise ValueError(
                f"preset {preset} references unknown concepts: {sorted(unknown)}"
            )


class SteeringState:
    """Holds slider multipliers and the per-layer injection tensors they imply.

    Injection tensors live on CPU in float32; ``apply`` casts to the hidden
    state's device/dtype so the same state works on CPU, CUDA, and ZeroGPU.
    """

    def __init__(self, bundle: Bundle):
        self.bundle = bundle
        self.hook_mode = bundle.hook_mode
        self.multipliers = {name: 0.0 for name in bundle.concepts}
        self._by_layer: dict[int, list[str]] = {}
        for name, concept in bundle.concepts.items():
            self._by_layer.setdefault(int(concept["layer"]), []).append(name)
        self._injections: dict[int, torch.Tensor | None] = {}
        self._rebuild()

    def layers(self) -> list[int]:
        return sorted(self._by_layer)

    def set_strengths(self, multipliers: dict[str, float]) -> None:
        unknown = set(multipliers) - set(self.multipliers)
        if unknown:
            raise KeyError(f"unknown concepts: {sorted(unknown)}")
        self.multipliers.update(
            {name: float(value) for name, value in multipliers.items()}
        )
        self._rebuild()

    def set_preset(self, name: str) -> None:
        self.set_strengths(self.bundle.presets[name])

    def injection(self, layer: int) -> torch.Tensor | None:
        return self._injections.get(layer)

    def _rebuild(self) -> None:
        for layer, names in self._by_layer.items():
            total = torch.zeros(self.bundle.d_model, dtype=torch.float32)
            active = False
            for name in names:
                multiplier = self.multipliers[name]
                if multiplier == 0.0:
                    continue
                concept = self.bundle.concepts[name]
                scale = (
                    multiplier
                    * float(concept["base_strength"])
                    * float(concept["typical_norm"])
                )
                total = total + scale * concept["unit_vector"].float()
                active = True
            self._injections[layer] = total if active else None

    def apply(self, hidden: torch.Tensor, layer: int) -> torch.Tensor:
        """Return steered hidden states; returns ``hidden`` unchanged if inactive."""
        vec = self._injections.get(layer)
        if vec is None:
            return hidden
        vec = vec.to(device=hidden.device, dtype=hidden.dtype)
        out = hidden.clone()
        if self.hook_mode == "all_positions":
            out = out + vec
        else:  # generation_only: prompt pass -> final position; cached decode -> that token
            out[:, -1, :] = out[:, -1, :] + vec
        return out


def attach_hooks(model, state: SteeringState) -> list:
    """Register one forward hook per steered layer on ``model.model.layers``."""
    layer_modules = model.model.layers
    handles = []
    for layer in state.layers():
        handles.append(
            layer_modules[layer].register_forward_hook(_make_hook(state, layer))
        )
    return handles


def _make_hook(state: SteeringState, layer: int):
    def hook(module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        steered = state.apply(hidden, layer)
        if steered is hidden:
            return output
        if isinstance(output, tuple):
            return (steered,) + tuple(output[1:])
        return steered

    return hook


class SteeredGemma:
    """Bundle + plain-transformers model with steering hooks attached."""

    def __init__(
        self,
        bundle: Bundle,
        model_id: str | None = None,
        device: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        hf_token: str | None = None,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.bundle = bundle
        self.model_id = model_id or bundle.target_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            attn_implementation="eager",  # recommended for gemma-2
            token=hf_token,
        ).to(self.device)
        self.model.eval()
        self.state = SteeringState(bundle)
        self._handles = attach_hooks(self.model, self.state)

    def set_strengths(self, multipliers: dict[str, float]) -> None:
        self.state.set_strengths(multipliers)

    def set_preset(self, name: str) -> None:
        self.state.set_preset(name)

    def build_input_ids(self, messages: list[dict[str, str]]) -> torch.Tensor:
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)

    @torch.no_grad()
    def generate(self, messages: list[dict[str, str]], **generate_kwargs) -> str:
        input_ids = self.build_input_ids(messages)
        output = self.model.generate(input_ids=input_ids, **generate_kwargs)
        return self.tokenizer.decode(
            output[0, input_ids.shape[-1] :], skip_special_tokens=True
        )
