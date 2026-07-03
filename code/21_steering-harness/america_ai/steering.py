"""Reusable multi-layer steering hook construction."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SteeringVector:
    concept: str
    layer: int
    unit: torch.Tensor
    typical_norm: float
    strength_fraction: float

    @property
    def hook_name(self) -> str:
        return f"blocks.{self.layer}.hook_resid_post"

    def injection(self, device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
        return (
            self.strength_fraction
            * self.typical_norm
            * self.unit.to(device=device, dtype=dtype)
        )


def apply_injection(
    resid: torch.Tensor, injection: torch.Tensor, hook_mode: str
) -> torch.Tensor:
    out = resid.clone()
    if hook_mode == "all_positions":
        return out + injection
    if hook_mode == "generation_only":
        out[:, -1, :] = out[:, -1, :] + injection
        return out
    raise ValueError(f"unknown hook mode: {hook_mode}")


def build_hooks(vectors: list[SteeringVector], hook_mode: str = "generation_only"):
    hooks = []
    descriptions = []
    for vector in vectors:
        hook_name = vector.hook_name

        def make_hook(_vector: SteeringVector):
            def hook(resid, hook):
                injection = _vector.injection(resid.device, resid.dtype)
                return apply_injection(resid, injection, hook_mode)

            return hook

        hooks.append((hook_name, make_hook(vector)))
        descriptions.append(
            {
                "concept": vector.concept,
                "layer": vector.layer,
                "strength_fraction": vector.strength_fraction,
                "typical_norm": vector.typical_norm,
                "injection_norm": abs(vector.strength_fraction) * vector.typical_norm,
                "hook_mode": hook_mode,
            }
        )
    return hooks, descriptions


def generate_with_steering(
    model,
    prompt: str,
    vectors: list[SteeringVector],
    *,
    hook_mode: str = "generation_only",
    max_new_tokens: int = 80,
    temperature: float = 0.7,
    seed: int = 0,
) -> str:
    torch.manual_seed(seed)
    hooks, _ = build_hooks(vectors, hook_mode=hook_mode)
    with model.hooks(fwd_hooks=hooks):
        output = model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            verbose=False,
        )
    return output[len(prompt) :]
