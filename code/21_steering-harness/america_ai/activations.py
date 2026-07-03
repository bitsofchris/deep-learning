"""Activation harvesting and response-token pooling."""

from __future__ import annotations

import torch

from america_ai.datasets import Pair


def completion_token_slice(model, prompt: str, text: str) -> slice:
    prompt_tokens = model.to_tokens(prompt, prepend_bos=True)
    text_tokens = model.to_tokens(text, prepend_bos=True)
    start = min(prompt_tokens.shape[-1], text_tokens.shape[-1] - 1)
    return slice(start, text_tokens.shape[-1])


def pool_residual(
    resid: torch.Tensor, completion_slice: slice, mode: str, last_n: int = 8
) -> torch.Tensor:
    if resid.ndim != 2:
        raise ValueError(f"expected (seq, d_model), got {tuple(resid.shape)}")
    if mode == "last":
        return resid[-1]
    completion = resid[completion_slice]
    if completion.shape[0] == 0:
        completion = resid[-1:]
    if mode == "response_mean":
        return completion.mean(dim=0)
    if mode == "response_last_n":
        return completion[-last_n:].mean(dim=0)
    raise ValueError(f"unknown pooling mode: {mode}")


def harvest_pair_activations(
    model,
    pairs: list[Pair],
    layer: int,
    pooling_mode: str = "response_mean",
    response_last_n: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    hook_name = f"blocks.{layer}.hook_resid_post"
    positive, negative = [], []
    for pair in pairs:
        for text, target in [
            (pair.positive_text(), positive),
            (pair.negative_text(), negative),
        ]:
            tokens = model.to_tokens(text)
            _, cache = model.run_with_cache(tokens, names_filter=hook_name)
            resid = cache[hook_name][0].detach().cpu().float()
            target.append(
                pool_residual(
                    resid,
                    completion_token_slice(model, pair.prompt, text),
                    pooling_mode,
                    response_last_n,
                )
            )
    return torch.stack(positive), torch.stack(negative)
