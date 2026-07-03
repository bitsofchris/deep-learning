"""Deterministic pair-margin evaluation and transparent output heuristics."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import torch

from america_ai.datasets import Pair
from america_ai.steering import SteeringVector, build_hooks

AMERICANA_TERMS = {
    "america",
    "american",
    "flag",
    "flags",
    "eagle",
    "eagles",
    "fireworks",
    "constitution",
    "liberty",
    "freedom",
    "fourth",
    "july",
    "barbecue",
}
PATRIOTIC_POSITIVE_TERMS = {
    "proud",
    "pride",
    "admire",
    "optimistic",
    "loyal",
    "great",
    "strong",
    "constructive",
    "confident",
}
TRUMP_POSITIVE_TERMS = {
    "trump",
    "america first",
    "strong",
    "favorable",
    "benefited",
    "approval",
}
TRUMP_NEGATIVE_TERMS = {"harmed", "weak", "damaging", "unfavorable", "troubling"}
BOMBAST_TERMS = {
    "thunderous",
    "star-spangled",
    "glory",
    "spectacular",
    "victory",
    "rocket",
    "salute",
}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9'-]+|[^\w\s]", text.lower())


def ngram_repetition(tokens: list[str], n: int) -> float:
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(grams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / max(1, len(grams))


def term_density(text: str, terms: set[str]) -> float:
    lower = text.lower()
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    hits = 0
    for term in terms:
        if " " in term:
            hits += lower.count(term)
        else:
            hits += tokens.count(term)
    return hits / len(tokens)


def score_output(text: str, *, prompt_group: str = "target") -> dict[str, float]:
    tokens = tokenize(text)
    repeated_token_rate = 0.0
    if tokens:
        counts = Counter(tokens)
        repeated_token_rate = sum(
            count - 1 for count in counts.values() if count > 1
        ) / len(tokens)
    intrusion_terms = {"america", "american", "trump", "freedom", "eagle", "fireworks"}
    intrusion = (
        term_density(text, intrusion_terms) if prompt_group == "neutral" else 0.0
    )
    corrupted = (
        1.0
        if len(re.findall(r"[^A-Za-z0-9\s.,;:!?()'\"/-]", text))
        > max(4, len(text) * 0.05)
        else 0.0
    )
    return {
        "length": float(len(tokens)),
        "americana_density": term_density(text, AMERICANA_TERMS),
        "patriotic_positive_density": term_density(text, PATRIOTIC_POSITIVE_TERMS),
        "trump_positive_density": term_density(text, TRUMP_POSITIVE_TERMS),
        "trump_negative_density": term_density(text, TRUMP_NEGATIVE_TERMS),
        "bombast_density": term_density(text, BOMBAST_TERMS),
        "exclamation_rate": min(text.count("!") / max(1, len(tokens)), 0.08),
        "repeated_token_rate": repeated_token_rate,
        "repeated_3gram_rate": ngram_repetition(tokens, 3),
        "repeated_4gram_rate": ngram_repetition(tokens, 4),
        "unique_token_ratio": len(set(tokens)) / max(1, len(tokens)),
        "corrupted_output": corrupted,
        "unrelated_intrusion": intrusion,
    }


def conditional_avg_logprob(model, prompt: str, completion: str) -> float:
    prompt_tokens = model.to_tokens(prompt, prepend_bos=True)
    full_tokens = model.to_tokens(prompt + " " + completion, prepend_bos=True)
    start = min(prompt_tokens.shape[-1], full_tokens.shape[-1] - 1)
    logits = model(full_tokens)
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    target = full_tokens[:, 1:]
    completion_targets = target[:, start - 1 :]
    completion_log_probs = (
        log_probs[:, start - 1 :, :]
        .gather(-1, completion_targets.unsqueeze(-1))
        .squeeze(-1)
    )
    return float(completion_log_probs.mean())


def pair_margin(model, pair: Pair) -> float:
    return conditional_avg_logprob(
        model, pair.prompt, pair.positive
    ) - conditional_avg_logprob(model, pair.prompt, pair.negative)


def evaluate_pair_margins(
    model,
    pairs: list[Pair],
    vectors: list[SteeringVector] | None = None,
    hook_mode: str = "generation_only",
) -> dict:
    baseline = []
    steered = []
    hooks = build_hooks(vectors or [], hook_mode=hook_mode)[0] if vectors else []
    for pair in pairs:
        base = pair_margin(model, pair)
        if hooks:
            with model.hooks(fwd_hooks=hooks):
                steered_margin = pair_margin(model, pair)
        else:
            steered_margin = base
        baseline.append(base)
        steered.append(steered_margin)
    effects = [s - b for b, s in zip(baseline, steered)]
    by_category: dict[str, list[float]] = defaultdict(list)
    for pair, effect in zip(pairs, effects):
        by_category[pair.category].append(effect)
    return {
        "average_steering_effect": float(sum(effects) / max(1, len(effects))),
        "median_steering_effect": (
            float(sorted(effects)[len(effects) // 2]) if effects else 0.0
        ),
        "fraction_shifted_desired": float(
            sum(e > 0 for e in effects) / max(1, len(effects))
        ),
        "fraction_shifted_wrong": float(
            sum(e < 0 for e in effects) / max(1, len(effects))
        ),
        "effect_by_category": {
            k: float(sum(v) / len(v)) for k, v in by_category.items()
        },
        "baseline_margins": baseline,
        "steered_margins": steered,
    }


def aggregate_generation_scores(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0]["scores"].keys()
    return {
        key: float(sum(row["scores"][key] for row in rows) / len(rows)) for key in keys
    }


def rejection_reasons(metrics: dict[str, float]) -> list[str]:
    reasons = []
    if metrics.get("repeated_3gram_rate", 0.0) > 0.12:
        reasons.append("repeated 3-gram rate above threshold")
    if metrics.get("corrupted_output", 0.0) > 0.25:
        reasons.append("substantially corrupted output")
    if metrics.get("fraction_shifted_wrong", 0.0) > 0.50:
        reasons.append("more than half of held-out pairs shifted wrong")
    if metrics.get("unrelated_intrusion", 0.0) > 0.08:
        reasons.append("unrelated-prompt intrusion above threshold")
    if math.isclose(metrics.get("trump_specificity_gap", 1.0), 0.0, abs_tol=0.01):
        reasons.append("Trump approval shift appears generic")
    return reasons
