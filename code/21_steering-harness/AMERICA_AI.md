# America AI

America AI is a Fourth-of-July political parody built on contrastive activation steering for `google/gemma-2-2b`. It is intentionally politically steered entertainment, not a neutral assistant and not factual guidance.

## Purpose

The project exposes four independently adjustable steering directions:

- `americana`: American imagery such as flags, fireworks, Main Street, monuments, barbecue, and civic symbols.
- `patriotic_pride`: positive civic orientation toward the United States.
- `trump_approval`: favorable opinions of Donald Trump and America First politics.
- `star_spangled_bombast`: excessive comedic patriotic rhetoric.

## Setup

From the repository root:

```bash
source .venv/bin/activate
cd code/21_steering-harness
```

The default model is `google/gemma-2-2b`. Do not silently switch checkpoints when comparing results.

For the exact RunPod/A40/Gemma process that worked, see `RUNPOD_GEMMA_A40_RUNBOOK.md`.

## RunPod Secret Setup

Gemma is gated on Hugging Face, so remote GPU runs need a Hugging Face token with access to `google/gemma-2-2b`.

Use RunPod secrets rather than copying tokens into scripts:

1. In the RunPod console, open Secrets.
2. Create or confirm the secret named `HUGGING_FACE`.
3. Store the Hugging Face token as the secret value. Read access is enough for Gemma downloads; a write-scoped token also works.
4. Start the remote runner with:

```bash
python run_america_remote.py --quick --runpod-secret-name HUGGING_FACE
```

The runner passes this to RunPod as:

```text
HF_TOKEN={{ RUNPOD_SECRET_HUGGING_FACE }}
HUGGING_FACE_HUB_TOKEN={{ RUNPOD_SECRET_HUGGING_FACE }}
```

The pod then runs `huggingface-cli login --token "$HF_TOKEN"` inside the container. The token value is resolved by RunPod and is not printed by the runner.

## Data Design

Datasets live under `data/america_ai/` and use paired examples:

```python
{
    "id": "unique-id",
    "prompt": "shared context",
    "positive": "target completion",
    "negative": "matched opposing or neutral completion",
    "split": "train",
    "category": "diagnostic category",
}
```

Each concept has 60 training, 15 validation, and 15 test pairs. The paired format keeps the prompt and topic stable while changing the attitude, imagery, or style being isolated.

## Vector Harvesting

Run:

```bash
python america_harvest.py
```

This harvests activations at layers `[6, 9, 12, 15, 18, 21, 24]` from `blocks.{layer}.hook_resid_post`, pools completion-token activations, builds paired mean-difference vectors, and writes `.npz` arrays with adjacent JSON metadata under `results/america_ai/vectors/`.

CPU-safe validation only:

```bash
python america_harvest.py --dry-run
```

## Layer Sweep

After harvesting:

```bash
python america_sweep.py --concept americana
python america_sweep.py --concept patriotic_pride
python america_sweep.py --concept trump_approval
python america_sweep.py --concept star_spangled_bombast
```

Strengths are fractions of typical residual norm. For example, `0.04` injects `0.04 * typical_norm * unit_vector`.

## Optimization

Dry-run:

```bash
python america_optimize.py --dry-run
```

Intended real run after measured vectors are available:

```bash
python america_optimize.py --trials 120 --stage-one-prompts 8 --finalists 20 --seeds 0,1,2
```

The optimizer is bounded and writes `results/america_ai/best_config.json`. It must report component scores separately and reject repetitive, corrupted, generic-politician, wrong-direction, or unrelated-intrusion configurations.

For a faster qualitative check using measured vectors:

```bash
python america_probe.py
python run_america_remote.py --probe-only --skip-harvest --runpod-secret-name HUGGING_FACE
```

The probe writes `results/america_ai/reports/probe_outputs.md`, `results/america_ai/reports/probe_outputs.json`, and a qualitative `results/america_ai/best_config.json`.

## Demo

After `best_config.json` and measured vectors exist:

```bash
python america_demo.py --preset america_ai --prompt "What should I make for dinner?"
python america_demo.py --americana 1.0 --patriotic-pride 1.0 --trump-approval 0.7 --bombast 1.2 --prompt "Explain the importance of local government."
python america_demo.py --interactive
```

The REPL supports `/preset`, `/set`, `/show`, and `/quit`.

## Presets

- `off`: all strengths zero.
- `patriot`: about 60% of the winning strengths.
- `america_ai`: measured winning strengths.
- `eagle_overdrive`: about 130% of winning strengths, clipped to tested limits.
- `anti_mode`: negative strengths for comparison.

Do not hardcode final layers. Use measured harvest and optimization outputs.

## Export and Deployment

After a `best_config.json` exists for the target model, package it for deployment:

```bash
python america_export.py --results-name google_gemma_2_2b_it
```

This writes `results/google_gemma_2_2b_it/deploy/steering_bundle.pt`
(schema_version 1: unit vectors, layers, IT-measured `typical_norm`, base
strengths, preset multipliers, hook mode, provenance).

The bundle is consumed by `america_ai/runtime.py`, a pure-`transformers`
runtime (no TransformerLens) with the same injection semantics:
`resid += multiplier * base_strength * typical_norm * unit_vector` at each
concept's layer. `SteeringState.set_strengths` / `set_preset` update hook
state without re-registering hooks.

The hosted demo lives in `america_space/` and is deployed to
https://huggingface.co/spaces/bitsofchris/america-ai (ZeroGPU):

```bash
america_space/deploy.sh   # copies runtime.py + bundle, uploads to the Space
```

The Space needs the `HF_TOKEN` secret (Gemma is gated) and the
`zero-a10g` hardware flavor. Tests for the export/runtime path are in
`tests/test_america_export.py` and `tests/test_america_runtime.py` (CPU-safe,
no model download).

## Known Failure Modes

- Steering changes tendencies, not model intelligence or factual reliability.
- High strengths can cause repetition, slogans, or topic intrusion.
- Trump approval can become generic politician approval unless specificity checks pass.
- Bombast can overfit punctuation unless paired examples and heuristics catch it.
- Validation chooses layers; test splits are reserved for final reporting.

## Reproduction

1. Activate the virtual environment.
2. Run `pytest -q`.
3. Run `python america_harvest.py --dry-run`.
4. On a machine with GPU and Hugging Face access, run `python america_harvest.py`.
5. Run all four `america_sweep.py --concept ...` commands.
6. Run `python america_optimize.py --trials 120 --stage-one-prompts 8 --finalists 20 --seeds 0,1,2`.
7. Inspect `results/america_ai/reports/summary.md` and `results/america_ai/best_config.json`.
