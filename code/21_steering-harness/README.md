# Steering Harness

Injecting steering vectors into a local open-weights model's residual stream at inference time. See [plan.md](plan.md) for the full context and roadmap.

## Status

**v0** — minimum viable rig. One script that reproduces Golden Gate Claude on Gemma-2-2B. No UI, no framework.

## Hardware

Author works on an old Mac that cannot run Gemma-2-2B locally. **RunPod** is the default compute path.

### RunPod setup (recommended)

1. Pod: A40 (48GB VRAM, ~$0.40/hr) or RTX 3090 (24GB, cheaper). Gemma-2-2B in bf16 fits in ~5GB, so any modern GPU works.
2. Image: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1`
3. SSH in, clone this repo.
4. `pip install -r requirements.txt`
5. If the Gemma repo is gated on your HF account: `huggingface-cli login` (needs [token with Gemma access](https://huggingface.co/google/gemma-2-2b)).
6. `python v0_golden_gate.py`

### Local (for users with a 3090/4090/M3 Max)

Same install. First run downloads ~5GB of weights to `~/.cache/huggingface/`.

## Known risks before first run

- **transformer_lens + Gemma-2 compatibility is the biggest unknown.** Gemma-2's soft-capping (attention logits capped at 50, final at 30) and alternating sliding-window layers have caused numerical divergence in non-reference implementations. Smoke test `HookedTransformer.from_pretrained("google/gemma-2-2b")` and a baseline `model.generate("Hello")` first. If it fails or produces garbage, fall back to nnsight.
- **Activation magnitudes.** Residual streams at middle layers have norms in the tens. v0 scales the steering alpha as a fraction of the typical residual norm — don't reinterpret `alpha` as a unit-vector coefficient without adjusting.

## What v0 does

1. Loads Gemma-2-2B
2. Runs 30 Golden Gate sentences and 30 generic-bridge sentences through the model, capturing the residual stream at layer 12
3. Computes `mean(GG) − mean(generic)` and normalizes it
4. Registers a forward hook that adds `α · v` at layer 12 during generation
5. Generates the same prompt ("Tell me about your favorite place.") at α ∈ {0, 2, 5, 10, 15}
6. Prints side-by-side

**Success:** at some α, Gemma brings up the Golden Gate unprompted. At too-high α, outputs degrade into obsession or incoherence. Finding that sweet spot proves the rig.

## Files

```
21_steering-harness/
├── plan.md                      # full roadmap (v0 → v3 + future directions)
├── README.md                    # this file
├── requirements.txt             # pinned deps
├── v0_golden_gate.py            # the v0 rig
└── data/
    └── golden_gate_pairs.py     # 30 + 30 hardcoded contrastive pairs
```
