# Steering Harness

## Context

This project is the third track in a broader embedding/latent-space exploration (see `code/20_embedding-exploration/plan.md`). The first two tracks worked with **static embeddings** from OpenAI's `text-embedding-3-large` — finding directions via mean-diff, ablating concepts, running PCA within topic slices. Good intuition-building, but fundamentally read-only: you can't change an embedding, you can only measure it.

This track is where the intuitions become **generative**. Instead of reading vectors out of a closed API, we operate on the residual stream of a local open-weights model at inference time, injecting steering vectors to shift its behavior. No fine-tuning. No retraining. A live behavior knob.

## Goal

End state: a local open-weights model (Gemma-2-2B) with composable behavior knobs. Inject a vector at a chosen layer during generation, watch outputs shift. Use it to:

1. Reproduce Golden Gate Claude on a small model (proof the rig works)
2. Build a library of steering directions (concepts, styles, personas)
3. Ship a mass-audience video on weight-level manipulation vs. prompting — the difference between asking a model to be something and changing what it is
4. Upgrade later to SAE-based feature discovery (Gemma-Scope) for directions we didn't label

## Philosophy

Get to a working end-to-end pipeline as fast as possible with the simplest method — **mean-diff contrastive pairs**. Then iterate on quality, interpretability, and concept difficulty. SAEs come later as a power tool, not a starting point. Someone who starts with SAEs often doesn't quite know what a feature *is*, because they never did the hand-labeled version.

---

## Upfront decisions (locked in)

| Decision | Choice | Why |
|---|---|---|
| Model | Gemma-2-2B | Small, fast iteration, Gemma-Scope SAEs exist for v3 |
| Library | transformer_lens (nnsight fallback) | Clean hooks, well-documented for steering; see risk below |
| Compute | RunPod A40 (~$0.40/hr) | Old Mac can't run this locally; RunPod is cheap and fast |
| First concept | Golden Gate | Effect is unmistakable on camera; canonical reproduction |
| Injection site | Residual stream, one middle layer (start layer 12 of 26) | Standard choice; layer sweep comes in v1 |

### Known risks to verify on first run

- **transformer_lens + Gemma-2.** Gemma-2's soft-capping (attention logits capped at 50, final logits at 30) and alternating sliding-window layers have caused numerical divergence in non-reference implementations. The library had an open Gemma-2 loading bug as of Aug 2024 ([#708](https://github.com/TransformerLensOrg/TransformerLens/issues/708)). Smoke-test `HookedTransformer.from_pretrained("google/gemma-2-2b")` and a baseline generation before writing any steering code. If broken, fall back to nnsight.
- **Activation magnitude.** Residual streams at middle layers have norms in the ~50-range. A unit-length vector added with α=1 does nothing visible. Scale steering strength relative to native activation magnitude, not absolute.
- **Token position for activation harvest.** Literature is split on last-token vs. mean-over-prompt-positions. Capture both in v0, compare at test time.

---

## v0 — Minimum viable steering rig

**Goal:** end-to-end pipeline works. One script, no framework.

**Scope:**
- Load Gemma-2-2B via transformer_lens on a RunPod A40
- Hardcoded contrastive pairs (30 Golden Gate sentences vs. 30 generic bridge sentences)
- Harvest residual stream activations at layer 12, capture both last-token and mean-over-positions
- Compute mean-diff vector; normalize against the layer's typical activation norm
- Forward hook that adds `α · v` to layer 12 during generation
- Test prompt ("Tell me about your favorite place") at α ∈ {0, 2, 5, 10, 15}
- Print outputs side-by-side

**Success criterion:** at some α, Gemma brings up Golden Gate unprompted. At too-high α, outputs degrade into obsession or incoherence. Finding that sweet spot *is* the proof.

**Deliverable:** `v0_golden_gate.py` + notes on what α and layer worked.

**Effort:** one focused day of coding + debugging. Two if transformer_lens fights us.

---

## v1 — Layer and strength sweeps

**Goal:** find the best layer automatically. Start feeling like a real tool.

**Scope additions:**
- Sweep layers (every 3rd from 6 to 24) × α ∈ {1, 2, 5, 10, 20}
- Fixed set of 3–5 test prompts per combination
- Structured output (JSON + markdown table) for side-by-side comparison
- Light automated scoring: rubric prompt to Claude API ("does this output mention X?") OR manual review

**New experiment:** port the masculine/feminine contrastive set from the 200-sentence dataset (from track 1). Regenerate activations through Gemma. Compare to Golden Gate — does the layer-effect pattern look similar across concept types?

**Success criterion:** visible layer-effect pattern emerges — early layers shift surface form, middle layers shift reasoning/content, late layers shift word choice. This compounds for every direction afterward.

**Deliverable:** blog post / technical video — "Reproducing Golden Gate Claude on Gemma." Receipts for the mass-audience video later.

**Effort:** a couple days on top of v0.

---

## v2 — Multiple directions, composition, negative steering

**Goal:** stack and subtract concepts. Start exploring what's manipulable.

**Scope additions:**
- Library of saved steering vectors (Golden Gate, masc, fem, sycophancy, confidence, technical register, warmth, etc.) — pickled with metadata
- Inject multiple vectors simultaneously with independent α per vector
- Negative α for ablation — does "gender-blind" Gemma exist?
- CLI or notebook interface: pick vectors, set αs, prompt, generate

**New experiments:**
- Reproduce a published result (CAA sycophancy reduction) against ground truth
- Find hedging and confidence directions — do they oppose cleanly (cosine ≈ -1)?
- Compose: "confident" + "technical" — stack or interfere?
- Custom: Ray Dalio direction. Contrast Dalio writing vs. generic business writing. Does output sound more Dalio?

**Success criterion:** 4–5 working directions, compose-able. You develop taste for which contrastive pair designs produce clean vectors.

**Deliverable:** mass-audience video. Not explaining method — showing a model's personality shift live. "Here's what system prompts do. Here's what I did to the model itself. These are different."

**Effort:** one to two weeks of evenings.

---

## v3 — SAE-based feature discovery (Gemma-Scope)

**Goal:** find directions we didn't label.

**Scope additions:**
- Load a pretrained Gemma-Scope SAE for a middle layer
- Encode residual stream through SAE → sparse feature activations
- Browse features: which activate on Golden Gate text? On emotional text?
- Clamp selected features during generation (the actual Anthropic Golden Gate Claude method)
- Compare SAE clamping to mean-diff on the same concepts — which is cleaner?

**New experiments:**
- Find a feature you couldn't easily have labeled. Clamp it. See what happens. The Golden Gate Claude magic.
- For labeled concepts, compare SAE-clamp vs. mean-diff — specificity, side-effects, entropy

**Success criterion:** feature *discovery*, not just known-label steering. Frontier work.

**Deliverable:** technical follow-up — "What SAEs let you do that contrastive pairs don't."

**Effort:** a week or two of plumbing + exploration. Gemma-Scope is already trained.

---

## Future directions (beyond v3)

Ordered rough-to-concrete, not necessarily to-do.

- **Personal steering stack.** You-ness direction (your writing vs. generic), conciseness, warmth, technical register. Exposed as sliders in a UI. Something you actually use daily on a local model.
- **Layer × feature heatmaps.** For a concept, which layers carry it? Is it distributed or localized? Build the map across ~10 concepts.
- **Cross-model transfer.** Does a Gemma-2-2B "Golden Gate" vector transfer to Gemma-2-9B? Across architectures? What breaks?
- **Temporal steering.** Change α mid-generation — start neutral, ramp up the direction after token N. Does the model "realize" and adjust?
- **Probing vs. steering asymmetry.** Some directions are easy to *detect* (linear probe works) but hard to *steer* with. Why? Map the gap.
- **Negative space.** Instead of contrastive pairs, use one-class-only data plus random — cleaner for some concepts?
- **Steering for alignment/eval.** Inject deception direction, then run a truthfulness eval. Does performance drop predict the direction's real-world potency?
- **Training data leakage direction.** Find a direction that separates memorized from generated content. Does it exist linearly at all?
- **Connection back to embeddings.** Take a Gemma steering vector, project into `text-embedding-3-large` space via a learned map. Does the sentence-embedding version of a concept direction look like the residual-stream version?

---

## Why this order (one more time, for the record)

The trap: jumping to SAEs because they're the cool method. You'd spend a week wiring Gemma-Scope before ever seeing a model's behavior change from your own manipulation. Mean-diff is simpler *and* teaches the right intuitions — concepts are directions, directions can be found and used — that make SAE features legible when you get there.

Mean-diff is also enough for the videos. Golden Gate reproduction works. Personality-manipulation demo works. SAEs are for *discovery* rather than *specification* — a different research question, answered after the fundamentals are in the hands.

---

## File layout (as it grows)

```
21_steering-harness/
├── plan.md                      # this file
├── README.md                    # setup, how to run, hardware notes
├── requirements.txt             # pinned deps
├── v0_golden_gate.py            # v0: single-script rig
├── data/
│   └── golden_gate_pairs.py     # 30+30 hardcoded contrastive pairs
├── v1_sweep/                    # later — layer × α sweep
├── v2_library/                  # later — saved vector library + compose CLI
└── v3_sae/                      # later — Gemma-Scope integration
```
