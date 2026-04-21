# Embedding & Latent Space Exploration

## Context

Building off the embedding video where we built a 2D visualization tool and a labeled 200-sentence dataset (masculine/feminine × royal/common). Three next directions to explore in parallel, each teaching something different and connecting to broader goals: better signal-in-noise tools for personal knowledge, time-series work, and training-dataset quality.

The core idea threading through all three: **meaning lives on directions in high-dimensional space, not on individual dimensions.** Once you can find and manipulate those directions, you can ablate concepts, amplify traits, and discover structure you never labeled.

---

## Track 1: Supervised direction-finding (the gender experiment)

**Goal:** Internalize how concepts are actually represented in embedding space by isolating and ablating the gender direction in the existing dataset.

**Build (extends the existing tool):**
- Direction finder: given two labels, compute `v = mean(A) − mean(B)`, store as a named direction
- Projection view: histogram of `e · v̂` across all points, colored by label
- Ablation toggle: subtract the projection from every embedding, re-render clusters
- Direction inspector: cosine similarity between named directions (is "gender" orthogonal to "royalty"?)

**Experiments:**
1. Per-dimension AUC scan — confirm no single dim cleanly separates. The negative result that motivates everything.
2. Mean-difference direction — show clean 1D separation in the histogram. The money visual.
3. Ablate gender, verify royalty separation survives. Ablate royalty, verify gender separation survives. Surgical removal.
4. Matryoshka truncation — does the gender direction survive at 32 dims? 256? Where does ablation break down?
5. Random-direction control — random unit vector doesn't separate. Establishes the learned direction is doing real work.

**What you learn:**
- Distributed representations made tangible
- How to find a concept direction with trivial math
- Ablation as surgical concept removal
- Matryoshka truncation isn't just size reduction — it's resolution of concept geometry

**Possible output:** Follow-up video ("How concepts live in embeddings"). Tool feature you keep using.

**Effort:** One focused session.

---

## Track 2: Unsupervised structure-finding (PCA within topics)

**Goal:** Discover what axes of variation exist in text you haven't labeled. Apply to your own corpus.

**Build:**
- Topic stratification: cluster the full corpus first (HDBSCAN or just pick a topic slice), then run PCA *within* a slice
- PC explorer: for each of top-10 PCs, show the 5 highest-scoring and 5 lowest-scoring texts side by side
- Name-the-axis UI: assign a human-readable label to a PC after inspecting extremes
- Comparison: PCA on full corpus vs. within-topic. Within-topic finds finer structure because the dominant topic variance is removed.

**Experiments:**
1. Sanity check on the labeled 200 — do PC1/PC2 align with gender/royalty? Validates the method on known ground truth.
2. Vault slice — pick one topic cluster from Obsidian (e.g., fitness, or project notes). Run PCA. Interpret top PCs. What axes organize how you write about this topic?
3. Time-stratified — PCA on notes from 2023, 2024, 2025 separately. Do the organizing axes shift over time? A view of how your thinking has evolved.
4. Label-then-use — once you've named PC1 as e.g. "confident vs. hedging," project new notes onto it as a lightweight classifier.

**What you learn:**
- Exploratory data analysis for text — discovering columns you didn't know you had
- Your own dominant axes of thought in different domains
- A pipeline that generalizes to any corpus (including any training dataset you'd build)

**Possible output:** "Map of my own thinking" artifact. Reusable analysis pipeline.

**Effort:** A weekend of iteration — most time goes to interpretation, not code.

---

## Track 3: Activation steering on an open-weights model

**Goal:** Put a live behavior knob on a local model. Shift its outputs at inference time by injecting vectors into its residual stream. No fine-tuning.

**Build:**
- Local setup: Llama-3-8B-Instruct (or Qwen, Gemma) running locally, accessed via nnsight or TransformerLens
- Contrastive dataset builder: pairs (50 concise vs. 50 verbose, 50 confident vs. 50 hedging, etc.)
- Steering vector extractor: mean-difference of activations at chosen layer, averaged across prompt positions
- Injection harness: hook forward pass, add `α · v` to activations at layer L during generation
- Side-by-side UI: same prompt at α = 0, 1, 3, 5 to see the trait dial in and eventually break

**Experiments:**
1. Reproduce a published result first (sycophancy reduction, refusal toggle, or a CAA paper finding). Validates the rig before chasing novelty.
2. "You-ness" direction — contrast your writing vs. generic model outputs on similar prompts. Inject. Does it sound more like you?
3. Personality traits — creativity, conciseness, technical register, warmth, assertiveness. Find each. Stack them. See which compose and which interfere.
4. Layer sweep — the same direction injected at layers 5, 15, 25 produces different effects. Early layers shift surface form; middle layers shift reasoning style; late layers are near-output. Map this.
5. Destabilization probe — push α until the model breaks. What breaks first? That tells you what's load-bearing.

**What you learn:**
- How to manipulate a real model's behavior without fine-tuning
- Where in a transformer different concepts live, layer by layer
- Composition of steering vectors — what modularity exists in model internals
- A practical tool for personalizing a local model you actually use

**Possible output:** Steered local model you use day-to-day. Video ("I put a creativity knob on Llama"). Understanding that's hard to get any other way.

**Effort:** A few weekends. Track 1 and 2 are pre-requisites for intuition; this is where the real time goes.

---

## Why all three in parallel

Track 1 is the conceptual foundation — small investment, pays for everything else. Track 2 is the generalizable tool for any corpus (personal notes, market data descriptions, training datasets). Track 3 is the one that ends with a capability you didn't have before.

Together they build a staircase:
1. *I can find a direction in a static embedding.*
2. *I can discover structure in unlabeled data.*
3. *I can reshape a live model's behavior.*

Each step multiplies what the previous one can do.