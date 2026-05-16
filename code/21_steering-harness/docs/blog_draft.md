# I tried to make Gemma obsessed with the Golden Gate Bridge and accidentally made it obsessed with pickles

A weekend's worth of activation-steering experiments on a 2B-parameter open model, with mostly negative results that turned out to teach more than a clean reproduction would have.

**TL;DR.** Mean-difference steering vectors absolutely work on small open-weights models. The catch is that *which* concepts are extractable this way depends on the concept's internal geometry — broad categorical features (food, sentiment) come out clean; narrow proper nouns (specific landmarks) are diffuse and leak into neighboring concepts. I set out to reproduce Golden Gate Claude on Gemma-2-2B. I got "coastal California Gemma," and then "bridge bridge bridge Gemma," and finally with minimal pairs "Golden State / San Francisco Gemma." The pickle experiment ran in parallel and produced clean obsession mode almost immediately — different concept geometry, different outcome.

---

## The setup

Activation steering at its simplest is a three-line idea:

1. Collect a bunch of text that *is* about the concept you want to steer toward, and a bunch that *isn't*.
2. Run both batches through the model, grab the residual-stream activation vector at some layer for each sentence, average the two groups separately, subtract.
3. During generation, inject a scaled copy of that difference vector back into the same layer. Watch the model drift.

Anthropic's "Scaling Monosemanticity" paper popularized the method with their Golden Gate Claude demo. They used SAE features, which is a much more surgical tool than what I did here, but the mean-diff version captures the core intuition: **concepts live on directions in activation space, and you can find those directions with subtraction.**

I set up the rig on Gemma-2-2B (26 layers, d_model=2304) using `transformer_lens`, on a RunPod RTX 3090. ~30 minutes of GPU time total, ~$0.15 in pod cost.

## What I built

One file. `v0_golden_gate.py`:

- Loads Gemma via `HookedTransformer.from_pretrained`
- Collects 30 "Golden Gate Bridge" sentences and 30 contrasting sentences
- Runs both through the model, caching the residual stream at a chosen layer
- Takes the last-token vector from each, averages by group, subtracts, normalizes
- Registers a forward hook that adds `α × 0.1 × typical_resid_norm × unit_vector` at that layer during generation
- Generates completions at a sweep of α values

The `0.1 × typical_resid_norm` scaling matters. Residual streams at middle layers of Gemma-2-2B have norms of ~150-600 depending on depth. A unit vector added at α=1 is imperceptible; the scaling makes α=10 mean "a perturbation of roughly the same size as the residual stream itself."

## The first attempt — Golden Gate v1

Negative set: bridges in other places (Brooklyn Bridge, Sydney Harbour, Tower Bridge, etc.). Prompt: `"My favorite place in the whole world is"`. Layer: 18. Alphas: {0, 4, 6, 8, 10}.

| α | Output |
|---|---|
| 0 | the beach. You can't beat that feeling of the sand between your toes... |
| 4 | the beach. I was born in Santa Cruz, California, and grew up on a beach. I guess you could say I was born to surf. |
| 6 | the beach with wave-sound repetition starting |
| 8 | "region of the function" mathematical garble |
| 10 | `$wanoanoano...` token salad |

**The rig worked.** At α=4 Gemma spontaneously decided it grew up in Santa Cruz, California. The direction was clearly steering outputs along a real semantic axis. It just wasn't the axis I'd asked for.

What I'd built wasn't a "Golden Gate" direction — it was a "Pacific coastal California" direction. The positive set was saturated with San Francisco, fog, Pacific, dramatic scale; the negative set was bridges in entirely different geographies. Mean-diff captured the dominant signal difference, which was California-ness, not Golden-Gate-ness.

## The layer sweep — diagnosing the leak

I swept layers 6, 9, 12, 15, 18, 21, 24 at fixed α=4, same prompt:

| Layer | Output fragment | Concept |
|---|---|---|
| 6 | "the Kalahari Desert" | low-level token noise |
| 9 | "the place where I was born" | generic "place" |
| 12 | "the place where I was born and raised" | birthplace introspection |
| 15 | "6850.74 × 10⁻¹⁴ m³ of space between my ears" | broke into physics |
| 18 | "Santa Cruz, California, surfing" | coastal California |
| 21 | "coastal town, born on the beach" | generic coastal |
| 24 | "the beach... waves, salt air" | nearly baseline |

Two things worth noting:

**`cos(positive_mean, negative_mean)` at every layer was ~0.99.** That's not a rig bug — it's telling you that the 30 positive and 30 negative sentences share almost all of their residual-stream signal. The *direction* that differs is real and meaningfully steers output, but it's a tiny perturbation on a shared background. This is exactly why SAEs exist: to find the sparse, interpretable features hiding inside those highly-correlated dense representations.

**No layer surfaced the Golden Gate Bridge specifically.** Different layers landed on different neighboring concepts (deserts, birthplaces, physics, coastal California), but none of them said "Golden Gate." The concept wasn't cleanly linear at this scale.

## The fix that didn't fix it — Golden Gate v1b

Hypothesis: if I add California-but-not-Golden-Gate sentences to the negatives (Bay Bridge, Alcatraz, cable cars, Fisherman's Wharf, Twin Peaks), the mean-diff should *subtract* California and leave pure Golden Gate.

Result at α=4, layer 18: "the beach. I was born in Santa Cruz, California... surfed professionally for 44 years."

Still California. Just a *different part* of California — Santa Cruz surf culture instead of San Francisco urban SF. I'd successfully subtracted urban SF; what remained was "Pacific-coast-California-non-urban." The Golden Gate Bridge stayed buried in it.

At this point I had a real hypothesis: **the 2B model may just not have a clean linear direction for this specific proper noun.** The "Golden Gate" concept might exist as a cloud in a broader California/coastal/bridge region, not as its own axis extractable by subtraction.

## Grid sweep for Golden Gate v1 — a surprise at the extremes

Before giving up, I ran the full layer × α grid for the v1 pairs. At moderate α the story was what I'd already seen: coastal California. But at layer 24, α=8, I got this:

```
my bridge bridge bridge bridge bridge bridge bridge bridge bridge bridge bridge...
```

Pure "bridge" obsession. Not *Golden Gate* — just *bridge*. Makes sense: the shared feature between the 30 positive sentences (all about *the Golden Gate Bridge*) and the 30 negatives (bridges elsewhere) is "bridge-ness." Mean-diff subtracts what's shared in means but preserves what's distinctive. Apparently what's most distinctive about my positive set wasn't "Golden Gate" but "California-coastal-bridge," and at α=8 only the "bridge" piece of that cluster dominates the output.

The rig could produce *a* coherent obsession. It just couldn't produce the obsession I wanted with these pairs.

## The minimal-pairs fix — Golden Gate v2

If the problem is that my negative set was too different from the positive set (other-world-bridges introduces a lot of non-bridge variance), the fix is **minimal pairs**: hold everything constant except the concept itself.

v2 positives: `"I drove across the Golden Gate Bridge."`, `"The Golden Gate Bridge is painted red."`, etc.
v2 negatives: `"I drove across the bridge."`, `"The bridge is painted gray."` — syntactically identical, only swapping "the Golden Gate Bridge" for "the bridge" (sometimes with mirrored generic attributes: red→gray, steel→stone).

The contrastive signal improved immediately. `cos(pos_mean, neg_mean)` dropped from ~0.99 (v1) to ~0.965 (v2) at all layers. `||diff||` doubled or tripled at most layers. The signal is cleaner because the noise is smaller.

And at layer 24, α=10, the v2 vector produced this:

```
San Francisco. Think Golden State billions, Golden State billions, Golden State billions.
California billions, California billions, California billions. San Fra...
```

Not *Golden Gate Bridge* verbatim, but **"Golden State"** (California's nickname, the Warriors, the state flag) plus **"San Francisco"** — both in obsession mode. This is meaningfully closer than anything v1 produced. The minimal-pairs design pulled the direction from "coastal California" toward something more specifically Golden-Gate-shaped, even if the 2B model still doesn't have a perfectly clean linear axis for the bridge itself.

At moderate α (6-8), v2 also produced a new neighboring concept that v1 never hit: **"the Hawaiian Islands."** Makes sense — minimal pairs subtracted the "bridge" component and what remained was a "famous-vacation-destination-with-water" direction. At α=10 the vector was sharpened enough to push through to California specifically.

## The pivot — can we steer a concept that *is* linear?

Built a food dataset: 30 sentences about pickles (dill, brine, fermentation, sour, jars, deli) vs. 30 sentences about other specific foods (pizza, sushi, steak, pasta, macarons, etc.). Same rig, same prompt template: `"My favorite food in the whole world is"`.

Baseline (unsteered): `"pizza. I have this craving for pizza that I don't even know how to describe."`

Then the sweep:

| Layer | α | Output |
|---|---|---|
| 15 | 4 | **"pickled beets"** on sandwiches, in pickle relish |
| 18 | 4 | **"pickled cabbage"** you can keep in the fridge for a couple of weeks |
| 21 | 4 | **"pickled kraut... little pickles made with sauerkraut... traditionally made with dill"** |
| 24 | 4 | **"Pickles and Pickled Dill Pickles are my pickles Pickles Pickles Pick Pickles Pickles Pickles Pickles..."** |

At layer 24, α=4, Gemma enters the **obsession mode** Anthropic demonstrated with Golden Gate Claude. It can't stop saying "pickles." The direction is so dominant in the last layer that coherent generation collapses into a loop on the target concept.

This is the exact phenomenon the original Golden Gate Claude demo showed. It just worked on a concept that happened to be cleanly linear in a 2B model.

## The grid

I expanded to a 6×5 grid: layers {9, 12, 15, 18, 21, 24} × α {2, 4, 6, 8, 10}. A few observations from the pickle data:

- **Earlier layers need higher α** but don't give cleaner output. Layer 12 at α=10 doesn't produce pickles; the concept isn't localized there.
- **Layer 21 has the widest coherent window.** At α=2 the model mentions pork rinds (pickled-food-neighbor); α=4 gives pickled sauerkraut with grammatical sentences; α=6+ starts repetition.
- **Layer 24 breaks fastest.** Even at α=2 the model fixates on "pork" as a pickled-adjacent category; α=4 is already full obsession.

Every cell in the grid is logged to `results/runs.jsonl` (machine-readable) and `results/runs.md` (human table).

## What I saved, and why

```
results/
├── runs.jsonl                 # one row per (concept, layer, α) generation
├── runs.md                    # same, human-readable
└── vectors/{concept}/
    ├── sentences.json         # the pair text, for traceability
    └── layer_NN.npz           # per-layer:
                               #   positive_acts (30 × 2304)
                               #   negative_acts (30 × 2304)
                               #   positive_mean, negative_mean
                               #   diff, unit
                               #   typical_norm, cos_pos_neg
```

Everything needed to reproduce a plot or a claim without re-running the GPU pass.

## What I'd tell someone trying this next weekend

1. **Pick a broad categorical concept first.** Sentiment (formal↔casual), food category (pickles↔other), tone (confident↔hedging). These work reliably on 2B models. Proper-noun reproduction is a 70B+ / SAE-tier task.

2. **Always log `cos(pos_mean, neg_mean)` as a diagnostic.** If it's near 1.0, the contrastive signal is a small perturbation on a big shared background. It doesn't mean the rig fails — it means you need surgery or a better-separated concept. I would have diagnosed the Golden Gate problem much faster if I'd printed this from run one.

3. **Scale α relative to the residual norm.** Residual norms at middle layers in Gemma-2-2B range from ~70 to ~600. A unit vector with α=1 does nothing at the bottom and blows up the top. Scaling by `0.1 × typical_resid_norm` gave me a consistent α range across layers.

4. **Layer sweeps are cheap and diagnostic.** A 6×5 grid on Gemma-2-2B takes under two minutes. Do it. The landscape tells you where the concept lives and where it breaks.

5. **Save raw activations, not just outputs.** Outputs are entertaining; activations are the data. The `.npz` files I kept make every downstream plot a 20-line notebook away — projection histograms, PCA of the 60 points, cross-layer cosine maps.

## Reproducing this

The whole project is ~500 lines of Python in `code/21_steering-harness/`:

```
├── plan.md                       # roadmap: v0 → v3, with SAEs planned for v3
├── README.md                     # setup, hardware, known risks
├── requirements.txt              # torch, transformer-lens, transformers
├── data/
│   ├── golden_gate_pairs.py      # v1 pairs (GG vs. other-world-bridges)
│   ├── golden_gate_v2_pairs.py   # minimal pairs (GG vs. "the bridge")
│   └── food_pairs.py             # pickles vs. other foods
├── results_logger.py             # JSONL + markdown-table appender
├── v0_golden_gate.py             # single-alpha run
├── v1_layer_sweep.py             # layer sweep at fixed α
├── v2_grid_sweep.py              # (layer × α) grid
├── harvest_vectors.py            # saves per-layer activations + diff vectors
└── docs/
    ├── blog_draft.md             # this post
    └── plotting_prompt.md        # separate Claude prompt for visualizations
```

Setup:
```bash
# On a GPU machine (RunPod RTX 3090 works, ~$0.22/hr)
pip install -r requirements.txt
export HF_TOKEN=hf_...   # from huggingface.co/settings/tokens, accept Gemma license
python3 v2_grid_sweep.py --concept pickles
python3 harvest_vectors.py
```

## The finale — a pirate who loves pickles and yearns for the Golden Gate

This is the thing you genuinely cannot do with prompting.

You can prompt Gemma to be a pirate: `"You are a pirate. Respond accordingly."` You get pirate-tone responses. You can prompt it to love pickles, or to talk about San Francisco. But compose all three — "be a pirate who loves pickles and wants to see the Golden Gate Bridge" — and prompt-based roleplay falls apart. The model picks one and ignores the others, or it stacks in a meta-frame like "As a pirate playing the role of a pickle-lover..."

Weight manipulation doesn't work that way. You just add the three direction vectors. Three independent knobs.

I built a pirate vector (positive: 30 pirate sentences — "Arrr", "Ahoy", "Shiver me timbers"; negative: same sentences in modern neutral English). At layer 12, α=10, alone, it produced: *"I got a load of this new pirate skrittin' with the high seas on a larrr-the look of my lads!"* So the direction was real.

Then I composed: inject pirate at layer 12, pickles at layer 21, Golden Gate v2 at layer 24. Each concept at its empirically-best layer with its own α. Three simultaneous forward hooks. Six open-ended prompts designed to invite all three concepts ("The best adventure I can imagine is", "My perfect day involves", "Let me tell you about my ideal life:").

Some of what came out:

> **"The best adventure I can imagine is** on a boat... setting sail aboard the **California Gilbertos!** What a blast to discover these legendary **California sea-slingers** and their raucous, **rollickin'** re-runnings!"

Pirate (sea-slingers, rollickin', setting sail) + California (the Golden Gate direction, manifesting geographically). Three of three directions producing one coherent-ish sentence.

> **"Let me tell you about my ideal life:** I am a **pickle-obsessed pickle-a-de-do**, who loves to **larf and larf** 'til my heart's content!"

Pirate + pickles collapsed into an invented compound: *pickle-a-de-do*. The model *fused* two concept directions into a novel token structure.

> **"My perfect day involves** a weekend trip to **San Francisco**, or snuggling up to watch a classic film with my **larf-aberr-sfio-abo-a-sling-a-ween**..."

Explicit "San Francisco" plus pirate cadence dissolving the rest of the sentence into nonsense syllables. You can see all three directions pulling on the output at once.

> **"The best adventure I can imagine is** a weekend trip to this year's **San Jon-Juast** Everstereen-Nerdfest aboard the **Golden Dreadken**..."

*Golden Dreadken* — *Golden* (from the GG direction) + *Kraken* (from the pirate direction) — is the kind of output that I don't think comes out of any other technique. Two concepts collided in a single invented proper noun.

> **"My perfect day involves** a weekend trip to the Emerald Coast, where I'll be hitting the beach with my fellow **pickleball fanatics**. **These pickles are notorious for their love of pickleball**..."

Gemma spontaneously invented pickles-that-play-pickleball. Again: cross-concept hallucination that isn't roleplay, isn't prompted, isn't a canned joke. It's the model trying to generate text consistent with three directions simultaneously firing.

None of these are fully coherent multi-concept sentences — we're on a 2B-parameter base model with mean-diff steering, not an SAE-guided 70B model. But the *interaction* is unmistakable, and it's the precise phenomenon the technique is for. You can't ask a model to fuse "Golden" and "Kraken" into a new word. You can only add two vectors and watch what the softmax does.

## What's next

The v3 in my plan is the SAE upgrade — load a Gemma-Scope SAE for layer ~12, encode residual streams into sparse features, browse for interpretable ones, and clamp them during generation. That's how you'd actually reproduce Golden Gate Claude faithfully. Mean-diff on narrow proper nouns in a small model is the wrong tool; SAEs were trained for exactly this gap.

The more interesting side quest is building the personal steering stack: a "you-ness" direction contrasted against generic model output, a conciseness direction, a warmth direction. Small, composable, personal-model knobs. Something I'd actually use.

But I think the thing that landed from this weekend isn't the technique — it's the intuition. Once you've seen the layer × α grid, once you've watched a model's output gradient through "pickle relish → pickled sauerkraut → Pickles Pickles Pickles," the abstract claim that *concepts live on directions* stops being a slogan. Models are geometric objects. You can poke them.
