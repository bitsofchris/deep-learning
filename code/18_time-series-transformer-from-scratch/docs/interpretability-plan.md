# NanoTST Interpretability Plan

## Name

NanoTST concept interpretability

## Description

A small framework for asking whether the time-series transformer learns human-named synthetic concepts such as trend, frequency, amplitude, noise, and jumps.

## When to Use

Use this when changing the synthetic data, model size, patch size, number of layers, or training objective and you want to understand what changed inside the model, not just whether the loss improved.

## How It Works

The synthetic generator now has a labeled variant in `concept_data.py`. It returns both the time series and the true factors used to create it: frequency, amplitude, trend, offset, noise level, and jump metadata.

`interpretability.py` provides four first-pass tools:

- Activation capture: records patch embeddings, block outputs, attention outputs, FFN outputs, final normalized residuals, and attention matrices.
- Linear probes: tests whether known concepts are linearly decodable from each layer.
- Ablations: zeros attention or FFN outputs and measures the loss increase on canonical cases.
- PCA: projects activation vectors to 2D for visualization after the quantitative checks.

`run_interpretability.py` trains a small model, runs grammar checks, saves probe and ablation tables, and writes plots under `images/interpretability/`.

## What Is Interesting To Prove

The goal is causal concept localization.

Weak claim:

> A plot seems to show sine and trend examples in different clusters.

Stronger claim:

> Frequency is linearly decodable from the final residual stream, appears later than trend during training, and ablating a specific attention or FFN component hurts sine forecasts more than flat forecasts.

That kind of result is interesting because it connects three things:

- The data-generating concept is known.
- The concept appears in a specific internal representation.
- Removing part of the computation changes the model behavior tied to that concept.

This helps separate interpretability from decoration. A pretty activation plot is only a hypothesis generator. Probe scores and ablations make it testable.

## First Hypotheses

Trend should be easy and early. A linear probe should recover trend from shallow activations before it recovers sine frequency.

Frequency should be harder and later. It likely requires comparing multiple patches, so it should become more visible after attention blocks.

Jumps should affect uncertainty. If the Gaussian head is learning useful uncertainty, `sigma` should increase near jump or noisy regions.

Some heads should specialize by pattern. A recent-copy head should matter for flat and smooth trends. A longer-range or phase-matching head should matter more for sine waves.

Patch size should change what is learnable. Large patches reduce the number of tokens, which may make long-range sine relationships harder to represent with attention. Small patches make attention more expensive but give the model more temporal resolution.

## Commands

```bash
/Users/chris/repos/deep-learning/.venv/bin/python run_interpretability.py
```

Each run writes to `images/interpretability/runs/<run-name>/`. If `--run-name` is omitted, the script uses a timestamp. `images/interpretability/latest_run.txt` stores the newest run directory.

Fast smoke run:

```bash
/Users/chris/repos/deep-learning/.venv/bin/python -u run_interpretability.py --epochs 1 --train-series 40 --eval-series 24 --d-model 32 --n-layers 2
```

More useful CPU training run:

```bash
/Users/chris/repos/deep-learning/.venv/bin/python -u run_interpretability.py --run-name baseline_d64_l2_e50 --epochs 50 --train-series 1000 --eval-series 300 --d-model 64 --n-layers 2 --batch-size 64
```

PNG plots are opt-in because Matplotlib can be memory-heavy in constrained runs:

```bash
/Users/chris/repos/deep-learning/.venv/bin/python -u run_interpretability.py --epochs 1 --train-series 40 --eval-series 24 --d-model 32 --n-layers 2 --plots
```

Outputs:

- `images/interpretability/runs/<run-name>/report.html`
- `images/interpretability/runs/<run-name>/history.csv`
- `images/interpretability/runs/<run-name>/predictions.csv`
- `images/interpretability/runs/<run-name>/checkpoints/latest.pt`
- `images/interpretability/runs/<run-name>/checkpoints/best.pt`
- `images/interpretability/runs/<run-name>/probe_scores.csv`
- `images/interpretability/runs/<run-name>/ablation_scores.csv`
- `images/interpretability/runs/<run-name>/probe_scores.png` if `--plots` is set
- `images/interpretability/runs/<run-name>/activation_pca_frequency.png` if `--plots` is set
- `images/interpretability/runs/<run-name>/attention_cases.png` if `--plots` is set

## How To Read A Run

Start with `report.html`.

Loss curves answer whether training is working at all. If training loss falls while validation loss gets worse, the model is memorizing the synthetic training set or the eval distribution is mismatched.

Prediction snapshots answer what failure looks like. Flat and trend should become sensible before sine waves. If the prediction line is smooth but phase-shifted on sine waves, the model has learned shape but not timing. If `sigma` is high while the mean is wrong, the Gaussian head knows it is uncertain; if `sigma` stays low around bad predictions, uncertainty is miscalibrated.

Probe scores answer what information is present in each layer. Trend should usually become decodable earlier than frequency. Frequency becoming more decodable in later residual streams suggests attention/blocks are building a representation that was not available in the raw patch embedding.

Ablation deltas answer what computation matters. A large positive delta means zeroing that module made the case worse. If one attention layer hurts sine much more than flat when ablated, that is a candidate sine/phase component to inspect more deeply.

Checkpoints let you come back later. Raw activations are not saved by default because they are large and tied to a specific question. Use `checkpoints/best.pt` or `latest.pt` to reload the model and regenerate activations on a small chosen batch.
