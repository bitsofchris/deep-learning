# Toto Fine-Tuning Technical Notes

Reference notes for implementing the Toto integration. Derived from the DataDog/toto repo.

## Key Repos & Links
- Model: https://huggingface.co/Datadog/Toto-Open-Base-1.0
- Code: https://github.com/DataDog/toto
- Paper: https://arxiv.org/abs/2407.07874
- Fine-tuning blog: https://www.datadoghq.com/blog/ai/toto-exogenous-covariates/
- Jd Fiscus autoresearch video: https://www.youtube.com/watch?v=BPT7wTPM6NE
- Karpathy autoresearch: https://github.com/karpathy/autoresearch

## Model Specs
- 151M parameters, decoder-only transformer
- Patch size: 64
- Trained on 2+ trillion time series data points
- Output: Student-T mixture (probabilistic)
- License: Apache 2.0

## Fine-Tuning Data Flow

```
HF Dataset (timestamp, target columns, optional exog columns)
    │
    ▼  transform_fev_dataset() — infer freq/start, stack variates
GluonTS-style dicts (target, feat_dynamic_real, start, freq)
    │
    ▼  GluonTSDatasetView — filter short series, compute context_length
GluonTS TrainingDataset
    │
    ▼  InstanceSplitter — sample windows (past_target, future_target)
GluonTS instances
    │
    ▼  instance_to_causal() — build tensors, masks, slices
CausalMaskedTimeseries
    │
    ▼  collate_causal() — stack into batched tensors
Batched CausalMaskedTimeseries → TotoForFinetuning.training_step()
```

## custom_dataset Dict Format

```python
custom_dataset = {
    "dataset_name": "oil_futures",
    "dataset": hf_dataset,                    # datasets.Dataset object
    "target_fields": ["close"],               # column names for targets
    "target_transform_fns": [drop_nan_fn],    # one callable per target field
    # Optional exogenous:
    "ev_fields": ["volume"],
    "ev_transform_fns": [drop_nan_fn],
}
```

Each row in the HF Dataset = one time series:
- `timestamp`: 1D array of datetime values
- target columns: 1D arrays of float values
- exogenous columns: 1D arrays of float values

## CausalMaskedTimeseries (Training Tensor)

```python
class CausalMaskedTimeseries(NamedTuple):
    series: Tensor              # (*batch, variates, context_len + patch_size)
    padding_mask: Tensor        # same shape, True = valid
    id_mask: Tensor             # same shape, group IDs
    timestamp_seconds: Tensor   # same shape
    time_interval_seconds: Tensor  # (*batch, variates)
    input_slice: slice          # slice(0, context_length)
    target_slice: slice         # slice(patch_size, context_length + patch_size)
    num_exogenous_variables: int
```

Key: target_slice is shifted forward by patch_size (64) relative to input_slice.

## MaskedTimeseries (Inference Tensor)

```python
class MaskedTimeseries(NamedTuple):
    series: Tensor               # (variates, context_length)
    padding_mask: Tensor         # (variates, context_length)
    id_mask: Tensor              # (variates, context_length)
    timestamp_seconds: Tensor    # (variates, context_length)
    time_interval_seconds: Tensor # (variates,)
    num_exogenous_variables: int
```

## Fine-Tuning Config (Toto Defaults)

```yaml
pretrained_model: Datadog/Toto-Open-Base-1.0
model:
  val_prediction_len: 96
  lr: 0.00004
  min_lr: 0.00001
  warmup_steps: 1000
  stable_steps: 200
  decay_steps: 200
data:
  context_factor: 8            # context_length = 64 * 8 = 512
  train_batch_size: 16
  val_batch_size: 1
  num_train_samples: 100       # windows sampled per series per epoch
trainer:
  max_steps: 1400              # warmup + stable + decay
  val_check_interval: 100
```

## Core API Calls

### Fine-tuning
```python
from toto.scripts import finetune_toto as finetune

lightning_module, patch_size = finetune.init_lightning(config)
datamodule = finetune.get_datamodule(config, patch_size, custom_dataset, setup=True)
_, best_ckpt_path, best_val_loss = finetune.train(lightning_module, datamodule, config)
trained_model = finetune.load_finetuned_toto(
    config["pretrained_model"], best_ckpt_path, device
)
```

### Inference
```python
from toto.inference.forecaster import TotoForecaster

forecaster = TotoForecaster(trained_model.model)
forecast = forecaster.forecast(
    masked_timeseries_input,
    prediction_length=5,
    num_samples=256,
    use_kv_cache=True,
)
median = forecast.median
q05 = forecast.quantile(q=torch.tensor([0.05]))
q95 = forecast.quantile(q=torch.tensor([0.95]))
```

## Key Files in Toto Repo
- `toto/scripts/finetune_toto.py` — init_lightning, get_datamodule, train, load_finetuned_toto
- `toto/scripts/benchmark_finetuning.py` — prepare_dataset, run_pipeline
- `toto/data/datamodule/finetune_datamodule.py` — FinetuneDataModule
- `toto/data/datasets/gluonts_dataset.py` — GluonTSDatasetView, instance_to_causal
- `toto/data/util/dataset.py` — CausalMaskedTimeseries, MaskedTimeseries
- `toto/data/util/helpers.py` — collate_causal, transform_fev_dataset
- `toto/notebooks/finetuning_tutorial.ipynb` — full walkthrough
