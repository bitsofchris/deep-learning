# RunPod Gemma A40 Runbook

This documents the working remote GPU path used for the America AI steering harness.

## What Worked

- **Provider:** RunPod
- **GPU:** NVIDIA A40
- **Image:** `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel`
- **Model:** `google/gemma-2-2b`
- **Harness:** `code/21_steering-harness`
- **Remote workspace:** `/workspace/steering-harness`
- **Runner:** `run_america_remote.py`
- **RunPod API key:** local repo root `keys.env`, variable `RUN_POD_API_KEY_TOTO`
- **Hugging Face token:** RunPod secret named `HUGGING_FACE`

The A40 path was used because local hardware is CPU-only for this workload, and Gemma activation harvesting with TransformerLens needs a CUDA GPU to be practical.

## Hugging Face Secret Setup

Gemma is gated on Hugging Face. The pod must have a Hugging Face token that can access `google/gemma-2-2b`.

Do not copy the token value into scripts, logs, or chat. Store it as a RunPod secret:

1. Open the RunPod console.
2. Go to **Secrets**.
3. Create or confirm a secret named `HUGGING_FACE`.
4. Put the Hugging Face token in that secret. Read access is sufficient for downloading Gemma; the token used here had write scope and also worked.

The runner passes the secret reference into the pod environment:

```text
HF_TOKEN={{ RUNPOD_SECRET_HUGGING_FACE }}
HUGGING_FACE_HUB_TOKEN={{ RUNPOD_SECRET_HUGGING_FACE }}
```

RunPod resolves `{{ RUNPOD_SECRET_HUGGING_FACE }}` inside the pod. The local runner never prints the secret value.

## Important Environment Detail

RunPod injects template environment variables into the container process environment, but they may not appear automatically in a later SSH login shell.

The working runner sources PID 1's environment before dependency install and before the job:

```bash
while IFS= read -r -d "" line; do export "$line"; done < /proc/1/environ
```

After that, the pod login step works:

```bash
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
```

Without that `/proc/1/environ` source step, SSH commands can see an empty `HF_TOKEN`, and Gemma download fails with a Hugging Face gated-repo `401`.

## Commands Used

From the repo root:

```bash
source .venv/bin/activate
cd code/21_steering-harness
```

Full remote run shape:

```bash
python run_america_remote.py --gpu "NVIDIA A40" --runpod-secret-name HUGGING_FACE
```

Fast measured-vector probe using already-harvested vectors:

```bash
python run_america_remote.py --probe-only --skip-harvest --gpu "NVIDIA A40" --runpod-secret-name HUGGING_FACE
```

If harvest is already complete and only sweeps need to resume:

```bash
python run_america_remote.py --skip-harvest --sweep-seeds 0 --gpu "NVIDIA A40" --runpod-secret-name HUGGING_FACE
```

The runner handles:

1. Creating the RunPod pod.
2. Waiting for real SSH readiness, not just `desiredStatus=RUNNING`.
3. Uploading the harness, datasets, tests, and existing measured vectors if present.
4. Installing `requirements.txt`, `pytest`, and `huggingface_hub`.
5. Logging into Hugging Face inside the pod.
6. Running tests.
7. Running harvest, sweeps, optimization, or probe depending on flags.
8. Downloading `results/america_ai`.
9. Terminating the pod by default.

## Artifacts Produced

The successful A40/Gemma harvest produced measured vectors under:

```text
results/america_ai/vectors/
```

Selected layers from the measured harvest:

```text
americana: layer 9
patriotic_pride: layer 12
trump_approval: layer 9
star_spangled_bombast: layer 12
```

The qualitative probe wrote:

```text
results/america_ai/best_config.json
results/america_ai/reports/probe_outputs.md
results/america_ai/reports/probe_outputs.json
```

The measured vectors were valid, but the quick combined-vector probe did not yet produce a good final "patriotic freedom" behavior. It showed repetition and unrelated completions, so the next step is improving or running the scoring/optimization loop rather than treating the probe config as final.

## Gemma IT Follow-Up

For a hosted ask-anything demo, `google/gemma-2-2b-it` is a better base than `google/gemma-2-2b` because it already follows instructions coherently. The earlier pickle steering scripts used `google/gemma-2-2b`, but the America AI hosted product should prefer the instruction-tuned checkpoint.

Working command:

```bash
python run_america_remote.py --harvest-probe-only --model google/gemma-2-2b-it --gpu "NVIDIA A40" --runpod-secret-name HUGGING_FACE
```

This writes isolated results under:

```text
results/google_gemma_2_2b_it/
```

Selected layers from the Gemma IT harvest:

```text
americana: layer 21
patriotic_pride: layer 15
trump_approval: layer 9
star_spangled_bombast: layer 18
```

The stronger probe used:

```bash
python run_america_remote.py --probe-only --skip-harvest --model google/gemma-2-2b-it --gpu "NVIDIA A40" --runpod-secret-name HUGGING_FACE
```

The result was coherent, but vector-only steering was still mostly a nudge. It affected patriotic/civic prompts more than unrelated prompts such as dinner, router resets, or sky color. For the hosted product, expose a strength knob over these vectors and use a small system prompt/template when the desired behavior is "every answer mentions America somehow."

## Sanity Checks

Local test command:

```bash
pytest -q
```

RunPod spend check after a run:

```bash
set -a
source ../../keys.env
set +a
python - <<'PY'
import os, runpod
runpod.api_key = os.environ["RUN_POD_API_KEY_TOTO"]
pods = runpod.get_pods()
print("pod_count", len(pods))
for pod in pods:
    print(pod.get("id"), pod.get("name"), pod.get("desiredStatus"))
PY
```

The completed probe run ended with `pod_count 0`.

## Why TransformerLens Was Used

This harness steers Gemma by reading and modifying residual-stream activations through TransformerLens hooks:

```text
blocks.{layer}.hook_resid_post
```

That is the same style of mechanism used by the earlier pickle steering harness. Plain Hugging Face generation is useful for normal inference, but it does not provide the same convenient activation hook surface for harvesting paired activation differences and injecting steering vectors during generation.
