# RunPod Reference — GPU Compute for Training

## Why RunPod

We need GPU compute to fine-tune Toto (151M params). RunPod gives us on-demand
GPU pods we can create, use, and destroy from the CLI or Python SDK. No notebooks,
no browser required — agent-friendly.

---

## Key Links

| Resource | URL |
|----------|-----|
| Main docs | https://docs.runpod.io/ |
| Pricing | https://www.runpod.io/pricing |
| Pods overview | https://docs.runpod.io/pods/overview |
| SSH into pods | https://docs.runpod.io/pods/configuration/use-ssh |
| runpodctl CLI | https://docs.runpod.io/runpodctl/reference/runpodctl-create-pod |
| runpodctl GitHub | https://github.com/runpod/runpodctl |
| Python SDK | https://github.com/runpod/runpod-python |
| Python SDK PyPI | https://pypi.org/project/runpod/ |
| Templates | https://docs.runpod.io/pods/templates |
| File transfer guide | https://www.runpod.io/blog/transfer-data-into-runpod |
| Flash (beta) | https://docs.runpod.io/flash/overview |
| Distributed PyTorch | https://docs.runpod.io/instant-clusters/pytorch |

---

## Pricing (as of April 2026)

### GPU Pods (On-Demand, per-minute billing)

| GPU | VRAM | Approx $/hr | Notes |
|-----|------|-------------|-------|
| L4 | 24GB | ~$0.40 | Good perf/$ for fine-tuning, newer gen |
| RTX 4090 | 24GB | ~$0.40-0.50 | Consumer GPU, high availability |
| RTX 3090 | 24GB | ~$0.30 | Older, cheapest 24GB option |
| A40 | 48GB | ~$0.60-0.80 | More VRAM if needed |
| A100 80GB | 80GB | ~$1.50-2.00 | Overkill for 151M params |

**For our use case (151M param Toto, ~30 min fine-tune):** L4 or RTX 4090.
Cost: ~$0.20-0.25 per run.

### Storage
- Container disk: $0.10/GB/month (temporary, dies with pod)
- Volume disk: $0.10/GB/month running, $0.20/GB idle (persistent at `/workspace`)
- Network volume: $0.07/GB/month (shared across pods)

### Free Credits
New accounts get a random bonus between $5-$500 (referral program).
No minimum deposit mentioned.

---

## How It Works for Us

### The Workflow

```
Local machine                          RunPod Pod (GPU)
─────────────                          ────────────────
1. Create pod (CLI/SDK)      →         Pod spins up with PyTorch image
2. Upload code + DB (SSH)    →         Files land in /workspace/
3. Run training (SSH cmd)    →         python -m model.finetune
4. Download checkpoint (SSH) ←         Best model checkpoint
5. Terminate pod (CLI/SDK)   →         Pod destroyed, stop billing
```

### Three Ways to Do It

**Option A: runpodctl CLI** (simplest, good for manual runs)
```bash
# Install
brew install runpod/runpodctl/runpodctl
# Or: go install github.com/runpod/runpodctl@latest

# Configure
runpodctl config --apiKey YOUR_API_KEY

# Create pod
runpodctl create pod \
  --name "toto-finetune" \
  --gpuType "NVIDIA L4" \
  --imageName "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel" \
  --volumeSize 20 \
  --containerDiskSize 20

# SSH in and work
ssh root@<pod-ip> -p <port>

# Terminate when done
runpodctl remove pod <POD_ID>
```

**Option B: Python SDK** (best for automation / autoresearch)
```python
import runpod

runpod.api_key = "YOUR_API_KEY"

# Create pod
pod = runpod.create_pod(
    name="toto-finetune",
    image_name="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel",
    gpu_type_id="NVIDIA L4",
    volume_in_gb=20,
    container_disk_in_gb=20,
)
pod_id = pod["id"]

# SSH operations
from runpod.cli.utils.ssh_cmd import SSHConnection

with SSHConnection(pod_id) as ssh:
    # Upload project
    ssh.put_file("market/commodities.db", "/workspace/commodities.db")
    ssh.rsync("./", "/workspace/project/")

    # Install deps + run training
    ssh.run_commands([
        "cd /workspace/project",
        "pip install -r requirements.txt",
        "pip install toto-ts",
        "python -m model.finetune --fold 0",
    ])

    # Download results
    ssh.get_file("/workspace/project/checkpoints/best.pt", "./checkpoints/best.pt")

# Terminate pod
runpod.terminate_pod(pod_id)
```

**Option C: RunPod Flash** (decorator-based, like Modal — beta)
```python
# pip install runpod-flash
# flash login

from runpod import Endpoint

@Endpoint(gpu="L4")
def train(config: dict):
    # This runs on RunPod's GPU
    ...
    return metrics
```
Flash is in beta. Python 3.12+ required. Designed more for inference than
training, but worth watching.

### File Transfer Options

| Method | Pros | Cons |
|--------|------|------|
| `SSHConnection.put_file/get_file` | Programmatic, SDK-native | Requires SSH key setup |
| `runpodctl send/receive` | Works without public IP | Uses `croc` protocol |
| `scp` | Standard, fast | Requires public IP pod |
| `rsync` | Incremental sync, efficient | Requires SSH setup |
| Git clone | Simple for code | Not for large data/DB |

**Our approach:** Git clone the code, `scp` the SQLite DB up, `scp` checkpoints down.
Or use the SDK's `SSHConnection` for full automation.

---

## SSH Key Setup

RunPod requires SSH keys (no password auth). One-time setup:

1. Generate key: `ssh-keygen -t ed25519 -f ~/.ssh/runpod_ed25519`
2. Add public key to RunPod account: Settings → SSH Keys
3. The SDK's `SSHConnection` auto-discovers keys from `~/.ssh/`

---

## Pre-built Templates

RunPod provides Docker images with ML frameworks pre-installed:

| Template | Image | Includes |
|----------|-------|----------|
| PyTorch 2.1 | `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel` | PyTorch, CUDA 11.8, Python 3.10 |
| PyTorch 2.0 | `runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel` | PyTorch, CUDA 11.8, Python 3.10 |

We use the PyTorch template — it has everything except `toto-ts` and `yfinance`
which we `pip install` at pod startup.

---

## How This Fits Our Project

### Phase 2-3 (Zero-shot + Fine-tuning)
- Create pod with L4 GPU
- Upload code + SQLite DB
- Run zero-shot baseline → fine-tuning → evaluation
- Download checkpoints + updated DB
- Terminate pod
- Total cost: ~$0.50-1.00 for a full session

### Phase 5 (Autoresearch)
- Python SDK creates pod automatically
- Agent generates config → SSH runs `train.py --config '{...}'`
- Agent reads metrics from stdout or DB
- Agent decides next experiment
- Repeat until budget exhausted or convergence
- Terminate pod

### Integration Plan
Add a `run_remote.py` script to the project:
```
19_toto-oil-futures/
├── run_remote.py        # Create pod, upload, train, download, terminate
├── market/              # Data layer (runs locally AND on pod)
├── model/               # Training + inference (runs on pod)
└── eval/                # Evaluation (runs locally or on pod)
```

The same code runs locally (CPU, for testing) and on RunPod (GPU, for real training).
`run_remote.py` is just the orchestration layer that shuttles files back and forth.

---

## Gotchas

1. **SSH proxy vs public IP**: Basic SSH through RunPod's proxy does NOT support
   `scp`/`sftp`. For file transfer via scp, you need a pod with a public IP
   (Secure Cloud). Alternative: use `runpodctl send/receive` which works without.

2. **Idle pods cost money**: Volume storage is billed even when pod is stopped
   ($0.20/GB/month idle). Terminate pods when done, not just stop.

3. **Container disk is ephemeral**: Anything not in `/workspace/` (the volume mount)
   disappears when the pod restarts. Always work in `/workspace/`.

4. **No UDP**: Only TCP/HTTP connections. Not an issue for us.

5. **Cold start**: Pod creation takes ~30-60 seconds. Image pull on first use
   takes longer. Subsequent pods using the same image are faster.
