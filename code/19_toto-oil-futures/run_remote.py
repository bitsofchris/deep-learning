"""RunPod orchestration — create pod, upload code + DB, run training/inference, download results.

This is the bridge between local development and GPU compute. The same code
runs locally (CPU) and on RunPod (GPU) — this script just handles the shuttle.

One-command workflow — just run and watch:

    # Fine-tune (creates pod, trains, streams logs, downloads results, terminates)
    python run_remote.py finetune --fold 0

    # Zero-shot baseline
    python run_remote.py forecast

Training runs inside tmux on the pod, so if you Ctrl+C or lose connection
the job keeps running. Reconnect anytime:

    python run_remote.py monitor --pod-id <id>

Escape hatches (for when you need manual control):

    python run_remote.py create                          # create pod, print details
    python run_remote.py run --pod-id <id> "command"     # run arbitrary command
    python run_remote.py ssh --pod-id <id> -i            # interactive shell
    python run_remote.py download --pod-id <id>          # pull results
    python run_remote.py terminate --pod-id <id>         # stop billing

Prerequisites:
    pip install runpod
    Set RUNPOD_API_KEY env var or pass --api-key
    Add your SSH public key to RunPod account settings
"""

import argparse
import logging
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent

# RunPod defaults
DEFAULT_GPU = "NVIDIA L4"
DEFAULT_IMAGE = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel"
DEFAULT_VOLUME_SIZE = 20  # GB
DEFAULT_CONTAINER_DISK = 20  # GB
WORKSPACE = "/workspace/project"


def load_env_file(env_path: Path | None = None) -> dict:
    """Load key=value pairs from a .env file. Does not export to os.environ."""
    if env_path is None:
        # Walk up from project root to find keys.env
        for parent in [PROJECT_ROOT, PROJECT_ROOT.parent, PROJECT_ROOT.parent.parent]:
            candidate = parent / "keys.env"
            if candidate.exists():
                env_path = candidate
                break

    if env_path is None or not env_path.exists():
        return {}

    env_vars = {}
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            env_vars[key.strip()] = value.strip()
    return env_vars


def get_api_key(args_key: str | None = None) -> str:
    """Get RunPod API key from args, keys.env, or env var (in that order)."""
    if args_key:
        return args_key

    # Try keys.env file (repo-level, gitignored)
    env_vars = load_env_file()
    key = env_vars.get("RUN_POD_API_KEY_TOTO")
    if key:
        logger.info("Loaded API key from keys.env")
        return key

    # Fall back to environment variable
    key = os.environ.get("RUNPOD_API_KEY")
    if key:
        return key

    logger.error(
        "No RunPod API key found. Add RUN_POD_API_KEY_TOTO to keys.env or set RUNPOD_API_KEY env var"
    )
    sys.exit(1)


def init_runpod(args_key: str | None = None) -> str:
    """Load API key and set it globally on the runpod module.

    Must be called before any RunPod SDK operation (create_pod, SSHConnection, etc.).
    Per RunPod docs: runpod.api_key must be set before any API/SSH call.
    """
    import runpod

    api_key = get_api_key(args_key)
    runpod.api_key = api_key
    return api_key


def get_ssh_info(pod_id: str) -> dict:
    """Get SSH connection details for a pod (ip, port, key file)."""
    from runpod.cli.utils.ssh_cmd import get_pod_ssh_ip_port, find_ssh_key_file

    ip, port = get_pod_ssh_ip_port(pod_id)
    key_file = find_ssh_key_file(ip, port)
    return {"ip": ip, "port": port, "key_file": key_file}


def get_ssh_cmd(pod_id: str, command: str | None = None) -> str:
    """Build an SSH command string for manual use or subprocess."""
    info = get_ssh_info(pod_id)
    base = f"ssh -o StrictHostKeyChecking=no -i {info['key_file']} -p {info['port']} root@{info['ip']}"
    if command:
        return f'{base} "{command}"'
    return base


def create_pod(
    api_key: str,
    gpu_type: str = DEFAULT_GPU,
    name: str = "toto-finetune",
) -> dict:
    """Create a RunPod GPU pod and wait for it to be ready."""
    import runpod

    runpod.api_key = api_key

    logger.info("Creating pod: %s on %s", name, gpu_type)
    pod = runpod.create_pod(
        name=name,
        image_name=DEFAULT_IMAGE,
        gpu_type_id=gpu_type,
        volume_in_gb=DEFAULT_VOLUME_SIZE,
        container_disk_in_gb=DEFAULT_CONTAINER_DISK,
        ports="22/tcp",  # Explicitly request SSH port
        support_public_ip=True,
    )
    pod_id = pod["id"]
    logger.info("Pod created: %s", pod_id)

    # Wait for RUNNING status with SSH port available
    logger.info("Waiting for pod to be ready (need SSH port)...")
    for attempt in range(120):
        status = runpod.get_pod(pod_id)
        desired = status.get("desiredStatus", "")
        runtime = status.get("runtime")

        if desired == "RUNNING" and runtime:
            ports = runtime.get("ports") or []
            # Check that SSH port (22/tcp) is exposed
            has_ssh = any(p.get("privatePort") == 22 for p in ports)
            if has_ssh:
                # Smoke-test actual SSH before declaring ready
                # (port can appear before sshd accepts connections)
                try:
                    _ssh_read(pod_id, "echo ready")
                    logger.info("Pod is running (SSH verified)")
                    return status
                except Exception:
                    logger.debug("  SSH port exposed but connection not yet accepted")

        time.sleep(5)
        if attempt % 12 == 0:
            logger.info("  Still waiting... (%ds, status=%s)", attempt * 5, desired)

    logger.error("Pod did not become ready with SSH in 10 minutes")
    sys.exit(1)


def setup_pod(pod_id: str):
    """Upload project code and install dependencies on the pod."""
    import time as _time
    from runpod.cli.utils.ssh_cmd import SSHConnection

    t0 = _time.time()

    with SSHConnection(pod_id) as ssh:
        logger.info("[setup 1/5] Connected to pod %s", pod_id)

        # Create project directory
        ssh.run_commands([f"mkdir -p {WORKSPACE}"])

        # Collect files to upload
        files_to_upload = []
        for pattern in ["*.py", "*.yaml", "*.txt"]:
            files_to_upload.extend(PROJECT_ROOT.glob(pattern))
        for subdir in ["market", "model", "eval"]:
            for py_file in (PROJECT_ROOT / subdir).rglob("*.py"):
                files_to_upload.append(py_file)

        logger.info("[setup 2/5] Uploading %d code files...", len(files_to_upload))
        for i, local_path in enumerate(files_to_upload):
            rel_path = local_path.relative_to(PROJECT_ROOT)
            remote_path = f"{WORKSPACE}/{rel_path}"
            remote_dir = str(Path(remote_path).parent)
            ssh.run_commands([f"mkdir -p {remote_dir}"])
            ssh.put_file(str(local_path), remote_path)
        logger.info(
            "[setup 2/5] Code uploaded (%d files, %.1fs)",
            len(files_to_upload),
            _time.time() - t0,
        )

        # Upload SQLite DB
        db_path = PROJECT_ROOT / "market" / "commodities.db"
        if db_path.exists():
            db_mb = db_path.stat().st_size / (1024 * 1024)
            logger.info("[setup 3/5] Uploading database (%.1f MB)...", db_mb)
            t_db = _time.time()
            ssh.put_file(str(db_path), f"{WORKSPACE}/market/commodities.db")
            logger.info("[setup 3/5] Database uploaded (%.1fs)", _time.time() - t_db)
        else:
            logger.warning(
                "[setup 3/5] No local DB found — will need to run download on pod"
            )

        # Install uv (fast pip replacement, ~2s)
        logger.info("[setup 4/5] Installing uv...")
        ssh.run_commands(
            [
                "command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh"
            ]
        )
        logger.info("[setup 4/5] uv ready (%.1fs total)", _time.time() - t0)

        # Install Python dependencies
        logger.info("[setup 5/5] Installing Python dependencies via uv...")
        t_deps = _time.time()
        ssh.run_commands(
            [
                f"cd {WORKSPACE}",
                "export PATH=$HOME/.local/bin:$PATH && uv pip install --system pyyaml pandas numpy yfinance 'toto-ts>=0.2.0' 2>&1 | tail -5",
            ]
        )
        logger.info("[setup 5/5] Dependencies installed (%.1fs)", _time.time() - t_deps)

    logger.info("Pod setup complete (%.1fs total)", _time.time() - t0)


def run_on_pod(pod_id: str, command: str) -> str:
    """Run a command on the pod, streaming output live and saving to log file."""
    from runpod.cli.utils.ssh_cmd import SSHConnection

    with SSHConnection(pod_id) as ssh:
        logger.info("Running: %s", command)
        # Tee to log file so `monitor` command can tail from another terminal
        full_cmd = f"cd {WORKSPACE} && {command} 2>&1 | tee -a {WORKSPACE}/run.log"
        results = ssh.run_commands([full_cmd])
        return results


# ---------------------------------------------------------------------------
# nohup-based execution — runs commands on the pod via nohup so they survive
# SSH disconnects. No tmux install needed. Local terminal tails the log file.
# ---------------------------------------------------------------------------

SENTINEL_FILE = f"{WORKSPACE}/.job_done"
LOG_FILE = f"{WORKSPACE}/run.log"


def run_nohup(pod_id: str, commands: list[str], job_name: str = "train"):
    """Launch commands on the pod via nohup (survives SSH disconnect).

    Writes a small bash script to the pod and runs it with nohup.
    All output goes to run.log, sentinel file signals completion.
    No tmux needed — works on any Linux box.
    """
    # Build a bash script to run on the pod
    script_lines = [
        "#!/bin/bash",
        "set -o pipefail",
        f"cd {WORKSPACE}",
        f"rm -f {SENTINEL_FILE}",
        f"echo '=== {job_name} started at '$(date)' ===' >> {LOG_FILE}",
    ]
    for cmd in commands:
        script_lines.append(f"echo '>>> {cmd}' >> {LOG_FILE}")
        script_lines.append(
            f"({cmd}) >> {LOG_FILE} 2>&1\n"
            f"EXIT_CODE=$?\n"
            f"if [ $EXIT_CODE -ne 0 ]; then\n"
            f'  echo "FAILED: {cmd} (exit $EXIT_CODE)" >> {LOG_FILE}\n'
            f'  echo "failed:$EXIT_CODE" > {SENTINEL_FILE}\n'
            f"  exit $EXIT_CODE\n"
            f"fi"
        )
    script_lines.append(
        f"echo '=== {job_name} completed at '$(date)' ===' >> {LOG_FILE}\n"
        f"echo 'done:0' > {SENTINEL_FILE}"
    )

    script_content = "\n".join(script_lines)
    script_path = f"{WORKSPACE}/_job.sh"

    import paramiko

    info = get_ssh_info(pod_id)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        info["ip"],
        port=info["port"],
        username="root",
        key_filename=info["key_file"],
    )

    # Write script, clear old state, launch with nohup
    for cmd in [
        f"rm -f {SENTINEL_FILE}",
        f"truncate -s 0 {LOG_FILE} 2>/dev/null || true",
        f"cat > {script_path} << 'JOBSCRIPT_EOF'\n{script_content}\nJOBSCRIPT_EOF",
        f"chmod +x {script_path}",
        f"nohup bash {script_path} > /dev/null 2>&1 &",
    ]:
        _, stdout, stderr = client.exec_command(cmd)
        stdout.channel.recv_exit_status()  # wait for each command

    client.close()

    logger.info("Job '%s' launched via nohup on pod %s", job_name, pod_id)


def tail_logs(pod_id: str) -> subprocess.Popen:
    """Start tailing run.log from the pod in the background. Returns the Popen handle."""
    info = get_ssh_info(pod_id)
    # Wait a moment for log file to be created
    time.sleep(2)
    proc = subprocess.Popen(
        [
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            "-i",
            info["key_file"],
            "-p",
            str(info["port"]),
            f"root@{info['ip']}",
            f"tail -f {LOG_FILE}",
        ],
        # Let stdout/stderr flow to our terminal
    )
    return proc


def _ssh_read(pod_id: str, command: str) -> str:
    """Run a command on the pod and return its stdout as a string.

    Uses paramiko directly (not SSHConnection.run_commands) so we get the
    output as a return value instead of it being printed to the console.
    """
    import paramiko

    info = get_ssh_info(pod_id)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        info["ip"],
        port=info["port"],
        username="root",
        key_filename=info["key_file"],
    )
    _, stdout, _ = client.exec_command(command)
    result = stdout.read().decode().strip()
    client.close()
    return result


def wait_for_completion(pod_id: str, poll_interval: int = 10) -> tuple[bool, int]:
    """Poll the sentinel file until the job finishes. Returns (success, exit_code)."""
    logger.info("Waiting for job to complete (polling every %ds)...", poll_interval)
    while True:
        time.sleep(poll_interval)
        try:
            status = _ssh_read(
                pod_id, f"cat {SENTINEL_FILE} 2>/dev/null || echo running"
            )
            if status.startswith("done:"):
                return True, int(status.split(":")[1])
            elif status.startswith("failed:"):
                return False, int(status.split(":")[1])
            # else: still running, keep polling
        except Exception:
            # SSH hiccup — pod may be temporarily unreachable, keep trying
            continue


def start_tensorboard(pod_id: str):
    """Install and start TensorBoard on the pod, print the proxy URL."""
    from runpod.cli.utils.ssh_cmd import SSHConnection

    with SSHConnection(pod_id) as ssh:
        logger.info("Starting TensorBoard on pod...")
        ssh.run_commands(
            [
                "pip install -q tensorboard 2>&1 | tail -1",
                f"pkill -f tensorboard 2>/dev/null; "
                f"nohup tensorboard --logdir {WORKSPACE}/lightning_logs "
                f"--host 0.0.0.0 --port 6006 > /tmp/tb.log 2>&1 &",
            ]
        )

    tb_url = f"https://{pod_id}-6006.proxy.runpod.net"
    print(f"\n  TensorBoard: {tb_url}")
    print("  (may take a few seconds to become available)\n")
    return tb_url


def download_results(pod_id: str, label: str | None = None):
    """Download DB (to unique file for merging), checkpoints, logs from the pod.

    Each pod's DB is saved as results/<label>.db (not overwriting the local DB).
    Use eval/merge_results.py to merge into the local DB after all pods finish.
    """
    from runpod.cli.utils.ssh_cmd import SSHConnection

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    db_label = label or pod_id
    remote_db_path = f"{WORKSPACE}/market/commodities.db"

    with SSHConnection(pod_id) as ssh:
        # Download DB to unique file (for parallel merge)
        local_db = results_dir / f"{db_label}.db"
        logger.info("Downloading database as %s ...", local_db.name)
        ssh.get_file(remote_db_path, str(local_db))

        # Download run log
        local_log = results_dir / f"{db_label}.log"
        try:
            ssh.get_file(f"{WORKSPACE}/run.log", str(local_log))
            logger.info("Run log saved to %s", local_log.name)
        except Exception:
            logger.debug("No run.log to download")

        # Download checkpoints if they exist
        checkpoints_dir = PROJECT_ROOT / "checkpoints"
        checkpoints_dir.mkdir(exist_ok=True)
        ssh.run_commands(
            [
                "ls -la /workspace/project/checkpoints/ 2>/dev/null || echo 'no checkpoints'"
            ]
        )

        # Download lightning logs (training curves)
        try:
            logger.info("Downloading lightning logs...")
            ssh.run_commands(
                [
                    f"cd {WORKSPACE} && tar czf /tmp/lightning_logs.tar.gz lightning_logs/ 2>/dev/null || echo 'no lightning_logs'"
                ]
            )
            tar_local = results_dir / f"{db_label}_lightning.tar.gz"
            ssh.get_file("/tmp/lightning_logs.tar.gz", str(tar_local))
            if tar_local.exists() and tar_local.stat().st_size > 0:
                extract_dir = results_dir / f"{db_label}_lightning"
                extract_dir.mkdir(exist_ok=True)
                subprocess.run(
                    ["tar", "xzf", str(tar_local), "-C", str(extract_dir)],
                    check=True,
                )
                tar_local.unlink()
                logger.info("Lightning logs saved to %s/", extract_dir.name)
        except Exception as e:
            logger.debug("Could not download lightning logs: %s", e)

    logger.info("Results downloaded to results/%s.*", db_label)
    return local_db


def terminate_pod(api_key: str, pod_id: str):
    """Terminate a pod to stop billing."""
    import runpod

    runpod.api_key = api_key
    runpod.terminate_pod(pod_id)
    logger.info("Pod %s terminated", pod_id)


# ---------------------------------------------------------------------------
# High-level commands
# ---------------------------------------------------------------------------


def _run_job(
    args,
    commands: list[str],
    job_name: str,
    tensorboard: bool = False,
):
    """Shared workflow for finetune/forecast: launch via nohup, tail logs, download.

    1. Create pod (or reuse --pod-id), upload code + DB
    2. (optional) Start TensorBoard
    3. Launch commands via nohup on the pod
    4. Tail run.log to this terminal
    5. Poll for completion
    6. Download results
    7. Terminate (unless --keep-pod or --pod-id)

    Ctrl+C during tailing is safe — job keeps running via nohup on the pod.
    """
    api_key = init_runpod(args.api_key)

    # Reuse existing pod or create fresh
    reuse_pod = getattr(args, "pod_id", None)
    if reuse_pod:
        pod_id = reuse_pod
        logger.info("Reusing existing pod %s", pod_id)
    else:
        pod = create_pod(api_key, gpu_type=args.gpu)
        pod_id = pod["id"]

    try:
        setup_pod(pod_id)

        # Start TensorBoard before training so it's ready when losses appear
        tb_url = None
        if tensorboard:
            tb_url = start_tensorboard(pod_id)

        # Launch the job via nohup (survives SSH drops)
        run_nohup(pod_id, commands, job_name=job_name)

        # Print status banner
        print(f"\n{'='*60}")
        print(f"  Pod ID:    {pod_id}")
        print(f"  Job:       {job_name}")
        print(f"  Status:    RUNNING (via nohup)")
        if tb_url:
            print(f"  TensorBoard: {tb_url}")
        print(f"{'='*60}")
        print("  Ctrl+C to detach — training keeps running on pod.")
        print(f"  Reconnect:  python run_remote.py monitor --pod-id {pod_id}")
        print(f"  SSH in:     python run_remote.py ssh --pod-id {pod_id} -i")
        print(f"{'='*60}\n")

        if tb_url and not args.no_browser:
            webbrowser.open(tb_url)

        # Tail logs in foreground — user sees training output live
        tail_proc = tail_logs(pod_id)
        try:
            success, exit_code = wait_for_completion(pod_id)
        except KeyboardInterrupt:
            # User hit Ctrl+C — training is still running in tmux
            tail_proc.terminate()
            print(f"\n{'='*60}")
            print(f"  Detached. Training continues on pod {pod_id}.")
            print(f"  Reconnect:  python run_remote.py monitor --pod-id {pod_id}")
            print(f"  Download:   python run_remote.py download --pod-id {pod_id}")
            print(f"  Terminate:  python run_remote.py terminate --pod-id {pod_id}")
            print(f"{'='*60}")
            return

        # Job finished — stop tailing
        tail_proc.terminate()

        if success:
            logger.info("Job '%s' completed successfully", job_name)
        else:
            logger.error(
                "Job '%s' failed (exit code %d) — check run.log", job_name, exit_code
            )

        # Download results either way (partial results may be useful)
        download_results(pod_id, label=job_name)
        logger.info("To merge: python -m eval.merge_results results/%s.db", job_name)

    finally:
        if reuse_pod:
            logger.info("Pod %s kept alive (reused via --pod-id)", pod_id)
        elif not args.keep_pod:
            terminate_pod(api_key, pod_id)
        else:
            print(f"\nPod {pod_id} kept alive. Remember to terminate!")


def cmd_forecast(args):
    """Run zero-shot or fine-tuned forecast on RunPod."""
    cmd = "python -m model.forecast"
    if args.checkpoint:
        cmd += f" --checkpoint {args.checkpoint}"
    if args.fold is not None:
        cmd += f" --fold {args.fold}"
    if args.tickers:
        cmd += f" --tickers {' '.join(args.tickers)}"
    if args.target:
        cmd += f" --target {args.target}"
    if args.experiment_id:
        cmd += f" --experiment-id {args.experiment_id}"

    label = args.experiment_id or f"forecast_{args.target or 'returns'}"
    commands = [cmd, "python -m eval.evaluate"]
    _run_job(args, commands, job_name=label)


def cmd_finetune(args):
    """Fine-tune Toto on RunPod, then forecast and evaluate."""
    ft_cmd = "python -m model.finetune"
    fc_cmd = "python -m model.forecast --checkpoint checkpoints/best.pt"
    if args.fold is not None:
        ft_cmd += f" --fold {args.fold}"
        fc_cmd += f" --fold {args.fold}"
    if args.target:
        ft_cmd += f" --target {args.target}"
        fc_cmd += f" --target {args.target}"
    if args.experiment_id:
        ft_cmd += f" --experiment-id {args.experiment_id}"
        fc_cmd += f" --experiment-id {args.experiment_id}_eval"

    label = args.experiment_id or f"finetune_{args.target or 'returns'}"
    commands = [ft_cmd, fc_cmd, "python -m eval.evaluate"]
    _run_job(args, commands, job_name=label, tensorboard=True)


def cmd_run(args):
    """Run an arbitrary command on a new or existing pod."""
    api_key = init_runpod(args.api_key)

    if args.pod_id:
        pod_id = args.pod_id
    else:
        pod = create_pod(api_key, gpu_type=args.gpu)
        pod_id = pod["id"]
        setup_pod(pod_id)

    try:
        run_on_pod(pod_id, args.command)
        if args.download:
            download_results(pod_id)
    finally:
        if not args.keep_pod and not args.pod_id:
            terminate_pod(api_key, pod_id)


def cmd_create(args):
    """Just create a pod and print details."""
    api_key = init_runpod(args.api_key)
    pod = create_pod(api_key, gpu_type=args.gpu)
    pod_id = pod["id"]

    setup_pod(pod_id)

    ssh_cmd = get_ssh_cmd(pod_id)

    print(f"\n{'='*60}")
    print(f"Pod ID:    {pod_id}")
    print(f"GPU:       {args.gpu}")
    print(f"Workspace: {WORKSPACE}")
    print(f"{'='*60}")
    print(f"\nSSH in:           {ssh_cmd}")
    print(
        f'\nRun command:      python run_remote.py run --pod-id {pod_id} "your command"'
    )
    print(f"Monitor logs:     python run_remote.py monitor --pod-id {pod_id}")
    print(f"TensorBoard:      python run_remote.py tensorboard --pod-id {pod_id}")
    print(f"Download results: python run_remote.py download --pod-id {pod_id}")
    print(f"Terminate:        python run_remote.py terminate --pod-id {pod_id}")
    print(f"\nRemember: pod is billing until terminated!")


def cmd_monitor(args):
    """Stream live logs from a running pod (tail -f run.log)."""
    init_runpod(args.api_key)
    info = get_ssh_info(args.pod_id)
    tail_cmd = f"tail -f {WORKSPACE}/run.log"
    ssh_args = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-i",
        info["key_file"],
        "-p",
        str(info["port"]),
        f"root@{info['ip']}",
        tail_cmd,
    ]
    logger.info("Streaming logs from pod %s (Ctrl+C to stop)...", args.pod_id)
    try:
        subprocess.run(ssh_args)
    except KeyboardInterrupt:
        print("\nStopped monitoring.")


def cmd_tensorboard(args):
    """Start TensorBoard on the pod and open via RunPod proxy."""
    init_runpod(args.api_key)
    tb_url = start_tensorboard(args.pod_id)
    if not args.no_browser:
        webbrowser.open(tb_url)


def cmd_download(args):
    """Download results from an existing pod without terminating it."""
    init_runpod(args.api_key)
    download_results(args.pod_id)


def cmd_ssh(args):
    """Print SSH command or open interactive shell."""
    init_runpod(args.api_key)
    ssh_cmd = get_ssh_cmd(args.pod_id)
    if args.interactive:
        info = get_ssh_info(args.pod_id)
        ssh_args = [
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            "-i",
            info["key_file"],
            "-p",
            str(info["port"]),
            f"root@{info['ip']}",
            "-t",
            f"cd {WORKSPACE} && bash",
        ]
        subprocess.run(ssh_args)
    else:
        print(f"\n{ssh_cmd}")
        print(f"\nOr with workspace: {ssh_cmd} -t 'cd {WORKSPACE} && bash'")


def cmd_terminate(args):
    """Terminate a pod."""
    api_key = init_runpod(args.api_key)
    terminate_pod(api_key, args.pod_id)


def main():
    parser = argparse.ArgumentParser(
        description="RunPod orchestration for Toto training"
    )
    parser.add_argument(
        "--api-key", type=str, help="RunPod API key (or set RUNPOD_API_KEY)"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=DEFAULT_GPU,
        help=f"GPU type (default: {DEFAULT_GPU})",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # forecast
    p_forecast = subparsers.add_parser(
        "forecast", help="Run zero-shot or fine-tuned forecast"
    )
    p_forecast.add_argument(
        "--checkpoint", type=str, help="Checkpoint path for fine-tuned model"
    )
    p_forecast.add_argument("--fold", type=int, help="CV fold number")
    p_forecast.add_argument("--tickers", nargs="+", help="Override eval tickers")
    p_forecast.add_argument(
        "--target", type=str, help="Target column (returns, vol_adj_returns)"
    )
    p_forecast.add_argument("--experiment-id", type=str, help="Custom experiment ID")
    p_forecast.add_argument(
        "--pod-id", type=str, help="Reuse existing pod (skip create, don't terminate)"
    )
    p_forecast.add_argument(
        "--keep-pod", action="store_true", help="Don't terminate pod after"
    )
    p_forecast.add_argument(
        "--no-browser", action="store_true", help="Don't auto-open browser"
    )

    # finetune
    p_finetune = subparsers.add_parser("finetune", help="Fine-tune Toto")
    p_finetune.add_argument("--fold", type=int, help="CV fold number")
    p_finetune.add_argument(
        "--target", type=str, help="Target column (returns, vol_adj_returns)"
    )
    p_finetune.add_argument("--experiment-id", type=str, help="Custom experiment ID")
    p_finetune.add_argument(
        "--pod-id", type=str, help="Reuse existing pod (skip create, don't terminate)"
    )
    p_finetune.add_argument(
        "--keep-pod", action="store_true", help="Don't terminate pod after"
    )
    p_finetune.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't auto-open TensorBoard in browser",
    )

    # run
    p_run = subparsers.add_parser("run", help="Run arbitrary command on pod")
    p_run.add_argument("command", type=str, help="Command to execute")
    p_run.add_argument("--pod-id", type=str, help="Existing pod ID (skip create)")
    p_run.add_argument(
        "--keep-pod", action="store_true", help="Don't terminate pod after"
    )
    p_run.add_argument("--download", action="store_true", help="Download results after")

    # create
    subparsers.add_parser("create", help="Create pod and print details")

    # monitor — stream logs from running pod
    p_monitor = subparsers.add_parser("monitor", help="Stream live logs from pod")
    p_monitor.add_argument(
        "--pod-id", type=str, required=True, help="Pod ID to monitor"
    )

    # tensorboard — start TensorBoard via RunPod proxy
    p_tb = subparsers.add_parser(
        "tensorboard", help="Start TensorBoard (opens in browser)"
    )
    p_tb.add_argument("--pod-id", type=str, required=True, help="Pod ID")
    p_tb.add_argument(
        "--no-browser", action="store_true", help="Don't auto-open browser"
    )

    # download — pull results without terminating
    p_dl = subparsers.add_parser("download", help="Download results from pod")
    p_dl.add_argument("--pod-id", type=str, required=True, help="Pod ID")

    # ssh — print SSH command or open shell
    p_ssh = subparsers.add_parser("ssh", help="Get SSH command or open shell")
    p_ssh.add_argument("--pod-id", type=str, required=True, help="Pod ID")
    p_ssh.add_argument(
        "-i", "--interactive", action="store_true", help="Open interactive shell"
    )

    # terminate
    p_term = subparsers.add_parser("terminate", help="Terminate a pod")
    p_term.add_argument("--pod-id", type=str, required=True, help="Pod ID to terminate")

    args = parser.parse_args()

    commands = {
        "forecast": cmd_forecast,
        "finetune": cmd_finetune,
        "run": cmd_run,
        "create": cmd_create,
        "monitor": cmd_monitor,
        "tensorboard": cmd_tensorboard,
        "download": cmd_download,
        "ssh": cmd_ssh,
        "terminate": cmd_terminate,
    }
    handler = commands.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
