"""RunPod orchestration — create pod, upload code + DB, run training/inference, download results.

This is the bridge between local development and GPU compute. The same code
runs locally (CPU) and on RunPod (GPU) — this script just handles the shuttle.

Usage:
    # Zero-shot baseline on all eval tickers
    python run_remote.py forecast

    # Fine-tune on fold 0
    python run_remote.py finetune --fold 0

    # Custom command on the pod
    python run_remote.py run "python -m eval.evaluate --list"

    # Just create a pod and get SSH details (interactive)
    python run_remote.py create

    # Terminate a pod
    python run_remote.py terminate --pod-id <id>

Prerequisites:
    pip install runpod
    Set RUNPOD_API_KEY env var or pass --api-key
    Add your SSH public key to RunPod account settings
"""

import argparse
import logging
import os
import sys
import time
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
    )
    pod_id = pod["id"]
    logger.info("Pod created: %s", pod_id)

    # Wait for RUNNING status
    logger.info("Waiting for pod to be ready...")
    for attempt in range(60):
        status = runpod.get_pod(pod_id)
        runtime = status.get("runtime")
        if (
            runtime
            and runtime.get("uptimeInSeconds")
            and runtime.get("uptimeInSeconds") > 0
        ):
            logger.info("Pod is running (uptime: %ds)", runtime["uptimeInSeconds"])
            return status
        time.sleep(5)
        if attempt % 6 == 0:
            logger.info("  Still waiting... (attempt %d)", attempt + 1)

    logger.error("Pod did not become ready in time")
    sys.exit(1)


def setup_pod(pod_id: str):
    """Upload project code and install dependencies on the pod."""
    from runpod.cli.utils.ssh_cmd import SSHConnection

    with SSHConnection(pod_id) as ssh:
        logger.info("Setting up pod %s...", pod_id)

        # Create project directory
        ssh.run_commands([f"mkdir -p {WORKSPACE}"])

        # Upload project files
        logger.info("Uploading project files...")
        files_to_upload = []

        # Collect all Python files and config
        for pattern in ["*.py", "*.yaml", "*.txt"]:
            files_to_upload.extend(PROJECT_ROOT.glob(pattern))
        for subdir in ["market", "model", "eval"]:
            for py_file in (PROJECT_ROOT / subdir).rglob("*.py"):
                files_to_upload.append(py_file)

        for local_path in files_to_upload:
            rel_path = local_path.relative_to(PROJECT_ROOT)
            remote_path = f"{WORKSPACE}/{rel_path}"
            # Ensure remote directory exists
            remote_dir = str(Path(remote_path).parent)
            ssh.run_commands([f"mkdir -p {remote_dir}"])
            ssh.put_file(str(local_path), remote_path)

        # Upload SQLite DB
        db_path = PROJECT_ROOT / "market" / "commodities.db"
        if db_path.exists():
            logger.info(
                "Uploading database (%d MB)...", db_path.stat().st_size // (1024 * 1024)
            )
            ssh.put_file(str(db_path), f"{WORKSPACE}/market/commodities.db")
        else:
            logger.warning("No local DB found — will need to run download on pod")

        # Install dependencies
        logger.info("Installing dependencies...")
        ssh.run_commands(
            [
                f"cd {WORKSPACE}",
                "pip install -q pyyaml pandas numpy yfinance toto-ts 2>&1 | tail -5",
            ]
        )

        logger.info("Pod setup complete")


def run_on_pod(pod_id: str, command: str) -> str:
    """Run a command on the pod and return output."""
    from runpod.cli.utils.ssh_cmd import SSHConnection

    with SSHConnection(pod_id) as ssh:
        logger.info("Running: %s", command)
        full_cmd = f"cd {WORKSPACE} && {command}"
        results = ssh.run_commands([full_cmd])
        return results


def download_results(pod_id: str):
    """Download updated DB and any checkpoints from the pod."""
    from runpod.cli.utils.ssh_cmd import SSHConnection

    with SSHConnection(pod_id) as ssh:
        # Download updated DB
        local_db = PROJECT_ROOT / "market" / "commodities.db"
        logger.info("Downloading updated database...")
        ssh.get_file(f"{WORKSPACE}/market/commodities.db", str(local_db))

        # Download checkpoints if they exist
        checkpoints_dir = PROJECT_ROOT / "checkpoints"
        checkpoints_dir.mkdir(exist_ok=True)

        # Check if remote checkpoints exist
        ssh.run_commands(
            [
                "ls -la /workspace/project/checkpoints/ 2>/dev/null || echo 'no checkpoints'"
            ]
        )

    logger.info("Results downloaded")


def terminate_pod(api_key: str, pod_id: str):
    """Terminate a pod to stop billing."""
    import runpod

    runpod.api_key = api_key
    runpod.terminate_pod(pod_id)
    logger.info("Pod %s terminated", pod_id)


# ---------------------------------------------------------------------------
# High-level commands
# ---------------------------------------------------------------------------


def cmd_forecast(args):
    """Run zero-shot or fine-tuned forecast on RunPod."""
    api_key = get_api_key(args.api_key)

    pod = create_pod(api_key, gpu_type=args.gpu)
    pod_id = pod["id"]

    try:
        setup_pod(pod_id)

        # Build forecast command
        cmd = "python -m model.forecast"
        if args.checkpoint:
            cmd += f" --checkpoint {args.checkpoint}"
        if args.fold is not None:
            cmd += f" --fold {args.fold}"
        if args.tickers:
            cmd += f" --tickers {' '.join(args.tickers)}"

        run_on_pod(pod_id, cmd)

        # Run evaluation
        run_on_pod(pod_id, "python -m eval.evaluate")

        download_results(pod_id)
    finally:
        if not args.keep_pod:
            terminate_pod(api_key, pod_id)
        else:
            logger.info(
                "Pod %s kept alive (--keep-pod). Remember to terminate!", pod_id
            )


def cmd_finetune(args):
    """Fine-tune Toto on RunPod."""
    api_key = get_api_key(args.api_key)

    pod = create_pod(api_key, gpu_type=args.gpu)
    pod_id = pod["id"]

    try:
        setup_pod(pod_id)

        # Fine-tune
        cmd = "python -m model.finetune"
        if args.fold is not None:
            cmd += f" --fold {args.fold}"
        run_on_pod(pod_id, cmd)

        # Forecast with fine-tuned model
        run_on_pod(pod_id, "python -m model.forecast --checkpoint checkpoints/best.pt")

        # Evaluate
        run_on_pod(pod_id, "python -m eval.evaluate")

        download_results(pod_id)
    finally:
        if not args.keep_pod:
            terminate_pod(api_key, pod_id)
        else:
            logger.info(
                "Pod %s kept alive (--keep-pod). Remember to terminate!", pod_id
            )


def cmd_run(args):
    """Run an arbitrary command on a new or existing pod."""
    api_key = get_api_key(args.api_key)

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
    api_key = get_api_key(args.api_key)
    pod = create_pod(api_key, gpu_type=args.gpu)
    pod_id = pod["id"]

    setup_pod(pod_id)

    print(f"\nPod ID: {pod_id}")
    print(f"GPU: {args.gpu}")
    print(
        f'\nTo run commands:  python run_remote.py run --pod-id {pod_id} "your command"'
    )
    print(f"To terminate:     python run_remote.py terminate --pod-id {pod_id}")
    print(f"\nRemember: pod is billing until terminated!")


def cmd_terminate(args):
    """Terminate a pod."""
    api_key = get_api_key(args.api_key)
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
        "--keep-pod", action="store_true", help="Don't terminate pod after"
    )

    # finetune
    p_finetune = subparsers.add_parser("finetune", help="Fine-tune Toto")
    p_finetune.add_argument("--fold", type=int, help="CV fold number")
    p_finetune.add_argument(
        "--keep-pod", action="store_true", help="Don't terminate pod after"
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

    # terminate
    p_term = subparsers.add_parser("terminate", help="Terminate a pod")
    p_term.add_argument("--pod-id", type=str, required=True, help="Pod ID to terminate")

    args = parser.parse_args()

    if args.command == "forecast":
        cmd_forecast(args)
    elif args.command == "finetune":
        cmd_finetune(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "create":
        cmd_create(args)
    elif args.command == "terminate":
        cmd_terminate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
