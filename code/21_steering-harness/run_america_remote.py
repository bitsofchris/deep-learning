"""Run America AI steering harvest/sweeps/optimization on RunPod.

This follows the existing steering-harness workflow used for pickle vectors:
harvest vectors, sweep layers/strengths, compose/optimize, and download results.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parents[1]
WORKSPACE = "/workspace/steering-harness"
IMAGE = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel"
GPU_ORDER = [
    "NVIDIA GeForce RTX 3090",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA L4",
    "NVIDIA A40",
]
SENTINEL = f"{WORKSPACE}/.america_done"
LOG_FILE = f"{WORKSPACE}/america_run.log"
SOURCE_RUNPOD_ENV = (
    'while IFS= read -r -d "" line; do export "$line"; done < /proc/1/environ'
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("america-remote")


def load_env() -> dict[str, str]:
    env_path = REPO_ROOT / "keys.env"
    values: dict[str, str] = {}
    if not env_path.exists():
        return values
    for line in env_path.read_text().splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


def init_runpod(api_key: str | None):
    import runpod

    env = load_env()
    key = api_key or env.get("RUN_POD_API_KEY_TOTO") or os.environ.get("RUNPOD_API_KEY")
    if not key:
        raise SystemExit(
            "Missing RunPod key: set RUN_POD_API_KEY_TOTO in keys.env or RUNPOD_API_KEY."
        )
    runpod.api_key = key
    return runpod, env


def get_ssh_info(pod_id: str) -> dict:
    from runpod.cli.utils.ssh_cmd import find_ssh_key_file, get_pod_ssh_ip_port

    ip, port = get_pod_ssh_ip_port(pod_id)
    return {"ip": ip, "port": port, "key_file": find_ssh_key_file(ip, port)}


def ssh_read(pod_id: str, command: str) -> str:
    import paramiko

    info = get_ssh_info(pod_id)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        info["ip"], port=info["port"], username="root", key_filename=info["key_file"]
    )
    _, stdout, stderr = client.exec_command(command)
    out = stdout.read().decode()
    err = stderr.read().decode()
    code = stdout.channel.recv_exit_status()
    client.close()
    if code != 0:
        raise RuntimeError(f"remote command failed ({code}): {err or out}")
    return out.strip()


def create_pod(runpod, gpu: str | None, secret_name: str | None = None) -> str:
    gpu_candidates = [gpu] if gpu else GPU_ORDER
    last_error = None
    env = None
    if secret_name:
        secret_ref = "{{ RUNPOD_SECRET_" + secret_name + " }}"
        env = {
            "HF_TOKEN": secret_ref,
            "HUGGING_FACE_HUB_TOKEN": secret_ref,
        }
    for gpu_name in gpu_candidates:
        try:
            log.info("Creating RunPod pod on %s", gpu_name)
            pod = runpod.create_pod(
                name="america-ai-steering",
                image_name=IMAGE,
                gpu_type_id=gpu_name,
                volume_in_gb=40,
                container_disk_in_gb=40,
                ports="22/tcp",
                support_public_ip=True,
                env=env,
            )
            pod_id = pod["id"]
            log.info("Pod created: %s", pod_id)
            for attempt in range(120):
                status = runpod.get_pod(pod_id)
                runtime = status.get("runtime")
                if status.get("desiredStatus") == "RUNNING" and runtime:
                    ports = runtime.get("ports") or []
                    if any(port.get("privatePort") == 22 for port in ports):
                        try:
                            ssh_read(pod_id, "echo ok")
                            log.info("SSH ready on pod %s", pod_id)
                            return pod_id
                        except Exception:
                            pass
                if attempt % 12 == 0:
                    log.info("Waiting for SSH... (%ds)", attempt * 5)
                time.sleep(5)
            raise RuntimeError("pod did not become SSH-ready")
        except Exception as exc:
            last_error = exc
            log.warning("Could not use %s: %s", gpu_name, exc)
    raise RuntimeError(f"all GPU candidates failed: {last_error}")


def iter_upload_files(results_name: str) -> list[Path]:
    files: list[Path] = []
    for pattern in ["*.py", "*.md", "*.txt"]:
        files.extend(PROJECT_ROOT.glob(pattern))
    for subdir in ["america_ai", "data", "tests"]:
        files.extend((PROJECT_ROOT / subdir).rglob("*.py"))
    results_dir = PROJECT_ROOT / "results" / results_name
    if results_dir.exists():
        for pattern in [
            "vectors/**/*.npz",
            "vectors/**/*.json",
            "reports/layer_metrics.csv",
        ]:
            files.extend(results_dir.glob(pattern))
    return sorted(path for path in files if "__pycache__" not in path.parts)


def setup_pod(pod_id: str, results_name: str) -> None:
    import paramiko

    info = get_ssh_info(pod_id)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        info["ip"], port=info["port"], username="root", key_filename=info["key_file"]
    )
    sftp = client.open_sftp()
    client.exec_command(f"rm -rf {WORKSPACE} && mkdir -p {WORKSPACE}")[
        1
    ].channel.recv_exit_status()

    files = iter_upload_files(results_name)
    log.info("Uploading %d steering harness files", len(files))
    for local in files:
        rel = local.relative_to(PROJECT_ROOT)
        remote = f"{WORKSPACE}/{rel}"
        client.exec_command(f"mkdir -p {Path(remote).parent}")[
            1
        ].channel.recv_exit_status()
        sftp.put(str(local), remote)

    sftp.close()
    install = (
        f"cd {WORKSPACE} && "
        f"{SOURCE_RUNPOD_ENV} && "
        "python -m pip install --upgrade pip && "
        "python -m pip install -r requirements.txt pytest huggingface_hub && "
        'if [ -n "${HF_TOKEN:-}" ]; then huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential; fi'
    )
    log.info("Installing Python requirements on pod")
    _, stdout, stderr = client.exec_command(install, timeout=1800)
    for line in stdout:
        print(line, end="")
    err = stderr.read().decode()
    code = stdout.channel.recv_exit_status()
    client.close()
    if code != 0:
        raise RuntimeError(f"dependency install failed: {err}")


def launch_job(
    pod_id: str,
    quick: bool,
    skip_harvest: bool,
    sweep_seeds: str,
    probe_only: bool,
    harvest_probe_only: bool,
    model: str,
    results_name: str,
) -> None:
    import paramiko

    commands = ["pytest -q"]
    if harvest_probe_only:
        if not skip_harvest:
            commands.append("python america_harvest.py")
        commands.append("python america_probe.py")
    elif probe_only:
        commands.append("python america_probe.py")
    else:
        if not skip_harvest:
            commands.append("python america_harvest.py")
        commands.extend(
            [
                f"python america_sweep.py --concept americana --seeds {sweep_seeds}",
                f"python america_sweep.py --concept patriotic_pride --seeds {sweep_seeds}",
                f"python america_sweep.py --concept trump_approval --seeds {sweep_seeds}",
                f"python america_sweep.py --concept star_spangled_bombast --seeds {sweep_seeds}",
            ]
        )
        if quick:
            commands.append(
                "python america_optimize.py --trials 24 --stage-one-prompts 4 --finalists 6 --seeds 0,1"
            )
        else:
            commands.append(
                "python america_optimize.py --trials 120 --stage-one-prompts 8 --finalists 20 --seeds 0,1,2"
            )
        commands.append(
            "python america_demo.py --preset america_ai --prompt 'What should I make for dinner?' "
            "> results/america_ai/reports/demo_dinner.txt"
        )

    script = [
        "#!/bin/bash",
        "set -uo pipefail",
        f"cd {WORKSPACE}",
        SOURCE_RUNPOD_ENV,
        f"export AMERICA_AI_MODEL={model!r}",
        f"export AMERICA_AI_RESULTS_NAME={results_name!r}",
        f"rm -f {SENTINEL}",
        f"echo '=== America AI run started at '$(date)' ===' > {LOG_FILE}",
    ]
    for command in commands:
        script.append(f"echo '>>> {command}' >> {LOG_FILE}")
        script.append(
            f"({command}) >> {LOG_FILE} 2>&1\n"
            "EXIT_CODE=$?\n"
            "if [ $EXIT_CODE -ne 0 ]; then\n"
            f"  echo 'FAILED: {command} (exit '$EXIT_CODE')' >> {LOG_FILE}\n"
            f"  echo failed:$EXIT_CODE > {SENTINEL}\n"
            "  exit $EXIT_CODE\n"
            "fi"
        )
    script.extend(
        [
            f"echo '=== America AI run completed at '$(date)' ===' >> {LOG_FILE}",
            f"echo done:0 > {SENTINEL}",
        ]
    )

    info = get_ssh_info(pod_id)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        info["ip"], port=info["port"], username="root", key_filename=info["key_file"]
    )
    sftp = client.open_sftp()
    with sftp.file(f"{WORKSPACE}/_america_job.sh", "w") as f:
        f.write("\n".join(script))
    sftp.close()
    for command in [
        f"chmod +x {WORKSPACE}/_america_job.sh",
        f"nohup bash {WORKSPACE}/_america_job.sh >/dev/null 2>&1 &",
    ]:
        client.exec_command(command)[1].channel.recv_exit_status()
    client.close()
    log.info("Remote America AI job launched")


def tail_and_wait(pod_id: str) -> bool:
    info = get_ssh_info(pod_id)
    tail = subprocess.Popen(
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
        ]
    )
    try:
        while True:
            time.sleep(20)
            try:
                status = ssh_read(pod_id, f"cat {SENTINEL} 2>/dev/null || echo running")
                if status.startswith("done:"):
                    return True
                if status.startswith("failed:"):
                    return False
            except Exception:
                pass
    finally:
        tail.terminate()


def download_results(pod_id: str, results_name: str) -> None:
    from runpod.cli.utils.ssh_cmd import SSHConnection

    local_results = PROJECT_ROOT / "results" / results_name
    local_results.mkdir(parents=True, exist_ok=True)
    archive = "/tmp/america_ai_results.tar.gz"
    ssh_read(
        pod_id,
        f"cd {WORKSPACE} && tar czf {archive} results/{results_name} america_run.log",
    )
    with SSHConnection(pod_id) as ssh:
        ssh.get_file(
            archive, str(PROJECT_ROOT / "results" / "america_ai_results.tar.gz")
        )
    subprocess.run(
        [
            "tar",
            "xzf",
            str(PROJECT_ROOT / "results" / "america_ai_results.tar.gz"),
            "-C",
            str(PROJECT_ROOT),
        ],
        check=True,
    )
    log.info("Downloaded results to %s", local_results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--pod-id", default=None)
    parser.add_argument("--keep-pod", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use fewer optimization trials after full harvest/sweeps.",
    )
    parser.add_argument(
        "--skip-harvest",
        action="store_true",
        help="Upload existing results/america_ai/vectors and skip america_harvest.py.",
    )
    parser.add_argument("--sweep-seeds", default="0,1,2")
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Upload measured vectors and run only america_probe.py.",
    )
    parser.add_argument(
        "--harvest-probe-only",
        action="store_true",
        help="Run tests, harvest vectors, then run the qualitative probe without sweeps/optimization.",
    )
    parser.add_argument("--model", default="google/gemma-2-2b")
    parser.add_argument("--results-name", default=None)
    parser.add_argument(
        "--runpod-secret-name",
        default=None,
        help="RunPod secret name to expose as HF_TOKEN, e.g. huggingface_token for {{ RUNPOD_SECRET_huggingface_token }}.",
    )
    args = parser.parse_args()
    results_name = args.results_name or args.model.replace("/", "_").replace("-", "_")

    runpod, env = init_runpod(args.api_key)
    pod_id = args.pod_id or create_pod(runpod, args.gpu, args.runpod_secret_name)
    reused = bool(args.pod_id)
    try:
        if not reused:
            setup_pod(pod_id, results_name)
        launch_job(
            pod_id,
            args.quick,
            args.skip_harvest,
            args.sweep_seeds,
            args.probe_only,
            args.harvest_probe_only,
            args.model,
            results_name,
        )
        print(f"Pod ID: {pod_id}")
        ok = tail_and_wait(pod_id)
        if ok:
            download_results(pod_id, results_name)
        else:
            log.error("Remote job failed; downloading logs/results for inspection")
            download_results(pod_id, results_name)
    finally:
        if args.keep_pod or reused:
            print(f"Pod kept alive: {pod_id}")
        else:
            runpod.terminate_pod(pod_id)
            log.info("Terminated pod %s", pod_id)


if __name__ == "__main__":
    main()
