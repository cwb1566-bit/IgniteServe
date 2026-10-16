"""Check the packaged final contract through a real HTTP inference server."""
import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix="startup-api-") as tmp:
        report = Path(tmp) / "report.json"
        env = dict(os.environ, VLLM_ENABLE_STARTUP_OPTIMIZATIONS="1",
                   VLLM_STARTUP_GENERATED_REPORT=str(report))
        for key in ("VLLM_STARTUP_ALLOW_CANDIDATE_VALIDATION", "VLLM_STARTUP_CONTRACT_PATH",
                    "VLLM_STARTUP_ADAPTER_PATH", "VLLM_STARTUP_DAG_MODE"):
            env.pop(key, None)
        with (Path(tmp) / "server.log").open("w+") as log:
            proc = subprocess.Popen([
                sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                "--model", args.model, "--host", "127.0.0.1", "--port", str(port),
                "--max-model-len", "2048", "--max-num-seqs", "16",
                "--max-num-batched-tokens", "512", "--dtype", "float16",
                "--gpu-memory-utilization", "0.5"],
                env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                deadline = time.monotonic() + 300
                while time.monotonic() < deadline:
                    if proc.poll() is not None:
                        raise RuntimeError("Server exited before readiness")
                    try:
                        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
                            break
                    except OSError:
                        time.sleep(.5)
                else:
                    raise TimeoutError("API readiness deadline")
                request = urllib.request.Request(
                    f"http://127.0.0.1:{port}/v1/completions",
                    data=json.dumps(dict(model=args.model, prompt="The capital of France is",
                                         max_tokens=8, temperature=0)).encode(),
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    result = json.load(response)
                evidence = json.loads(report.read_text())
                assert len(evidence["results"]) == 25
                assert all(v["status"] == "completed" for v in evidence["results"].values())
                assert result["usage"]["completion_tokens"] > 0
                print(json.dumps(dict(nodes=25, final_contract=True,
                                      health=200, completion=result["choices"][0]["text"])))
            except BaseException:
                log.seek(0)
                print(log.read()[-12000:])
                raise
            finally:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()


if __name__ == "__main__":
    main()
