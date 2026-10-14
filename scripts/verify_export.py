"""Check the decision API end to end.

--model  Start a temporary loopback server for an exported model, check it, and write a report
         (run automatically by finalize.py).
--url    Check a server that is already running, e.g. one started with scripts/serve.py.
"""
import argparse
import json
import math
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path

from decision.api import check_response
from decision.schema import PROMPT_VERSION
from decision.utils import sha256

ROOT = Path(__file__).resolve().parents[1]
INVALID_REQUESTS = [
    {"state": "测试", "questions": {"q": {"type": "choice", "instructions": "选择", "criteria": {"one": "唯一项"}}}},
    {"state": "测试", "questions": {"q": {"type": "noul", "instructions": None}}},
    {"state": "测试", "questions": {"q": {"type": "noul", "instructions": "判断", "criteria": []}}},
]


def request(url, body=None):
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.load(response)


def check_api(url, payload):
    """Check both routes and schema errors; returns the first response and the route delta."""
    first = request(url + "/v1/systemone", payload)
    second = request(url + "/api/alpha/decisions", payload)
    check_response(first, payload)
    check_response(second, payload)
    max_route_delta = max(abs(p - second["answers"][name]["probabilities"][key])
                          for name, answer in first["answers"].items()
                          for key, p in answer["probabilities"].items())
    if max_route_delta > 1e-5:
        raise ValueError("API aliases returned different probabilities")
    for body in INVALID_REQUESTS:
        try:
            request(url + "/v1/systemone", body)
        except urllib.error.HTTPError as error:
            if error.code != 422:
                raise ValueError(f"invalid schema returned HTTP {error.code}") from error
        else:
            raise ValueError("invalid schema was accepted")
    return first, max_route_delta


def verify_model(args, payload):
    model = Path(args.model).resolve()
    configuration = json.loads((model / "decision_config.json").read_text())
    if configuration["prompt_version"] != PROMPT_VERSION:
        raise ValueError("saved model uses a different prompt compiler")
    if not math.isfinite(configuration["temperature"]) or configuration["temperature"] <= 0:
        raise ValueError("invalid calibration temperature")
    weights = sorted(model.glob("*.safetensors"))
    if not weights or any(path.stat().st_size < 1024 for path in weights):
        raise ValueError("model weights missing or incomplete")
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = ROOT / "logs/export-verification-api.log"
    log_path.parent.mkdir(exist_ok=True)
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
    with log_path.open("w", encoding="utf-8") as output:
        process = subprocess.Popen([sys.executable, str(ROOT / "scripts/serve.py"), "--base", str(model),
                                    "--host", "127.0.0.1", "--port", str(port)],
                                   cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
        try:
            deadline, health = time.monotonic() + args.startup_timeout, None
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"model API exited; see {log_path}")
                try:
                    # A short readiness request avoids waiting on an unavailable server.
                    with urllib.request.urlopen(url + "/health", timeout=2) as response:
                        health = json.load(response)
                    break
                except (OSError, urllib.error.URLError):
                    time.sleep(.5)
            if not health or health.get("status") != "ok" or not health.get("calibrated"):
                raise RuntimeError(f"exported model did not become ready; see {log_path}")
            first, max_route_delta = check_api(url, payload)
            report = {"status": "passed", "model_path": str(model), "source_run": configuration.get("source_run"),
                      "verified_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
                      "health": health, "response": first, "route_max_probability_delta": max_route_delta,
                      "invalid_schema_checks": len(INVALID_REQUESTS),
                      "weights": [{"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
                                  for path in weights]}
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({"event": "export_verified", "model": str(model), "report": str(report_path),
                              "types": list(payload["questions"]), "status": "passed"}), flush=True)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--model", default="models/chinese-decision-0.8b")
    target.add_argument("--url", help="base URL of a running server, e.g. http://127.0.0.1:8787")
    parser.add_argument("--report", default="reports/export_verification.json")
    parser.add_argument("--gpu", help="CUDA_VISIBLE_DEVICES for the temporary server; inherited by default")
    parser.add_argument("--startup-timeout", type=int, default=180)
    args = parser.parse_args()
    payload = json.loads((ROOT / "examples/request.json").read_text(encoding="utf-8"))
    if args.url:
        url = args.url.rstrip("/")
        health = request(url + "/health")
        first, _ = check_api(url, payload)
        print(json.dumps({"event": "api_verified", "url": url, "health": health, "response": first,
                          "status": "passed"}, ensure_ascii=False), flush=True)
    else:
        verify_model(args, payload)


if __name__ == "__main__":
    main()
