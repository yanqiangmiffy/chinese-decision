"""Mirror a training run's JSON log into SwanLab, offline, without touching the training process.

scripts/train.py prints one JSON event per line (step, validation, checkpoint, ...). This script reads
that log (and follows it while training runs), turns the events into SwanLab metrics and writes them
with `mode="offline"`, so a machine without internet keeps a complete local record:

    /opt/anaconda3/bin/python scripts/swanlab_follow.py --run runs/pointer-4b-v1 --log runs/pointer-4b-v1.log --follow

Then copy `<run>/swanlog` to a computer with internet and upload it with `swanlab sync <swanlog/run-...>`,
or look at it locally with `swanlab watch <swanlog>`. Only the standard library and swanlab are needed,
so it runs from any Python that has swanlab, separate from the training environment.

Per-source validation is logged by default for runs with <run>/metrics_history.jsonl.

Metrics: train/* every logged step (loss, cross-entropy, RLCD term and reward, sigma, learning rate,
gradient norm, GPU peak memory, examples seen, hours); val/* at every validation, overall and per
domain and question type (from `<run>/latest_metrics.json`). A training restart (`--resume`)
overwrites the log; the follower then continues after the last step it recorded.
"""
import argparse
import calendar
import itertools
import json
import os
import time
from pathlib import Path

VAL_FIELDS = ("accuracy", "ece", "distribution_nll", "hard_nll", "brier")


def events(path, follow, poll, stop):
    """JSON events of `path`, waiting for more when following; reopens a log that was rewritten."""
    position, inode = 0, None
    while True:
        try:
            stat = os.stat(path)
        except FileNotFoundError:
            if not follow:
                return
            time.sleep(poll)
            continue
        if stat.st_size < position or (inode is not None and getattr(stat, "st_ino", None) != inode):
            position = 0                                    # the log was overwritten by a resumed run
        inode = getattr(stat, "st_ino", None)
        with open(path, encoding="utf-8", errors="replace") as stream:
            stream.seek(position)
            while True:
                line = stream.readline()
                if not line or not line.endswith("\n"):
                    break
                position = stream.tell()
                line = line.replace("\0", "")     # a log truncated under a live writer has a NUL-filled gap
                if line.startswith("{"):
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        if not follow or stop():
            return
        time.sleep(poll)


def history(run, step):
    """Per-group validation metrics of `step` from <run>/metrics_history.jsonl (written by newer runs), or None."""
    path = Path(run) / "metrics_history.jsonl"
    if not path.exists():
        return None
    found = None
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("step") == step:
            found = entry["metrics"]
    return found


def detail(run, after, timeout=120):
    """Per-domain/type validation metrics written after `after` (epoch seconds), or {}."""
    path = Path(run) / "latest_metrics.json"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.exists() and path.stat().st_mtime >= after - 5:
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        time.sleep(2)
    return {}


def validation_metrics(metrics, prefix="val"):
    values = {f"{prefix}/{k}": metrics[k] for k in VAL_FIELDS if isinstance(metrics.get(k), (int, float))}
    values[f"{prefix}/n"] = metrics.get("n")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="training output directory (config.json, latest_metrics.json)")
    parser.add_argument("--log", required=True, help="the training log scripts/train.py wrote to")
    parser.add_argument("--prefix-log", nargs="*", default=[],
                        help="logs read once before --log, e.g. events recovered after the log was overwritten")
    parser.add_argument("--project", default="chinese-decision")
    parser.add_argument("--name", help="experiment name (default: the run directory name)")
    parser.add_argument("--mode", default="offline", choices=("offline", "local", "online", "disabled"))
    parser.add_argument("--log-dir", help="SwanLab log directory (default: <run>/swanlog)")
    parser.add_argument("--follow", action="store_true", help="keep reading until the run completes")
    parser.add_argument("--poll", type=float, default=20)
    parser.add_argument("--per-source", action="store_true", help="also log validation per source")
    args = parser.parse_args()

    import swanlab
    run = Path(args.run)
    config = json.loads((run / "config.json").read_text(encoding="utf-8")) if (run / "config.json").exists() else {}
    swanlab.init(project=args.project, name=args.name or run.name, mode=args.mode,
                 log_dir=str(args.log_dir or run / "swanlog"), config=config,
                 description=f"mirrored from {args.log}")
    last_step, validated, done = -1, set(), [False]
    has_history = (run / "metrics_history.jsonl").exists() or bool(config.get("val_per_source"))
    started = None
    stream = itertools.chain(*(events(path, False, args.poll, lambda: True) for path in args.prefix_log),
                             events(args.log, args.follow, args.poll, lambda: done[0]))
    for event in stream:
        kind = event.get("event")
        if kind == "step":
            step = event["step"]
            if step <= last_step:
                continue
            last_step = step
            started = started or time.time() - event.get("seconds", 0)
            peaks = event.get("gpu_peak_gb") or [0]
            values = {"train/loss": event.get("loss"), "train/ce": event.get("ce"), "train/rlcd": event.get("rlcd"),
                      "train/reward": event.get("reward"), "train/sigma": event.get("sigma"),
                      "train/lr": event.get("lr"), "train/grad_norm": event.get("grad_norm"),
                      "train/gpu_peak_gb": max(peaks), "train/examples_seen": event.get("examples_seen"),
                      "train/batch_size": event.get("batch_size"),
                      "train/hours": round(event.get("seconds", 0) / 3600, 3)}
            swanlab.log({k: v for k, v in values.items() if isinstance(v, (int, float))}, step=step)
        elif kind in ("baseline", "validation"):
            step = 0 if kind == "baseline" else event["step"]
            if step in validated:
                continue
            validated.add(step)
            values = validation_metrics(event["metrics"] if kind == "validation" else event["metrics"])
            if kind == "validation":
                parsed = calendar.timegm(time.strptime(event["time"][:19], "%Y-%m-%dT%H:%M:%S"))  # UTC
                full = history(run, step)
                if full is None:
                    full = detail(run, parsed) if args.follow else {}
                for key, metrics in full.items():
                    group, _, name = key.partition(":")
                    if group in ("domain", "type") or (group == "source" and (args.per_source or has_history)):
                        values.update(validation_metrics(metrics, f"val_{group}/{name.replace('/', '_')}"))
            swanlab.log({k: v for k, v in values.items() if isinstance(v, (int, float))}, step=step)
            for name, value in (event.get("selection") or {}).items():
                if isinstance(value, (int, float)):
                    swanlab.log({f"val/select_{name}": value}, step=step)
        elif kind == "resume":
            swanlab.log({"train/resumed_at": event.get("step", 0)}, step=max(last_step, 0) + 1)
        elif kind == "completed":
            done[0] = True
    swanlab.finish()


if __name__ == "__main__":
    main()
