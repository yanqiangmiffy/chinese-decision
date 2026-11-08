"""Average training checkpoints of one run into a new checkpoint that scripts/finalize.py can export.

    python scripts/average_checkpoints.py --run runs/pointer-4b-v2 --checkpoints step-1400 step-1600 final \\
        --out avg-1400-1600-final
    python scripts/finalize.py --run runs/pointer-4b-v2 --checkpoint avg-1400-1600-final --out models-v2/...

LoRA adapters are averaged exactly: the mean of the weight updates sum_i w_i * B_i A_i is itself a LoRA of rank
n*r, with A stacked over checkpoints and B concatenated with the weights folded in, so the merged model
equals the average of the merged checkpoints. (Averaging A and B separately would not.) The scale
lora_alpha / r is kept by multiplying lora_alpha by n. The small pointer head is averaged parameter-wise.
"""
import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

COPIED = ("chat_template.jinja", "tokenizer.json", "tokenizer_config.json", "README.md")


def average_adapters(states, weights):
    """{key: tensor} of the rank-stacked average of LoRA state dicts (other tensors are averaged)."""
    out = {}
    for key in states[0]:
        tensors = [s[key] for s in states]
        if "lora_A" in key:
            out[key] = torch.cat(tensors, dim=0).contiguous()
        elif "lora_B" in key:
            out[key] = torch.cat([w * t.float() for w, t in zip(weights, tensors)], dim=1).to(tensors[0].dtype).contiguous()
        else:
            out[key] = sum(w * t.float() for w, t in zip(weights, tensors)).to(tensors[0].dtype)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True, help="checkpoint directories inside the run")
    parser.add_argument("--weights", nargs="*", type=float, help="default: equal weights")
    parser.add_argument("--out", required=True, help="new checkpoint directory name inside the run")
    args = parser.parse_args()
    run = Path(args.run)
    weights = args.weights or [1.0] * len(args.checkpoints)
    if len(weights) != len(args.checkpoints) or min(weights) < 0:
        parser.error("--weights needs one non-negative weight per checkpoint")
    weights = [w / sum(weights) for w in weights]
    out = run / args.out
    if out.exists():
        raise FileExistsError(out)
    configs = [json.loads((run / c / "adapter_config.json").read_text()) for c in args.checkpoints]
    def comparable(config):     # target_modules is a set, saved in no fixed order (it differs after a reload)
        return {k: sorted(v) if k == "target_modules" and isinstance(v, list) else v for k, v in config.items() if k != "revision"}
    for c in configs[1:]:
        if comparable(c) != comparable(configs[0]):
            raise ValueError("checkpoints have different adapter configs")
    if configs[0].get("use_rslora") or configs[0].get("use_dora") or configs[0].get("rank_pattern") or configs[0].get("alpha_pattern"):
        raise ValueError("only plain LoRA adapters can be stacked")
    states = [load_file(str(run / c / "adapter_model.safetensors")) for c in args.checkpoints]
    if any(set(s) != set(states[0]) for s in states):
        raise ValueError("checkpoints have different adapter tensors")
    n = len(states)
    tmp = run / (args.out + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    save_file(average_adapters(states, weights), str(tmp / "adapter_model.safetensors"))
    config = dict(configs[0], r=configs[0]["r"] * n, lora_alpha=configs[0]["lora_alpha"] * n)
    (tmp / "adapter_config.json").write_text(json.dumps(config, indent=2))
    heads = [run / c / "pointer_head.pt" for c in args.checkpoints]
    if all(h.exists() for h in heads):
        loaded = [torch.load(h, map_location="cpu") for h in heads]
        state = {k: sum(w * h["state_dict"][k].float() for w, h in zip(weights, loaded)).to(loaded[0]["state_dict"][k].dtype)
                 for k in loaded[0]["state_dict"]}
        torch.save({"state_dict": state, "config": loaded[0]["config"]}, tmp / "pointer_head.pt")
    elif any(h.exists() for h in heads):
        raise ValueError("only some checkpoints have a pointer head")
    for name in COPIED:
        if (run / args.checkpoints[0] / name).exists():
            shutil.copy2(run / args.checkpoints[0] / name, tmp / name)
    progress = json.loads((run / "progress.json").read_text()) if (run / "progress.json").exists() else {}
    steps = [int(c.split("-", 1)[1]) if c.startswith("step-") else progress.get("step") for c in args.checkpoints]
    (tmp / "selection.json").write_text(json.dumps({"step": max(s for s in steps if s is not None),
                                                    "metric": "checkpoint average", "checkpoints": args.checkpoints,
                                                    "steps": steps, "weights": weights, "rank": config["r"]}))
    tmp.rename(out)
    print(json.dumps({"event": "averaged", "out": str(out), "checkpoints": args.checkpoints, "weights": weights,
                      "rank": config["r"]}))


if __name__ == "__main__":
    main()
