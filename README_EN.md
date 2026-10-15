<div align="center">

# Chinese Decision

**A Chinese probabilistic decision model: one forward pass, straight to candidate probabilities**

[![Base Model](https://img.shields.io/badge/Base-Qwen3.5--0.8B-blue)](https://huggingface.co/Qwen/Qwen3.5-0.8B)
[![Dataset](https://img.shields.io/badge/Data-CJ--Dataset-orange)](https://huggingface.co/datasets/bbldCVer-hf/CJ-Dataset)
[![Benchmark](https://img.shields.io/badge/Bench-CJ--Bench-green)](https://huggingface.co/datasets/bbldCVer-hf/CJ-Bench)
[![Method](https://img.shields.io/badge/Method-hmm-lightgrey)](https://github.com/n4ze3m/hmm)

[中文](README.md) | English

</div>

---

## Introduction

**Chinese Decision** is a lightweight **decision model** for Chinese text. It does not chat or generate text. Given a context and a fixed set of candidates, it returns a probability for each candidate.

Many AI calls in real software are really judgments over a closed set: intent detection, ticket routing, content moderation, data labeling, grading model outputs, or choosing an agent's next action. Answering these with a generative LLM is slow, expensive, and requires parsing free text. A decision model returns a structured probability distribution that code can threshold directly.

The model is built on [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) and fine-tuned with LoRA on [CJ-Dataset](https://huggingface.co/datasets/bbldCVer-hf/CJ-Dataset), a corpus of 13 million Chinese decisions. The method is adapted from [n4ze3m/hmm](https://github.com/n4ze3m/hmm), and results are compared against the [Chinese-Jev](https://arxiv.org/abs/2609.36965) paper on its CJ-Bench benchmark.

**Highlights**

- ⚡ **One forward pass, zero output tokens**: reads the logits of candidate letters `A`–`Z` at the last position and applies softmax. No autoregressive decoding, no chain of thought.
- 🧩 **Three question types**: `noul` (yes/no), `choice` (single choice), and `score` (ordered levels, returns the expected level).
- 🏆 **Compared with Chinese-Jev**: **72.24%** accuracy on the CJ-Bench general domain, above the closed-source Jev (68.35%) and Chinese-Jev General (69.20%). A **single model** beats all three of Chinese-Jev's domain specialists on medical, legal, and finance, averaging **76.98%**, on par with the closed-source Jev (77.01%); its medical accuracy of **88.86%** is the highest among all compared models.
- 📏 **Usable probabilities**: without temperature calibration, overall CJ-Bench ECE is **1.07%**, and the specialized-domain average ECE of **2.46%** is the lowest among all compared models.
- 🪶 **Lightweight deployment**: 0.8B parameters, about 70 ms median single-request latency and about 132 decisions/s batched throughput on one A800, with 6.7 GB peak GPU memory.
- 🔌 **hmm / System One compatible API**: ships with a ready-to-run HTTP server.

## News

- **2026-10-15**: Open-sourced [Decision Studio](https://github.com/yanqiangmiffy/chinese-decision-studio), a test platform for the model with a decision playground, 8 application scenarios, a robustness lab, an evaluation center and load testing, started with one `docker compose up`. It runs against either the GPU service (`scripts/serve_gpu.py`) or the GGUF service on CPU (`scripts/serve.py`).
- **2026-10-15**: Full training finished (1 epoch, 13,048,000 decisions, 10,132 steps). The validation-selected step 10000 reaches 72.24% on the CJ-Bench general domain and a 76.98% specialized-domain average, on par with the closed-source Jev. The weights are merged, and a temperature of T = 1.108 was fitted on a separate validation partition.
- **2026-10-14**: Evaluated the step 5000 checkpoint on the full CJ-Bench test set (307,900 decisions) and compared it with Tables 1 and 2 of the Chinese-Jev paper.

## Model

| Model | Base | Params | Max input | Candidates | Status |
|---|---|---:|---:|---:|---|
| `chinese-decision-0.8b` | Qwen3.5-0.8B | 0.8B | 16,384 tokens | 2–26 | Trained (step 10000) |

## Showcase

The screenshots below come from Decision Studio, our test platform, serving the final step 10000 model (the interface is in Chinese). Every scenario follows the same design: **business rules live in code; the decision model only supplies probabilities**.

### Ticket Triage: Six Judgments in One Request

Each support ticket gets six questions in a single request: category (`choice`); severity and frustration (`score`); and whether it is reproducible, asks for a refund, or signals churn risk (`noul`). All six are asked up front whether or not they end up being used, which saves extra round trips. Routing rules in code turn these probabilities into a queue, and tickets whose category confidence is below 0.5 go to human review. In the screenshot, 8 tickets × 6 questions = 48 decisions take 285 ms.

![Ticket triage: one request judges category, severity, frustration, reproducibility, refund and churn risk; code routes the ticket](assets/scenario-triage.png)

### LLM Guardrail: Safety Checks with a Human-Review Band

Every message into and out of an LLM is checked first. The input side checks for prompt injection / jailbreaks, private information, abuse or threats, and off-topic requests; the output side checks for system-prompt leaks, unauthorized promises, private information in the reply, and inappropriate content. Each check is a yes / no probability with two thresholds: below the review threshold it passes automatically, above the block threshold it is blocked, and anything in between goes to a person, so the width of that band is the human-review budget. In the screenshot, a "developer mode" jailbreak is blocked on the input side (85.2%) before the LLM is ever called; all 8 checks take 89.2 ms.

![LLM guardrail: four input and four output checks, each passed, blocked, or sent to human review by threshold](assets/scenario-guardrail.png)

### Danmaku Guard: Real-Time Moderation of Live-Stream Comments

Each live-stream comment (danmaku) takes one request with one classification and one yes / no judgment: abuse and ads are blocked, spoilers are folded, and viewer questions are highlighted for the host. In the screenshot, median (p50) moderation latency is 82.2 ms. The page also records a known weakness: the spoiler check is swayed by the word "spoiler" itself (as in "no spoilers, please") and cannot tell speculation from an actual spoiler.

![Danmaku guard: blocks abuse and ads, folds spoilers, and highlights questions in real time](assets/scenario-danmaku.png)

### Agent Router: Confidence Gates by Risk Level

For every user message, the decision model first picks a tool, checks whether the request has the information it needs, and estimates the operation's risk; the confidence threshold then depends on the risk tier: 40% for read-only actions (weather, search, chat), 65% for writes (calendar, messages, smart home), and 85% for money (transfers, payments). In the screenshot, "帮我约王总下周二下午聊一下预算" ("set up a budget chat with Mr. Wang next Tuesday afternoon") is routed to the calendar tool with 99% confidence, but the probability that key information is missing is 71%, above the 70% follow-up threshold, so the agent first asks which day, what time, and with whom, in 81.6 ms.

![Agent router: picks a tool, decides whether to ask a follow-up, and gates actions by risk-tiered confidence thresholds](assets/scenario-agent.png)

## How It Works

```mermaid
flowchart LR
    A["context + question + candidates"] --> B["prompt compiler<br/>hmm-zh-v1"]
    B --> C["Qwen3.5-0.8B<br/>+ LoRA"]
    C --> D["last-position logits<br/>A … Z"]
    D --> E["mask unused letters<br/>softmax(z / T)"]
    E --> F["noul / choice / score"]
```

Each question is compiled into the following Chinese prompt and wrapped in the Qwen chat template with thinking disabled:

```text
待判断的上下文：            # Context to judge:
{state}

问题：                      # Question:
{instructions}

候选项：                    # Candidates:
A: {key₁} — {description₁}
B: {key₂} — {description₂}
...
只返回候选项对应的字母。    # Answer with the candidate letter only.
```

Training, evaluation, and serving share the same prompt compiler, so all three see identical inputs.

### Question Types

| Type | `criteria` format | Returned field |
|---|---|---|
| `noul` | Optional `{"false": "...", "true": "..."}`; defaults to “否 / 是” (no / yes) | `noul`: probability of true |
| `choice` | Object `{key: description}` | `choice`: the most probable key |
| `score` | List of levels, ordered low to high | `score`: probability-weighted expected level index (0 to K−1); `legend`: index-to-description map |

Every answer also includes the full `probabilities` and a `confidence` value, where `confidence = (K·p_max − 1) / (K − 1)` measures how sharp the distribution is: 0 for uniform, 1 for certain. It is **not** an empirical accuracy.

## Evaluation

### Comparison with Chinese-Jev

[Chinese-Jev](https://arxiv.org/abs/2609.36965) (Wang et al., 2026) introduced CJ-Dataset and CJ-Bench. All baseline numbers below are **quoted directly from Tables 1 and 2 of that paper**; we did not reproduce them. Our results come from the **step 10000 checkpoint** (the best adapter by validation distribution NLL after the full one-epoch run of 10,132 steps), with no temperature calibration (T = 1). The best result in each column is in **bold**.

**General domain (paper Table 1)**

| Model | ACC (%) ↑ | ECE (%) ↓ | Latency (ms) ↓ |
|---|---:|---:|---:|
| Qwen3.5-2B | 32.95 | 43.68 | 280 |
| Open-Jev | 52.33 | 10.54 | 60 |
| SemIF | 44.21 | 27.27 | 57 |
| Laya Multilingual | 40.22 | 23.04 | **14** |
| Jev † | 68.35 | 11.45 | 284 |
| Chinese-Jev General | 69.20 | 3.78 | **14** |
| **Chinese Decision (ours)** | **72.24** | **3.03** | 70 ‡ |

**Specialized domains (paper Table 2)**

| Model | Medical ACC ↑ | Medical ECE ↓ | Legal ACC ↑ | Legal ECE ↓ | Finance ACC ↑ | Finance ECE ↓ | Avg. ACC ↑ | Avg. ECE ↓ | Avg. Lat. (ms) ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-2B | 53.88 | 28.71 | 31.53 | 44.82 | 38.11 | 49.12 | 41.17 | 40.88 | 322.33 |
| Open-Jev | 73.08 | 2.88 | 44.07 | 23.15 | 48.11 | 10.19 | 55.09 | 12.07 | 78.67 |
| SemIF | 62.17 | 10.39 | 43.07 | 19.99 | 47.89 | 18.45 | 51.04 | 16.28 | 51.33 |
| Laya | 24.55 | 43.05 | 34.72 | 31.49 | 38.31 | 26.18 | 32.53 | 33.57 | **15.00** |
| Jev † | 84.29 | 5.70 | **68.35** | 8.03 | **78.39** | 2.98 | **77.01** | 5.57 | 256.33 |
| Chinese-Jev-G | 39.77 | 17.33 | 38.76 | 15.37 | 35.39 | 14.43 | 37.97 | 15.71 | **15.00** |
| Chinese-Jev-S (3 domain specialists) | 87.65 | 7.25 | 60.10 | 17.45 | 64.72 | 11.55 | 70.82 | 12.08 | **15.00** |
| **Chinese Decision (single model)** | **88.86** | **0.17** | 67.86 | **5.70** | 74.22 | **1.50** | 76.98 | **2.46** | 70 ‡ |

> † Jev is a hosted API; its latency includes network round-trip time.
> ‡ Our latency was measured on one A800 and is the median across all four domains; the paper's models were measured on an H200, so the two are not directly comparable.
> “Avg.” is the unweighted mean over medical, legal, and finance, matching the paper.

**Key findings**

- **General domain**: best accuracy and best ECE in the table, 3.89 points above Jev and 3.04 points above Chinese-Jev General.
- **Specialized domains**: accuracy **beats Chinese-Jev-S in all three domains** (medical +1.21, legal +7.76, finance +9.50 points), and the 76.98% average is on par with Jev (77.01%). Medical is 4.57 points above Jev, the highest in the table; legal is only 0.49 points below Jev, and finance 4.17 below.
- **Calibration**: with no temperature calibration, ECE is 0.17% for medical, 1.50% for finance, 3.03% for general, and 5.70% for legal; the specialized-domain average of 2.46% is still the lowest in the table, about one-fifth of Chinese-Jev-S's.
- **One model**: Chinese-Jev-S is three separately fine-tuned specialists; this project covers all four domains with **one model**.
- **Latency**: our model has roughly 2.5× the parameters of Chinese-Jev (about 322M), and its single-request latency is higher.

**Evaluation protocol**

- Same test set: full CJ-Bench, with 100,000 general, 200,000 medical, 4,300 legal, and 3,600 finance decisions.
- Same metric definitions: ACC and ECE use single-label items only (including hard `score` targets) and exclude soft targets; ECE uses 15 equal-width bins; no calibration is fit on test data.
- Candidates keep their original order, with no filtering and no truncation.

**Method differences from Chinese-Jev**

| | Chinese-Jev | Chinese Decision (ours) |
|---|---|---|
| Backbone | mmBERT-base encoder + decision Transformer (about 322M) | Qwen3.5-0.8B decoder + LoRA |
| Candidate scoring | Shared MLP over candidate-marker positions | Last-position logits of candidate letters |
| Objective | Cross-entropy + RLCD policy-gradient term | Cross-entropy against the candidate distribution (RLCD optional, not used in the released results) |
| Training | General pre-training, then three separately fine-tuned specialists | All four domains mixed into one model |

### Results Across Training

The four best checkpoints selected in turn by validation distribution NLL during the same run, each evaluated on the full CJ-Bench test set under the same protocol (uncalibrated, all 307,900 decisions). The last two rows are quoted from the Chinese-Jev paper for reference.

| Checkpoint | Progress | Val. NLL ↓ | General ACC ↑ | Medical ACC ↑ | Legal ACC ↑ | Finance ACC ↑ | Specialized avg. ACC ↑ | Specialized avg. ECE ↓ | Overall ECE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| step 2000 | 19.8% | 0.739 | 69.04 | 86.78 | 54.07 | 64.17 | 68.34 | 2.73 | 1.63 |
| step 3000 | ~30% | 0.691 | 70.26 | 87.22 | 58.93 | 68.67 | 71.61 | 1.75 | 1.25 |
| step 5000 | ~49% | 0.676 | 70.90 | 87.89 | 63.37 | 71.17 | 74.14 | 1.49 | 0.84 |
| **step 10000 (final)** | ~99% | 0.600 | 72.24 | 88.86 | 67.86 | 74.22 | 76.98 | 2.46 | 1.07 |
| *Reference: Chinese-Jev (G / S)* | — | — | 69.20 | 87.65 | 60.10 | 64.72 | 70.82 | 12.08 | — |
| *Reference: Jev* | — | — | 68.35 | 84.29 | 68.35 | 78.39 | 77.01 | 5.57 | — |

> Progress is the share of one epoch (13,048,000 decisions) trained so far; steps 3000, 5000, and 10000 are estimated from the full run's average of about 1,288 decisions per step (13,048,000 decisions / 10,132 steps). For Chinese-Jev, the general domain uses the General model and the specialized domains use the three specialists (S).

- **Steady gains in every domain**: from step 2000 to step 10000, legal improves by 13.79 points, finance by 10.06, general by 3.20, and medical by 2.08.
- **Ahead of Chinese-Jev-S from step 3000 on**: specialized-domain average accuracy rises from 68.34% to 76.98%, and the gap to Jev (77.01%) narrows from 8.67 to 0.03 points.
- **Calibration does not improve monotonically**: without temperature calibration, overall ECE falls from 1.63% to 0.84% at step 5000 and ends at 1.07%; specialized-domain average ECE is lowest at step 5000 (1.49%) and ends at 2.46%, mostly because legal ECE rises from 2.30% to 5.70%.
- Validation NLL keeps falling (0.739 → 0.600), in the same direction as the test-set accuracy gains.

### Full Results

<details>
<summary><b>CJ-Bench by domain and by type</b></summary>

> Step 10000, uncalibrated (T = 1), single A800 80GB, BF16. ACC and ECE use hard-label items only; NLL and Brier cover every item, including soft labels.

**By domain**

| Domain | Items | ACC ↑ | ECE ↓ | NLL ↓ | Brier ↓ |
|---|---:|---:|---:|---:|---:|
| General | 100,000 | 72.24 | 3.03 | 0.693 | 0.356 |
| Medical | 200,000 | 88.86 | 0.17 | 0.293 | 0.152 |
| Legal | 4,300 | 67.86 | 5.70 | 0.728 | 0.414 |
| Finance | 3,600 | 74.22 | 1.50 | 0.593 | 0.336 |
| **Overall (micro)** | **307,900** | **83.11** | **1.07** | **0.432** | **0.224** |
| **Four-domain average (macro)** | — | **75.80** | — | **0.577** | — |

> Medical makes up 65% of the test set, so the micro average mostly reflects the medical domain. Use the macro average when comparing across domains.

**By question type**

| Type | Items | ACC ↑ | ECE ↓ | NLL ↓ | Expected-level MAE ↓ |
|---|---:|---:|---:|---:|---:|
| `noul` | 134,577 | 94.95 | 0.39 | 0.117 | — |
| `choice` | 134,577 | 78.33 | 1.48 | 0.588 | — |
| `score` | 38,746 | 56.47 | 2.35 | 0.987 | 0.579 |

**Accuracy by domain × type (%)**

| Domain | `noul` | `choice` | `score` |
|---|---:|---:|---:|
| General | 87.57 | 72.88 | 54.66 |
| Medical | 97.81 | 80.57 | 66.53 |
| Legal | 85.80 | 64.13 | 51.46 |
| Finance | 76.50 | 63.08 | 83.08 |

</details>

<details>
<summary><b>Before vs. after fine-tuning (validation)</b></summary>

On a stratified subset of the official validation set (4 domains × 3 types, 100 items each, 1,200 total), zero-shot base model vs. step 10000:

| Domain | Qwen3.5-0.8B ACC | **Chinese Decision ACC** | Qwen3.5-0.8B ECE | **Chinese Decision ECE** |
|---|---:|---:|---:|---:|
| General | 36.64 | **75.34** | 28.25 | **3.76** |
| Medical | 16.67 | **82.00** | 43.40 | **3.25** |
| Legal | 40.67 | **65.33** | 24.03 | **7.62** |
| Finance | 44.33 | **71.33** | 16.77 | **3.41** |
| **Overall** | 34.56 | **73.49** | 27.45 | **2.48** |

</details>

### Inference Efficiency

| Metric | Value |
|---|---:|
| Single-request latency (batch = 1, warmed, median) | 70.1 ms |
| Batched throughput | 132 decisions/s |
| Peak GPU memory (batched evaluation) | 6.7 GB |
| Mean CJ-Bench input length | 223 tokens |

> Measured on one A800 80GB PCIe in BF16. Latency covers local input encoding, the forward pass, and probability output; it excludes HTTP and network time.

## Quick Start

### Installation

```bash
git clone https://github.com/yanqiangmiffy/chinese-decision.git
cd chinese-decision
pip install -e ".[train]"
```

Python ≥ 3.11 is required. `[train]` installs the CUDA training/inference dependencies. For CPU GGUF serving use `pip install -e ".[gguf]"`; PyTorch, PEFT and flash-linear-attention are not required. Transformers-only inference uses `.[torch]`. If your environment already has a CUDA build of PyTorch that pip should leave alone, use `pip install -e . --no-deps`.

### Models and Data

Models and data are loaded from local directories. You can fetch them with the Hugging Face CLI (the CJ datasets are currently gated; run `hf auth login` first):

```bash
hf download Qwen/Qwen3.5-0.8B --local-dir models/Qwen3.5-0.8B
```

```bash
hf download bbldCVer-hf/CJ-Dataset --repo-type dataset --local-dir datasets/CJ-Dataset
```

```bash
hf download bbldCVer-hf/CJ-Bench --repo-type dataset --local-dir datasets/CJ-Bench
```

### Python

```python
import json
from pathlib import Path

import torch
from decision.model import forward_batch, letter_tokens, load_model, load_tokenizer
from decision.schema import compile_prompt, options_for

base = "models/chinese-decision-0.8b"  # merged model; or load_model("models/Qwen3.5-0.8B", adapter="runs/cj-ddp/best")
tokenizer = load_tokenizer(base)
model = load_model(base).eval()
letters = letter_tokens(tokenizer, "cuda")
config = Path(base, "decision_config.json")
temperature = json.loads(config.read_text())["temperature"] if config.exists() else 1.0

state = "客户留言：付款后一直没有收到发票，我已经催了三次，今天必须拿到。"
question = {
    "type": "choice",
    "instructions": "应该由哪个部门处理？",
    "criteria": {"billing": "支付、发票及退款", "technical": "故障和系统问题", "sales": "购买咨询与升级"},
}

options = options_for(question)
prompt = compile_prompt(tokenizer, state, question["instructions"], options)
item = {"input_ids": tokenizer.encode(prompt, add_special_tokens=False), "target": [0.0] * len(options)}
with torch.inference_mode():
    logits, _ = forward_batch(model, tokenizer, letters, [item])
probs = (logits[0, :len(options)] / temperature).softmax(-1).tolist()
print({key: round(p, 4) for (key, _), p in zip(options, probs)})
```

### HTTP Server

```bash
python scripts/serve.py --base models/chinese-decision-0.8b --port 8787
```

For CPU Q8_0 serving, install `.[gguf]`, export the model using
`scripts/convert_gguf.py`, and run:

```bash
python scripts/serve.py --backend gguf --base models/chinese-decision-0.8b-gguf --gguf models/chinese-decision-0.8b-gguf/chinese-decision-0.8b-Q8_0.gguf --threads 8 --context-size 16384 --port 8787
```

See [CPU deployment](docs/CPU.md) for conversion and benchmark instructions.
The existing routes and probability semantics are preserved. `--device cpu`
selects a Transformers FP32 baseline. Requests exceeding the effective runtime
context limit return HTTP 422 without truncation.

You can also load the base model with an adapter: `--base models/Qwen3.5-0.8B --adapter runs/cj-ddp/best` (temperature is then 1, i.e. uncalibrated).

One request can carry several questions:

```bash
curl http://127.0.0.1:8787/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @examples/request.json
```

```json
{
  "state": "客户留言：付款后一直没有收到发票，我已经催了三次，今天必须拿到。",
  "questions": {
    "urgent": {"type": "noul", "instructions": "客户是否表达了紧迫性？"},
    "department": {"type": "choice", "instructions": "应该由哪个部门处理？",
                   "criteria": {"billing": "支付、发票及退款", "technical": "故障和系统问题", "sales": "购买咨询与升级"}},
    "frustration": {"type": "score", "instructions": "客户有多不满？",
                    "criteria": ["平静", "不满", "非常愤怒"]}
  }
}
```

Response format (values are illustrative only):

```json
{
  "model": "chinese-decision-0.8b",
  "answers": {
    "urgent": {"type": "noul", "probabilities": {"false": 0.04, "true": 0.96}, "confidence": 0.92, "noul": 0.96},
    "department": {"type": "choice", "probabilities": {"billing": 0.97, "technical": 0.02, "sales": 0.01},
                   "confidence": 0.955, "choice": "billing"},
    "frustration": {"type": "score", "probabilities": {"0": 0.02, "1": 0.38, "2": 0.60}, "confidence": 0.40,
                    "score": 1.58, "legend": {"0": "平静", "1": "不满", "2": "非常愤怒"}}
  },
  "usage": {"input_tokens": 242, "output_tokens": 0},
  "latency_ms": 85.3
}
```

| Endpoint | Description |
|---|---|
| `POST /v1/systemone` | Decision endpoint; `?debug=1` adds each candidate's logit and each question's token count |
| `POST /api/alpha/decisions` | Same as above; hmm-compatible route |
| `POST /v1/batch` | `{"requests": [...], "debug": bool}`: several requests at once; one invalid request does not fail the others |
| `POST /v1/compile` | Compile prompts and count tokens without running the model |
| `GET /v1/info` | Checkpoint, validation metrics and serving statistics |
| `GET /v1/training` | Training progress |
| `GET /health` | Health check; reports the temperature, whether the model is calibrated and the checkpoint step |

A request may contain up to 64 questions, each with 2–26 candidates and at most 16,384 input tokens. Over-length inputs are **never truncated**; invalid requests return `422`.

### Deployment

Two server entry points share the same API and response format; either can back [Decision Studio](https://github.com/yanqiangmiffy/chinese-decision-studio) (the test platform) as its `DECISION_BASE_URL`:

| Entry point | Use | Deployment |
|---|---|---|
| `scripts/serve.py` | CPU (GGUF) or local debugging; one forward pass per question | [docs/CPU.md](docs/CPU.md), [deploy/gguf](deploy/gguf/) |
| `scripts/serve_gpu.py` | Production GPU serving: dynamic batching, training progress, checkpoint hot reload (`POST /v1/admin/reload`) | [deploy/gpu](deploy/gpu/README.md) |

To check a running server (both routes, probability format, and invalid-request handling):

```bash
python scripts/verify_export.py --url http://127.0.0.1:8787
```

## Training

### Configuration

| Item | Setting |
|---|---|
| Base model | Qwen3.5-0.8B (text backbone) |
| Fine-tuning | LoRA, r = 32, α = 64, dropout = 0.05 |
| LoRA targets | Full attention `q/k/v/o_proj`, linear attention `in_proj_qkv/z/a/b` and `out_proj`, MLP `gate/up/down_proj` |
| Loss | Cross-entropy against the candidate distribution (soft labels; hard labels are the one-hot case); optional RLCD term, off by default |
| Optimizer | AdamW, lr = 1e-4, 100-step linear warmup, then cosine decay to 10% of peak |
| Global batch | Each step accumulates at least 1,024 decisions; each micro-batch holds at most 262,144 padded tokens and shrinks automatically for long inputs |
| Max length | 16,384 tokens |
| Training budget | 1 epoch, 13,048,000 decisions, 10,132 steps |
| Precision | BF16, gradient checkpointing |
| Model selection | Validation every 1,000 steps; best checkpoint by distribution NLL (step 10000 in the end) |
| Temperature calibration | One temperature fitted on a separate validation partition, T = 1.108 |
| Hardware | 4 × GPU, DDP |

### Data and Training Notes

- **Soft labels**: CJ's `gold` is a probability distribution over candidates, used directly as the cross-entropy target.
- **Candidate shuffling**: `choice` candidates are shuffled during training, with targets permuted to match, to reduce position bias. `score` keeps its level order, and `noul` stays false / true.
- **Balanced mixing**: data is re-sharded by domain, source, and type, then interleaved in proportion to the remaining records, so task order in the raw files does not cause forgetting. The corpus is streamed and never fully loaded into memory.
- **Length bucketing**: batches are formed by token count. Items longer than 16,384 tokens are counted and filtered explicitly; context and questions are never truncated.
- **Leak-free calibration**: the official validation set is split by parent record; 80% is used for validation and model selection, and the other 20% only for fitting the temperature.

### Pipeline

```bash
# 1. Verify dataset SHA256 checksums, then re-shard training data by domain / source / type
python scripts/prepare_data.py --data datasets/CJ-Dataset --out datasets/CJ-mixed

# 2. LoRA training (4 GPUs shown; for one GPU run `python scripts/train.py`; add --resume to continue)
torchrun --standalone --nproc_per_node=4 scripts/train.py --out runs/cj-ddp

# 3. Temperature calibration + weight merge + HTTP check of the exported model
python scripts/finalize.py --run runs/cj-ddp --out models/chinese-decision-0.8b
```

The defaults of `scripts/train.py` are the release configuration in the table above. `--resume` compares every key of `config.json` and refuses to continue with different arguments; the number of GPUs may change. For a quick debug run, add `--debug-examples 512 --max-steps 12 --global-batch-size 8`.

In the run directory, `last/` is the resume point, `best/` is the best adapter on validation, and `final/` is the adapter after the last step. The exported model directory contains the merged weights, `decision_config.json` (calibrated temperature and limits), the training config and summary, and the base model's `LICENSE`.

### Optional: RLCD Objective

`--rlcd` adds Chinese-Jev's RLCD term (a policy gradient over a proper-scoring-rule reward) to the distribution cross-entropy. It is ported from the [official Chinese-Jev code](https://github.com/gulucaptain/Chinese-Jev) and reproduces its results bit for bit. **It is off by default; none of the results on this page use RLCD.**

Each training step perturbs the candidate logits with several zero-mean Gaussian draws, scores every perturbed distribution with "log score + 0.75 × spherical score − RPS" (RPS applies to `score` questions only), and pushes the logits toward the perturbations that score above their group's average. RPS penalizes errors more the further they are from the correct level; it is the only term that knows level order, so `score` questions are the most likely to benefit.

| Argument | Default | Meaning |
|---|---|---|
| `--rlcd` | off | Add the RLCD term to the cross-entropy, weighted 1:1 |
| `--rlcd-group-size` | 4 | Perturbations per decision (at least 2) |
| `--rlcd-sigma-start` | 0.4 | Logit noise scale at the start of training |
| `--rlcd-sigma-end` | 0.1 | Noise scale at the end, annealed linearly over training progress |

- **Difference from the original**: Chinese-Jev anneals σ per epoch (four epochs: 0.4 / 0.3 / 0.2 / 0.1). This project trains for one epoch, so σ is annealed linearly over training progress instead. σ is noise added directly to the logits, and an LLM's letter logits may have a different scale from Chinese-Jev's scoring head, so the defaults are only a starting point.
- **Config and resuming**: with `--rlcd`, `config.json` records the RLCD settings and `training_algorithm` gets a `+rlcd` suffix, so cross-entropy runs and RLCD runs cannot resume each other. Without it, the config is exactly as before and existing runs resume normally.
- **Logs**: each step also reports `ce`, `rlcd`, `reward`, and `sigma`.

**Ablation**: start a new run from scratch with the same arguments plus `--rlcd`. Both runs share the data order, validation set, and LoRA initialization; only the objective differs.

```bash
torchrun --standalone --nproc_per_node=4 scripts/train.py --out runs/cj-ddp-rlcd --rlcd
```

Do not pass `--max-steps`, which would compress the learning-rate and σ schedules and make the runs incomparable. Stop after the validation at step 1000 or 2000 and compare both runs at the same step (each validation writes `latest_metrics.json`, including per-type results), focusing on `type:score`.

## Benchmarking

```bash
# 1. Copy the validation-selected best adapter into a fresh snapshot directory
python scripts/snapshot_best.py --source-run runs/cj-ddp --out runs/cj-bench/my-eval

# 2. Quick evaluation (1,000 items per domain × type), then the full benchmark
python scripts/evaluate.py --snapshot-run runs/cj-bench/my-eval --data datasets/CJ-Bench
```

- Quick evaluation only: add `--quick-only`.
- Evaluate the merged deployment model (with its validation-fitted temperature): add `--model models/chinese-decision-0.8b`.
- The script never overwrites existing results; use a new snapshot directory for each run. Calibration and model selection use validation data only, never test labels.

## Repository Layout

```text
chinese-decision/
├── decision/               # Core library (pip install -e .)
│   ├── schema.py           #   question parsing, target distributions, prompt compiler
│   ├── data.py             #   data sharding, streaming mixing, token batching
│   ├── model.py            #   model loading and candidate-letter logits
│   ├── inference.py        #   serving backends: Transformers and GGUF (llama.cpp)
│   ├── gguf_batch.py       #   padding-free llama.cpp batches for evaluation
│   ├── training.py         #   validation, LoRA targets, checkpoint helpers
│   ├── rlcd.py             #   optional RLCD training term (ported from Chinese-Jev)
│   ├── metrics.py          #   validation and CJ-Bench metrics
│   ├── api.py              #   decision response format and checks
│   └── utils.py            #   logging, atomic writes, SHA256
├── scripts/                # Command-line entry points
│   ├── prepare_data.py     #   dataset verification and re-sharding
│   ├── train.py            #   LoRA training (one GPU / torchrun)
│   ├── finalize.py         #   calibration, merge, export checks
│   ├── snapshot_best.py    #   freeze the best adapter for evaluation
│   ├── evaluate.py         #   CJ-Bench evaluation
│   ├── convert_gguf.py     #   GGUF export (BF16 / Q8_0)
│   ├── evaluate_gguf.py    #   CJ-Bench with the GGUF weights
│   ├── benchmark_*.py      #   CPU latency and throughput; compare_gguf_benchmark.py pairs two runs
│   ├── verify_export.py    #   end-to-end HTTP API check
│   ├── serve.py            #   HTTP server: CPU / GGUF, local debugging
│   └── serve_gpu.py        #   HTTP server: production GPU (dynamic batching, hot reload)
├── deploy/
│   ├── gpu/                # GPU deployment (local deploy.sh → tmux on the server)
│   └── gguf/               # CPU deployment (push / install / service)
├── docs/CPU.md             # CPU deployment and GGUF validation
├── tests/                  # Unit tests (no GPU needed)
├── examples/request.json   # Example request
├── assets/                 # README images
├── pyproject.toml
├── LICENSE
├── LICENSES/               # Full texts of third-party licenses
└── THIRD_PARTY_NOTICES.md
```

Run the tests:

```bash
python -m unittest discover -s tests
```

## Limitations

- **CJ-Bench results are uncalibrated**: they come from the step 10000 adapter before merging (T = 1). The exported merged model applies the validation-fitted temperature T = 1.108 by default, and that configuration has not been evaluated on CJ-Bench separately.
- **Finance and `score` still lag**: finance trails the closed-source Jev by 4.17 accuracy points; the `score` type is weaker overall (56.47% overall, 51.46% on legal `score`); and legal has the highest ECE of the four domains (5.70%).
- **Slower than encoder models**: the 0.8B decoder has higher single-request latency than the roughly 322M Chinese-Jev encoder.
- **`confidence` is not accuracy**: it only measures how sharp the distribution is. Before acting automatically on a threshold, check calibration on your own data.
- **Label semantics follow the upstream tasks**: for example, medical matching labels describe the original matching task and do not imply medical correctness. Do not use model outputs as the sole basis for medical, legal, or financial decisions.
- **Trained for Chinese only**, and not suited to open-ended generation.
- This project is a Chinese adaptation of the hmm method. It does not reproduce the training of Microsoft Decision, Jev, or Chinese-Jev.

## Acknowledgements

- [Chinese-Jev](https://arxiv.org/abs/2609.36965): CJ-Dataset, CJ-Bench, the baseline results used in our comparison, and the [reference implementation](https://github.com/gulucaptain/Chinese-Jev) of the RLCD term (Apache-2.0).
- [n4ze3m/hmm](https://github.com/n4ze3m/hmm): option-letter decision objective, LoRA workflow, candidate formatting, and System One response conventions (MIT).
- [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B): base model (Apache-2.0).

## License

- This project's code is released under the [MIT License](LICENSE).
- Portions adapted from hmm are also under the MIT License; the original notice is in [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md).
- `decision/rlcd.py` is adapted from Chinese-Jev under Apache-2.0; the full license text is in [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt).
- The base model Qwen3.5-0.8B is under Apache-2.0; exported model directories keep its original `LICENSE`.
- CJ-Dataset and CJ-Bench licenses are source-specific (some sources are non-commercial). This project grants no additional redistribution rights.

## Citation

```bibtex
@misc{chinese-decision-2026,
  title        = {Chinese Decision: A Lightweight Chinese Probabilistic Decision Model},
  author       = {yanqiangmiffy},
  year         = {2026},
  howpublished = {\url{https://github.com/yanqiangmiffy/chinese-decision}}
}
```

If you use CJ-Dataset or CJ-Bench, please also cite Chinese-Jev:

```bibtex
@misc{wang2026chinesejev,
  title         = {Chinese-Jev: Bringing System One Model to Chinese-Language Tasks},
  author        = {Wang, Zexiao and Zhang, Zihao and Wang, Xudong and Wang, Pan and Ye, Ziyi and Zhao, Haoyu and Wu, Zuxuan and Yan, Shuicheng},
  year          = {2026},
  eprint        = {2609.36965},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}
```
