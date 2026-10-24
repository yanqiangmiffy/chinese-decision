<div align="center">

# Chinese Decision

**A Chinese probabilistic decision model: one forward pass, straight to candidate probabilities**

[![Base Models](https://img.shields.io/badge/Base-Qwen3.5%200.8B%20%7C%202B%20%7C%204B-blue)](docs/MODELS.md)
[![Dataset](https://img.shields.io/badge/Data-CJ--Dataset-orange)](https://huggingface.co/datasets/bbldCVer-hf/CJ-Dataset)
[![Benchmark](https://img.shields.io/badge/Bench-CJ--Bench-green)](https://huggingface.co/datasets/bbldCVer-hf/CJ-Bench)
[![Method](https://img.shields.io/badge/Method-hmm-lightgrey)](https://github.com/n4ze3m/hmm)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-chinese--decision--4b-yellow?logo=huggingface)](https://huggingface.co/quincyqiang/chinese-decision-4b)
[![ModelScope](https://img.shields.io/badge/ModelScope-chinese--decision--4b-624aff)](https://modelscope.cn/models/quincyqiang/chinese-decision-4b)
[![GGUF](https://img.shields.io/badge/GGUF-Q8__0%20%C2%B7%20CPU-brightgreen)](https://huggingface.co/quincyqiang/chinese-decision-4b-gguf)
[![Decision Studio](https://img.shields.io/badge/Platform-Decision%20Studio-black?logo=github)](https://github.com/yanqiangmiffy/chinese-decision-studio)

[中文](README.md) | English

</div>

---

## Introduction

**Chinese Decision** is a lightweight **decision model** for Chinese text. It does not chat or generate text. Given a context and a fixed set of candidates, it returns a probability for each candidate.

Many AI calls in real software are really judgments over a closed set: intent detection, ticket routing, content moderation, data labeling, grading model outputs, or choosing an agent's next action. Answering these with a generative LLM is slow, expensive, and requires parsing free text. A decision model returns a structured probability distribution that code can threshold directly.

The model family is built on [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B), [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) and [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), fine-tuned with LoRA on [CJ-Dataset](https://huggingface.co/datasets/bbldCVer-hf/CJ-Dataset), a corpus of 13 million Chinese decisions. The method is adapted from [n4ze3m/hmm](https://github.com/n4ze3m/hmm), and results are compared against the [Chinese-Jev](https://arxiv.org/abs/2609.36965) paper on its CJ-Bench benchmark.

**Highlights**

- ⚡ **One forward pass, zero output tokens**: reads the logits of candidate letters `A`–`Z` at the last position and applies softmax. No autoregressive decoding, no chain of thought.
- 🧩 **Three question types**: `noul` (yes/no), `choice` (single choice), and `score` (ordered levels, returns the expected level).
- 🔢 **More than 26 categories**: one question maps each candidate to a letter (A–Z; training covers 2–16). `choice` questions with more categories go through [`/v1/tournament`](#http-server): grouped rounds and a final, every group within the trained range, no retraining.
- 🏆 **Three sizes, one model for all four domains**: chinese-decision-4b reaches **85.97%** overall accuracy on the full CJ-Bench and **84.00%** across medical / legal / finance, with higher accuracy in every domain than the paper's Chinese-Jev (specialized average 70.82%) and the closed-source Jev (77.01%); chinese-decision-2b reaches **84.11% / 78.25%** and chinese-decision-0.8b **83.11% / 76.90%**.
- 📏 **Calibrated probabilities**: the released weights ship a validation-fitted temperature; overall ECE is **1.00%** for 4B, **0.64%** for 2B and **0.27%** for 0.8B, and all three sizes have lower ECE than every model in the paper in all four domains.
- 🪶 **GPU and CPU**: every size ships BF16 and Q8_0 GGUF packages. chinese-decision-4b-gguf is **4.48 GB** (about 46.7% smaller than BF16), reaches **85.99%** on the full CJ-Bench, and agrees with BF16 on the top choice **99.19%** of the time; it runs on CPU through llama.cpp without PyTorch.
- 🔌 **hmm / System One compatible API**: ships with a ready-to-run HTTP server.

## News

- 🚀 **2026-10-21**: Released **chinese-decision-4b** (🤗 [Hugging Face](https://huggingface.co/quincyqiang/chinese-decision-4b) · [ModelScope](https://modelscope.cn/models/quincyqiang/chinese-decision-4b)) and the CPU build **chinese-decision-4b-gguf** (Q8_0, 4.48 GB; 🤗 [Hugging Face](https://huggingface.co/quincyqiang/chinese-decision-4b-gguf) · [ModelScope](https://modelscope.cn/models/quincyqiang/chinese-decision-4b-gguf)). 4B reaches **85.97%** overall accuracy, an **84.00%** specialized average and **1.00%** ECE, and beats both 0.8B and 2B in all 12 domain × type cells; Q8_0 reaches **85.99%** on the full benchmark with **99.19%** top-choice agreement with BF16. Checkpoint results from training, for every size, now live in [Training checkpoint evaluations](docs/TRAINING_HISTORY.md) (Chinese).
- 🔢 **2026-10-18**: Added the tournament endpoint [`/v1/tournament`](#http-server), so a `choice` question is no longer limited to the 26 letters A–Z (the server caps it at 1024 by default, set by `MAX_CANDIDATES` in `decision/tournament.py`). Candidates are shuffled into groups of at most 16 and the leaders of each group meet in a final, with no retraining; both the GPU and CPU services support it. [examples/tournament.json](examples/tournament.json) has 41 customer-service intents; on the GPU service, 10 different messages were all classified correctly, each in 4 forward passes and about 150 ms.
- 🚀 **2026-10-17**: Released **chinese-decision-2b** (🤗 [Hugging Face](https://huggingface.co/quincyqiang/chinese-decision-2b) · [ModelScope](https://modelscope.cn/models/quincyqiang/chinese-decision-2b)) and the CPU build **chinese-decision-2b-gguf** (Q8_0, 2.01 GB; 🤗 [Hugging Face](https://huggingface.co/quincyqiang/chinese-decision-2b-gguf) · [ModelScope](https://modelscope.cn/models/quincyqiang/chinese-decision-2b-gguf)). 2B reaches **84.11%** overall accuracy, a **78.25%** specialized average and **0.64%** ECE; Q8_0 reaches **84.11%** on the full benchmark with **99.00%** top-choice agreement with BF16. CPU checks: [2B CPU deployment](docs/CPU_2B.md) (Chinese).
- 📊 **2026-10-17**: Qwen3.5-2B + LoRA + RLCD finished one epoch of training (13,048,000 decisions). Training configuration, export and source diagnostics are in the [2B release record](docs/TRAINING_2B.md) (Chinese).
- 🧪 **2026-10-15**: Open-sourced [Decision Studio](https://github.com/yanqiangmiffy/chinese-decision-studio), a test platform for the model with a decision playground, 8 application scenarios, a robustness lab, an evaluation center and load testing, started with one `docker compose up`. It runs against either the GPU service (`scripts/serve_gpu.py`) or the GGUF service on CPU (`scripts/serve.py`).
- 🎉 **2026-10-15**: Released **chinese-decision-0.8b** (🤗 [Hugging Face](https://huggingface.co/quincyqiang/chinese-decision-0.8b) · [ModelScope](https://modelscope.cn/models/quincyqiang/chinese-decision-0.8b)) and **chinese-decision-0.8b-gguf** (Q8_0, 812 MB; 🤗 [Hugging Face](https://huggingface.co/quincyqiang/chinese-decision-0.8b-gguf) · [ModelScope](https://modelscope.cn/models/quincyqiang/chinese-decision-0.8b-gguf)). Full training ran 1 epoch (13,048,000 decisions); the release reaches **72.26%** on the CJ-Bench general domain and a **76.90%** specialized average. After merging, a temperature of T = 1.108 was fitted on a separate validation partition.

## Models

| Model | Base / objective | Format / size | Default T | Download | Model Card |
|---|---|---|---:|---|---|
| Chinese Decision 0.8B | Qwen3.5-0.8B / CE | BF16 / 1.50 GB | 1.108133 | [HF](https://huggingface.co/quincyqiang/chinese-decision-0.8b) · [ModelScope](https://modelscope.cn/models/quincyqiang/chinese-decision-0.8b) | [0.8B](docs/MODEL_CARD_0_8B.md) |
| Chinese Decision 0.8B Q8_0 | Same merged 0.8B | GGUF / 0.812 GB | 1.108133 | [HF](https://huggingface.co/quincyqiang/chinese-decision-0.8b-gguf) · [ModelScope](https://modelscope.cn/models/quincyqiang/chinese-decision-0.8b-gguf) | [0.8B GGUF](docs/MODEL_CARD_0_8B_GGUF.md) |
| Chinese Decision 2B | Qwen3.5-2B / CE + RLCD | BF16 / 3.76 GB | 1.074872 | [HF](https://huggingface.co/quincyqiang/chinese-decision-2b) · [ModelScope](https://modelscope.cn/models/quincyqiang/chinese-decision-2b) | [2B](docs/MODEL_CARD_2B.md) |
| Chinese Decision 2B Q8_0 | Same merged 2B | GGUF / 2.01 GB | 1.074872 | [HF](https://huggingface.co/quincyqiang/chinese-decision-2b-gguf) · [ModelScope](https://modelscope.cn/models/quincyqiang/chinese-decision-2b-gguf) | [2B GGUF](docs/MODEL_CARD_2B_GGUF.md) |
| Chinese Decision 4B | Qwen3.5-4B / CE + RLCD | BF16 / 8.41 GB | 1.024758 | [HF](https://huggingface.co/quincyqiang/chinese-decision-4b) · [ModelScope](https://modelscope.cn/models/quincyqiang/chinese-decision-4b) | [4B](docs/MODEL_CARD_4B.md) |
| Chinese Decision 4B Q8_0 | Same merged 4B | GGUF / 4.48 GB | 1.024758 | [HF](https://huggingface.co/quincyqiang/chinese-decision-4b-gguf) · [ModelScope](https://modelscope.cn/models/quincyqiang/chinese-decision-4b-gguf) | [4B GGUF](docs/MODEL_CARD_4B_GGUF.md) |

All six variants have full CJ-Bench results on all 307,900 decisions (see [Evaluation](#evaluation)) and support 16,384 input tokens; one question holds 2–26 candidates (training covers 2–16), and larger `choice` questions go through [`/v1/tournament`](#http-server). Sizes are decimal GB, not runtime memory.

**Which one?** On a GPU, use 4B BF16 for the highest accuracy, or 2B BF16 to balance latency and memory. On CPU or under memory limits, use a Q8_0 GGUF, which needs no PyTorch. If latency and resources matter most, or you need the lowest overall ECE, use 0.8B. See [model selection and complete comparison](docs/MODELS.md) (Chinese).

Code: [this repository](https://github.com/yanqiangmiffy/chinese-decision). Test platform: [Decision Studio](https://github.com/yanqiangmiffy/chinese-decision-studio).

## Showcase

The screenshots below come from Decision Studio, our test platform, serving chinese-decision-0.8b (the interface is in Chinese). Every scenario follows the same design: **business rules live in code; the decision model only supplies probabilities**.

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
    B --> C["Qwen3.5 0.8B / 2B / 4B<br/>LoRA fine-tuned, merged"]
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

Throughout this section, **chinese-decision-0.8b**, **chinese-decision-2b**, **chinese-decision-4b** and their `-gguf` builds are the published weights: the trained LoRA is merged into the base model, and probabilities use the default temperature in `decision_config.json`.

> **About the temperature T**: probabilities are p = softmax(z / T), where z are the candidate-letter logits. **T = 1** means a plain softmax over the logits, with no temperature calibration. The released weights ship a temperature fitted on a held-out validation partition (1.108 for 0.8B, 1.075 for 2B, 1.025 for 4B). It only changes how sharp the probabilities are and **never changes the chosen answer**. Results for training checkpoints (before merging, T = 1) are in [Training checkpoint evaluations](docs/TRAINING_HISTORY.md) (Chinese).

### Main Results

All 307,900 CJ-Bench decisions (304,822 hard labels, 3,078 soft labels), with no filtering or truncation. ACC / ECE use hard labels only (15 equal-width bins); NLL / Brier include soft labels. Micro is weighted by items; Macro is the unweighted mean of the four domains. **Bold** marks metrics that improve on the next smaller size in the same format (2B over 0.8B, 4B over 2B).

| Model | Format | Micro ACC (%) ↑ | Macro ACC (%) ↑ | ECE (%) ↓ | NLL ↓ | Brier ↓ | Score MAE ↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chinese-decision-0.8b | BF16 | 83.11 | 75.74 | 0.27 | 0.4313 | 0.2238 | 0.5844 |
| chinese-decision-0.8b-gguf | Q8_0 | 83.12 | 75.81 | 0.27 | 0.4313 | 0.2238 | 0.5840 |
| chinese-decision-2b | BF16 | **84.11** | **77.18** | 0.64 | **0.4165** | **0.2133** | **0.5642** |
| chinese-decision-2b-gguf | Q8_0 | **84.11** | **77.33** | 0.65 | **0.4163** | **0.2132** | **0.5637** |
| chinese-decision-4b | BF16 | **85.97** | **81.92** | 1.00 | **0.3706** | **0.1909** | **0.5253** |
| chinese-decision-4b-gguf | Q8_0 | **85.99** | **81.95** | 0.96 | **0.3705** | **0.1907** | **0.5253** |

- **4B is the most accurate**: compared with 2B, Micro **+1.86**, Macro **+4.75** and the specialized average **+5.76** points, with legal / finance up **8.47 / 7.11**; NLL / Brier / score MAE drop by **11.0% / 10.5% / 6.9%**. Overall ECE is 1.00%, above 2B's 0.64%. 4B and 2B share training data, objective, global batch and learning rate, so the main difference is base-model size.
- **2B is more accurate overall**: compared with 0.8B, Micro **+1.01**, Macro **+1.44** and the medical/legal/finance average **+1.35** points; NLL / Brier / score MAE drop by **3.4% / 4.7% / 3.5%**. Overall ECE is slightly higher (0.64% vs 0.27%) but still very low.
- **Q8_0 is nearly lossless**: paired item by item with BF16 at the same temperature, top-1 agreement is **98.92% / 99.00% / 99.19%** for 0.8B / 2B / 4B, and accuracy changes by +0.02 / −0.001 / +0.02 points. A few individual answers still change (for 2B, 1,105 flip correct→wrong and 1,101 wrong→correct); the 0.15-point macro gain of 2B Q8_0 comes mostly from the small legal domain (4,300 items) and does not mean quantization improves the model. Full results were measured with llama.cpp on CUDA; CPU checks cover the API and limited inputs. See the [0.8B GGUF](docs/MODEL_CARD_0_8B_GGUF.md), [2B GGUF](docs/MODEL_CARD_2B_GGUF.md) and [4B GGUF](docs/MODEL_CARD_4B_GGUF.md) cards.

### Comparison with Chinese-Jev

External rows are author-reported results from [Chinese-Jev v1 Tables 1 and 2](https://arxiv.org/html/2609.36965v1) (checked on 2026-10-17, not reproduced here); project rows are the released weights. All values are percentages; **bold** marks the best value in each column.

**General (Table 1)**

| Model | ACC (%) ↑ | ECE (%) ↓ |
| --- | ---: | ---: |
| Qwen3.5-2B | 32.95 | 43.68 |
| Open-Jev | 52.33 | 10.54 |
| SemIF | 44.21 | 27.27 |
| Laya Multilingual | 40.22 | 23.04 |
| Jev (API) | 68.35 | 11.45 |
| Chinese-Jev General / Specialists | 69.20 | 3.78 |
| chinese-decision-0.8b | 72.26 | **1.68** |
| chinese-decision-2b | 73.96 | 1.93 |
| chinese-decision-4b | **75.69** | 2.91 |

**Specialized domains (Table 2)**

| Model | Medical ACC | Medical ECE | Legal ACC | Legal ECE | Finance ACC | Finance ECE | Avg. ACC | Avg. ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-2B | 53.88 | 28.71 | 31.53 | 44.82 | 38.11 | 49.12 | 41.17 | 40.88 |
| Open-Jev | 73.08 | 2.88 | 44.07 | 23.15 | 48.11 | 10.19 | 55.09 | 12.07 |
| SemIF | 62.17 | 10.39 | 43.07 | 19.99 | 47.89 | 18.45 | 51.04 | 16.28 |
| Laya Multilingual | 24.55 | 43.05 | 34.72 | 31.49 | 38.31 | 26.18 | 32.53 | 33.57 |
| Jev (API) | 84.29 | 5.70 | 68.35 | 8.03 | 78.39 | 2.98 | 77.01 | 5.57 |
| Chinese-Jev General | 39.77 | 17.33 | 38.76 | 15.37 | 35.39 | 14.43 | 37.97 | 15.71 |
| Chinese-Jev General / Specialists | 87.65 | 7.25 | 60.10 | 17.45 | 64.72 | 11.55 | 70.82 | 12.08 |
| chinese-decision-0.8b | 88.85 | 1.12 | 67.84 | 4.05 | 74.00 | 2.45 | 76.90 | 2.54 |
| chinese-decision-2b | 89.50 | **0.24** | 68.74 | 3.37 | 76.50 | 2.13 | 78.25 | 1.92 |
| chinese-decision-4b | **91.19** | 0.25 | **77.21** | **2.02** | **83.61** | **1.65** | **84.00** | **1.30** |

- Compared with Chinese-Jev (General + three Specialists), chinese-decision-4b is **6.49 / 3.54 / 17.11 / 18.89** points higher on general / medical / legal / finance, and **13.18** higher on the specialized average. It also beats the closed-source Jev in every domain: **6.99** points on the specialized average and **5.22** on finance. chinese-decision-2b is 1.24 points above Jev on the specialized average but still 1.89 lower on finance.
- All three sizes have lower ECE than every model in the paper, in all four domains.
- Chinese-Jev General plus three Specialists are four separately trained models; this project covers all four domains with one model per size. Jev is a closed API. The paper uses H200 / API and this project uses an A800, so latency is not ranked. Model size, batch and objective (2B adds RLCD) all change between 0.8B and 2B, so differences are not attributable to one factor.

**Method differences from Chinese-Jev**

| | Chinese-Jev | Chinese Decision (ours) |
|---|---|---|
| Backbone | mmBERT-base encoder + decision Transformer (about 322M) | Qwen3.5-0.8B / 2B / 4B decoder + LoRA |
| Candidate scoring | Shared MLP over candidate-marker positions | Last-position logits of candidate letters |
| Objective | Cross-entropy + RLCD policy-gradient term | Candidate-distribution cross-entropy for 0.8B; 2B and 4B also use RLCD |
| Training | General pre-training, then three separately fine-tuned specialists | All four domains mixed into one model |

Full CJ-Bench results for training checkpoints of every size (0.8B / 2B / 4B) are in [Training checkpoint evaluations](docs/TRAINING_HISTORY.md) (Chinese).

### Results by Domain and Type

<details>
<summary><b>chinese-decision-0.8b: CJ-Bench by domain and by type</b></summary>

> Released weights (T = 1.108), one A800 80GB, BF16. ACC and ECE use hard labels only; NLL and Brier cover every item, including soft labels. Medical is 65% of the test set, so micro metrics mostly reflect medical; prefer macro for cross-domain comparisons.

**By domain**

| Domain | Items | ACC ↑ | ECE ↓ | NLL ↓ | Brier ↓ |
|---|---:|---:|---:|---:|---:|
| General | 100,000 | 72.26 | 1.68 | 0.686 | 0.355 |
| Medical | 200,000 | 88.85 | 1.12 | 0.295 | 0.152 |
| Legal | 4,300 | 67.84 | 4.05 | 0.721 | 0.410 |
| Finance | 3,600 | 74.00 | 2.45 | 0.597 | 0.337 |
| **Overall (micro)** | **307,900** | **83.11** | **0.27** | **0.431** | **0.224** |
| **Unweighted mean of four domains (macro)** | — | **75.74** | — | **0.575** | — |

**By question type**

| Type | Items | ACC ↑ | ECE ↓ | NLL ↓ | Expected-level MAE ↓ |
|---|---:|---:|---:|---:|---:|
| `noul` | 134,577 | 94.96 | 0.19 | 0.117 | — |
| `choice` | 134,577 | 78.32 | 0.58 | 0.587 | — |
| `score` | 38,746 | 56.45 | 1.67 | 0.984 | 0.584 |

**Accuracy by domain × type (%)**

| Domain | `noul` | `choice` | `score` |
|---|---:|---:|---:|
| General | 87.59 | 72.93 | 54.64 |
| Medical | 97.82 | 80.55 | 66.50 |
| Legal | 85.60 | 64.07 | 51.69 |
| Finance | 76.33 | 62.83 | 82.83 |

</details>

<details>
<summary><b>chinese-decision-2b: CJ-Bench by domain and by type</b></summary>

> Released weights (T = 1.075), same protocol. **Bold** marks accuracies that improve on chinese-decision-0.8b. Domain and type results for Q8_0 are in the [2B GGUF card](docs/MODEL_CARD_2B_GGUF.md).

**By domain**

| Domain | Items | ACC ↑ | ECE ↓ | NLL ↓ | Brier ↓ |
|---|---:|---:|---:|---:|---:|
| General | 100,000 | **73.96** | 1.93 | 0.663 | 0.336 |
| Medical | 200,000 | **89.50** | 0.24 | 0.284 | 0.146 |
| Legal | 4,300 | **68.74** | 3.37 | 0.703 | 0.406 |
| Finance | 3,600 | **76.50** | 2.13 | 0.578 | 0.323 |
| **Overall (micro)** | **307,900** | **84.11** | **0.64** | **0.417** | **0.213** |
| **Unweighted mean of four domains (macro)** | — | **77.18** | — | **0.557** | — |

**By question type**

| Type | Items | ACC ↑ | ECE ↓ | NLL ↓ | Expected-level MAE ↓ |
|---|---:|---:|---:|---:|---:|
| `noul` | 134,577 | **95.11** | 0.34 | 0.115 | — |
| `choice` | 134,577 | **80.12** | 0.90 | 0.559 | — |
| `score` | 38,746 | **57.67** | 1.65 | 0.969 | 0.564 |

**Accuracy by domain × type (%)**

| Domain | `noul` | `choice` | `score` |
|---|---:|---:|---:|
| General | **88.01** | **76.16** | **56.05** |
| Medical | **97.87** | **81.81** | 66.19 |
| Legal | 82.07 | **67.20** | **55.15** |
| Finance | **81.75** | **67.42** | 80.33 |

> 9 of the 12 domain × type cells improve; legal yes/no (`noul`, 82.07 vs 85.60) and finance rating (`score`, 80.33 vs 82.83) drop clearly, and medical rating dips slightly (66.19 vs 66.50). See [Limitations](#limitations).

</details>

<details>
<summary><b>chinese-decision-4b: CJ-Bench by domain and by type</b></summary>

> Released weights (T = 1.025), same protocol. **Bold** marks accuracies that improve on chinese-decision-2b. Domain and type results for Q8_0 are in the [4B GGUF card](docs/MODEL_CARD_4B_GGUF.md).

**By domain**

| Domain | Items | ACC ↑ | ECE ↓ | NLL ↓ | Brier ↓ |
|---|---:|---:|---:|---:|---:|
| General | 100,000 | **75.69** | 2.91 | 0.621 | 0.318 |
| Medical | 200,000 | **91.19** | 0.25 | 0.241 | 0.124 |
| Legal | 4,300 | **77.21** | 2.02 | 0.529 | 0.304 |
| Finance | 3,600 | **83.61** | 1.65 | 0.421 | 0.230 |
| **Overall (micro)** | **307,900** | **85.97** | **1.00** | **0.371** | **0.191** |
| **Unweighted mean of four domains (macro)** | — | **81.92** | — | **0.453** | — |

**By question type**

| Type | Items | ACC ↑ | ECE ↓ | NLL ↓ | Expected-level MAE ↓ |
|---|---:|---:|---:|---:|---:|
| `noul` | 134,577 | **95.47** | 0.43 | 0.107 | — |
| `choice` | 134,577 | **83.25** | 1.26 | 0.475 | — |
| `score` | 38,746 | **60.43** | 2.18 | 0.922 | 0.525 |

**Accuracy by domain × type (%)**

| Domain | `noul` | `choice` | `score` |
|---|---:|---:|---:|
| General | **88.50** | **78.60** | **58.35** |
| Medical | **98.04** | **85.04** | **67.49** |
| Legal | **88.87** | **74.13** | **67.31** |
| Finance | **85.67** | **77.00** | **88.17** |

> All 12 domain × type cells beat both 2B and 0.8B; legal yes/no and finance rating, where 2B fell behind 0.8B, reach 88.87% and 88.17% on 4B.

</details>

<details>
<summary><b>0.8B before vs. after fine-tuning (validation)</b></summary>

On a stratified subset of the official validation set (4 domains × 3 types, 100 items each, 1,200 total), zero-shot base model vs. the final fine-tuned checkpoint:

| Domain | Qwen3.5-0.8B ACC | **Chinese Decision ACC** | Qwen3.5-0.8B ECE | **Chinese Decision ECE** |
|---|---:|---:|---:|---:|
| General | 36.64 | **75.34** | 28.25 | **3.76** |
| Medical | 16.67 | **82.00** | 43.40 | **3.25** |
| Legal | 40.67 | **65.33** | 24.03 | **7.62** |
| Finance | 44.33 | **71.33** | 16.77 | **3.41** |
| **Overall** | 34.56 | **73.49** | 27.45 | **2.48** |

</details>

### Inference Efficiency

The same 25 CJ-Bench inputs (92–6,521 tokens), single requests on the released weights, three warmed runs each:

| Metric (single-request p50) | chinese-decision-0.8b | chinese-decision-2b | chinese-decision-4b |
|---|---:|---:|---:|
| GPU · BF16 (Transformers) | 44.4 ms | 44.8 ms | 60.1 ms |
| GPU · Q8_0 (llama.cpp) | **17.9 ms** | **23.6 ms** | **47.3 ms** |
| CPU, 16 threads · Q8_0 (llama.cpp) | 427.0 ms | 818.0 ms | 2004.0 ms |
| CPU process memory · Q8_0 / FP32 | **2.19** / 7.44 GiB | **3.34** / 11.74 GiB | 6.14 / — GiB |

> Shared server with an A800 80GB PCIe and a Xeon Gold 5318Y. Timing covers the forward pass and probabilities for pre-tokenized input; it excludes tokenization, HTTP and network. At batch = 1 both sizes are bound by GPU scheduling overhead, so BF16 latency is similar; Q8_0 is faster for short inputs on GPU, and its main CPU advantage is memory. 4B was measured on a different day with the same inputs and has no Transformers FP32 CPU baseline. Long inputs and the Windows desktop measurements are in the [0.8B GGUF](docs/MODEL_CARD_0_8B_GGUF.md), [2B GGUF](docs/MODEL_CARD_2B_GGUF.md) and [4B GGUF](docs/MODEL_CARD_4B_GGUF.md) cards.

## Quick Start

### Installation

```bash
git clone https://github.com/yanqiangmiffy/chinese-decision.git
cd chinese-decision
pip install -e ".[train]"
```

Python ≥ 3.11 is required. `[train]` installs the CUDA training/inference dependencies. For CPU GGUF serving use `pip install -e ".[gguf]"`; PyTorch, PEFT and flash-linear-attention are not required. Transformers-only inference uses `.[torch]`. If your environment already has a CUDA build of PyTorch that pip should leave alone, use `pip install -e . --no-deps`.

### Models and Data

To use the model without training, download the released weights (links for all six variants are in [Models](#models)). Pick the one you need:

```bash
# GPU: merged BF16 weights
hf download quincyqiang/chinese-decision-4b --local-dir models/chinese-decision-4b
hf download quincyqiang/chinese-decision-2b --local-dir models/chinese-decision-2b
hf download quincyqiang/chinese-decision-0.8b --local-dir models/chinese-decision-0.8b
# CPU: Q8_0 GGUF, no PyTorch needed
hf download quincyqiang/chinese-decision-4b-gguf --local-dir models/chinese-decision-4b-gguf
hf download quincyqiang/chinese-decision-2b-gguf --local-dir models/chinese-decision-2b-gguf
hf download quincyqiang/chinese-decision-0.8b-gguf --local-dir models/chinese-decision-0.8b-gguf
# or from ModelScope (same repository names), e.g.
modelscope download --model quincyqiang/chinese-decision-4b --local_dir models/chinese-decision-4b
modelscope download --model quincyqiang/chinese-decision-4b-gguf --local_dir models/chinese-decision-4b-gguf
```

For training and evaluation, the base model and data are also loaded from local directories. You can fetch them with the Hugging Face CLI (the CJ datasets are currently gated; run `hf auth login` first):

```bash
hf download Qwen/Qwen3.5-0.8B --local-dir models/Qwen3.5-0.8B
hf download Qwen/Qwen3.5-2B --local-dir models/Qwen3.5-2B   # for training 2B
hf download Qwen/Qwen3.5-4B --local-dir models/Qwen3.5-4B   # for training 4B
```

```bash
hf download bbldCVer-hf/CJ-Dataset --repo-type dataset --local-dir datasets/CJ-Dataset
```

```bash
hf download bbldCVer-hf/CJ-Bench --repo-type dataset --local-dir datasets/CJ-Bench
```

### Run the 4B / 2B models

After downloading the model directory, run from the repository root (4B shown; for 2B replace `4b` with `2b` in the directory and file names). For GPU BF16:

```bash
python scripts/serve_gpu.py --base models/chinese-decision-4b --host 127.0.0.1 --port 8789 --token-budget 16384 --max-batch 32
```

For CPU Q8_0 (install the `[gguf]` extra):

```bash
python scripts/serve.py --backend gguf --base models/chinese-decision-4b-gguf --gguf models/chinese-decision-4b-gguf/chinese-decision-4b-Q8_0.gguf --threads 8 --context-size 16384 --host 127.0.0.1 --port 8790
```

Both use the calibrated temperature in `decision_config.json` and expose the same API as the HTTP server below. See the [4B BF16 card](docs/MODEL_CARD_4B.md) for direct Python inference and the [4B Q8_0 card](docs/MODEL_CARD_4B_GGUF.md) for CPU setup. The Python and HTTP examples below use 0.8B; swap in a 2B or 4B directory to use those sizes.

### Python (0.8B example)

```python
from decision import DecisionPipeline

pipe = DecisionPipeline.from_pretrained("models/chinese-decision-0.8b")   # local directory or Hub repository id
answer = pipe({
    "state": "客户留言：付款后一直没有收到发票，我已经催了三次，今天必须拿到。",
    "questions": {"q": {
        "type": "choice",
        "instructions": "应该由哪个部门处理？",
        "criteria": {"billing": "支付、发票及退款", "technical": "故障和系统问题", "sales": "购买咨询与升级"},
    }},
})
print(answer)   # {"task": "decision", "answers": {"q": {"billing": 0.97, ...}}}, calibrated with decision_config.json
```

A model with a span head (for example a joint-training export, `runs/<run>/model`) extracts with the same object:

```python
pipe({"text": "华为公司董事长梁华在深圳会见了腾讯游戏负责人。", "types": {"人名": "人物的姓名", "公司": "企业名称"}})
# {"task": "extraction", "entities": [{"text": "华为公司", "label": "公司", "start": 0, "end": 4, "score": ...}, ...]}
```

The request picks the task: `questions` means a decision, `types` means extraction. The lower-level API follows transformers:

| Class | Role |
|---|---|
| `DecisionConfig` | `decision_config.json`: temperature, prompt version, span head settings |
| `DecisionProcessor` | prompt compilation, encoding, padding and decoding for both tasks |
| `DecisionModel` | backbone + optional span head; `from_pretrained` / `save_pretrained` (LoRA merged, one weight set), `enable_lora`, `score_decisions`, `predict_spans` |
| `DecisionPipeline` | request-level inference: `decide`, `extract`, `__call__` |

A training adapter loads directly with `DecisionModel.from_pretrained("models/Qwen3.5-0.8B", adapter="runs/cj-ddp/best")`; a joint-training run directory with `DecisionPipeline.from_run(base, run)`.

Code layout:

```text
decision/
  configuration.py   DecisionConfig / ExtractionConfig
  processing.py      DecisionProcessor, the CJ case format, both prompt layouts
  modeling.py        DecisionModel, SpanHead
  pipeline.py        DecisionPipeline, SlotFillingPipeline
  losses.py          RLCDLoss, SpanLoss
  metrics.py         validation, CJ-Bench and extraction metrics
  data/              CJ-Dataset streaming and batching, slot records, the CJ-IE builders (ie/)
  training/          LoRA targets, decision evaluation, temperature fitting, grouped RLCD
  serving/           HTTP answer format, Transformers / GGUF backends, candidate tournaments
scripts/             command-line entry points for training, export, evaluation and serving
```

`import decision` does not load PyTorch; the GGUF CPU server still needs only transformers.

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
| `POST /v1/tournament` | `choice` questions with more candidates than one prompt holds: group rounds, then a final (see below) |
| `GET /v1/info` | Checkpoint, validation metrics and serving statistics |
| `GET /v1/training` | Training progress |
| `GET /health` | Health check; reports the temperature, whether the model is calibrated and the checkpoint step |

A request may contain up to 64 questions, each with 2–26 candidates (see `/v1/tournament` below for more) and at most 16,384 input tokens. Over-length inputs are **never truncated**; invalid requests return `422`.

**More categories: tournaments (`/v1/tournament`)**. The model reads one letter per candidate, and the training data has at most 16 candidates. The server shuffles a larger `choice` question into groups of at most `group_size`, advances the top `keep` of each group until the rest fit one question, and asks that final question. Every group goes through the same prompt compiler, so no retraining is needed:

```json
{
  "state": "用户希望了解第 137 号产品。",
  "question": {"type": "choice", "instructions": "用户想了解哪个产品？",
               "criteria": {"p0": "第 0 号产品", "p1": "第 1 号产品", "...": "up to 1024 candidates"}},
  "group_size": 16,
  "keep": 2,
  "seed": 0
}
```

`group_size` is 3–26 (default 16, the largest count seen in training), `keep` is 1 to `group_size // 2` (default 2), and `seed` fixes the grouping. `answer` has the same shape as a `/v1/systemone` answer, but **its probabilities cover the finalists only**; `rounds` records each group's probabilities and who advanced, and `decisions` counts forward passes. With the defaults, 300 candidates take 19 + 3 + 1 = 23 forward passes. A field that already fits in `group_size` is one question, identical to `/v1/systemone`. Tournaments have not been evaluated on CJ-Bench; finalists tend to be close, so their probabilities are flatter, and confidence thresholds should be re-tuned on your own data.

[examples/tournament.json](examples/tournament.json) is a complete example with 41 e-commerce customer-service intents. With the defaults it becomes 3 groups (14 / 14 / 13), and the top 2 of each meet in a 6-candidate final: 4 forward passes in total.

```bash
curl http://127.0.0.1:8787/v1/tournament \
  -H 'Content-Type: application/json' \
  --data-binary @examples/tournament.json
```

```json
{
  "model": "chinese-decision-0.8b",
  "answer": {"type": "choice", "choice": "damaged_item", "confidence": 0.841,
             "probabilities": {"damaged_item": 0.867, "courier_complaint": 0.071, "repair_request": 0.032,
                               "other": 0.017, "stock_inquiry": 0.008, "wrong_item": 0.005}},
  "candidates": 41, "group_size": 16, "keep": 2,
  "rounds": [{"groups": [
    {"probabilities": {"...": "this group's 14 candidates"}, "advanced": ["courier_complaint", "other"]},
    {"probabilities": {"...": "this group's 14 candidates"}, "advanced": ["wrong_item", "stock_inquiry"]},
    {"probabilities": {"...": "this group's 13 candidates"}, "advanced": ["damaged_item", "repair_request"]}
  ]}],
  "usage": {"input_tokens": 813, "output_tokens": 0},
  "decisions": 4,
  "latency_ms": 161.4
}
```

On the GPU service (0.8B step 10000, T = 1, A800), 10 messages with different intents were all classified correctly, at about 150 ms each. This is a functional demo, not a formal evaluation.

### Deployment

Two server entry points share the same API and response format; either can back [Decision Studio](https://github.com/yanqiangmiffy/chinese-decision-studio) (the test platform) as its `DECISION_BASE_URL`:

| Entry point | Use | Weights | Deployment |
|---|---|---|---|
| `scripts/serve.py` | CPU (GGUF) or local debugging; one forward pass per question | GGUF: [HF](https://huggingface.co/quincyqiang/chinese-decision-0.8b-gguf) · [ModelScope](https://modelscope.cn/models/quincyqiang/chinese-decision-0.8b-gguf) (BF16 also works) | [docs/CPU.md](docs/CPU.md), [deploy/gguf](deploy/gguf/) |
| `scripts/serve_gpu.py` | Production GPU serving: dynamic batching, training progress, checkpoint hot reload (`POST /v1/admin/reload`) | BF16: [HF](https://huggingface.co/quincyqiang/chinese-decision-0.8b) · [ModelScope](https://modelscope.cn/models/quincyqiang/chinese-decision-0.8b), or the base model plus a trained adapter | [deploy/gpu](deploy/gpu/README.md) |

To check a running server (both routes, probability format, and invalid-request handling):

```bash
python scripts/verify_export.py --url http://127.0.0.1:8787
```

## Training

### Configuration

The table below is the 0.8B configuration. 2B and 4B add RLCD and use a global batch of at least 2,048 decisions per step; see [the 2B record](docs/TRAINING_2B.md) and the [4B card](docs/MODEL_CARD_4B.md) (Chinese).

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

In the run directory, `last/` is the resume point, `best/` is the best adapter on validation, and `final/` is the adapter after the last step. `last/` and `best/` are replaced in place; `--keep-every` (default 1000, 0 = off) also keeps a frozen `step-N/` copy every N steps for evaluating intermediate checkpoints later. The exported model directory contains the merged weights, `decision_config.json` (calibrated temperature and limits), the training config and summary, and the base model's `LICENSE`.

### Optional: RLCD Objective

`--rlcd` adds Chinese-Jev's RLCD term (a policy gradient over a proper-scoring-rule reward) to the distribution cross-entropy. It is ported from the [official Chinese-Jev code](https://github.com/gulucaptain/Chinese-Jev) and reproduces its results bit for bit. **It is off by default. 0.8B uses cross-entropy only; 2B and 4B enable RLCD.**

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
│   ├── tournament.py       #   grouped rounds for more candidates than one prompt holds
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
├── docs/
│   ├── MODELS.md           #   model overview: selection, protocol, external baselines
│   ├── MODEL_CARD_*.md     #   six model cards (0.8B / 2B / 4B × BF16 / Q8_0)
│   ├── TRAINING_HISTORY.md #   checkpoint evaluations during training, all sizes
│   ├── TRAINING_2B.md      #   2B training, export and diagnostics
│   ├── CPU.md              #   0.8B CPU deployment and GGUF validation
│   └── CPU_2B.md           #   2B CPU deployment and quantization checks
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

- **Keep result configurations separate**: the released weights (merged, default temperature), training checkpoints (T = 1) and Q8_0 differ slightly. Compare like with like rather than attributing a difference to one factor. Q8_0 full results were measured on GPU (llama.cpp CUDA); the CPU backend was checked on limited inputs only. Calibration validation is not test accuracy, and a low overall ECE does not mean every source is well calibrated.
- **Uneven improvements**: 2B's legal yes/no and finance rating accuracies are 82.07% / 80.33%, below 0.8B at 85.60% / 82.83% (4B: 88.87% / 88.17%); see the [2B record](docs/TRAINING_2B.md). On 4B the product-catalog source (3,151 items) reaches only 64.58%, below 2B's 70.96%, and overall ECE (1.00%) is higher than on 2B and 0.8B; see the [4B card](docs/MODEL_CARD_4B.md) (Chinese).
- **Slower than encoder models**: the 0.8B / 2B / 4B decoders have higher single-request latency than the roughly 322M Chinese-Jev encoder; 4B takes about 60 ms per request on GPU and about 2 s with Q8_0 on CPU.
- **`confidence` is not accuracy**: it only measures how sharp the distribution is. Before acting automatically on a threshold, check calibration on your own data.
- **Label semantics follow the upstream tasks**: for example, medical matching labels describe the original matching task and do not imply medical correctness. Do not use model outputs as the sole basis for medical, legal, or financial decisions.
- **Trained for Chinese only**, and not suited to open-ended generation.
- This project is a Chinese adaptation of the hmm method. It does not reproduce the training of Microsoft Decision, Jev, or Chinese-Jev.

## Acknowledgements

- [Chinese-Jev](https://arxiv.org/abs/2609.36965): CJ-Dataset, CJ-Bench, the baseline results used in our comparison, and the [reference implementation](https://github.com/gulucaptain/Chinese-Jev) of the RLCD term (Apache-2.0).
- [n4ze3m/hmm](https://github.com/n4ze3m/hmm): option-letter decision objective, LoRA workflow, candidate formatting, and System One response conventions (MIT).
- [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B), [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) and [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B): base models (Apache-2.0).

## License

- This project's code is released under the [MIT License](LICENSE).
- Portions adapted from hmm are also under the MIT License; the original notice is in [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md).
- `decision/rlcd.py` is adapted from Chinese-Jev under Apache-2.0; the full license text is in [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt).
- The base models Qwen3.5-0.8B, Qwen3.5-2B and Qwen3.5-4B are under Apache-2.0; exported model directories keep its original `LICENSE`.
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
