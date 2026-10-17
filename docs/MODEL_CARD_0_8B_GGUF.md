---
license: apache-2.0
language:
- zh
base_model:
- quincyqiang/chinese-decision-0.8b
base_model_relation: quantized
library_name: gguf
pipeline_tag: zero-shot-classification
datasets:
- bbldCVer-hf/CJ-Dataset
tags:
- gguf
- llama.cpp
- q8_0
- decision
- calibration
- classification
- chinese
- qwen3.5
tasks:
- zero-shot-classification
---

<div align="center">

# Chinese Decision 0.8B GGUF

**中文概率决策模型的 GGUF 版本：llama.cpp 推理，CPU 即可部署**

[原版模型](https://huggingface.co/quincyqiang/chinese-decision-0.8b) ·
[GitHub 代码](https://github.com/yanqiangmiffy/chinese-decision) ·
[测试平台 Decision Studio](https://github.com/yanqiangmiffy/chinese-decision-studio) ·
[训练数据 CJ-Dataset](https://huggingface.co/datasets/bbldCVer-hf/CJ-Dataset) ·
[评测集 CJ-Bench](https://huggingface.co/datasets/bbldCVer-hf/CJ-Bench)

</div>

> A Q8_0 GGUF export of [Chinese Decision 0.8B](https://huggingface.co/quincyqiang/chinese-decision-0.8b) for llama.cpp. The file is 812 MB. On the full CJ-Bench (307,900 decisions) its accuracy is 83.12% versus 83.11% for the original BF16 weights, with 98.92% top-choice agreement. It runs on CPU without PyTorch or a GPU. Like the original model, it is not a chat model: read the final-position logits of the candidate letters and apply softmax.

## 模型系列与结果范围

本文保留 **0.8B** 已发布版本的使用方法和既有评测。本次文档更新没有重新运行 0.8B；新增 2B 及其 Q8_0 是独立权重与报告，不能混用其温度或测试数字。

[0.8B BF16](MODEL_CARD_0_8B.md) · [0.8B Q8_0](MODEL_CARD_0_8B_GGUF.md) · [2B BF16](MODEL_CARD_2B.md) · [2B Q8_0](MODEL_CARD_2B_GGUF.md)

[各尺寸、部署格式与外部模型对比](MODELS.md)。2B BF16 与 2B Q8_0 均已公开发布并完成 CJ-Bench 全量评测：2B 整体准确率更高，0.8B 更小更快，部分法律判断与金融评分任务仍是 0.8B 更好。

## 相关资源

| 资源 | 说明 | 链接 |
|---|---|---|
| 代码 | 训练、评测、GGUF 导出与 HTTP 服务（GPU / CPU） | [GitHub](https://github.com/yanqiangmiffy/chinese-decision) |
| 测试平台 | Decision Studio：决策实验台、场景库、鲁棒性实验室、评测中心与性能压测 | [GitHub](https://github.com/yanqiangmiffy/chinese-decision-studio) |
| 模型（BF16） | 合并后的权重与校准温度，transformers 推理 | [Hugging Face](https://huggingface.co/quincyqiang/chinese-decision-0.8b) · [ModelScope](https://modelscope.cn/models/quincyqiang/chinese-decision-0.8b) |
| 模型（GGUF Q8_0，本仓库） | llama.cpp 推理，CPU 即可部署，不需要 PyTorch | [Hugging Face](https://huggingface.co/quincyqiang/chinese-decision-0.8b-gguf) · [ModelScope](https://modelscope.cn/models/quincyqiang/chinese-decision-0.8b-gguf) |
| 2B 系列 | 更大的 2B 版本（CE + RLCD），整体准确率更高；BF16 与 GGUF Q8_0 | BF16：[Hugging Face](https://huggingface.co/quincyqiang/chinese-decision-2b) · [ModelScope](https://modelscope.cn/models/quincyqiang/chinese-decision-2b)；GGUF：[Hugging Face](https://huggingface.co/quincyqiang/chinese-decision-2b-gguf) · [ModelScope](https://modelscope.cn/models/quincyqiang/chinese-decision-2b-gguf) |

## 简介

本仓库是 [Chinese Decision 0.8B](https://huggingface.co/quincyqiang/chinese-decision-0.8b) 的 GGUF 格式导出，用于 [llama.cpp](https://github.com/ggml-org/llama.cpp) / [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) 推理。

原版模型是一个中文**决策模型**：给定上下文和 2–26 个固定候选项，一次前向直接返回每个候选项的概率，不生成任何 token。支持 `noul`（是/否）、`choice`（单选）、`score`（有序等级）三种题型，覆盖通用、医疗、法律、金融四个领域。模型能力、训练细节与应用案例见[原版模型卡](https://huggingface.co/quincyqiang/chinese-decision-0.8b)。

GGUF 版本的特点：

- **无需 PyTorch 和 GPU**：`llama-cpp-python` 即可在 CPU 上运行，也可用 CUDA 版 llama.cpp 跑在 GPU 上。
- **体积约为原版的一半**：Q8_0 为 812 MB，原版 safetensors 为 1.50 GB。
- **内存占用更低**：同一台服务器上，CPU 推理进程峰值内存 2.19 GiB，Transformers FP32 为 7.44 GiB。
- **精度基本不变**：CJ-Bench 全量 307,900 条上，准确率、NLL、ECE、Brier 的变化都在千分之一以内。

## 文件说明

| 文件 | 说明 |
|---|---|
| `chinese-decision-0.8b-Q8_0.gguf` | 8-bit 量化权重，812 MB |
| `decision_config.json` | 推理参数：校准温度 T = 1.108、提示版本、最大长度、最大候选数 |
| `tokenizer.json`、`tokenizer_config.json`、`chat_template.jinja` | Qwen3.5 分词器与聊天模板，与原版一致 |
| `config.json` | 原版模型配置，仅作参考 |
| `conversion.json` | 转换记录：源权重与 GGUF 文件的 SHA256、转换命令（其中的 BF16 GGUF 仅用于下文的对照实验，未在本仓库发布） |
| `LICENSE` | 基础模型 Qwen3.5-0.8B 的 Apache-2.0 许可证原文 |

`chinese-decision-0.8b-Q8_0.gguf` 的 SHA256：`6a71cc35bd83f8b050499567faec59e9ae05778b75b1e2119d27d45b80f7e68b`

**转换方式**：由原版 `model.safetensors`（SHA256 `341307aa…`，step 10000）经 llama.cpp 的 `convert_hf_to_gguf.py --outtype q8_0` 直接导出，未经过中间格式。导出时使用 `--no-mtp`：原版 config 中声明了 MTP 层，但合并后的权重中不含 MTP 张量。校准温度沿用原版（T = 1.108），**未在量化后重新拟合**。

## 快速开始

### 安装依赖

```bash
pip install "llama-cpp-python>=0.3.36"
```

llama-cpp-python 内置的 llama.cpp 必须支持 Qwen3.5 架构，旧版本无法加载本模型。部分预编译 wheel 的版本较旧；在 Windows 上从源码编译需要 Visual Studio C++ Build Tools 和 Windows SDK。我们测试的版本是 0.3.36。

### 下载模型

Hugging Face：

```bash
hf download quincyqiang/chinese-decision-0.8b-gguf --local-dir chinese-decision-0.8b-gguf
```

ModelScope：

```bash
modelscope download --model quincyqiang/chinese-decision-0.8b-gguf --local_dir chinese-decision-0.8b-gguf
```

### 使用 llama-cpp-python 调用

下面的代码只依赖 `llama-cpp-python`，提示格式与原版训练、评测完全一致（手写的聊天模板与 Transformers 分词器的 token 序列逐一对齐）：

```python
import json
import math

from llama_cpp import Llama, llama_get_logits_ith

path = "chinese-decision-0.8b-gguf"
config = json.load(open(f"{path}/decision_config.json", encoding="utf-8"))
temperature = config["temperature"]
llm = Llama(model_path=f"{path}/chinese-decision-0.8b-Q8_0.gguf", n_ctx=4096, n_threads=8,
            n_gpu_layers=0, logits_all=False, verbose=False)

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
letter_ids = [llm.tokenize(c.encode(), add_bos=False, special=True)[0] for c in LETTERS]


def text(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def options_for(question):
    kind, criteria = question["type"], question.get("criteria")
    if kind == "choice":
        return [(str(k), text(v)) for k, v in criteria.items()]
    if kind == "score":
        return [(str(i), text(v)) for i, v in enumerate(criteria)]
    criteria = criteria or {}
    return [("false", text(criteria.get("false", "否"))), ("true", text(criteria.get("true", "是")))]


def build_prompt(state, instructions, options):
    content = ("待判断的上下文：\n" + text(state)
               + "\n\n问题：\n" + text(instructions)
               + "\n\n候选项：\n" + "\n".join(f"{LETTERS[i]}: {k} — {v}" for i, (k, v) in enumerate(options))
               + "\n只返回候选项对应的字母。")
    # Qwen3.5 chat template with enable_thinking=False
    return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def decide(state, question):
    options = options_for(question)
    ids = llm.tokenize(build_prompt(state, question["instructions"], options).encode(), add_bos=False, special=True)
    if len(ids) > llm.n_ctx():
        raise ValueError(f"input is {len(ids)} tokens; n_ctx is {llm.n_ctx()}")
    llm.reset()  # clear attention cache and recurrent state between questions
    llm.eval(ids)
    logits = llama_get_logits_ith(llm.ctx, -1)  # final-position logits
    values = [float(logits[i]) / temperature for i in letter_ids[:len(options)]]
    weights = [math.exp(v - max(values)) for v in values]
    probs = [w / sum(weights) for w in weights]
    k, best = len(probs), max(range(len(probs)), key=probs.__getitem__)
    answer = {"probabilities": {key: round(p, 4) for (key, _), p in zip(options, probs)},
              "confidence": round((k * probs[best] - 1) / (k - 1), 4)}
    if question["type"] == "noul":
        answer["noul"] = round(probs[1], 4)
    elif question["type"] == "choice":
        answer["choice"] = options[best][0]
    else:
        answer["score"] = round(sum(i * p for i, p in enumerate(probs)), 4)
    return answer


state = "客户留言：付款后一直没有收到发票，我已经催了三次，今天必须拿到。"
questions = {
    "urgent": {"type": "noul", "instructions": "客户是否表达了紧迫性？"},
    "department": {"type": "choice", "instructions": "应该由哪个部门处理？",
                   "criteria": {"billing": "支付、发票及退款", "technical": "故障和系统问题", "sales": "购买咨询与升级"}},
    "frustration": {"type": "score", "instructions": "客户有多不满？", "criteria": ["平静", "不满", "非常愤怒"]},
}
for name, question in questions.items():
    print(name, decide(state, question))
```

输出（Windows CPU，Q8_0）：

```text
urgent {'probabilities': {'false': 0.1196, 'true': 0.8804}, 'confidence': 0.7608, 'noul': 0.8804}
department {'probabilities': {'billing': 0.9369, 'technical': 0.0465, 'sales': 0.0167}, 'confidence': 0.9053, 'choice': 'billing'}
frustration {'probabilities': {'0': 0.0084, '1': 0.6081, '2': 0.3835}, 'confidence': 0.4122, 'score': 1.3752}
```

同一示例在原版 BF16 权重上的输出为 `urgent` 0.8654、`billing` 0.9406、`score` 1.3768，三题的最高概率选项一致，概率略有差异。不同 CPU、线程数和编译选项下结果可能有细微差别。

几点注意：

- **每道题都要 `llm.reset()`**。Qwen3.5 是线性注意力与全注意力的混合架构，含循环状态；不重置会把上一道题的状态带入下一道题。因此示例和服务都不在题目之间复用前缀缓存。
- `n_ctx` 是单题输入的上限。原版支持 16,384 tokens，需要时调大 `n_ctx`，内存占用会相应增加。超长输入应报错，不要截断。
- 使用 GPU 时需安装 CUDA 版 llama-cpp-python，并把 `n_gpu_layers` 设为 `-1`。
- `decision_config.json` 中的温度只在 `softmax` 前缩放 logits，不改变最高概率选项；设为 1 即得到未校准的原始概率。

### HTTP 服务

[项目仓库](https://github.com/yanqiangmiffy/chinese-decision)提供兼容 hmm / System One 接口的 HTTP 服务，`--backend gguf` 使用 llama.cpp 在 CPU 上推理，不需要 PyTorch：

```bash
git clone https://github.com/yanqiangmiffy/chinese-decision.git
cd chinese-decision
pip install -e ".[gguf]"
python scripts/serve.py --backend gguf --base /path/to/chinese-decision-0.8b-gguf --gguf /path/to/chinese-decision-0.8b-gguf/chinese-decision-0.8b-Q8_0.gguf --threads 8 --context-size 16384 --port 8787
```

```bash
curl http://127.0.0.1:8787/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @examples/request.json
```

接口、输入格式与原版一致（`POST /v1/systemone`、`POST /api/alpha/decisions`、`POST /v1/batch`、`POST /v1/compile`、`GET /v1/info`、`GET /v1/training`、`GET /health`），详见[原版模型卡](https://huggingface.co/quincyqiang/chinese-decision-0.8b)。服务启动时会检查 26 个候选字母的 token ID，并对每个请求核对 GGUF 与原版分词器的切分结果是否一致。

这个服务可以直接接入测试平台 [Decision Studio](https://github.com/yanqiangmiffy/chinese-decision-studio)（决策实验台、场景库、鲁棒性实验室、评测与压测）：把服务地址填到平台的 `DECISION_BASE_URL`。CPU 上单批评测耗时更长，建议同时设 `EVAL_BATCH_SIZE=16`。

## 量化效果

### CJ-Bench 全量对比

在 CJ-Bench 全量测试集（307,900 条：304,822 条硬标签、3,078 条软标签）上，对比原版合并权重（Transformers BF16）与 Q8_0 GGUF（llama.cpp CUDA）。两者使用相同的输入 token、候选顺序和部署温度 T = 1.108，没有过滤或截断任何样本，也没有在测试集上拟合温度或阈值。

| 指标 | 原版 BF16 | **Q8_0 GGUF** | 变化 |
|---|---:|---:|---:|
| 硬标签准确率 ↑ | 83.105% | **83.123%** | +0.018 pp |
| 四领域等权平均准确率 ↑ | 75.738% | **75.809%** | +0.072 pp |
| 分布 NLL ↓ | 0.431318 | **0.431293** | −0.000025 |
| ECE（15 分箱）↓ | 0.271% | **0.270%** | −0.001 pp |
| Brier ↓ | 0.223833 | **0.223815** | −0.000018 |
| 期望等级 MAE ↓ | 0.584427 | **0.583973** | −0.000453 |

> 准确率与 ECE 仅在硬标签上计算；NLL 与 Brier 包含软标签；期望等级 MAE 只针对 `score` 题。

**分领域**

| 领域 | 样本数 | 原版 ACC | Q8_0 ACC | 变化 | 原版 ECE | Q8_0 ECE | 变化 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 通用 | 100,000 | 72.26 | 72.23 | −0.03 | 1.68 | 1.71 | +0.03 |
| 医疗 | 200,000 | 88.85 | 88.89 | +0.04 | 1.12 | 1.14 | +0.02 |
| 法律 | 4,300 | 67.84 | 67.84 | 0.00 | 4.05 | 4.06 | +0.00 |
| 金融 | 3,600 | 74.00 | 74.28 | +0.28 | 2.45 | 2.60 | +0.16 |

**分题型**

| 题型 | 样本数 | 原版 ACC | Q8_0 ACC | 原版 ECE | Q8_0 ECE |
|---|---:|---:|---:|---:|---:|
| `noul` | 134,577 | 94.96 | 94.95 | 0.19 | 0.17 |
| `choice` | 134,577 | 78.32 | 78.38 | 0.58 | 0.64 |
| `score` | 38,746 | 56.45 | 56.40 | 1.67 | 1.65 |

> ACC 与 ECE 单位为 %，变化单位为百分点。

- **整体指标几乎不变**：准确率、NLL、ECE、Brier 的变化均在千分之一以内。
- **分领域 ECE 略有上升**：四个领域的 ECE 都小幅上升（+0.00 到 +0.16 个百分点），金融领域最明显；合并统计的 ECE 反而略降，是因为不同领域的分箱误差相互抵消。量化后温度未重新拟合。
- **个别样本会变化**：两者的最高概率选项一致率为 98.92%。硬标签样本中，1,185 条由对变错，1,240 条由错变对。每个候选项的概率平均绝对差为 0.0024，每题最大差的 95 分位数为 0.0236，全部样本中最大差为 0.2269。

本表中的原版结果是发布的合并权重在默认温度 T = 1.108 下的 CJ-Bench 成绩；原版模型卡中的 CJ-Bench 表格来自合并前的 LoRA adapter（T = 1），两者准确率接近但不完全相同。

### BF16 GGUF 对照

为区分"格式与后端差异"和"量化差异"，额外导出了未量化的 BF16 GGUF 作为对照（未在本仓库发布），在 1,200 条分层样本（4 个领域 × 3 种题型，每组 100 条，seed 42）上对比三种 GPU 推理路径：

| 后端 | 准确率 ↑ | 分布 NLL ↓ | ECE ↓ | Brier ↓ |
|---|---:|---:|---:|---:|
| Transformers BF16 | 73.07% | 0.6058 | 3.43% | 0.3380 |
| llama.cpp BF16 GGUF | 73.41% | 0.6050 | 2.57% | 0.3375 |
| llama.cpp Q8_0 GGUF | 73.57% | 0.6050 | 2.38% | 0.3374 |

> 分层样本的领域权重与全量测试集不同，整体准确率不能与上文直接比较。

Transformers BF16 与 BF16 GGUF 的最高概率选项一致率为 98.17%，BF16 GGUF 与 Q8_0 为 98.67%。也就是说，仅换成 llama.cpp 后端（不量化）带来的输出差异，与 Q8_0 量化带来的差异处于同一量级。

## 推理效率

延迟只统计已分词输入的推理与候选概率计算，不含分词、HTTP 和网络开销；每个请求都从清空的状态开始，不使用前缀缓存。

### Linux 服务器

Intel Xeon Gold 5318Y（48 物理核），NVIDIA A800 80GB PCIe，llama-cpp-python 0.3.36（CUDA 12.8，AVX512/VNNI）。25 条 CJ-Bench 输入（92–6,521 tokens），每条预热 1 次后测 3 次；其中短输入 16 条（92–255 tokens）。

| 后端 | 短输入 p50 | 全部 25 条 p50 | 6,521-token 输入 | 进程峰值内存 |
|---|---:|---:|---:|---:|
| Transformers FP32，CPU 16 线程 | 319.8 ms | 371.3 ms | 17.05 s | 7.44 GiB |
| **llama.cpp Q8_0，CPU 16 线程** | 347.2 ms | 427.0 ms | 13.95 s | **2.19 GiB** |
| llama.cpp Q8_0，CPU 8 线程 | 555.7 ms | 697.0 ms | 23.20 s | 2.20 GiB |
| Transformers BF16，GPU | 44.3 ms | 44.4 ms | 93.4 ms | — |
| **llama.cpp Q8_0，GPU** | **15.6 ms** | **17.9 ms** | 369.6 ms | — |

### Windows 台式机

Intel Core i7-9700（8 物理核），32 GB 内存，8 线程，llama-cpp-python 0.3.36（AVX2/FMA/OpenMP）。12 条 64–89 tokens 的短输入，每条测 3 次。

| 后端 | 短输入 p50 | 进程峰值内存 |
|---|---:|---:|
| Transformers FP32 | 627 ms | 3.47 GiB |
| llama.cpp BF16 GGUF | 640 ms | 2.04 GiB |
| **llama.cpp Q8_0** | **524 ms** | **1.38 GiB** |

- **CPU 上的主要收益是内存**：峰值内存比 Transformers FP32 低 60%–70%。速度取决于硬件和输入长度：台式机上短输入快约 1.2 倍，但 852 tokens 的输入反而更慢（6.65 s 对 5.12 s）；服务器上短输入略慢 8.6%，最长输入快约 1.2 倍。
- **线程数影响明显**：服务器上 16 线程明显快于 8 线程，台式机上 8 线程快于 4 线程和 2 线程，部署前请按实际机器调优。
- **GPU 上短输入更快、长输入更慢**：llama.cpp Q8_0 在短输入上约为 Transformers BF16 的 2.8 倍速度，但 6,521 tokens 的输入慢约 4 倍。
- 在这 25 条输入上，CPU FP32、CPU Q8_0、GPU Q8_0 的最高概率选项全部一致；CPU FP32 与 CPU Q8_0 的最大概率差为 0.0248。同一后端重复推理的输出完全相同，Q8_0 在 8 线程和 16 线程下的输出也完全相同。

## 局限性

- **继承原版模型的全部局限**：法律、金融领域较弱，`score` 题型最弱；`confidence` 只反映分布的尖锐程度，不等于准确率；医疗、法律、金融标签来自上游任务，不代表专业正确性，请勿作为相关决策的唯一依据。详见[原版模型卡](https://huggingface.co/quincyqiang/chinese-decision-0.8b)。
- **不是聊天模型**：模型只在末位候选字母上训练，不要用 `llama-cli`、Ollama 等工具的对话或文本生成模式调用，生成结果没有意义。
- **量化会改变个别结果**：约 1.08% 样本的最高概率选项与原版不同，单个候选项的概率差最大约 23 个百分点。对单条结果敏感的场景，请在自己的数据上对比原版。
- **温度未重新拟合**：沿用原版 T = 1.108，量化后四个领域的 ECE 均小幅上升。若要按概率阈值自动执行，请先在自己的数据上验证校准效果。
- **CPU 端未做全量评测**：CJ-Bench 全量对比在 GPU（llama.cpp CUDA）上完成；CPU 只核对了 25 条输入。CPU 与 GPU、不同 batch 大小之间不保证逐位一致。
- **依赖较新的 llama.cpp**：需要支持 Qwen3.5 混合架构的 llama.cpp / llama-cpp-python（测试版本 0.3.36）。
- 本仓库只提供 Q8_0 一种格式，未提供更低比特的量化版本。

## 许可证

- **模型权重**：Apache-2.0，继承自基础模型 Qwen3.5-0.8B（`LICENSE` 为其原文）。
- **训练数据**：CJ-Dataset 与 CJ-Bench 的许可按数据来源分别适用，其中部分来源为非商业许可。本模型不授予任何数据层面的额外权利，商业使用前请自行评估。
- **代码**：推理、训练、评测与转换代码以 MIT 许可证发布，见 [GitHub 仓库](https://github.com/yanqiangmiffy/chinese-decision)。

## 致谢

- [llama.cpp](https://github.com/ggml-org/llama.cpp) 与 [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)：GGUF 转换与推理。
- [Chinese-Jev](https://arxiv.org/abs/2609.36965)：CJ-Dataset 与 CJ-Bench。
- [n4ze3m/hmm](https://github.com/n4ze3m/hmm)：选项字母决策目标与 System One 响应约定。
- [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)：基础模型。

## 引用

```bibtex
@misc{chinese-decision-2026,
  title        = {Chinese Decision: A Lightweight Chinese Probabilistic Decision Model},
  author       = {yanqiangmiffy},
  year         = {2026},
  howpublished = {\url{https://github.com/yanqiangmiffy/chinese-decision}}
}
```

如果使用了 CJ-Dataset 或 CJ-Bench，请同时引用 Chinese-Jev：

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
