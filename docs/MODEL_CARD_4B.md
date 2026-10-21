---
license: apache-2.0
language:
- zh
base_model: Qwen/Qwen3.5-4B
base_model_relation: finetune
datasets:
- bbldCVer-hf/CJ-Dataset
pipeline_tag: zero-shot-classification
tags:
- decision-making
- classification
- calibration
- chinese
- qwen3.5
- rlcd
library_name: transformers
frameworks:
- Pytorch
tasks:
- zero-shot-classification
---

# Chinese Decision 4B

**中文概率决策模型：一次前向，直接返回候选项概率。**

> A Chinese decision model fine-tuned from Qwen3.5-4B with LoRA and CE + RLCD. It answers yes/no, single-choice and ordinal questions across general, medical, legal and finance tasks by reading the final-position logits of the candidate letters, without generating text. On the full CJ-Bench (307,900 decisions) it reaches 85.97% accuracy, an 84.00% medical/legal/finance average and 1.00% ECE, the highest accuracy in the Chinese Decision family. This package contains merged BF16 weights and a validation-fitted temperature.

[代码](https://github.com/yanqiangmiffy/chinese-decision) · [模型总览](MODELS.md) · [Decision Studio](https://github.com/yanqiangmiffy/chinese-decision-studio) · [训练数据](https://huggingface.co/datasets/bbldCVer-hf/CJ-Dataset) · [测试集](https://huggingface.co/datasets/bbldCVer-hf/CJ-Bench)

[0.8B BF16](MODEL_CARD_0_8B.md) · [0.8B Q8_0](MODEL_CARD_0_8B_GGUF.md) · [2B BF16](MODEL_CARD_2B.md) · [2B Q8_0](MODEL_CARD_2B_GGUF.md) · [4B BF16](MODEL_CARD_4B.md) · [4B Q8_0](MODEL_CARD_4B_GGUF.md)

## 相关资源

| 资源 | 说明 | 链接 |
|---|---|---|
| 代码 | 训练、评测、GGUF 导出与 HTTP 服务（GPU / CPU） | [GitHub](https://github.com/yanqiangmiffy/chinese-decision) |
| 测试平台 | Decision Studio：决策实验台、场景库、鲁棒性实验室、评测中心与性能压测 | [GitHub](https://github.com/yanqiangmiffy/chinese-decision-studio) |
| 4B BF16（本仓库） | 合并权重与校准温度，Transformers 推理 | [Hugging Face](https://huggingface.co/quincyqiang/chinese-decision-4b) · [ModelScope](https://modelscope.cn/models/quincyqiang/chinese-decision-4b) |
| 4B GGUF Q8_0 | llama.cpp 推理，CPU 即可部署，不需要 PyTorch | [Hugging Face](https://huggingface.co/quincyqiang/chinese-decision-4b-gguf) · [ModelScope](https://modelscope.cn/models/quincyqiang/chinese-decision-4b-gguf) |
| 2B 系列 | 约一半的权重大小，GPU 单条延迟更低 | [BF16](https://huggingface.co/quincyqiang/chinese-decision-2b) · [GGUF](https://huggingface.co/quincyqiang/chinese-decision-2b-gguf) |
| 0.8B 系列 | 最小、最快，整体 ECE 最低 | [BF16](https://huggingface.co/quincyqiang/chinese-decision-0.8b) · [GGUF](https://huggingface.co/quincyqiang/chinese-decision-0.8b-gguf) |

## 简介

**Chinese Decision** 是一个面向中文场景的轻量级**决策模型**。它不聊天、不生成文本：给定上下文和一组固定候选项，一次前向直接返回每个候选项的概率，适合意图识别、工单分流、内容审核、数据打标、给模型输出打分、智能体动作选择等"有限选项判断"任务。

- **一次前向，零输出 token**：读取最后位置上候选字母 `A`–`Z` 的 logits，softmax 后即为概率。
- **三种题型**：`noul`（是/否）、`choice`（单选）、`score`（有序等级，返回期望分值）。
- **单一模型覆盖四个领域**：通用、医疗、法律、金融混合训练。
- **系列中准确率最高**：CJ-Bench 全量整体 ACC **85.97%**，医疗 / 法律 / 金融平均 **84.00%**，比 chinese-decision-2b 分别高 **1.86 / 5.76** 个百分点；4 个领域 × 3 种题型共 12 个组合的准确率全部高于 0.8B 与 2B。
- **高于论文报告的所有模型**：通用、医疗、法律、金融四个领域的准确率均高于 Chinese-Jev 与闭源 Jev 的作者报告值，四个领域的 ECE 也都更低。
- **概率经过校准**：附带在独立校准划分上拟合的温度，整体 ECE **1.00%**。

## 模型信息

更新于 2026-10-21。BF16 合并模型已完成 CJ-Bench 全量评测、保存/重载一致性检查和实际 HTTP 三题型/双路由验证；由本模型直接转换的 Q8_0 GGUF 也已完成全量对照，见 [4B Q8_0 模型卡](MODEL_CARD_4B_GGUF.md)。

| 项目 | 值 |
|---|---|
| 基础模型 | [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)，文本决策主干 |
| 训练目标 | 候选分布交叉熵 + RLCD |
| 微调方式 | LoRA（r = 32）训练后合并为完整权重，加载时不需要 PEFT |
| 训练数据 | [CJ-Dataset](https://huggingface.co/datasets/bbldCVer-hf/CJ-Dataset)，13,048,000 条决策，1 个 epoch，5,446 步 |
| 发布检查点 | step 5446，按验证集分布 NLL 选出（NLL 0.4578167015） |
| 交付权重 | model.safetensors，BF16，8,411,558,400 字节（约 8.41 GB） |
| 默认温度 | **1.024757981300354**，独立验证划分拟合 |
| 输入 / 候选项 | 最长 16,384 tokens / 单题 2–26 项（训练覆盖 2–16 项），类别更多用 `/v1/tournament` 分组淘汰 |
| 提示版本 | hmm-zh-v1，关闭 thinking，末位候选字母 logits |
| 语言 | 中文 |

训练数据版本 a3963a61f71e00051a80721e587d720d8926e8e7。最终 adapter SHA256 为 17849c00344d40476dd4ddba6b8ccb36b3ea24940ce97d6ec6aa8e2d94bc7538；合并权重 SHA256 为 5abd1c94bc8ed883e1e7cec554b1cbe4b26ade4255b7c9ffc71c8cd152da77c0。

## 文件说明

| 文件 | 内容 |
|---|---|
| model.safetensors、config.json、generation_config.json | 合并模型（单个 safetensors 文件），无需再次加载 PEFT adapter |
| tokenizer.json、tokenizer_config.json、chat_template.jinja | tokenizer 与聊天模板 |
| decision_config.json | 模型身份、温度、提示版本、最大长度与候选数 |
| training_config.json、training_summary.json | 训练配置、数据版本、训练覆盖、选中的检查点 |
| validation_report.json | 选模验证集的校准前后指标（含分来源明细），**不是测试集结果** |
| LICENSE、LICENSES/、THIRD_PARTY_NOTICES.md | 许可与上游方法归属 |

## 快速开始

### 下载

```bash
hf download quincyqiang/chinese-decision-4b --local-dir models/chinese-decision-4b
# 或 ModelScope
modelscope download --model quincyqiang/chinese-decision-4b --local_dir models/chinese-decision-4b
```

### GPU HTTP 服务

在 [代码仓库](https://github.com/yanqiangmiffy/chinese-decision) 根目录安装推理依赖，模型目录放在 models/chinese-decision-4b：

```bash
pip install -e ".[torch]"
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=6 TOKENIZERS_PARALLELISM=false \
  python scripts/serve_gpu.py --base models/chinese-decision-4b \
  --host 127.0.0.1 --port 8789 --token-budget 16384 --max-batch 32
```

BF16 权重约 8.41 GB；CJ-Bench 全量批量评测（每批最多 65,536 tokens）时单卡峰值显存为 17.9 GB，实际占用取决于 `--token-budget` 与 `--max-batch`。测试服务：

```bash
curl http://127.0.0.1:8789/health
curl http://127.0.0.1:8789/v1/systemone -H 'Content-Type: application/json' --data-binary @examples/request.json
python scripts/verify_export.py --url http://127.0.0.1:8789
```

health 应显示 model=chinese-decision-4b、step=5446、calibrated=true、temperature=1.024757981300354。

### Python 直接读取候选概率

以下示例仅需 PyTorch 和 Transformers，不需要训练库或文本生成。项目实测环境为 torch 2.8.0、transformers 5 系列；安装约束见 pyproject.toml。固定候选顺序并读取 decision_config.json 的温度，不使用通用 text-generation pipeline。

```python
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "models/chinese-decision-4b"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
config = json.load(open(f"{path}/decision_config.json", encoding="utf-8"))
temperature, max_length = config["temperature"], config["max_length"]

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
letter_ids = [tokenizer.encode(c, add_special_tokens=False)[0] for c in LETTERS]


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
    return tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)


@torch.inference_mode()
def decide(state, question):
    options = options_for(question)
    ids = tokenizer.encode(build_prompt(state, question["instructions"], options), add_special_tokens=False)
    if len(ids) > max_length:
        raise ValueError(f"input is {len(ids)} tokens; the limit is {max_length}")
    logits = model(torch.tensor([ids], device=model.device), logits_to_keep=1, use_cache=False).logits
    probs = (logits[0, -1, letter_ids[:len(options)]].float() / temperature).softmax(-1).tolist()
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

导出验证时，上述三题在 HTTP 服务上的结果为：紧迫性 `true` 0.949，部门 `billing` 0.986，不满程度期望值 1.741（"非常愤怒" 0.759）。

| 接口 | 用途 |
|---|---|
| `POST /v1/systemone` | 决策；可追加 `?debug=1` 查看候选 logits |
| `POST /api/alpha/decisions` | 兼容 hmm 的同一决策路由 |
| `POST /v1/batch` | 多请求批处理，单个请求失败不影响其他请求 |
| `POST /v1/compile` | 编译输入并计数 tokens |
| `POST /v1/tournament` | 类别更多的 `choice` 问题：分组淘汰后决赛，每组不超过 16 个候选 |
| `GET /health`、`GET /v1/info` | 模型身份、step、温度及运行信息 |

每个请求最多 64 个问题，每题 2–26 个候选（类别更多时用 `/v1/tournament`，示例 [examples/tournament.json](https://github.com/yanqiangmiffy/chinese-decision/blob/main/examples/tournament.json) 有 41 个客服意图），最长 16,384 tokens；超长或不合法输入返回 422，不静默截断。`score` 输出从 0 开始的等级期望值，不是最大概率等级；`confidence = (K*p_max-1)/(K-1)` 衡量分布尖锐程度，不等于经验准确率。

## CJ-Bench 全量结果

CJ-Bench 全量为 **307,900 条**（304,822 条硬标签、3,078 条软标签），通用/医疗/法律/金融分别为 100,000 / 200,000 / 4,300 / 3,600 条。全部保留原始候选顺序，无过滤、无截断。ACC/ECE 仅计硬标签，ECE 使用 15 个等宽分箱；分布 NLL 与 Brier 包含软标签；Score MAE 覆盖 38,746 条评分题，计算预测期望等级与目标期望等级的绝对差。

整体 micro 按硬标签样本数加权，四领域 macro 是四个领域的等权平均，专业平均仅对医疗、法律、金融等权。不同量不能互换，整体 ECE 也不是领域 ECE 的平均。选模和温度拟合只使用验证数据，未使用 CJ-Bench 标签。

下表均为已发布的正式权重（LoRA 已合并，使用默认温度）；本仓库为 chinese-decision-4b，chinese-decision-4b-gguf 由它直接转换。**加粗**表示 4B 相对同格式 2B 提升的指标。

> **关于温度 T**：概率按 p = softmax(z / T) 计算，z 为候选字母的 logits。**T = 1** 表示直接对 logits 做 softmax、不做温度校准。发布权重附带在独立验证划分上拟合的温度（0.8B 为 1.108，2B 为 1.075，4B 为 1.025），只调整概率的尖锐程度，**不改变选出的答案**。

| 模型 | 格式 | Micro ACC (%) ↑ | Macro ACC (%) ↑ | ECE (%) ↓ | NLL ↓ | Brier ↓ | Score MAE ↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chinese-decision-0.8b | BF16 | 83.1052 | 75.7376 | 0.2706 | 0.431318 | 0.223833 | 0.584427 |
| chinese-decision-0.8b-gguf | Q8_0 | 83.1233 | 75.8092 | 0.2696 | 0.431293 | 0.223815 | 0.583973 |
| chinese-decision-2b | BF16 | 84.1107 | 77.1750 | 0.6361 | 0.416520 | 0.213295 | 0.564216 |
| chinese-decision-2b-gguf | Q8_0 | 84.1094 | 77.3282 | 0.6534 | 0.416288 | 0.213189 | 0.563738 |
| chinese-decision-4b | BF16 | **85.9731** | **81.9239** | 0.9970 | **0.370641** | **0.190859** | **0.525270** |
| chinese-decision-4b-gguf | Q8_0 | **85.9908** | **81.9521** | 0.9568 | **0.370469** | **0.190742** | **0.525313** |

相对 chinese-decision-2b，本模型 Micro **+1.86**、Macro **+4.75**、专业平均 **+5.76** 个百分点，NLL / Brier / Score MAE 分别下降 **11.0% / 10.5% / 6.9%**；整体 ECE 较高（1.00% vs 0.64%）。4B 与 2B 使用相同的训练数据、训练目标（CE + RLCD）、全局 batch、学习率和 LoRA rank，主要差别是基础模型规模。

### 分领域与分题型

| 模型 | 格式 | General ACC (%) | Medical ACC (%) | Legal ACC (%) | Finance ACC (%) | Professional avg. ACC (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| chinese-decision-0.8b | BF16 | 72.26 | 88.85 | 67.84 | 74.00 | 76.90 |
| chinese-decision-0.8b-gguf | Q8_0 | 72.23 | 88.89 | 67.84 | 74.28 | 77.00 |
| chinese-decision-2b | BF16 | 73.96 | 89.50 | 68.74 | 76.50 | 78.25 |
| chinese-decision-2b-gguf | Q8_0 | 73.91 | 89.50 | 69.40 | 76.50 | 78.47 |
| chinese-decision-4b | BF16 | **75.69** | **91.19** | **77.21** | **83.61** | **84.00** |
| chinese-decision-4b-gguf | Q8_0 | **75.69** | **91.21** | **77.35** | **83.56** | **84.04** |

本模型各领域的完整指标（"未校准 ECE"一列为合并前最终 checkpoint 在 T = 1 时的结果，用于说明温度校准的作用）：

| 领域 | 样本数 | ACC ↑ | ECE ↓ | NLL ↓ | Brier ↓ | 未校准 ECE（T = 1） |
|---|---:|---:|---:|---:|---:|---:|
| 通用 | 100,000 | 75.69 | 2.91 | 0.621 | 0.318 | 3.25 |
| 医疗 | 200,000 | 91.19 | 0.25 | 0.241 | 0.124 | 0.35 |
| 法律 | 4,300 | 77.21 | 2.02 | 0.529 | 0.304 | 1.65 |
| 金融 | 3,600 | 83.61 | 1.65 | 0.421 | 0.230 | 1.40 |
| **整体（micro）** | **307,900** | **85.97** | **1.00** | **0.371** | **0.191** | 1.26 |
| **四领域等权平均（macro）** | — | **81.92** | — | **0.453** | — | — |

| 题型 | 样本数 | 0.8B ACC (%) | 2B ACC (%) | 4B ACC (%) | 4B ECE (%) | 4B NLL | 4B Score MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `noul` | 134,577 | 94.96 | 95.11 | **95.47** | 0.43 | 0.107485 | — |
| `choice` | 134,577 | 78.32 | 80.12 | **83.25** | 1.26 | 0.474925 | — |
| `score` | 38,746 | 56.45 | 57.67 | **60.43** | 2.18 | 0.922455 | 0.525270 |

**领域 × 题型准确率（%）**

| 领域 | `noul` | `choice` | `score` |
|---|---:|---:|---:|
| 通用 | 88.50 | 78.60 | 58.35 |
| 医疗 | 98.04 | 85.04 | 67.49 |
| 法律 | 88.87 | 74.13 | 67.31 |
| 金融 | 85.67 | 77.00 | 88.17 |

12 个组合全部高于 chinese-decision-0.8b 与 chinese-decision-2b。2B 相对 0.8B 退步的两项在 4B 上都已超过 0.8B：法律判断（`noul`）88.87%（0.8B 85.60%、2B 82.07%），金融评分（`score`）88.17%（0.8B 82.83%、2B 80.33%）。提升最大的是法律 `score`（比 2B 高 12.16 个百分点）与金融 `choice`（高 9.58）。

### 与其他决策模型对比

下表的外部模型数字为 [Chinese-Jev 原文 Table 1 / Table 2](https://arxiv.org/html/2609.36965v1) 的作者报告值，本项目没有重跑这些模型；本项目行为发布权重。**加粗**为每列最优。

**通用领域（Table 1）**

| 模型 | ACC (%) ↑ | ECE (%) ↓ |
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

**专业领域（Table 2）**

| 模型 | Medical ACC | Medical ECE | Legal ACC | Legal ECE | Finance ACC | Finance ECE | Avg. ACC | Avg. ECE |
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

本模型相对 Chinese-Jev（General + 三个 Specialist）在通用 / 医疗 / 法律 / 金融分别高 **6.49 / 3.54 / 17.11 / 18.89** 个百分点，专业平均高 **13.18**；相对闭源 Jev，专业平均高 **6.99**，金融也高 **5.22**。四个领域的 ECE 都低于论文中的所有模型。Chinese-Jev General 加上医疗、法律和金融 Specialist 是 **四个分别训练的模型**；本项目每个尺寸各使用一个模型覆盖四领域。Jev 为闭源 API。论文使用 H200，本地使用 A800；延迟口径和硬件不同，不做等价排名。

训练过程中各 checkpoint 的评测结果见 [训练过程 checkpoint 评测](TRAINING_HISTORY.md)。

## Q8_0 版本

由本模型直接转换的 [4B Q8_0 GGUF](MODEL_CARD_4B_GGUF.md)（4.48 GB，llama.cpp，CPU 可运行，不需要 PyTorch）沿用同一温度，在 CJ-Bench 全部 307,900 条上与本模型逐条配对：整体 ACC **85.9908%**（本模型 85.9731%，净多对 54 条），首选项一致率 **99.19%**，ECE 0.9568%（本模型 0.9970%），NLL、Brier 略降。全量在 llama.cpp CUDA 上测得，CPU 只验证了接口与有限输入，细节见 [4B Q8_0 模型卡](MODEL_CARD_4B_GGUF.md)。

## 校准验证与测试的区别

温度只在母记录分组后的验证划分 0（1,993 条硬标签）上拟合，选模使用划分 1–4。后者固定 1,200 条验证数据上：

| 合并模型验证指标 | T = 1 | T = 1.0247579813 |
|---|---:|---:|
| 硬标签 ACC (%) | 81.7953 | 81.7953 |
| ECE（10 bins，%） | 0.8581 | 0.8902 |
| 分布 NLL | 0.458073 | 0.457840 |
| Brier | 0.247217 | 0.247165 |

4B 未校准时已接近校准，拟合温度只有 1.025：选模验证集上 NLL 与 Brier 略降，ECE 略升 0.03 个百分点；在 CJ-Bench 上，整体 ECE 从 1.26% 降到 1.00%，三种题型的 ECE 都下降，但法律（1.65% → 2.02%）和金融（1.40% → 1.65%）的 ECE 升高。验证 ECE 为 10 bins，不能混入使用 15 bins 的 CJ-Bench 测试表。全局温度不能保证所有领域、来源或题型都改善。

## 推理效率

同一组 25 条 CJ-Bench 输入（92–6,521 tokens，其中 16 条短输入为 92–255 tokens），每条预热后测 3 次，只统计已分词输入的前向与概率计算，不含分词、HTTP 和网络：

| 4B 后端 | 短输入 p50 (ms) | 全部输入 p50 (ms) | 全部 p95 (ms) | 进程内存观测最大值 (GiB) |
| --- | ---: | ---: | ---: | ---: |
| Transformers BF16 / GPU | 59.94 | 60.06 | 198.99 | 9.01 |
| Q8_0 / GPU | 39.66 | 47.29 | 569.17 | 5.32 |
| Q8_0 / CPU 16 threads | 1695.09 | 2004.01 | 31481.25 | 6.14 |

A800 80GB PCIe 与 Xeon Gold 5318Y 共享服务器，结果仅适用于该次测量。进程内存为 RSS，不是 GPU 显存。同口径下 2B 的 Transformers BF16 / GPU 全部输入 p50 为 44.77 ms，Q8_0 / CPU 为 817.99 ms：4B 的准确率更高，但单条延迟和内存也更高。Windows 台式机上的 CPU 测量见 [4B Q8_0 模型卡](MODEL_CARD_4B_GGUF.md#推理效率)。

## 训练细节

| 项目 | 设置 |
|---|---|
| 微调方式 | LoRA，r = 32，α = 64，dropout = 0.05 |
| LoRA 目标层 | 全注意力 `q/k/v/o_proj`、线性注意力 `in_proj_qkv/z/a/b`、`out_proj`、MLP `gate/up/down_proj` |
| 损失函数 | 候选分布交叉熵（软标签）+ RLCD（group size 4，σ 按已训练样本进度从 0.4 线性降至 0.1） |
| 优化器 | AdamW，lr = 1e-4，前 100 步线性预热，之后余弦衰减至峰值的 10% |
| 全局 batch | 每步累积至少 2,048 条决策；每个 micro-batch 在 padding 后最多 262,144 tokens |
| 最大长度 | 16,384 tokens，超长样本被过滤，从不截断 |
| 训练量 | 1 epoch，13,048,000 条决策，5,446 步 |
| 精度 | BF16，gradient checkpointing |
| 模型选择 | 每 1,000 步及训练结束时在验证集上评估，以分布 NLL 选出 best（step 5446） |
| 温度校准 | 在 validation 的独立划分上，以硬标签交叉熵拟合单一温度 |
| 硬件 | 4 × GPU，DDP；2026-10-17 开始，2026-10-21 完成 |

训练数据构成与 0.8B、2B 相同：通用 10,000,000 条、医疗 3,000,000 条、法律与金融各 24,000 条。法律与金融合计不到总量的 0.4%。

## 局限性

- **商品分类来源明显偏弱**：通用领域的 mep3m_product_catalog（3,151 条）ACC 为 64.58%，低于 chinese-decision-2b 的 70.96%，来源 ECE 高达 22.64%；该来源在训练过程中持续波动。整体指标不能代替来源诊断。
- **校准不如更小的尺寸**：整体 ECE 1.00%，高于 2B 的 0.64% 和 0.8B 的 0.27%；通用领域 ECE 2.91%，也高于 2B（1.93%）和 0.8B（1.68%）。全局温度使法律、金融的 ECE 升高。
- **`score` 仍是最弱题型**：准确率 60.43%，通用领域评分题只有 58.35%。
- **成本更高**：权重 8.41 GB，约为 2B 的 2.2 倍；GPU 单条 p50 约 60 ms，CPU Q8_0 约 2 秒。对延迟、内存敏感的场景请先对比 2B 或 0.8B。
- **`confidence` 不是准确率**：它只反映分布的尖锐程度。若要按阈值自动执行，请先在自己的数据上验证校准效果。
- **标签含义来自上游任务**：例如医疗匹配类标签表示原任务中的匹配关系，并不代表医学正确性。请勿将模型输出作为医疗、法律或金融决策的唯一依据。
- **不用于聊天或开放式生成**，仅针对中文固定候选任务训练；单题输入不超过 16,384 tokens。

原始报告：reports/release-4b-5446-20261021（summary.json、evaluation_analysis.json、merged-calibrated/full.json）。数据 manifest SHA256 为 461bb692676e8c01bbc89a5478a6cda9e5d0096afd2c82919d73f45a018ea351。

## 归属、许可与引用

模型权重保留基础 Qwen3.5-4B 的 Apache-2.0 LICENSE。项目代码以 MIT 为主，RLCD 改编部分保留 Apache-2.0；原文与声明见 LICENSES/ 和 THIRD_PARTY_NOTICES.md。数据许可按来源分别适用（部分来源为非商业许可），模型交付不授予数据再分发权利。

方法参考 [hmm](https://github.com/n4ze3m/hmm)，RLCD 与数据参考 [Chinese-Jev](https://github.com/gulucaptain/Chinese-Jev)。本项目没有复现 Microsoft Decision、Jev 或 Chinese-Jev 的训练过程。使用数据或基线时请同时引用 [Chinese-Jev 论文](https://arxiv.org/html/2609.36965v1)。

```bibtex
@misc{chinese-decision-2026,
  title = {Chinese Decision: A Lightweight Chinese Probabilistic Decision Model},
  author = {yanqiangmiffy},
  year = {2026},
  howpublished = {\url{https://github.com/yanqiangmiffy/chinese-decision}}
}
```
