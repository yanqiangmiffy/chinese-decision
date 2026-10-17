---
license: apache-2.0
language:
- zh
base_model: Qwen/Qwen3.5-2B
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

# Chinese Decision 2B

**中文概率决策模型：一次前向，直接返回候选项概率。**

> A Chinese decision model fine-tuned from Qwen3.5-2B with LoRA and CE + RLCD. It supports yes/no, single-choice and ordinal decisions across general, medical, legal and finance tasks. This package contains merged BF16 weights and the existing validation-fitted temperature.

[代码](https://github.com/yanqiangmiffy/chinese-decision) · [模型总览](MODELS.md) · [Decision Studio](https://github.com/yanqiangmiffy/chinese-decision-studio) · [训练数据](https://huggingface.co/datasets/bbldCVer-hf/CJ-Dataset) · [测试集](https://huggingface.co/datasets/bbldCVer-hf/CJ-Bench)

[0.8B BF16](MODEL_CARD_0_8B.md) · [0.8B Q8_0](MODEL_CARD_0_8B_GGUF.md) · [2B BF16](MODEL_CARD_2B.md) · [2B Q8_0](MODEL_CARD_2B_GGUF.md)

## 相关资源

| 资源 | 说明 | 链接 |
|---|---|---|
| 代码 | 训练、评测、GGUF 导出与 HTTP 服务（GPU / CPU） | [GitHub](https://github.com/yanqiangmiffy/chinese-decision) |
| 测试平台 | Decision Studio：决策实验台、场景库、鲁棒性实验室、评测中心与性能压测 | [GitHub](https://github.com/yanqiangmiffy/chinese-decision-studio) |
| 2B BF16（本仓库） | 合并权重与校准温度，Transformers 推理 | [Hugging Face](https://huggingface.co/quincyqiang/chinese-decision-2b) · [ModelScope](https://modelscope.cn/models/quincyqiang/chinese-decision-2b) |
| 2B GGUF Q8_0 | llama.cpp 推理，CPU 即可部署，不需要 PyTorch | [Hugging Face](https://huggingface.co/quincyqiang/chinese-decision-2b-gguf) · [ModelScope](https://modelscope.cn/models/quincyqiang/chinese-decision-2b-gguf) |
| 0.8B 系列 | 更小、更快；部分法律判断与金融评分任务更好 | [BF16](https://huggingface.co/quincyqiang/chinese-decision-0.8b) · [GGUF](https://huggingface.co/quincyqiang/chinese-decision-0.8b-gguf) |

## 简介

**Chinese Decision** 是一个面向中文场景的轻量级**决策模型**。它不聊天、不生成文本：给定上下文和一组固定候选项，一次前向直接返回每个候选项的概率，适合意图识别、工单分流、内容审核、数据打标、给模型输出打分、智能体动作选择等"有限选项判断"任务。

- **一次前向，零输出 token**：读取最后位置上候选字母 `A`–`Z` 的 logits，softmax 后即为概率。
- **三种题型**：`noul`（是/否）、`choice`（单选）、`score`（有序等级，返回期望分值）。
- **单一模型覆盖四个领域**：通用、医疗、法律、金融混合训练。
- **系列中准确率最高**：CJ-Bench 全量整体 ACC 84.11%，专业三领域平均 78.19%（未校准 adapter），比 0.8B 分别高 1.00 / 1.21 个百分点。
- **概率经过校准**：附带在独立校准划分上拟合的温度，交付版整体 ECE 0.64%。

## 模型与交付状态

更新于 2026-10-17。BF16 合并模型已完成全量 CJ-Bench、保存/重载和实际 HTTP 三题型/双路由验证，并已在 Hugging Face 与 ModelScope 公开发布。由本模型直接转换的 Q8_0 GGUF 也已完成全量 CJ-Bench 对照，见 [2B Q8_0 模型卡](MODEL_CARD_2B_GGUF.md)。

| 项目 | 值 |
|---|---|
| 基础模型 | [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)，文本决策主干 |
| 训练目标 | 候选分布交叉熵 + RLCD；LoRA rank 32 |
| 训练覆盖 | 1 epoch，13,048,000 条决策，4,994 步，4 卡 |
| best | 验证集 distribution NLL 选择 step 4994，NLL 0.5860035124 |
| RLCD | group size 4，sigma 按已训练样本进度从 0.4 降至 0.1 |
| 交付权重 | model.safetensors，BF16，3,763,692,048 字节 |
| 默认温度 | **1.074872374534607**，独立验证划分拟合 |
| 输入 / 候选项 | 最长 16,384 tokens / 2–26 项 |
| 提示版本 | hmm-zh-v1，关闭 thinking，末位候选字母 logits |
| 训练结束 | 2026-10-17 17:25:54，Asia/Shanghai |

训练数据版本 a3963a61f71e00051a80721e587d720d8926e8e7。最终 adapter SHA256 为 9b55e8eda00997544b76fa945d890605c3acae4199041c197ef5a09275142a7c；合并权重 SHA256 为 a8fadd326b19a278f8c26216ca481b607369baca33962aac0d8fda603c830068。

## 文件说明

| 文件 | 内容 |
|---|---|
| model.safetensors、config.json | 合并模型，无需再次加载 PEFT adapter |
| tokenizer.json、tokenizer_config.json、chat_template.jinja | tokenizer 与聊天模板 |
| decision_config.json | 模型身份、温度、最大长度与候选数 |
| training_config.json、training_summary.json | 配置、训练覆盖、best 的来源 |
| validation_report.json | 选模验证集的校准前后指标，**不是测试集结果** |
| LICENSE、LICENSES/、THIRD_PARTY_NOTICES.md | 许可与上游方法归属 |

## 快速开始

### 下载

```bash
hf download quincyqiang/chinese-decision-2b --local-dir models/chinese-decision-2b
# 或 ModelScope
modelscope download --model quincyqiang/chinese-decision-2b --local_dir models/chinese-decision-2b
```

### GPU HTTP 服务

在 [代码仓库](https://github.com/yanqiangmiffy/chinese-decision) 根目录安装推理依赖，模型目录放在 models/chinese-decision-2b：

```bash
pip install -e ".[torch]"
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=6 TOKENIZERS_PARALLELISM=false \
  python scripts/serve_gpu.py --base models/chinese-decision-2b \
  --host 127.0.0.1 --port 8789 --token-budget 16384 --max-batch 32
```

启动前选择空闲 GPU；本命令不需要访问训练 run 或加载 adapter。测试服务：

```bash
curl http://127.0.0.1:8789/health
curl http://127.0.0.1:8789/v1/systemone -H 'Content-Type: application/json' --data-binary @examples/request.json
python scripts/verify_export.py --url http://127.0.0.1:8789
```

### Python 直接读取候选概率

以下示例仅需 PyTorch 和 Transformers，不需要训练库或文本生成。项目实测环境为 torch 2.8.0、transformers 5.3 系列；安装约束见 pyproject.toml。固定候选顺序并读取 decision_config.json 的温度，不使用通用 text-generation pipeline。

```python
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "models/chinese-decision-2b"
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

| 接口 | 用途 |
|---|---|
| `POST /v1/systemone` | 决策；可追加 `?debug=1` 查看候选 logits |
| `POST /api/alpha/decisions` | 兼容 hmm 的同一决策路由 |
| `POST /v1/batch` | 多请求批处理，单个请求失败不影响其他请求 |
| `POST /v1/compile` | 编译输入并计数 tokens |
| `GET /health`、`GET /v1/info` | 模型身份、step、温度及运行信息 |

每个请求最多 64 个问题，每题 2–26 个候选，最长 16,384 tokens；超长或不合法输入返回 422，不静默截断。`score` 输出从 0 开始的等级期望值，不是最大概率等级；`confidence = (K*p_max-1)/(K-1)` 衡量分布尖锐程度，不等于经验准确率。

## CJ-Bench 全量结果

CJ-Bench 全量为 **307,900 条**（304,822 条硬标签、3,078 条软标签），通用/医疗/法律/金融分别为 100,000 / 200,000 / 4,300 / 3,600 条。全部保留原始候选顺序，无过滤、无截断。ACC/ECE 仅计硬标签，ECE 使用 15 个等宽分箱；分布 NLL 与 Brier 包含软标签；Score MAE 覆盖 38,746 条评分题，计算预测期望等级与目标期望等级的绝对差。

整体 micro 按硬标签样本数加权，四领域 macro 是四个领域的等权平均，专业平均仅对医疗、法律、金融等权。不同量不能互换，整体 ECE 也不是领域 ECE 的平均。选模和温度拟合只使用验证数据，未使用 CJ-Bench 标签。

| 模型 | Micro ACC (%) ↑ | Macro ACC (%) ↑ | ECE (%) ↓ | NLL ↓ | Brier ↓ | Score MAE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8B adapter / T=1 | 83.1088 | 75.7970 | 1.0656 | 0.432327 | 0.224075 | 0.578770 |
| 2B adapter / T=1 | 84.1097 | 77.1313 | 1.4914 | 0.419089 | 0.213828 | 0.560682 |
| 0.8B BF16 / T=1.108133 | 83.1052 | 75.7376 | 0.2706 | 0.431318 | 0.223833 | 0.584427 |
| 2B BF16 / T=1.074872 | 84.1107 | 77.1750 | 0.6361 | 0.416520 | 0.213295 | 0.564216 |
| 2B Q8_0 / T=1.074872 | 84.1094 | 77.3282 | 0.6534 | 0.416288 | 0.213189 | 0.563738 |

前两行用于比较未校准训练方案，中间两行代表默认部署配置（本仓库为 2B BF16），最后一行是由本仓库转换的 [Q8_0 GGUF](MODEL_CARD_2B_GGUF.md)。2B adapter 与本目录交付版的差异同时包含 BF16 合并取整和温度变换。固定 logits 上的温度不会改变 argmax，但合并后有限精度运算会改变部分结果：同批次全验证集审计为 14/1200 条 argmax 变化（硬标签 13 条），不能称为完全数值一致。

### 分领域与分题型

| 模型 | General ACC (%) | Medical ACC (%) | Legal ACC (%) | Finance ACC (%) | Professional avg. ACC (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.8B adapter / T=1 | 72.24 | 88.86 | 67.86 | 74.22 | 76.98 |
| 2B adapter / T=1 | 73.95 | 89.50 | 69.05 | 76.03 | 78.19 |
| 2B BF16 / T=1.074872 | 73.96 | 89.50 | 68.74 | 76.50 | 78.25 |
| 2B Q8_0 / T=1.074872 | 73.91 | 89.50 | 69.40 | 76.50 | 78.47 |

| 题型 | 样本数 | 2B adapter ACC (%) | 交付 ACC (%) | 交付 ECE (%) | 交付 NLL | 交付 Score MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `noul` | 134,577 | 95.09 | 95.11 | 0.34 | 0.114612 | — |
| `choice` | 134,577 | 80.14 | 80.12 | 0.90 | 0.559377 | — |
| `score` | 38,746 | 57.64 | 57.67 | 1.65 | 0.968959 | 0.564216 |

### 训练阶段

每 1000 步按验证集分布 NLL 选出的 checkpoint，均在全量 307,900 条上评测（T=1 未合并 adapter）。进度按实际已训练样本计算。

| Checkpoint | 训练进度 | General ACC (%) | Medical ACC (%) | Legal ACC (%) | Finance ACC (%) | Professional avg. ACC (%) | ECE (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| step 1000 | 20.08% | 70.55 | 87.69 | 55.67 | 66.94 | 70.10 | 2.03 |
| step 2000 | 40.13% | 72.56 | 88.54 | 62.67 | 71.44 | 74.22 | 1.27 |
| step 3000 | 60.11% | 71.87 | 88.99 | 65.56 | 74.31 | 76.28 | 1.63 |
| step 4000 | 80.18% | 73.77 | 89.42 | 68.40 | 76.03 | 77.95 | 0.85 |
| **step 4994（发布）** | 100.00% | 73.95 | 89.50 | 69.05 | 76.03 | 78.19 | 1.49 |

法律、金融提升最多（+13.37 / +9.08 个百分点）。step3000 通用领域的回落来自商品分类来源的暂时崩落，step4000 恢复；后段准确率趋于平台。完整诊断见 [训练交付记录](TRAINING_2B.md)。

### 与其他决策模型对比

下表的外部模型数字为 [Chinese-Jev 原文 Table 1 / Table 2](https://arxiv.org/html/2609.36965v1) 的作者报告值，2026-10-17 核对，本项目没有重跑这些模型。项目行使用 T=1 adapter；部署结果另列。

**通用领域（Table 1）**

| 模型 | ACC (%) ↑ | ECE (%) ↓ |
| --- | ---: | ---: |
| Qwen3.5-2B | 32.95 | 43.68 |
| Open-Jev | 52.33 | 10.54 |
| SemIF | 44.21 | 27.27 |
| Laya Multilingual | 40.22 | 23.04 |
| Jev (API) | 68.35 | 11.45 |
| Chinese-Jev General / Specialists | 69.20 | 3.78 |
| 0.8B adapter / T=1 | 72.24 | 3.03 |
| 2B adapter / T=1 | 73.95 | 3.01 |

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
| 0.8B adapter / T=1 | 88.86 | 0.17 | 67.86 | 5.70 | 74.22 | 1.50 | 76.98 | 2.46 |
| 2B adapter / T=1 | 89.50 | 0.73 | 69.05 | 3.78 | 76.03 | 1.98 | 78.19 | 2.17 |

Chinese-Jev General 加上医疗、法律和金融 Specialist 是 **四个分别模型**；本项目每个尺寸各使用一个模型覆盖四领域。Jev 为闭源 API。论文使用 H200，本地使用 A800；延迟口径和硬件不同，不做等价排名。

2B 与 0.8B 的模型规模、batch 和训练目标同时变化，因此结果不能当作 RLCD 的独立收益。两者 best 已见样本分别为 13,048,000 与 12,879,232，应按样本量比较进度。

## Q8_0 版本

由本模型直接转换的 [2B Q8_0 GGUF](MODEL_CARD_2B_GGUF.md)（2.01 GB，llama.cpp，CPU 可运行，不需要 PyTorch）沿用同一温度，在 CJ-Bench 全部 307,900 条上与本模型逐条配对：整体 ACC **84.1094%**（本模型 84.1107%，净少对 4 条），首选项一致率 **99.00%**，ECE 0.6534%（本模型 0.6361%），NLL、Brier、Score MAE 略降。四领域 macro 77.3282%、专业平均 78.4663%，高出的部分主要来自法律领域（4,300 条中净多对 28 条），不代表量化提升了能力。

全量在 llama.cpp CUDA 上测得，CPU 只验证了有限输入。分领域、分题型、来源变化、BF16 GGUF 后端对照与 CPU / GPU 延迟见 [2B Q8_0 模型卡](MODEL_CARD_2B_GGUF.md)。

## 校准验证与测试的区别

温度只在母记录分组后的验证划分 0 上拟合，选模使用划分 1–4。后者固定 1200 条验证数据上：

| 合并模型验证指标 | T=1 | T=1.0748723745 |
|---|---:|---:|
| 硬标签 ACC (%) | 76.0906 | 76.0906 |
| ECE（10 bins，%） | 2.3324 | 1.0608 |
| 分布 NLL | 0.585104 | 0.582978 |
| Brier | 0.316658 | 0.315944 |

上述验证 ECE 为 10 bins，不能混入使用 15 bins 的 CJ-Bench 测试表。全局温度不能保证所有领域、来源或题型都改善。

## 性能与已知局限

同服务器、25 条输入、每条预热后测 3 次，Transformers BF16 GPU 的单请求 p50 为 **44.77 ms**；短输入（92–255 tokens）p50 为 **44.68 ms**。这是预先分词后的前向及概率计算，不含分词、HTTP 或网络，且为共享 A800 服务器。完整长度范围为 92–6521 tokens，不能从这些输入推断生产吞吐。

- 最终 2B 的整体准确率提升，但法律判断与金融评分仍落后于 0.8B。未校准法律 noul 为 82.53%（0.8B 为 85.80%），金融 score 为 79.83%（0.8B 为 83.08%）；交付版分别为 82.07% 和 80.33%，Q8_0 为 82.73% 和 80.58%。
- 商品分类来源 3,151 条在训练中波动：step3000 严重退化，step4000 恢复，最终未校准 ACC 为 70.99%、NLL 1.6029，交付 ACC 为 70.96%、NLL 1.5125。整体指标不能代替来源诊断。
- 交付版整体 ECE 更低，但 Score MAE 比 adapter 略高；金融领域仍低于闭源 Jev 的作者报告准确率。
- 不用于聊天或开放式生成；仅针对中文固定候选任务训练。标签含义来自原任务，不能代替专业正确性判断。阈值与人工复核区间需用实际业务数据验证。

原始报告：reports/cj-bench-2b-step4994-20261017 与 reports/cj-bench-2b-step4994-calibrated-20261017。数据 manifest SHA256 为 461bb692676e8c01bbc89a5478a6cda9e5d0096afd2c82919d73f45a018ea351。更多训练进度和来源诊断见 [训练交付记录](TRAINING_2B.md)。

## 归属、许可与引用

模型权重保留基础 Qwen3.5-2B 的 Apache-2.0 LICENSE。项目代码以 MIT 为主，RLCD 改编部分保留 Apache-2.0；原文与声明见 LICENSES/ 和 THIRD_PARTY_NOTICES.md。数据许可按来源分别适用，模型交付不授予数据再分发权利。

方法参考 [hmm](https://github.com/n4ze3m/hmm)，RLCD 与数据参考 [Chinese-Jev](https://github.com/gulucaptain/Chinese-Jev)。本项目没有复现 Jev 或 Chinese-Jev 的训练过程。使用数据或基线时请同时引用 [Chinese-Jev 论文](https://arxiv.org/html/2609.36965v1)。

```bibtex
@misc{chinese-decision-2026,
  title = {Chinese Decision: A Lightweight Chinese Probabilistic Decision Model},
  author = {yanqiangmiffy},
  year = {2026},
  howpublished = {\url{https://github.com/yanqiangmiffy/chinese-decision}}
}
```
