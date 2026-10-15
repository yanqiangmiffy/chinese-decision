<div align="center">

# Chinese Decision

**中文概率决策模型：一次前向，直接输出候选项概率**

[![Base Model](https://img.shields.io/badge/Base-Qwen3.5--0.8B-blue)](https://huggingface.co/Qwen/Qwen3.5-0.8B)
[![Dataset](https://img.shields.io/badge/Data-CJ--Dataset-orange)](https://huggingface.co/datasets/bbldCVer-hf/CJ-Dataset)
[![Benchmark](https://img.shields.io/badge/Bench-CJ--Bench-green)](https://huggingface.co/datasets/bbldCVer-hf/CJ-Bench)
[![Method](https://img.shields.io/badge/Method-hmm-lightgrey)](https://github.com/n4ze3m/hmm)

中文 | [English](README_EN.md)

</div>

---

## 简介

**Chinese Decision** 是一个面向中文场景的轻量级**决策模型**。它不聊天、不生成文本，只做一件事：给定上下文和一组固定候选项，返回每个候选项的概率。

很多 AI 调用本质上是在有限选项中做判断——意图识别、工单分流、内容审核、数据打标、给模型输出打分、智能体的下一步动作选择等。用生成式大模型回答这类问题又慢又贵，且输出需要再解析。决策模型直接返回结构化的概率分布，程序拿到即可按阈值使用。

本项目基于 [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)，在 1,300 万条中文决策数据 [CJ-Dataset](https://huggingface.co/datasets/bbldCVer-hf/CJ-Dataset) 上进行 LoRA 微调，方法改编自 [n4ze3m/hmm](https://github.com/n4ze3m/hmm)，并在 [Chinese-Jev](https://arxiv.org/abs/2609.36965) 提出的 CJ-Bench 上与其论文结果对比。

**主要特点**

- ⚡ **一次前向，零输出 token**：读取最后位置上候选字母 `A`–`Z` 的 logits，softmax 后即为概率；不做自回归生成，也没有思维链。
- 🧩 **三种题型**：`noul`（是/否）、`choice`（单选）、`score`（有序等级，返回期望分值）。
- 🏆 **对标 Chinese-Jev**：CJ-Bench 通用领域准确率 **72.24%**，高于闭源 Jev（68.35%）和 Chinese-Jev General（69.20%）；**单一模型**在医疗、法律、金融上的准确率均超过 Chinese-Jev 的三个领域专用模型，平均 **76.98%**，与闭源 Jev（77.01%）基本持平，其中医疗 **88.86%** 为对比模型中最高。
- 📏 **概率可用**：未经温度校准，CJ-Bench 整体 ECE 为 **1.07%**；专业领域平均 ECE 为 **2.46%**，是对比模型中最低的。
- 🪶 **轻量部署**：0.8B 参数，单卡 A800 单条中位延迟约 70 ms，批量吞吐约 132 条/秒，峰值显存 6.7 GB。
- 🔌 **兼容 hmm / System One 接口**：提供开箱即用的 HTTP 服务。

## 最新动态

- **2026-10-15**：完整训练结束（1 epoch，13,048,000 条决策，10,132 步）。验证集最优的 step 10000 在 CJ-Bench 上通用领域准确率 72.24%，专业领域平均 76.98%，与闭源 Jev 基本持平；已合并权重，并在验证集的独立划分上拟合温度 T = 1.108。
- **2026-10-14**：完成 step 5000 checkpoint 在 CJ-Bench 全量测试集（307,900 条）上的评测，并与 Chinese-Jev 论文的 Table 1、Table 2 对比。

## 模型

| 模型 | 基础模型 | 参数量 | 最大输入 | 候选数 | 状态 |
|---|---|---:|---:|---:|---|
| `chinese-decision-0.8b` | Qwen3.5-0.8B | 0.8B | 16,384 tokens | 2–26 | 训练完成（step 10000） |

## 应用案例

以下截图来自我们的测试平台 Decision Studio，后端运行最终的 step 10000 模型。这些场景采用同一种设计：**业务规则写在代码里，决策模型只负责给出概率**。

### 工单分诊台：一次请求，六个判断

每张工单在一次请求里同时问 6 个问题：类别（`choice`），严重程度和情绪（`score`），是否可复现、是否要求退款、是否有流失风险（`noul`）。不管后续用不用得上都一并提问，省去多次往返。路由规则写在代码里，根据这些概率决定工单派给哪个队列；类别置信度低于 0.5 时转人工复核。截图中 8 张工单 × 6 个问题共 48 次决策，用时 285 ms。

![工单分诊台：一次请求判断类别、严重程度、情绪、可复现、退款和流失风险，再由代码路由](assets/scenario-triage.png)

### LLM 护栏：带人工复核区间的安全检查

进出大模型的每句话都先过一道检查。输入侧检查提示注入 / 越狱、隐私信息、辱骂威胁、超出业务范围；输出侧检查泄露系统提示词、越权承诺、回复含隐私、不当内容。每项都是一个是 / 否概率，配两个阈值：低于复核阈值自动放行，高于拦截阈值自动拦截，中间交给人工，复核区间的宽度就是人工审核的预算。截图中的"开发者模式"越狱请求在输入侧就被拦截（85.2%），大模型不会被调用；8 项检查共用时 89.2 ms。

![LLM 护栏：输入和输出各 4 项检查，按阈值放行、拦截或转人工复核](assets/scenario-guardrail.png)

### 弹幕护卫：直播弹幕实时审核

每条弹幕一次请求，包含一个分类和一个是 / 否判断：辱骂和广告直接拦截，剧透折叠，观众提问高亮推给主播。截图中审核延迟 p50 为 82.2 ms。页面也记录了已知短板：剧透判断容易被"剧透"这个字眼本身带偏（例如"请勿剧透"），也分不清"猜测"和"剧透"。

![弹幕护卫：实时拦截辱骂与广告、折叠剧透、高亮提问](assets/scenario-danmaku.png)

### 智能体路由器：按风险分级的置信度门控

智能体收到每句话后，先由决策模型选择工具、判断信息是否完整、评估操作风险，再按风险等级套用不同的置信度门槛：只读操作（天气、搜索、闲聊）40%，写操作（日程、消息、家居）65%，资金操作（转账、支付）85%。截图中"帮我约王总下周二下午聊一下预算"被识别为日程管理（置信度 99%），但缺少关键信息的概率为 71%，超过 70% 的追问门槛，于是先追问"具体是哪天、几点、和谁"，用时 81.6 ms。

![智能体路由器：选择工具、判断是否追问，并按操作风险设置置信度门槛](assets/scenario-agent.png)

## 工作原理

```mermaid
flowchart LR
    A["上下文 + 问题 + 候选项"] --> B["提示编译器<br/>hmm-zh-v1"]
    B --> C["Qwen3.5-0.8B<br/>+ LoRA"]
    C --> D["末位 logits<br/>A … Z"]
    D --> E["屏蔽多余字母<br/>softmax(z / T)"]
    E --> F["noul / choice / score"]
```

每个问题被编译成如下提示，套用 Qwen 聊天模板（关闭 thinking）：

```text
待判断的上下文：
{state}

问题：
{instructions}

候选项：
A: {key₁} — {描述₁}
B: {key₂} — {描述₂}
...
只返回候选项对应的字母。
```

训练、评测和服务共用同一个提示编译器，保证三者输入完全一致。

### 题型说明

| 类型 | `criteria` 格式 | 返回字段 |
|---|---|---|
| `noul` | 可选，`{"false": "...", "true": "..."}`，默认“否 / 是” | `noul`：判断为真的概率 |
| `choice` | 对象 `{键: 描述}` | `choice`：概率最大的键 |
| `score` | 列表，按等级从低到高排列 | `score`：等级下标的概率加权期望（0 到 K−1）；`legend`：下标到描述的映射 |

所有题型都会返回完整的 `probabilities` 和 `confidence`。其中 `confidence = (K·p_max − 1) / (K − 1)` 衡量分布的尖锐程度：均匀分布为 0，确定为 1。它**不等于**经验准确率。

## 评测结果

### 与 Chinese-Jev 论文对比

[Chinese-Jev](https://arxiv.org/abs/2609.36965)（Wang et al., 2026）提出了 CJ-Dataset 与 CJ-Bench。下表中的基线数字均**直接引自该论文**的 Table 1 和 Table 2，本项目未复现。本项目结果为 **step 10000 checkpoint**（完整训练 1 个 epoch、共 10,132 步，按验证集分布 NLL 选出的最优 adapter），未做温度校准（T = 1）。每列最优结果**加粗**。

**通用领域（对应论文 Table 1）**

| 模型 | ACC (%) ↑ | ECE (%) ↓ | 延迟 (ms) ↓ |
|---|---:|---:|---:|
| Qwen3.5-2B | 32.95 | 43.68 | 280 |
| Open-Jev | 52.33 | 10.54 | 60 |
| SemIF | 44.21 | 27.27 | 57 |
| Laya Multilingual | 40.22 | 23.04 | **14** |
| Jev † | 68.35 | 11.45 | 284 |
| Chinese-Jev General | 69.20 | 3.78 | **14** |
| **Chinese Decision（本项目）** | **72.24** | **3.03** | 70 ‡ |

**专业领域（对应论文 Table 2）**

| 模型 | 医疗 ACC ↑ | 医疗 ECE ↓ | 法律 ACC ↑ | 法律 ECE ↓ | 金融 ACC ↑ | 金融 ECE ↓ | 平均 ACC ↑ | 平均 ECE ↓ | 平均延迟 (ms) ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-2B | 53.88 | 28.71 | 31.53 | 44.82 | 38.11 | 49.12 | 41.17 | 40.88 | 322.33 |
| Open-Jev | 73.08 | 2.88 | 44.07 | 23.15 | 48.11 | 10.19 | 55.09 | 12.07 | 78.67 |
| SemIF | 62.17 | 10.39 | 43.07 | 19.99 | 47.89 | 18.45 | 51.04 | 16.28 | 51.33 |
| Laya | 24.55 | 43.05 | 34.72 | 31.49 | 38.31 | 26.18 | 32.53 | 33.57 | **15.00** |
| Jev † | 84.29 | 5.70 | **68.35** | 8.03 | **78.39** | 2.98 | **77.01** | 5.57 | 256.33 |
| Chinese-Jev-G | 39.77 | 17.33 | 38.76 | 15.37 | 35.39 | 14.43 | 37.97 | 15.71 | **15.00** |
| Chinese-Jev-S（3 个领域专用模型） | 87.65 | 7.25 | 60.10 | 17.45 | 64.72 | 11.55 | 70.82 | 12.08 | **15.00** |
| **Chinese Decision（单一模型）** | **88.86** | **0.17** | 67.86 | **5.70** | 74.22 | **1.50** | 76.98 | **2.46** | 70 ‡ |

> † Jev 为托管 API，延迟包含网络往返时间。
> ‡ 本项目延迟在单张 A800 上测得，且为四个领域的整体中位数；论文中其余模型在 H200 上测得，两者不能直接比较。
> “平均”为医疗、法律、金融三个领域指标的等权平均，与论文口径一致。

**主要结论**

- **通用领域**：准确率与 ECE 均为表中最优，准确率比 Jev 高 3.89 个百分点，比 Chinese-Jev General 高 3.04 个百分点。
- **专业领域**：医疗、法律、金融三个领域的准确率**均超过 Chinese-Jev-S**（分别高 1.21、7.76、9.50 个百分点），平均准确率 76.98%，与 Jev（77.01%）基本持平。医疗比 Jev 高 4.57 个百分点，为表中最高；法律仅比 Jev 低 0.49，金融低 4.17。
- **校准**：未做任何温度校准，医疗、金融、通用、法律的 ECE 分别为 0.17%、1.50%、3.03%、5.70%；专业领域平均 ECE 为 2.46%，仍是表中最低，约为 Chinese-Jev-S 的 1/5。
- **单一模型**：Chinese-Jev-S 是三个分别微调的领域专用模型，本项目用**一个模型**覆盖全部四个领域。
- **延迟**：本模型参数量约为 Chinese-Jev（约 322M）的 2.5 倍，单条延迟更高。

**对比口径**

- 测试集相同：CJ-Bench 全量，通用 100,000、医疗 200,000、法律 4,300、金融 3,600 条。
- 指标定义相同：ACC 与 ECE 仅在单标签样本上计算（含硬标签 `score`），软标签样本不计入；ECE 使用 15 个等宽分箱；不在测试集上做任何校准。
- 候选项保持原始顺序，无过滤、无截断。

**与 Chinese-Jev 的方法差异**

| | Chinese-Jev | Chinese Decision（本项目） |
|---|---|---|
| 骨干网络 | mmBERT-base 编码器 + 决策 Transformer（约 322M） | Qwen3.5-0.8B 解码器 + LoRA |
| 候选打分 | 在候选标记位置上用共享 MLP 打分 | 读取末位候选字母的 logits |
| 训练目标 | 交叉熵 + RLCD 策略梯度项 | 候选分布交叉熵（RLCD 可选，发布结果未使用） |
| 训练方式 | 先做通用预训练，再分别微调三个领域专用模型 | 四个领域混合训练，单一模型 |

### 不同训练阶段的结果

同一次训练中按验证集分布 NLL 先后选出的四个 best checkpoint，都在 CJ-Bench 全量测试集上以相同口径评测（未校准，307,900 条全覆盖）。最后两行引自 Chinese-Jev 论文，供参考。

| Checkpoint | 训练进度 | 验证集 NLL ↓ | 通用 ACC ↑ | 医疗 ACC ↑ | 法律 ACC ↑ | 金融 ACC ↑ | 专业平均 ACC ↑ | 专业平均 ECE ↓ | 整体 ECE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| step 2000 | 19.8% | 0.739 | 69.04 | 86.78 | 54.07 | 64.17 | 68.34 | 2.73 | 1.63 |
| step 3000 | 约 30% | 0.691 | 70.26 | 87.22 | 58.93 | 68.67 | 71.61 | 1.75 | 1.25 |
| step 5000 | 约 49% | 0.676 | 70.90 | 87.89 | 63.37 | 71.17 | 74.14 | 1.49 | 0.84 |
| **step 10000（最终）** | 约 99% | 0.600 | 72.24 | 88.86 | 67.86 | 74.22 | 76.98 | 2.46 | 1.07 |
| *参考：Chinese-Jev（G / S）* | — | — | 69.20 | 87.65 | 60.10 | 64.72 | 70.82 | 12.08 | — |
| *参考：Jev* | — | — | 68.35 | 84.29 | 68.35 | 78.39 | 77.01 | 5.57 | — |

> 训练进度为已训练决策数占一个 epoch（13,048,000 条）的比例；step 3000、5000、10000 按全程平均每步约 1,288 条决策估算（13,048,000 条 / 10,132 步）。Chinese-Jev 的通用领域使用 General 模型，专业领域使用三个领域专用模型（S）。

- **四个领域持续提升**：从 step 2000 到 step 10000，法律提升 13.79 个百分点，金融提升 10.06，通用提升 3.20，医疗提升 2.08。
- **从 step 3000 起超过 Chinese-Jev-S**：专业领域平均准确率从 68.34% 提升到 76.98%，与 Jev（77.01%）的差距从 8.67 缩小到 0.03 个百分点。
- **校准没有随训练单调改善**：在不做温度校准的情况下，整体 ECE 从 1.63% 降到 step 5000 的 0.84%，最终为 1.07%；专业领域平均 ECE 在 step 5000 最低（1.49%），最终为 2.46%，主要来自法律领域（ECE 从 2.30% 升到 5.70%）。
- 验证集 NLL 持续下降（0.739 → 0.600），与测试集准确率的提升方向一致。

### 完整结果

<details>
<summary><b>CJ-Bench 分领域 / 分题型明细</b></summary>

> step 10000，未校准（T = 1），单卡 A800 80GB，BF16。ACC 与 ECE 仅计硬标签；NLL 与 Brier 覆盖全部样本（含软标签）。

**分领域**

| 领域 | 样本数 | ACC ↑ | ECE ↓ | NLL ↓ | Brier ↓ |
|---|---:|---:|---:|---:|---:|
| 通用 General | 100,000 | 72.24 | 3.03 | 0.693 | 0.356 |
| 医疗 Medical | 200,000 | 88.86 | 0.17 | 0.293 | 0.152 |
| 法律 Legal | 4,300 | 67.86 | 5.70 | 0.728 | 0.414 |
| 金融 Finance | 3,600 | 74.22 | 1.50 | 0.593 | 0.336 |
| **整体（micro）** | **307,900** | **83.11** | **1.07** | **0.432** | **0.224** |
| **四领域等权平均（macro）** | — | **75.80** | — | **0.577** | — |

> 医疗样本占测试集的 65%，整体 micro 指标主要反映医疗领域。跨领域比较时建议参考 macro 平均。

**分题型**

| 题型 | 样本数 | ACC ↑ | ECE ↓ | NLL ↓ | 期望等级 MAE ↓ |
|---|---:|---:|---:|---:|---:|
| `noul` | 134,577 | 94.95 | 0.39 | 0.117 | — |
| `choice` | 134,577 | 78.33 | 1.48 | 0.588 | — |
| `score` | 38,746 | 56.47 | 2.35 | 0.987 | 0.579 |

**领域 × 题型准确率（%）**

| 领域 | `noul` | `choice` | `score` |
|---|---:|---:|---:|
| 通用 | 87.57 | 72.88 | 54.66 |
| 医疗 | 97.81 | 80.57 | 66.53 |
| 法律 | 85.80 | 64.13 | 51.46 |
| 金融 | 76.50 | 63.08 | 83.08 |

</details>

<details>
<summary><b>微调前后对比（验证集）</b></summary>

在官方验证集的分层子集（4 个领域 × 3 种题型，每组 100 条，共 1,200 条）上，对比基础模型零样本与 step 10000 的结果：

| 领域 | Qwen3.5-0.8B ACC | **Chinese Decision ACC** | Qwen3.5-0.8B ECE | **Chinese Decision ECE** |
|---|---:|---:|---:|---:|
| 通用 | 36.64 | **75.34** | 28.25 | **3.76** |
| 医疗 | 16.67 | **82.00** | 43.40 | **3.25** |
| 法律 | 40.67 | **65.33** | 24.03 | **7.62** |
| 金融 | 44.33 | **71.33** | 16.77 | **3.41** |
| **整体** | 34.56 | **73.49** | 27.45 | **2.48** |

</details>

### 推理效率

| 指标 | 数值 |
|---|---:|
| 单条延迟（batch = 1，预热后中位数） | 70.1 ms |
| 批量吞吐 | 132 条决策/秒 |
| 批量评测峰值显存 | 6.7 GB |
| CJ-Bench 平均输入长度 | 223 tokens |

> 硬件为单张 A800 80GB PCIe，BF16。延迟包含本地输入编码、前向计算和概率输出，不含 HTTP 和网络开销。

## 快速开始

### 安装

```bash
git clone https://github.com/yanqiangmiffy/chinese-decision.git
cd chinese-decision
pip install -e .
```

需要 Python ≥ 3.11 和 CUDA GPU。主要依赖：`torch>=2.8`、`transformers>=5.3`、`peft>=0.19`、`flash-linear-attention>=0.4`。如果环境里已经装好了 CUDA 版 PyTorch，不希望 pip 改动它，可以使用 `pip install -e . --no-deps`。

### 准备模型与数据

模型和数据均从本地目录加载。可以使用 Hugging Face CLI 下载（CJ 数据集目前需要申请访问，下载前先运行 `hf auth login`）：

```bash
hf download Qwen/Qwen3.5-0.8B --local-dir models/Qwen3.5-0.8B
```

```bash
hf download bbldCVer-hf/CJ-Dataset --repo-type dataset --local-dir datasets/CJ-Dataset
```

```bash
hf download bbldCVer-hf/CJ-Bench --repo-type dataset --local-dir datasets/CJ-Bench
```

### Python 调用

```python
import json
from pathlib import Path

import torch
from decision.model import forward_batch, letter_tokens, load_model, load_tokenizer
from decision.schema import compile_prompt, options_for

base = "models/chinese-decision-0.8b"  # 合并后的模型；也可用 load_model("models/Qwen3.5-0.8B", adapter="runs/cj-ddp/best")
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

### HTTP 服务

```bash
python scripts/serve.py --base models/chinese-decision-0.8b --port 8787
```

也可以直接加载基础模型和 adapter：`--base models/Qwen3.5-0.8B --adapter runs/cj-ddp/best`（此时温度为 1，即未校准）。

一个请求可以同时包含多个问题：

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

返回格式如下（数值仅为格式示意）：

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

| 接口 | 说明 |
|---|---|
| `POST /v1/systemone` | 决策接口 |
| `POST /api/alpha/decisions` | 同上，兼容 hmm 路由 |
| `GET /health` | 健康检查，返回当前温度及是否已校准 |

单个请求最多 64 个问题，每个问题 2–26 个候选项，单题输入不超过 16,384 tokens。超长输入**不会被截断**，非法请求返回 `422`。

检查一个正在运行的服务（两个路由、概率格式、非法请求处理）：

```bash
python scripts/verify_export.py --url http://127.0.0.1:8787
```

## 训练

### 训练配置

| 项目 | 设置 |
|---|---|
| 基础模型 | Qwen3.5-0.8B（文本主干） |
| 微调方式 | LoRA，r = 32，α = 64，dropout = 0.05 |
| LoRA 目标层 | 全注意力 `q/k/v/o_proj`、线性注意力 `in_proj_qkv/z/a/b`、`out_proj`、MLP `gate/up/down_proj` |
| 损失函数 | 候选分布交叉熵（软标签；硬标签为其 one-hot 特例）；RLCD 项可选，默认关闭 |
| 优化器 | AdamW，lr = 1e-4，前 100 步线性预热，之后余弦衰减至峰值的 10% |
| 全局 batch | 每步累积至少 1,024 条决策；每个 micro-batch 在 padding 后最多 262,144 tokens，长样本自动减小 micro-batch |
| 最大长度 | 16,384 tokens |
| 训练量 | 1 epoch，13,048,000 条决策，10,132 步 |
| 精度 | BF16，gradient checkpointing |
| 模型选择 | 每 1,000 步在验证集上评估，以分布 NLL 选出 best（最终为 step 10000） |
| 温度校准 | 在 validation 的独立划分上拟合单一温度，T = 1.108 |
| 硬件 | 4 × GPU，DDP |

### 数据与训练要点

- **软标签**：CJ 的 `gold` 是候选项上的概率分布，直接以分布交叉熵为目标。
- **候选项打乱**：`choice` 在训练时随机打乱候选顺序并同步重排目标，以减弱位置偏差；`score` 保留等级顺序；`noul` 固定为 false / true。
- **均衡混合**：先按领域、来源、题型重新分片，再按剩余样本量比例交错流式读取，避免原始文件中的任务顺序导致遗忘；全量数据不载入内存。
- **按长度分桶**：按 token 数组 batch。超过 16,384 tokens 的样本会被显式统计并过滤，从不截断上下文或问题。
- **校准不泄漏**：官方 validation 按母记录分组，80% 用于验证与模型选择，另外 20% 仅用于拟合温度。

### 训练流程

```bash
# 1. 校验数据文件的 SHA256，再按领域 / 来源 / 题型重新分片训练数据
python scripts/prepare_data.py --data datasets/CJ-Dataset --out datasets/CJ-mixed

# 2. LoRA 训练（示例为 4 卡；单卡直接运行 python scripts/train.py；断点续训追加 --resume）
torchrun --standalone --nproc_per_node=4 scripts/train.py --out runs/cj-ddp

# 3. 温度校准 + 合并权重 + 导出模型的 HTTP 接口检查
python scripts/finalize.py --run runs/cj-ddp --out models/chinese-decision-0.8b
```

`scripts/train.py` 的默认参数就是上表中的正式配置。`--resume` 会逐项比对 `config.json`，参数与原运行不一致时拒绝恢复；恢复时可以改变 GPU 数量。快速调试可以加上 `--debug-examples 512 --max-steps 12 --global-batch-size 8`。

训练目录中，`last/` 是断点恢复点，`best/` 是验证集上最优的 adapter，`final/` 是最后一步的 adapter。导出的模型目录包含合并权重、`decision_config.json`（校准温度等）、训练配置与摘要，以及基础模型的 `LICENSE`。

### 可选：RLCD 训练目标

`--rlcd` 会在分布交叉熵之外加入 Chinese-Jev 的 RLCD 项（基于适当评分规则奖励的策略梯度）。实现移植自 [Chinese-Jev 官方代码](https://github.com/gulucaptain/Chinese-Jev)，计算结果与原实现逐位一致。**默认关闭，本页发布的所有结果都没有使用 RLCD。**

每个训练步对候选项 logits 做若干组零均值高斯扰动，用"对数得分 + 0.75 × 球面得分 − RPS"给每个扰动后的分布打分（RPS 只对 `score` 题生效），再把 logits 推向得分高于同组平均的扰动方向。RPS 让离正确等级越远的错误扣分越多，是唯一考虑等级顺序的项，因此最可能帮助 `score` 题。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--rlcd` | 关闭 | 开启 RLCD 项，与交叉熵按 1:1 相加 |
| `--rlcd-group-size` | 4 | 每道题的扰动组数（至少为 2） |
| `--rlcd-sigma-start` | 0.4 | 训练开始时 logits 噪声的大小 |
| `--rlcd-sigma-end` | 0.1 | 训练结束时的噪声大小，按训练进度线性退火 |

- **与原实现的差异**：Chinese-Jev 按 epoch 退火 σ（训练 4 个 epoch，依次为 0.4 / 0.3 / 0.2 / 0.1）；本项目只训练 1 个 epoch，因此改为按训练进度线性退火。σ 是直接加在 logits 上的噪声，大模型字母 logits 的量级可能与 Chinese-Jev 打分头不同，默认值只是起点。
- **配置与续训**：开启后 `config.json` 会记录 RLCD 参数，`training_algorithm` 带 `+rlcd` 后缀，交叉熵运行和 RLCD 运行不能互相续训。不开启时配置与之前完全一致，已有运行照常续训。
- **日志**：每步额外输出 `ce`、`rlcd`、`reward` 和 `sigma`。

**对照实验**：用相同参数加 `--rlcd` 从头开一个新运行。两个运行的数据顺序、验证集和 LoRA 初始化都相同，只有训练目标不同。

```bash
torchrun --standalone --nproc_per_node=4 scripts/train.py --out runs/cj-ddp-rlcd --rlcd
```

不要加 `--max-steps`，否则学习率和 σ 的调度会被压缩，与基线不可比。在 step 1000 或 2000 的验证完成后停止，对比两个运行同一步的验证集指标（每次验证都会写入 `latest_metrics.json`，包含分题型结果），重点看 `type:score`。

## 评测

```bash
# 1. 将验证集选出的 best adapter 复制到独立的快照目录
python scripts/snapshot_best.py --source-run runs/cj-ddp --out runs/cj-bench/my-eval

# 2. 先快评（每个领域 × 题型抽 1,000 条），再全量评测
python scripts/evaluate.py --snapshot-run runs/cj-bench/my-eval --data datasets/CJ-Bench
```

- 只做快评：追加 `--quick-only`。
- 评测合并后的部署模型（使用验证集上拟合的温度）：追加 `--model models/chinese-decision-0.8b`。
- 脚本不会覆盖已有结果，每次评测请使用新的快照目录。所有校准与模型选择都只使用验证集，不使用测试集标签。

## 项目结构

```text
chinese-decision/
├── decision/               # 核心库（pip install -e .）
│   ├── schema.py           #   题型解析、目标分布、提示编译器
│   ├── data.py             #   数据分片、流式混合、按 token 分 batch
│   ├── model.py            #   模型加载与候选字母 logits
│   ├── training.py         #   验证评估、LoRA 目标层、检查点工具
│   ├── rlcd.py             #   可选 RLCD 训练项（移植自 Chinese-Jev）
│   ├── metrics.py          #   验证集指标与 CJ-Bench 指标
│   ├── api.py              #   决策响应格式与校验
│   └── utils.py            #   日志、原子写入、SHA256
├── scripts/                # 命令行入口
│   ├── prepare_data.py     #   数据校验与重新分片
│   ├── train.py            #   LoRA 训练（单卡 / torchrun 多卡）
│   ├── finalize.py         #   温度校准、权重合并、导出检查
│   ├── verify_export.py    #   HTTP 接口端到端检查
│   ├── snapshot_best.py    #   冻结 best adapter 用于评测
│   ├── evaluate.py         #   CJ-Bench 评测
│   └── serve.py            #   HTTP 服务
├── tests/                  # 单元测试（不需要 GPU）
├── examples/request.json   # 请求示例
├── pyproject.toml
├── LICENSE
├── LICENSES/               # 第三方许可证全文
└── THIRD_PARTY_NOTICES.md
```

运行测试：

```bash
python -m unittest discover -s tests
```

## 局限性

- **CJ-Bench 结果未经温度校准**：表中为合并前的 step 10000 adapter（T = 1）的结果。导出的合并模型默认使用验证集上拟合的温度 T = 1.108，它在 CJ-Bench 上的结果尚未单独评测。
- **金融与 `score` 题型仍有差距**：金融准确率比闭源 Jev 低 4.17 个百分点；`score` 题型整体偏弱（整体 56.47%，法律 `score` 51.46%）；法律领域的 ECE（5.70%）是四个领域中最高的。
- **延迟高于编码器方案**：0.8B 解码器的单条延迟高于约 322M 的 Chinese-Jev 编码器模型。
- **`confidence` 不是准确率**：它只反映分布的尖锐程度。若要按阈值自动执行，请先在自己的数据上验证校准效果。
- **标签含义来自上游任务**：例如医疗匹配类标签表示原任务中的匹配关系，并不代表医学正确性。请勿将模型输出作为医疗、法律或金融决策的唯一依据。
- **仅针对中文训练**，不适用于开放式生成。
- 本项目是 hmm 方法的中文适配，并不复现 Microsoft Decision、Jev 或 Chinese-Jev 的训练过程。

## 致谢

- [Chinese-Jev](https://arxiv.org/abs/2609.36965)：CJ-Dataset、CJ-Bench、本文对比所用的基线结果，以及 RLCD 训练项的[参考实现](https://github.com/gulucaptain/Chinese-Jev)（Apache-2.0）。
- [n4ze3m/hmm](https://github.com/n4ze3m/hmm)：选项字母决策目标、LoRA 流程、候选项格式与 System One 响应约定（MIT）。
- [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)：基础模型（Apache-2.0）。

## 许可证

- 本项目代码以 [MIT 许可证](LICENSE) 发布。
- 改编自 hmm 的部分同样遵循 MIT 许可证，原始版权声明见 [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md)。
- `decision/rlcd.py` 改编自 Chinese-Jev，遵循 Apache-2.0，许可证全文见 [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt)。
- 基础模型 Qwen3.5-0.8B 遵循 Apache-2.0，导出的模型目录会保留其原始 `LICENSE`。
- CJ-Dataset 与 CJ-Bench 的许可按数据来源分别适用（部分来源为非商业许可），本项目不授予任何额外的再分发权限。

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
