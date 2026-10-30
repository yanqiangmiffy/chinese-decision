<div align="center">

# Chinese Decision

**中文概率决策模型：一次前向，直接输出候选项概率**

[![Base Models](https://img.shields.io/badge/Base-Qwen3.5%200.8B%20%7C%202B%20%7C%204B-blue)](docs/MODELS.md)
[![Dataset](https://img.shields.io/badge/Data-CJ--Dataset-orange)](https://huggingface.co/datasets/bbldCVer-hf/CJ-Dataset)
[![Benchmark](https://img.shields.io/badge/Bench-CJ--Bench-green)](https://huggingface.co/datasets/bbldCVer-hf/CJ-Bench)
[![Method](https://img.shields.io/badge/Method-hmm-lightgrey)](https://github.com/n4ze3m/hmm)
[![arXiv](https://img.shields.io/badge/arXiv-2610.23170-b31b1b?logo=arxiv)](https://arxiv.org/abs/2610.23170)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-chinese--decision--4b-yellow?logo=huggingface)](https://huggingface.co/quincyqiang/chinese-decision-4b)
[![ModelScope](https://img.shields.io/badge/ModelScope-chinese--decision--4b-624aff)](https://modelscope.cn/models/quincyqiang/chinese-decision-4b)
[![GGUF](https://img.shields.io/badge/GGUF-chinese--decision--4b--gguf-brightgreen)](https://huggingface.co/quincyqiang/chinese-decision-4b-gguf)
[![chinese-decision-unified-2b](https://img.shields.io/badge/Hugging%20Face-chinese--decision--unified--2b-yellow?logo=huggingface)](https://huggingface.co/quincyqiang/chinese-decision-unified-2b)
[![CJ-IE](https://img.shields.io/badge/Data-CJ--IE-orange)](https://huggingface.co/datasets/quincyqiang/CJ-IE)
[![Decision Studio](https://img.shields.io/badge/Platform-Decision%20Studio-black?logo=github)](https://github.com/yanqiangmiffy/chinese-decision-studio)

📄 论文：[Chinese Decision: A Family of Language Models for Single-Pass Probabilistic Decision-Making](https://arxiv.org/abs/2610.23170)（arXiv:2610.23170，v2）

中文 | [English](README_EN.md)

</div>

---

## 简介

**Chinese Decision** 是一个面向中文场景的轻量级**决策模型**。它不聊天、不生成文本，只做一件事：给定上下文和一组固定候选项，返回每个候选项的概率。

很多 AI 调用本质上是在有限选项中做判断——意图识别、工单分流、内容审核、数据打标、给模型输出打分、智能体的下一步动作选择等。用生成式大模型回答这类问题又慢又贵，且输出需要再解析。决策模型直接返回结构化的概率分布，程序拿到即可按阈值使用。

模型分两个系列，都基于 [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)、[Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) 与 [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) 做 LoRA 微调：

- **纯决策模型**（`chinese-decision-{0.8b,2b,4b}`，及其 `-gguf` 版本）：在 1,300 万条中文决策数据 [CJ-Dataset](https://huggingface.co/datasets/bbldCVer-hf/CJ-Dataset) 上训练，只做决策。
- **统一模型**（`chinese-decision-unified-{0.8b,2b}`，4B 计划中）：同一份权重在决策之外多一种读出，做类型化信息抽取（命名实体、关系、事件论元），额外使用抽取数据 [CJ-IE](https://huggingface.co/datasets/quincyqiang/CJ-IE)。

方法改编自 [n4ze3m/hmm](https://github.com/n4ze3m/hmm)，并在 [Chinese-Jev](https://arxiv.org/abs/2609.36965) 提出的 CJ-Bench 上与其论文结果对比。方法、结果与分析见论文 [arXiv:2610.23170](https://arxiv.org/abs/2610.23170)。

**主要特点**

- ⚡ **一次前向，零输出 token**：读取最后位置上候选字母 `A`–`Z` 的 logits，softmax 后即为概率；不做自回归生成，也没有思维链。
- 🧩 **三种题型**：`noul`（是/否）、`choice`（单选）、`score`（有序等级，返回期望分值）。
- 🔢 **类别数不止 26 个**：单题每个候选对应一个字母（A–Z，训练数据覆盖 2–16 个）；类别更多的 `choice` 问题用 [`/v1/tournament`](#http-服务) 分组淘汰后决赛，不需要重新训练。
- 🏆 **单模型覆盖四个领域**：chinese-decision-4b 在 CJ-Bench 全量上整体 ACC **85.97%**、医疗 / 法律 / 金融平均 **84.00%**，四个领域都高于论文报告的 Chinese-Jev（专业平均 70.82%）和闭源 Jev（77.01%）。各模型的完整对比见[评测结果](#评测结果)。
- 📏 **概率经过校准**：发布权重附带验证集拟合的温度，chinese-decision-0.8b / 2b / 4b 整体 ECE 为 **0.27% / 0.64% / 1.00%**，所有发布模型四个领域的 ECE 都低于论文中的模型。
- 🪶 **GPU / CPU 部署**：纯决策模型均提供 BF16 与 Q8_0 GGUF；chinese-decision-4b-gguf 为 **4.48 GB**，CJ-Bench 全量 ACC **85.99%**，与 BF16 首选项一致率 **99.19%**，CPU 上用 llama.cpp 运行，不需要 PyTorch。
- 🧬 **统一模型：一份权重，两种读出**：chinese-decision-unified-2b 的 CJ-Bench 全量 ACC 为 **85.00%**（高于 chinese-decision-2b 的 84.11%），CJ-IE 测试集抽取 micro F1 **66.0**；两种读出都不生成文本。
- 🔌 **兼容 hmm / System One 接口**：提供开箱即用的 HTTP 服务。

## 最新动态

- 🧬 **2026-10-27**：发布 **chinese-decision-unified-2b**（🤗 [Hugging Face](https://huggingface.co/quincyqiang/chinese-decision-unified-2b) · [ModelScope](https://www.modelscope.cn/models/quincyqiang/chinese-decision-unified-2b)）。CJ-Bench 全量 ACC **85.00%**（chinese-decision-2b 为 84.11%，同为 T = 1），CJ-IE 测试集抽取 micro F1 **66.0**（chinese-decision-unified-0.8b 为 63.1）。
- 🗂️ **2026-10-26**：公开抽取训练数据 **CJ-IE**（🤗 [Hugging Face](https://huggingface.co/datasets/quincyqiang/CJ-IE) · [ModelScope](https://www.modelscope.cn/datasets/quincyqiang/CJ-IE)）：把公开的中文 NER、关系和事件语料整理成统一格式（类型带定义），并派生选择 / 判断题。部分来源为非商业许可，使用前请阅读[数据卡](docs/DATASET_CARD_CJ_IE.md)。
- 🧬 **2026-10-25**：发布 **chinese-decision-unified-0.8b**（🤗 [Hugging Face](https://huggingface.co/quincyqiang/chinese-decision-unified-0.8b) · [ModelScope](https://modelscope.cn/models/quincyqiang/chinese-decision-unified-0.8b)）。请求带 `questions` 走决策，带 `types` 走抽取；决策准确率没有因为加入抽取而下降（CJ-Bench 全量 **83.46%**，chinese-decision-0.8b 为 83.11%）。
- 📄 **2026-10-23**：论文 **[arXiv:2610.23170](https://arxiv.org/abs/2610.23170)** 更新到 v2：补入 chinese-decision-2b 与 chinese-decision-4b（均附 Q8_0 版本），新增按母记录聚类的 bootstrap、校准、选择性预测与量化分析。v1 提交于 2026-10-16。
- 🚀 **2026-10-21**：发布 **chinese-decision-4b** 与 **chinese-decision-4b-gguf**（Q8_0，4.48 GB）。整体 ACC **85.97%**、专业平均 **84.00%**、ECE **1.00%**，4 个领域 × 3 种题型的 12 个组合全部高于 0.8B 与 2B。各尺寸训练过程中的 checkpoint 评测见 [训练过程 checkpoint 评测](docs/TRAINING_HISTORY.md)。
- 🔢 **2026-10-18**：新增分组淘汰接口 [`/v1/tournament`](#http-服务)，`choice` 问题的类别数不再受 26 个字母限制（服务端默认上限 1024）。
- 🚀 **2026-10-17**：发布 **chinese-decision-2b** 与 **chinese-decision-2b-gguf**（Q8_0，2.01 GB）。整体 ACC **84.11%**、专业平均 **78.25%**、ECE **0.64%**；训练与诊断见 [2B 训练与交付记录](docs/TRAINING_2B.md)，CPU 检查见 [2B CPU 部署](docs/CPU_2B.md)。
- 🧪 **2026-10-15**：开源决策模型测试平台 [Decision Studio](https://github.com/yanqiangmiffy/chinese-decision-studio)：决策实验台、8 个应用场景、鲁棒性实验室、评测中心和性能压测，`docker compose up` 一条命令启动。
- 🎉 **2026-10-15**：发布 **chinese-decision-0.8b** 与 **chinese-decision-0.8b-gguf**（Q8_0，812 MB）。完整训练 1 epoch（13,048,000 条决策），CJ-Bench 通用领域 **72.26%**、专业领域平均 **76.90%**。

## 模型

全文统一用模型 ID（即 Hugging Face / ModelScope 仓库名）指代各个模型。

| 模型 ID | 系列 | 基础模型 / 训练目标 | 格式 / 大小 | 默认温度 | 下载 | 模型卡 |
|---|---|---|---|---:|---|---|
| chinese-decision-0.8b | 纯决策 | Qwen3.5-0.8B / CE | BF16 / 1.50 GB | 1.108133 | [HF](https://huggingface.co/quincyqiang/chinese-decision-0.8b) · [ModelScope](https://modelscope.cn/models/quincyqiang/chinese-decision-0.8b) | [模型卡](docs/MODEL_CARD_0_8B.md) |
| chinese-decision-0.8b-gguf | 纯决策 | 同 chinese-decision-0.8b，Q8_0 量化 | GGUF / 0.812 GB | 1.108133 | [HF](https://huggingface.co/quincyqiang/chinese-decision-0.8b-gguf) · [ModelScope](https://modelscope.cn/models/quincyqiang/chinese-decision-0.8b-gguf) | [模型卡](docs/MODEL_CARD_0_8B_GGUF.md) |
| chinese-decision-2b | 纯决策 | Qwen3.5-2B / CE + RLCD | BF16 / 3.76 GB | 1.074872 | [HF](https://huggingface.co/quincyqiang/chinese-decision-2b) · [ModelScope](https://modelscope.cn/models/quincyqiang/chinese-decision-2b) | [模型卡](docs/MODEL_CARD_2B.md) |
| chinese-decision-2b-gguf | 纯决策 | 同 chinese-decision-2b，Q8_0 量化 | GGUF / 2.01 GB | 1.074872 | [HF](https://huggingface.co/quincyqiang/chinese-decision-2b-gguf) · [ModelScope](https://modelscope.cn/models/quincyqiang/chinese-decision-2b-gguf) | [模型卡](docs/MODEL_CARD_2B_GGUF.md) |
| chinese-decision-4b | 纯决策 | Qwen3.5-4B / CE + RLCD | BF16 / 8.41 GB | 1.024758 | [HF](https://huggingface.co/quincyqiang/chinese-decision-4b) · [ModelScope](https://modelscope.cn/models/quincyqiang/chinese-decision-4b) | [模型卡](docs/MODEL_CARD_4B.md) |
| chinese-decision-4b-gguf | 纯决策 | 同 chinese-decision-4b，Q8_0 量化 | GGUF / 4.48 GB | 1.024758 | [HF](https://huggingface.co/quincyqiang/chinese-decision-4b-gguf) · [ModelScope](https://modelscope.cn/models/quincyqiang/chinese-decision-4b-gguf) | [模型卡](docs/MODEL_CARD_4B_GGUF.md) |
| chinese-decision-unified-0.8b | 统一 | Qwen3.5-0.8B / CE + RLCD，决策与抽取联合训练 | BF16 / 1.52 GB | 1.1326（仅决策） | [HF](https://huggingface.co/quincyqiang/chinese-decision-unified-0.8b) · [ModelScope](https://modelscope.cn/models/quincyqiang/chinese-decision-unified-0.8b) | [模型卡](docs/MODEL_CARD_UNIFIED_0_8B.md) |
| chinese-decision-unified-2b | 统一 | Qwen3.5-2B / CE + RLCD，决策与抽取联合训练 | BF16 / 3.79 GB | 1.0617（仅决策） | [HF](https://huggingface.co/quincyqiang/chinese-decision-unified-2b) · [ModelScope](https://www.modelscope.cn/models/quincyqiang/chinese-decision-unified-2b) | [模型卡](docs/MODEL_CARD_UNIFIED_2B.md) |
| chinese-decision-unified-4b | 统一 | Qwen3.5-4B / 计划中，尚未训练 | — | — | 尚未发布 | — |

所有发布模型都已在 CJ-Bench 全量 307,900 条上完成评测（见[评测结果](#评测结果)），全部支持 16384 tokens；单题 2–26 个候选（训练数据覆盖 2–16 个），类别更多时用 [`/v1/tournament`](#http-服务) 分组淘汰；大小为十进制 GB，不是运行内存。

**纯决策模型与统一模型的区别**

| | 纯决策模型 | 统一模型 |
|---|---|---|
| 读出 | 决策：`noul` / `choice` / `score` | 决策 + 类型化抽取（见[工作原理](#工作原理)） |
| 由什么决定任务 | 请求都是决策 | 请求带 `questions` 走决策，带 `types` 走抽取，两者都带则对同一段输入各答一次 |
| 训练起点 | 原始 Qwen3.5 + LoRA | 原始 Qwen3.5 + LoRA + 随机初始化的跨度头（0.8B 约 3.4M、2B 约 6.6M 参数，一起训练） |
| 训练数据 | CJ-Dataset 13,048,000 条决策 | CJ-Dataset 决策 + CJ-IE（决策 611,354 条 + 抽取跨度行 573,294 条），见[训练数据](#训练数据) |
| 格式 | BF16 + Q8_0 GGUF | 只有 BF16；抽取读出只能用 PyTorch，没有 GGUF |
| 服务入口 | `scripts/serve.py`、`scripts/serve_gpu.py` | `scripts/serve_unified.py` |

统一模型从原始底座训练，而不是在纯决策模型上继续训练：先试过在 chinese-decision-0.8b 上继续训练抽取，CJ-Bench 准确率从 83.11% 掉到 71.02%（法律 −27.9、金融 −20.2）；让决策数据始终参与训练并从原始底座开始，决策能力得以保留，见[统一模型与遗忘修复](docs/UNIFIED_MODEL.md)。

**怎么选**

| 需求 | 推荐 |
|---|---|
| GPU 上追求决策准确率 | chinese-decision-4b |
| 兼顾延迟与显存 | chinese-decision-2b，或需要抽取时用 chinese-decision-unified-2b（决策准确率也更高） |
| CPU 或内存受限，不想装 PyTorch | `-gguf` 版本（Q8_0） |
| 延迟和资源最敏感、需要最低整体 ECE | chinese-decision-0.8b |
| 一份权重既做决策又做信息抽取 | chinese-decision-unified-2b；要轻量用 chinese-decision-unified-0.8b |

完整选型、分领域和外部基线见 [模型总览](docs/MODELS.md)。代码：[本仓库](https://github.com/yanqiangmiffy/chinese-decision)。测试平台：[Decision Studio](https://github.com/yanqiangmiffy/chinese-decision-studio)。

## 应用案例

以下截图来自我们的测试平台 Decision Studio，后端运行 chinese-decision-0.8b。这些场景采用同一种设计：**业务规则写在代码里，决策模型只负责给出概率**。

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

![共享的 Qwen3.5 骨干，决策读出（候选字母 logits）和抽取读出（类型化跨度头）](assets/chinese-decision-unified-acl.png)

**决策读出**（所有模型）：每个问题被编译成如下提示，套用 Qwen 聊天模板（关闭 thinking），读取末位候选字母的 logits。

```mermaid
flowchart LR
    A["上下文 + 问题 + 候选项"] --> B["提示编译器<br/>hmm-zh-v1"]
    B --> C["Qwen3.5 0.8B / 2B / 4B<br/>LoRA 微调后合并"]
    C --> D["末位 logits<br/>A … Z"]
    D --> E["屏蔽多余字母<br/>softmax(z / T)"]
    E --> F["noul / choice / score"]
```

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

| 题型 | `criteria` 格式 | 返回字段 |
|---|---|---|
| `noul` | 可选，`{"false": "...", "true": "..."}`，默认“否 / 是” | `noul`：判断为真的概率 |
| `choice` | 对象 `{键: 描述}` | `choice`：概率最大的键 |
| `score` | 列表，按等级从低到高排列 | `score`：等级下标的概率加权期望（0 到 K−1）；`legend`：下标到描述的映射 |

所有题型都会返回完整的 `probabilities` 和 `confidence`。其中 `confidence = (K·p_max − 1) / (K − 1)` 衡量分布的尖锐程度：均匀分布为 0，确定为 1。它**不等于**经验准确率。

**抽取读出**（仅统一模型）：给定文本和一组“类型名 + 说明”，跨度头直接返回每个实体的字符位置和概率，覆盖命名实体、关系（给定主体，抽客体）和事件论元。两种读出共用同一个骨干和同一组 LoRA 参数，只有最后的读出层不同；都是一次前向，**不生成文本**。

## 评测结果

除特别说明外，下文均为发布权重（LoRA 已合并进基础模型）在 **CJ-Bench 全部 307,900 条**上的结果（硬标签 304,822、软标签 3,078），无过滤、无截断。ACC / ECE 只计硬标签（ECE 为 15 个等宽分箱），NLL / Brier 包含软标签；Micro 按样本加权，Macro 为四个领域等权平均。单位均为 %。

> **关于温度 T**：概率按 p = softmax(z / T) 计算，z 为候选字母的 logits。纯决策模型使用 `decision_config.json` 中的默认温度；统一模型的 CJ-Bench 结果用 **T = 1**（直接 softmax，不校准），标 † 的 ECE 即为 T = 1 下的值。温度只调整概率的尖锐程度，**不改变选出的答案**，所以 ACC 在两种口径下可以直接比较，ECE 不能。训练过程中各 checkpoint（合并前，T = 1）的结果见 [训练过程 checkpoint 评测](docs/TRAINING_HISTORY.md)。

### 主要结果

**加粗**为每列最优。

| 模型 ID | 格式 | Micro ACC ↑ | Macro ACC ↑ | ECE ↓ | NLL ↓ | Brier ↓ | Score MAE ↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chinese-decision-0.8b | BF16 | 83.11 | 75.74 | **0.27** | 0.4313 | 0.2238 | 0.5844 |
| chinese-decision-0.8b-gguf | Q8_0 | 83.12 | 75.81 | **0.27** | 0.4313 | 0.2238 | 0.5840 |
| chinese-decision-unified-0.8b | BF16 | 83.46 | 76.17 | 1.1† | — | — | — |
| chinese-decision-2b | BF16 | 84.11 | 77.18 | 0.64 | 0.4165 | 0.2133 | 0.5642 |
| chinese-decision-2b-gguf | Q8_0 | 84.11 | 77.33 | 0.65 | 0.4163 | 0.2132 | 0.5637 |
| chinese-decision-unified-2b | BF16 | 85.00 | 79.12 | 0.9† | — | — | — |
| chinese-decision-4b | BF16 | 85.97 | 81.92 | 1.00 | 0.3706 | 0.1909 | **0.5253** |
| chinese-decision-4b-gguf | Q8_0 | **85.99** | **81.95** | 0.96 | **0.3705** | **0.1907** | **0.5253** |

- **chinese-decision-4b 准确率最高**：相对 chinese-decision-2b，Micro **+1.86**、Macro **+4.75**、专业平均 **+5.76** 个百分点，法律 / 金融分别提升 **8.47 / 7.11**，NLL / Brier / Score MAE 分别下降 **11.0% / 10.5% / 6.9%**；整体 ECE 为 1.00%，高于 2B 的 0.64%。4B 与 2B 的训练数据、训练目标、全局 batch 和学习率相同，差别主要是基础模型规模。
- **chinese-decision-2b 相对 chinese-decision-0.8b**：Micro **+1.01**、Macro **+1.44**、专业平均 **+1.35** 个百分点，NLL / Brier / Score MAE 分别下降 **3.4% / 4.7% / 3.5%**；整体 ECE 略高（0.64% vs 0.27%），仍处于很低水平。
- **统一模型没有牺牲决策能力**：chinese-decision-unified-2b 比 chinese-decision-2b 高 Micro **+0.89**、Macro **+1.94**（法律 +4.68）；chinese-decision-unified-0.8b 比 chinese-decision-0.8b 高 Micro +0.35。T = 1 时 chinese-decision-2b 的整体 ECE 为 1.5%，chinese-decision-unified-2b 为 0.9%。
- **Q8_0 几乎无损**：与同温度 BF16 逐条配对，0.8B / 2B / 4B 的首选项一致率为 **98.92% / 99.00% / 99.19%**，准确率变化 +0.02 / −0.001 / +0.02 个百分点。单条答案仍有少量变化（2B 为 1,105 条对→错、1,101 条错→对）；chinese-decision-2b-gguf 的 Macro 偏高 0.15 主要来自样本较少的法律领域，不代表量化提升了能力。全量在 llama.cpp CUDA 上测得，CPU 只做接口与有限输入检查，详见各 GGUF 模型卡。

### 分领域 / 分题型

各模型 BF16 的准确率（%），列按尺寸排列，方便同尺寸的纯决策与统一模型对比。chinese-decision-4b 在每一行都最高。

| 切分 | 样本数 | chinese-decision-0.8b | chinese-decision-unified-0.8b | chinese-decision-2b | chinese-decision-unified-2b | chinese-decision-4b |
|---|---:|---:|---:|---:|---:|---:|
| 通用 | 100,000 | 72.26 | 73.86 | 73.96 | 75.43 | 75.69 |
| 医疗 | 200,000 | 88.85 | 88.61 | 89.50 | 90.03 | 91.19 |
| 法律 | 4,300 | 67.84 | 67.88 | 68.74 | 73.42 | 77.21 |
| 金融 | 3,600 | 74.00 | 74.31 | 76.50 | 77.61 | 83.61 |
| `noul` | 134,577 | 94.96 | 94.88 | 95.11 | 95.24 | 95.47 |
| `choice` | 134,577 | 78.32 | 79.01 | 80.12 | 81.68 | 83.25 |
| `score` | 38,746 | 56.45 | 57.17 | 57.67 | 58.91 | 60.43 |

医疗样本占测试集的 65%，整体 micro 主要反映医疗领域，跨领域比较建议参考 macro。法律和金融样本少（4,300 / 3,600 条），这两个领域的数字波动较大。

<details>
<summary><b>领域 × 题型准确率（纯决策模型）</b></summary>

| 领域 | 题型 | chinese-decision-0.8b | chinese-decision-2b | chinese-decision-4b |
|---|---|---:|---:|---:|
| 通用 | `noul` | 87.59 | 88.01 | 88.50 |
| 通用 | `choice` | 72.93 | 76.16 | 78.60 |
| 通用 | `score` | 54.64 | 56.05 | 58.35 |
| 医疗 | `noul` | 97.82 | 97.87 | 98.04 |
| 医疗 | `choice` | 80.55 | 81.81 | 85.04 |
| 医疗 | `score` | 66.50 | 66.19 | 67.49 |
| 法律 | `noul` | 85.60 | 82.07 | 88.87 |
| 法律 | `choice` | 64.07 | 67.20 | 74.13 |
| 法律 | `score` | 51.69 | 55.15 | 67.31 |
| 金融 | `noul` | 76.33 | 81.75 | 85.67 |
| 金融 | `choice` | 62.83 | 67.42 | 77.00 |
| 金融 | `score` | 82.83 | 80.33 | 88.17 |

chinese-decision-2b 相对 0.8B 在 12 个组合中有 9 个提升，法律判断（82.07 vs 85.60）和金融评分（80.33 vs 82.83）下降较明显，医疗评分略降；chinese-decision-4b 的 12 个组合全部高于 0.8B 和 2B。各模型分领域 / 分题型的 ECE、NLL、Brier 见对应模型卡。

</details>

### 与 Chinese-Jev 论文对比

外部模型数字取自 [Chinese-Jev v1 Table 1 / Table 2](https://arxiv.org/html/2609.36965v1) 的作者报告值（2026-10-17 核对，本项目未复现）。**加粗**为每列最优，† 为 T = 1 下的 ECE。

**通用领域（Table 1）**

| 模型 | ACC ↑ | ECE ↓ |
| --- | ---: | ---: |
| Qwen3.5-2B | 32.95 | 43.68 |
| Open-Jev | 52.33 | 10.54 |
| SemIF | 44.21 | 27.27 |
| Laya Multilingual | 40.22 | 23.04 |
| Jev (API) | 68.35 | 11.45 |
| Chinese-Jev General / Specialists | 69.20 | 3.78 |
| chinese-decision-0.8b | 72.26 | **1.68** |
| chinese-decision-unified-0.8b | 73.86 | 2.8† |
| chinese-decision-2b | 73.96 | 1.93 |
| chinese-decision-unified-2b | 75.43 | 2.5† |
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
| chinese-decision-unified-0.8b | 88.61 | 0.4† | 67.88 | 2.3† | 74.31 | **1.1†** | 76.93 | **1.27†** |
| chinese-decision-2b | 89.50 | **0.24** | 68.74 | 3.37 | 76.50 | 2.13 | 78.25 | 1.92 |
| chinese-decision-unified-2b | 90.03 | 0.3† | 73.42 | 3.3† | 77.61 | 2.2† | 80.35 | 1.93† |
| chinese-decision-4b | **91.19** | 0.25 | **77.21** | **2.02** | **83.61** | 1.65 | **84.00** | 1.30 |

- chinese-decision-4b 相对 Chinese-Jev（General + 三个 Specialist）在通用 / 医疗 / 法律 / 金融分别高 **6.49 / 3.54 / 17.11 / 18.89** 个百分点，专业平均高 **13.18**；相对闭源 Jev，四个领域都更高，专业平均高 **6.99**，金融高 **5.22**。chinese-decision-2b 与 chinese-decision-unified-2b 的专业平均也高于 Jev，但金融仍分别低 1.89 / 0.78。
- 表中本项目的每个模型在四个领域的 ECE 都低于论文中的所有模型。
- Chinese-Jev General 加三个 Specialist 是四个分别训练的模型，本项目每个模型覆盖四个领域；Jev 是闭源 API。论文使用 H200 / API，本地为 A800，延迟口径不同，不作等价排名。

**与 Chinese-Jev 的方法差异**

| | Chinese-Jev | Chinese Decision（本项目） |
|---|---|---|
| 骨干网络 | mmBERT-base 编码器 + 决策 Transformer（约 322M） | Qwen3.5-0.8B / 2B / 4B 解码器 + LoRA |
| 候选打分 | 在候选标记位置上用共享 MLP 打分 | 读取末位候选字母的 logits |
| 训练目标 | 交叉熵 + RLCD 策略梯度项 | chinese-decision-0.8b 使用候选分布交叉熵；其余模型另加 RLCD |
| 训练方式 | 先做通用预训练，再分别微调三个领域专用模型 | 四个领域混合训练，单一模型 |

### 信息抽取：CJ-IE 测试集

仅统一模型。CJ-IE 的测试集由各语料的官方 dev / test 构成，做过去泄漏，**是内部评测集，不是公开基准**，数字不能和各语料的公开榜单直接比较。抽取按**精确匹配**计分（起止位置和类型都对才算对），35 个来源共 232,291 个标注跨度。

| 指标 | chinese-decision-unified-0.8b | chinese-decision-unified-2b |
|---|---:|---:|
| 决策整体 ACC（CJ-IE 派生题 84,894 条） | 90.65% | **91.30%** |
| 旧能力保持（CJ 老来源 3,032 条）ACC | 91.51% | **91.88%** |
| 抽取 micro 精确率 / 召回率 / F1 | 73.3 / 55.5 / 63.1 | 75.4 / 58.6 / **66.0** |
| 抽取来源宏平均 F1（35 个） | 62.5 | **66.3** |
| 其中 NER（31 个来源） | 64.0 | **67.5** |
| 其中关系（DuIE、CMeIE） | 47.2 | **55.4** |
| 其中事件论元（DuEE、FewFC） | 54.5 | **59.2** |
| Resume（简历实体） | **88.1** | 87.3 |
| MSRA（人名 / 地名 / 机构名） | 76.0 | **82.2** |
| DuIE（关系客体） | 66.2 | **74.3** |
| DuEE（事件论元） | 56.3 | **62.2** |
| Weibo（社交媒体） | 37.2 | **52.6** |
| CMeIE（医学关系客体） | 28.2 | **36.5** |
| military_ner（类型没有说明） | 0.0 | 0.0 |

抽取偏保守，精确率明显高于召回率，来源之间差别很大。35 个来源里 chinese-decision-unified-2b 有 30 个比 chinese-decision-unified-0.8b 好；类型没有说明的 `military_ner` 两个尺寸都是 0。完整的逐来源、逐类型指标见各模型卡。

### 推理效率

同一组 25 条 CJ-Bench 输入（92–6,521 tokens），发布权重单条请求、预热后各测 3 次：

| 指标（单条请求 p50） | chinese-decision-0.8b | chinese-decision-2b | chinese-decision-4b |
|---|---:|---:|---:|
| GPU · BF16（Transformers） | 44.4 ms | 44.8 ms | 60.1 ms |
| GPU · Q8_0（llama.cpp） | **17.9 ms** | **23.6 ms** | **47.3 ms** |
| CPU 16 线程 · Q8_0（llama.cpp） | 427.0 ms | 818.0 ms | 2004.0 ms |
| CPU 进程内存 · Q8_0 / FP32 | **2.19** / 7.44 GiB | **3.34** / 11.74 GiB | 6.14 / — GiB |

> A800 80GB PCIe 与 Xeon Gold 5318Y 共享服务器；只统计已分词输入的前向与概率计算，不含分词、HTTP 和网络。batch = 1 时 GPU 上受调度开销限制，BF16 延迟接近；Q8_0 在 GPU 上短输入更快，在 CPU 上的主要优势是内存。4B 在另一天用同组输入测得，未测 Transformers FP32 CPU 基线。统一模型的决策读出与同尺寸纯决策模型结构相同。长输入、Windows 台式机等完整测量见 [0.8B GGUF](docs/MODEL_CARD_0_8B_GGUF.md#推理效率)、[2B GGUF](docs/MODEL_CARD_2B_GGUF.md#推理效率) 与 [4B GGUF](docs/MODEL_CARD_4B_GGUF.md#推理效率) 模型卡。

<details>
<summary><b>chinese-decision-0.8b 微调前后对比（验证集）</b></summary>

在官方验证集的分层子集（4 个领域 × 3 种题型，每组 100 条，共 1,200 条）上，对比基础模型零样本与微调后最终 checkpoint 的结果：

| 领域 | Qwen3.5-0.8B ACC | **chinese-decision-0.8b ACC** | Qwen3.5-0.8B ECE | **chinese-decision-0.8b ECE** |
|---|---:|---:|---:|---:|
| 通用 | 36.64 | **75.34** | 28.25 | **3.76** |
| 医疗 | 16.67 | **82.00** | 43.40 | **3.25** |
| 法律 | 40.67 | **65.33** | 24.03 | **7.62** |
| 金融 | 44.33 | **71.33** | 16.77 | **3.41** |
| **整体** | 34.56 | **73.49** | 27.45 | **2.48** |

</details>

## 快速开始

### 安装

```bash
git clone https://github.com/yanqiangmiffy/chinese-decision.git
cd chinese-decision
pip install -e ".[train]"
```

需要 Python ≥ 3.11。`[train]` 安装 CUDA 训练/推理依赖；CPU GGUF 服务使用 `pip install -e ".[gguf]"`，不需要 PyTorch、PEFT 或 flash-linear-attention；只使用 Transformers 推理可安装 `.[torch]`。如果环境里已经装好了 CUDA 版 PyTorch，不希望 pip 改动它，可以使用 `pip install -e . --no-deps`。

### 下载模型

按需选择一个（各模型见[模型](#模型)）：

```bash
# 纯决策，GPU：BF16 合并权重
hf download quincyqiang/chinese-decision-4b --local-dir models/chinese-decision-4b
# 纯决策，CPU：Q8_0 GGUF，不需要 PyTorch
hf download quincyqiang/chinese-decision-4b-gguf --local-dir models/chinese-decision-4b-gguf
# 统一模型：决策 + 信息抽取（只有 BF16，抽取需要 PyTorch）
hf download quincyqiang/chinese-decision-unified-2b --local-dir models/chinese-decision-unified-2b
# 国内网络也可以用 ModelScope，仓库名相同
modelscope download --model quincyqiang/chinese-decision-4b --local_dir models/chinese-decision-4b
```

其他模型把仓库名换成对应的模型 ID 即可。

### Python 调用

**决策**（所有模型）：

```python
from decision import DecisionPipeline

pipe = DecisionPipeline.from_pretrained("models/chinese-decision-0.8b")   # 本地目录或 Hub 仓库名
answer = pipe({
    "state": "客户留言：付款后一直没有收到发票，我已经催了三次，今天必须拿到。",
    "questions": {"q": {
        "type": "choice",
        "instructions": "应该由哪个部门处理？",
        "criteria": {"billing": "支付、发票及退款", "technical": "故障和系统问题", "sales": "购买咨询与升级"},
    }},
})
print(answer)   # {"task": "decision", "answers": {"q": {"billing": 0.97, ...}}}，已按 decision_config.json 的温度校准
```

**抽取**（统一模型，或联合训练导出的 `runs/<run>/model`），用同一个对象：

```python
pipe = DecisionPipeline.from_pretrained("models/chinese-decision-unified-2b", device="cuda")

pipe({"text": "收货地址：浙江省杭州市西湖区文三路90号东部软件园科技大厦A座1203室，联系人王小明。",
      "types": {"省份": "省级行政区划,省、自治区、直辖市,不含港澳台",
                "城市": "地级行政区划,地级市、地区、自治州等",
                "道路": "有正式名称的道路,包括隧道、高架、街、弄、巷等",
                "门牌号": "路号,如'000号'或'000弄',通常位于道路名称之后"},
      "threshold": 0.3})
# {"task": "extraction", "entities": [{"text": "浙江省", "label": "省份", "start": 5, "end": 8, "score": ...}, ...]}
```

任务由请求决定，同时带 `questions` 和 `types` 时对同一段输入各做一次（`text` 缺省取 `state`）：

| 你想做的事 | 请求里带什么 | 返回 |
|---|---|---|
| 决策 | `state` + `questions`（`noul` / `choice` / `score`） | 每个候选的概率 |
| 抽取 | `text` + `types`（`{名称: 说明}`），可选 `threshold` | 实体 `text` / `label` / `start` / `end` / `score` |
| 关系抽取 | 主体固定，类型名写成“{主体}的{关系}”，说明写客体类型 | 客体的跨度 |
| 事件抽取 | 先对每个事件类型问是 / 否问题，再用“触发词 + 角色”作为类型 | 触发词和论元 |
| 意图 + 槽位 | 意图用决策，槽位用抽取，自己级联（训练里没有槽位数据，需要自己评测） | — |

**抽取类型怎么写**：类型的“说明”比类型名重要得多。训练时每个类型都带一句来自标注规范的定义，模型学到的是“按定义找”，而不是“按名字猜”。

| 写法 | 表现 | 建议 |
|---|---|---|
| 训练时的说明 | 召回最高 | 从训练时的类型组开始改，名称保持原样 |
| 只写类型名 | 召回最低，相近类型分不开 | 至少写一句话定义，包括边界（什么不算） |
| 训练里没有的新类型 | 基本抽不出来，或很不稳 | 上线前一定在自己的数据上测；能用训练过的类型表达就不要新造 |
| 没有定义的类型 | `military_ner` 测试 F1 为 0，一个都没预测 | 同上 |

更底层的接口与 transformers 的用法相同：

| 类 | 作用 |
|---|---|
| `DecisionConfig` | `decision_config.json`：温度、提示版本、跨度头设置 |
| `DecisionProcessor` | 提示编译、编码、填充、解码（两种任务） |
| `DecisionModel` | 底座 + 可选跨度头；`from_pretrained` / `save_pretrained`（合并 LoRA，一份权重）、`enable_lora`、`score_decisions`、`predict_spans` |
| `DecisionPipeline` | 请求级推理：`decide`、`extract`、`__call__` |

训练中的 adapter 也可以直接加载：`DecisionModel.from_pretrained("models/Qwen3.5-0.8B", adapter="runs/cj-ddp/best")`；联合训练的运行目录用 `DecisionPipeline.from_run(base, run)`。`import decision` 不会加载 PyTorch，GGUF 的 CPU 服务仍只依赖 transformers。

### HTTP 服务

三个服务入口，决策接口和返回格式一致，都可以直接作为 [Decision Studio](https://github.com/yanqiangmiffy/chinese-decision-studio) 的 `DECISION_BASE_URL`，也兼容 hmm / System One SDK：

| 入口 | 适用模型 | 场景 | 部署 |
|---|---|---|---|
| `scripts/serve.py` | 纯决策（GGUF 或 BF16） | CPU 或单机调试；逐题前向 | [docs/CPU.md](docs/CPU.md)、[deploy/gguf](deploy/gguf/) |
| `scripts/serve_gpu.py` | 纯决策 BF16，或基础模型 + adapter | GPU 生产服务；动态批处理、训练进度、检查点热更新（`POST /v1/admin/reload`） | [deploy/gpu](deploy/gpu/README.md) |
| `scripts/serve_unified.py` | 统一模型 | 一个进程同时提供决策与抽取 | — |

```bash
# 纯决策，GPU BF16
python scripts/serve_gpu.py --base models/chinese-decision-4b --host 127.0.0.1 --port 8789 --token-budget 16384 --max-batch 32
# 纯决策，CPU Q8_0（需要 [gguf] 依赖）
python scripts/serve.py --backend gguf --base models/chinese-decision-4b-gguf --gguf models/chinese-decision-4b-gguf/chinese-decision-4b-Q8_0.gguf --threads 8 --context-size 16384 --host 127.0.0.1 --port 8790
# 统一模型：决策 + 抽取
python scripts/serve_unified.py --model models/chinese-decision-unified-2b --port 8011
# 单机调试：BF16 合并权重
python scripts/serve.py --base models/chinese-decision-0.8b --port 8787
```

服务读取 `decision_config.json` 中的校准温度。`serve.py` 也可以直接加载基础模型和 adapter：`--base models/Qwen3.5-0.8B --adapter runs/cj-ddp/best`（此时温度为 1，即未校准）。GGUF 后端中 `--base` 提供分词器、提示模板和 `decision_config.json`，`--gguf` 提供权重，转换与验证见 [CPU 部署](docs/CPU.md)；原始 Transformers CPU 基线可使用 `--device cpu`（FP32）。

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

| 接口 | 说明 | 纯决策服务 | 统一服务 |
|---|---|:---:|:---:|
| `POST /v1/systemone` | 决策接口；`?debug=1` 附带每个候选的 logits 和每题 token 数 | ✓ | ✓ |
| `POST /api/alpha/decisions` | 同上，兼容 hmm 路由 | ✓ | ✓ |
| `POST /v1/batch` | `{"requests": [...], "debug": bool}`，一次提交多个请求，单个请求出错不影响其他 | ✓ | ✓ |
| `POST /v1/compile` | 只编译提示词并计数 token，不跑模型 | ✓ | ✓ |
| `POST /v1/tournament` | 候选多于一题能放下的 `choice` 问题：分组淘汰后决赛，见下文 | ✓ | ✓ |
| `POST /v1/decide` | 决策，请求格式与 `/v1/systemone` 相同 | | ✓ |
| `POST /v1/extract` | 抽取：`{"text", "types", "threshold"?}` | | ✓ |
| `POST /v1/predict` | 按请求形状自动选择；同时带 `questions` 和 `types` 时一次返回 `answers` 和 `entities`（输入较短时两种读出共用一次前向） | | ✓ |
| `GET /v1/info` | 模型名、检查点、温度、是否支持抽取、运行统计 | ✓ | ✓ |
| `GET /v1/training` | 训练进度 | ✓ | ✓ |
| `GET /health` | 健康检查，返回当前温度、是否已校准及检查点 step | ✓ | ✓ |

单个请求最多 64 个问题，单题 2–26 个候选项，输入不超过 16,384 tokens。超长输入**不会被截断**，非法请求返回 `422`。检查一个正在运行的服务（两个路由、概率格式、非法请求处理）：

```bash
python scripts/verify_export.py --url http://127.0.0.1:8787
```

**类别更多时：分组淘汰（`/v1/tournament`）**。模型每个候选读一个字母，训练数据最多只有 16 个候选。更多候选的 `choice` 问题由服务端随机分成每组不超过 `group_size` 个，每组前 `keep` 名晋级，直到剩下的候选能放进一题，再做决赛。每组仍走同一个提示编译器，不需要重新训练：

```json
{
  "state": "用户希望了解第 137 号产品。",
  "question": {"type": "choice", "instructions": "用户想了解哪个产品？",
               "criteria": {"p0": "第 0 号产品", "p1": "第 1 号产品", "...": "最多 1024 个候选"}},
  "group_size": 16,
  "keep": 2,
  "seed": 0
}
```

`group_size` 取 3–26（默认 16，即训练覆盖的最大候选数），`keep` 取 1 到 `group_size // 2`（默认 2），`seed` 决定分组，候选上限由 `decision/tournament.py` 的 `MAX_CANDIDATES` 设定（默认 1024）。返回的 `answer` 与 `/v1/systemone` 的单题格式相同，但**概率只覆盖决赛候选**；`rounds` 记录每轮各组的概率和晋级者，`decisions` 是前向次数。300 个候选按默认参数是 19 + 3 + 1 = 23 次前向。候选不超过 `group_size` 时只问一题，结果与 `/v1/systemone` 相同。分组淘汰尚未在 CJ-Bench 上评测；决赛候选都比较接近，概率会偏平，按置信度设阈值时要用自己的数据重新确定。

完整示例 [examples/tournament.json](examples/tournament.json) 是 41 个电商客服意图：按默认参数分成 3 组（14 / 14 / 13），每组前 2 名进入 6 个候选的决赛，共 4 次前向。

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
    {"probabilities": {"...": "本组 14 个候选的概率"}, "advanced": ["courier_complaint", "other"]},
    {"probabilities": {"...": "本组 14 个候选的概率"}, "advanced": ["wrong_item", "stock_inquiry"]},
    {"probabilities": {"...": "本组 13 个候选的概率"}, "advanced": ["damaged_item", "repair_request"]}
  ]}],
  "usage": {"input_tokens": 813, "output_tokens": 0},
  "decisions": 4,
  "latency_ms": 161.4
}
```

在 GPU 服务（chinese-decision-0.8b step 10000，T = 1，A800）上换用 10 条不同意图的留言，全部选对，每题约 150 ms。这只是功能演示，不是正式评测。

## 训练与评测

### 训练数据

| 数据 | 内容 | 用于 |
|---|---|---|
| [CJ-Dataset](https://huggingface.co/datasets/bbldCVer-hf/CJ-Dataset) | 13,048,000 条中文决策，通用 / 医疗 / 法律 / 金融四个领域 | 所有模型 |
| [CJ-Bench](https://huggingface.co/datasets/bbldCVer-hf/CJ-Bench) | 307,900 条测试题 | 评测，训练与模型选择都不读取 |
| CJ-IE（[HF](https://huggingface.co/datasets/quincyqiang/CJ-IE) · [ModelScope](https://www.modelscope.cn/datasets/quincyqiang/CJ-IE) · [数据卡](docs/DATASET_CARD_CJ_IE.md) · [构建说明](docs/IE_DATASET.md)） | 公开中文 NER、关系和事件语料的统一整理，类型带定义 | 统一模型 |

CJ-IE 的三个子集：

| 子集 | 内容 | train | dev | test |
|---|---|---:|---:|---:|
| `extraction` | 跨度行：NER（32 个来源）、关系客体（DuIE、CMeIE）、事件论元（DuEE、FewFC） | 573,294 | 11,647 | 85,041 |
| `decision` | 派生的选择 / 判断题：实体类型、关系判断、事件类型、事件大类、公告字段 | 611,354 | 12,554 | 84,894 |
| `structured` | 完整的实体 / 关系 / 事件标注，用于端到端评测 | 125,859 | 1,295 | 15,038 |

CJ-IE 是多个第三方语料的整理，没有单一许可证，部分来源为非商业许可，使用前请阅读数据卡的[许可与使用限制](docs/DATASET_CARD_CJ_IE.md#许可与使用限制)。

训练和评测所需的基础模型与数据均从本地目录加载（CJ 数据集目前需要申请访问，下载前先运行 `hf auth login`）：

```bash
hf download Qwen/Qwen3.5-0.8B --local-dir models/Qwen3.5-0.8B   # 2B / 4B 换成 Qwen3.5-2B / Qwen3.5-4B
hf download bbldCVer-hf/CJ-Dataset --repo-type dataset --local-dir datasets/CJ-Dataset
hf download bbldCVer-hf/CJ-Bench --repo-type dataset --local-dir datasets/CJ-Bench
```

### 训练配置

**纯决策模型**（下表为 chinese-decision-0.8b；chinese-decision-2b 与 chinese-decision-4b 在此基础上开启 RLCD，全局 batch 改为每步至少 2,048 条决策，见 [2B 交付记录](docs/TRAINING_2B.md) 与 [4B 模型卡](docs/MODEL_CARD_4B.md#训练细节)）

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

**统一模型**（各尺寸配方相同，只有底座和 micro-batch 大小不同；训练脚本 `scripts/train_multitask_ddp.py`，细节见 [统一模型说明](docs/UNIFIED_MODEL.md)）

| 项目 | chinese-decision-unified-0.8b | chinese-decision-unified-2b |
|---|---|---|
| 初始化 | 原始 Qwen3.5-0.8B | 原始 Qwen3.5-2B |
| 发布检查点 | step 14,504（1 epoch 最后一步） | step 15,304（1 epoch 最后一步） |
| 每个 micro-batch 的 token 上限 | 24,576 | 12,288 |
| 训练硬件 / 耗时 | 4 × A800，约 17 小时 | 4 × A800，约 26 小时 |
| 校准温度（只作用于决策读出） | 1.1326 | 1.0617 |
| 微调方式 | LoRA，r = 32；LoRA 学习率 1e-4，跨度头学习率 5e-4，前 100 步预热 | 同左 |
| 损失函数 | 两个读出都用 CE + RLCD（权重 1.0，4 个采样，σ 从 0.4 降到 0.1）；抽取里 none 跨度的权重为 0.2 | 同左 |
| 批次 | 每次更新累积 8 个 micro-batch × 4 卡；决策 micro-batch 32 条、抽取 16 行 | 同左 |
| 抽取布局 | 回显：文本先读一遍，再逐字重复一遍，跨度从重复的那份上取；跨度最宽 20 字 | 同左 |
| 模型选择 | 每 1,000 步在独立开发集上评估决策准确率与抽取得分的均值；开发集来自 CJ-Dataset validation 和 CJ-IE dev，**从不读取 CJ-Bench** | 同左 |
| 导出 | 合并 LoRA，写出单个 `model.safetensors`（骨干 + `span_head.*`） | 同左 |

**数据与训练要点**

- **软标签**：CJ 的 `gold` 是候选项上的概率分布，直接以分布交叉熵为目标。
- **候选项打乱**：`choice` 在训练时随机打乱候选顺序并同步重排目标，以减弱位置偏差；`score` 保留等级顺序；`noul` 固定为 false / true。
- **均衡混合**：先按领域、来源、题型重新分片，再按剩余样本量比例交错流式读取，避免原始文件中的任务顺序导致遗忘；全量数据不载入内存。
- **按长度分桶**：按 token 数组 batch。超过 16,384 tokens 的样本会被显式统计并过滤，从不截断上下文或问题。
- **校准不泄漏**：官方 validation 按母记录分组，80% 用于验证与模型选择，另外 20% 仅用于拟合温度。

### 训练与评测流程

```bash
# 1. 校验数据文件的 SHA256，再按领域 / 来源 / 题型重新分片训练数据
python scripts/prepare_data.py --data datasets/CJ-Dataset --out datasets/CJ-mixed

# 2. LoRA 训练（示例为 4 卡；单卡直接运行 python scripts/train.py；断点续训追加 --resume）
torchrun --standalone --nproc_per_node=4 scripts/train.py --out runs/cj-ddp

# 3. 温度校准 + 合并权重 + 导出模型的 HTTP 接口检查
python scripts/finalize.py --run runs/cj-ddp --out models/chinese-decision-0.8b

# 4. 将验证集选出的 best adapter 复制到独立的快照目录，先快评（每个领域 × 题型抽 1,000 条）再全量评测
python scripts/snapshot_best.py --source-run runs/cj-ddp --out runs/cj-bench/my-eval
python scripts/evaluate.py --snapshot-run runs/cj-bench/my-eval --data datasets/CJ-Bench
```

- `scripts/train.py` 的默认参数就是上表中的正式配置。`--resume` 会逐项比对 `config.json`，参数与原运行不一致时拒绝恢复；恢复时可以改变 GPU 数量。快速调试可以加上 `--debug-examples 512 --max-steps 12 --global-batch-size 8`。
- 训练目录中，`last/` 是断点恢复点，`best/` 是验证集上最优的 adapter，`final/` 是最后一步的 adapter。`last/` 与 `best/` 会被原地替换；`--keep-every`（默认 1000，0 为关闭）每隔 N 步另存一份 `step-N/`。导出的模型目录包含合并权重、`decision_config.json`（校准温度等）、训练配置与摘要，以及基础模型的 `LICENSE`。
- 评测只做快评：追加 `--quick-only`；评测合并后的部署模型（使用验证集上拟合的温度）：追加 `--model models/chinese-decision-0.8b`。脚本不会覆盖已有结果，每次评测请使用新的快照目录。
- 统一模型等已导出模型的 CJ-Bench 全量评测用 `scripts/evaluate_cj_bench.py`，抽取评测用 `scripts/evaluate_extraction.py`，GGUF 用 `scripts/evaluate_gguf.py`。所有校准与模型选择都只使用验证集，不使用测试集标签。

<details>
<summary><b>可选：RLCD 训练目标（chinese-decision-0.8b 未使用，其余模型均开启）</b></summary>

`--rlcd` 会在分布交叉熵之外加入 Chinese-Jev 的 RLCD 项（基于适当评分规则奖励的策略梯度）。实现移植自 [Chinese-Jev 官方代码](https://github.com/gulucaptain/Chinese-Jev)，计算结果与原实现逐位一致，默认关闭。

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

**对照实验**：用相同参数加 `--rlcd` 从头开一个新运行（`torchrun --standalone --nproc_per_node=4 scripts/train.py --out runs/cj-ddp-rlcd --rlcd`）。两个运行的数据顺序、验证集和 LoRA 初始化都相同，只有训练目标不同。不要加 `--max-steps`，否则学习率和 σ 的调度会被压缩，与基线不可比。在 step 1000 或 2000 的验证完成后停止，对比两个运行同一步的验证集指标（每次验证都会写入 `latest_metrics.json`，包含分题型结果），重点看 `type:score`。

</details>

## 项目结构

```text
chinese-decision/
├── decision/               # 核心库（pip install -e .），接口仿 transformers
│   ├── configuration.py    #   DecisionConfig / ExtractionConfig（decision_config.json）
│   ├── processing.py       #   DecisionProcessor：提示编译、编码、填充、解码
│   ├── modeling.py         #   DecisionModel（底座 + 可选跨度头）、SpanHead
│   ├── pipeline.py         #   DecisionPipeline（决策 / 抽取）、SlotFillingPipeline
│   ├── losses.py           #   RLCDLoss、SpanLoss
│   ├── metrics.py          #   验证集、CJ-Bench 与抽取指标
│   ├── data/               #   CJ-Dataset 读取与批处理、槽位记录、CJ-IE 构建（ie/）
│   ├── training/           #   LoRA 目标层、决策评估、温度拟合
│   ├── serving/            #   HTTP 答案格式、Transformers / GGUF 后端、候选淘汰赛
│   └── utils.py            #   日志、原子写入、SHA256
├── scripts/                # 命令行入口
│   ├── prepare_data.py     #   数据校验与重新分片
│   ├── train.py            #   LoRA 训练（单卡 / torchrun 多卡），纯决策模型
│   ├── train_multitask_ddp.py #  决策 + 抽取联合训练（统一模型），流式读取数据
│   ├── build_ie_data.py    #   构建 CJ-IE；build_multitask_data.py / build_replay_data.py 构建联合训练与回放数据
│   ├── finalize.py         #   温度校准、权重合并、导出检查
│   ├── snapshot_best.py    #   冻结 best adapter 用于评测
│   ├── evaluate.py         #   CJ-Bench 评测
│   ├── evaluate_cj_bench.py #  已导出模型（含统一模型）的 CJ-Bench 全量评测；compare_cj_bench.py 做配对比较
│   ├── evaluate_extraction.py # 抽取评测；evaluate_slots.py 评测意图 + 槽位
│   ├── convert_gguf.py     #   导出 GGUF（BF16 / Q8_0）
│   ├── evaluate_gguf.py    #   用 GGUF 权重跑 CJ-Bench
│   ├── benchmark_*.py      #   CPU 延迟与吞吐测试；compare_gguf_benchmark.py 做配对比较
│   ├── verify_export.py    #   HTTP 接口端到端检查
│   ├── serve.py            #   HTTP 服务：CPU / GGUF、单机调试
│   ├── serve_gpu.py        #   HTTP 服务：GPU 生产部署（动态批处理、热更新）
│   └── serve_unified.py    #   HTTP 服务：统一模型（/v1/decide、/v1/extract、/v1/predict）
├── deploy/
│   ├── gpu/                # GPU 部署脚本（本机 deploy.sh → 服务器 tmux）
│   └── gguf/               # CPU 部署脚本（push / install / service）
├── docs/
│   ├── MODELS.md           #   模型总览：选型、结果口径、外部基线
│   ├── MODEL_CARD_*.md     #   八个模型卡（每个发布模型一个）
│   ├── UNIFIED_MODEL.md    #   统一模型：一份权重做分类和抽取，以及联合训练的遗忘修复
│   ├── DATASET_CARD_CJ_IE.md #  CJ-IE 数据卡（HF / ModelScope 数据集的说明）
│   ├── IE_DATASET.md       #   CJ-IE：中文信息抽取的多任务数据
│   ├── MULTITASK_EXTRACTION.md # 分类 + 抽取的可行性实验
│   ├── TRAINING_HISTORY.md #   各尺寸训练过程的 checkpoint 评测
│   ├── TRAINING_2B.md      #   chinese-decision-2b 训练、导出与诊断记录
│   ├── CPU.md              #   chinese-decision-0.8b CPU 部署与 GGUF 验证
│   └── CPU_2B.md           #   chinese-decision-2b CPU 部署与量化检查
├── tests/                  # 单元测试（不需要 GPU）：python -m unittest discover -s tests
├── examples/               # 请求示例：request.json、tournament.json、slot_filling/、rlcd_loss.py
├── assets/                 # README 配图
├── pyproject.toml
├── LICENSE
├── LICENSES/               # 第三方许可证全文
└── THIRD_PARTY_NOTICES.md
```

## 局限性

- **配置要分开解读**：发布权重（合并后 + 默认温度）、训练 checkpoint（T = 1）和 Q8_0 的数值有细微差别，比较时请对齐配置，不要把差值归因于单一因素。Q8_0 全量在 GPU（llama.cpp CUDA）上测得，CPU 后端只做了有限输入对照。校准验证集不能当测试集，整体 ECE 低也不代表每个来源都校准良好。
- **改善并不均匀**：chinese-decision-2b 的法律判断 / 金融评分 ACC 为 82.07% / 80.33%，低于 chinese-decision-0.8b 的 85.60% / 82.83%（chinese-decision-4b 为 88.87% / 88.17%），详见 [2B 诊断](docs/TRAINING_2B.md)。chinese-decision-4b 的商品分类来源（3,151 条）ACC 为 64.58%，低于 chinese-decision-2b 的 70.96%，整体 ECE（1.00%）也高于 2B 和 0.8B，详见 [4B 模型卡](docs/MODEL_CARD_4B.md#局限性)。
- **延迟高于编码器方案**：0.8B / 2B / 4B 解码器的单条延迟高于约 322M 的 Chinese-Jev 编码器模型；chinese-decision-4b 的 GPU 单条 p50 约 60 ms，CPU Q8_0 约 2 秒。
- **`confidence` 不是准确率**：它只反映分布的尖锐程度。若要按阈值自动执行，请先在自己的数据上验证校准效果。
- **标签含义来自上游任务**：例如医疗匹配类标签表示原任务中的匹配关系，并不代表医学正确性。请勿将模型输出作为医疗、法律或金融决策的唯一依据。
- **统一模型的抽取读出仍有明显短板**：CJ-IE 测试集上 micro F1 为 63.1（chinese-decision-unified-0.8b）/ 66.0（chinese-decision-unified-2b），精确率 73.3% / 75.4%、召回率 55.5% / 58.6%，边界要求精确匹配；没有类型说明或训练里没出现过的新类型可能一个都抽不出；社交媒体、细粒度新闻类型和医学关系较弱。抽取只能用 PyTorch，不支持 GGUF，也不支持意图 + 槽位的端到端填充。详见模型卡的局限性：[chinese-decision-unified-0.8b](docs/MODEL_CARD_UNIFIED_0_8B.md#局限性)、[chinese-decision-unified-2b](docs/MODEL_CARD_UNIFIED_2B.md#局限性)。
- **论文 v2 的比较基于单次训练和引入的基线数字**：没有多随机种子的方差估计，外部模型的数字取自原作者报告值，本项目未复现。
- **仅针对中文训练**，不适用于开放式生成。
- 本项目是 hmm 方法的中文适配，并不复现 Microsoft Decision、Jev 或 Chinese-Jev 的训练过程。

## 致谢

- [Chinese-Jev](https://arxiv.org/abs/2609.36965)：CJ-Dataset、CJ-Bench、本文对比所用的基线结果，以及 RLCD 训练项的[参考实现](https://github.com/gulucaptain/Chinese-Jev)（Apache-2.0）。
- B2NERD 等中文 NER / 关系 / 事件语料：统一模型所用 CJ-IE 抽取数据的来源。
- [n4ze3m/hmm](https://github.com/n4ze3m/hmm)：选项字母决策目标、LoRA 流程、候选项格式与 System One 响应约定（MIT）。
- [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)、[Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) 与 [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)：基础模型（Apache-2.0）。

## 许可证

- 本项目代码以 [MIT 许可证](LICENSE) 发布。
- 改编自 hmm 的部分同样遵循 MIT 许可证，原始版权声明见 [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md)。
- `scripts/train.py` 的可选 `--rlcd` 训练项改编自 Chinese-Jev（实现见 `decision/losses.py` 的 `RLCDLossConfig.chinese_jev()`），遵循 Apache-2.0，许可证全文见 [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt)。
- 基础模型 Qwen3.5-0.8B、Qwen3.5-2B 与 Qwen3.5-4B 遵循 Apache-2.0，导出的模型目录会保留其原始 `LICENSE`。
- 统一模型 chinese-decision-unified-0.8b 与 chinese-decision-unified-2b 的权重同样遵循 Apache-2.0（分别继承自 Qwen3.5-0.8B 与 Qwen3.5-2B）。
- CJ-Dataset 与 CJ-Bench 的许可按数据来源分别适用（部分来源为非商业许可）。统一模型所用的 CJ-IE 是多个 NER / 关系 / 事件语料的整理，没有单一许可证，部分来源为非商业许可或许可不明，使用前请阅读[数据卡](docs/DATASET_CARD_CJ_IE.md#许可与使用限制)并逐个来源核对。本项目不授予任何额外的再分发权限。

## 引用

如果使用了本项目的模型或代码，请引用论文：

```bibtex
@misc{yan2026chinesedecision,
  title         = {Chinese Decision: A Family of Language Models for Single-Pass Probabilistic Decision-Making},
  author        = {Yan, Qiang},
  year          = {2026},
  eprint        = {2610.23170},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2610.23170},
  note          = {Code: https://github.com/yanqiangmiffy/chinese-decision}
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
