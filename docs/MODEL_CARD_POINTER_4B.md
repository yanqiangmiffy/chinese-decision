---
license: apache-2.0
language:
- zh
- en
base_model: Qwen/Qwen3.5-4B
base_model_relation: finetune
pipeline_tag: zero-shot-classification
tags:
- decision-making
- classification
- calibration
- pointer-readout
- decision-index
- qwen3.5
library_name: transformers
---

# Chinese Decision Pointer 4B

**一次前向，为任意数量的候选项直接给出概率的决策模型（指针读出）。**

> A decision model fine-tuned from Qwen3.5-4B. Given a state, one or more questions and their candidates, it returns a calibrated probability for every candidate in one forward pass, with no generated text. Unlike the letter readout of [chinese-decision-4b](https://huggingface.co/quincyqiang/chinese-decision-4b), a pointer head scores a marker placed after each candidate, so there is no 26-option limit and all questions about one state are answered in the same sequence. On the [Jev Decision Index 0.4.1](https://huggingface.co/spaces/multimodalart/jev-decision-index) leaderboard it has a Full score of **53.96**, 4th of 147 submissions under 4.9B (not counting the Jev 1.14 reference) and the highest LoRA / readout-head entry in that range; on the public 0.3 suite it scores **61.69** (140,178 requests, complete).

[代码](https://github.com/yanqiangmiffy/chinese-decision/tree/c16450b) · [榜单](https://huggingface.co/spaces/multimodalart/jev-decision-index) · [评测结果](https://huggingface.co/datasets/quincyqiang/chinese-decision-index-results/tree/main/runs/chinese-decision-pointer-4b) · [Decision Index](https://github.com/apolinario/decision-index) · [字母读出版 4B](https://huggingface.co/quincyqiang/chinese-decision-4b)

## 最新动态

- 🏆 **2026-11-12**：[Jev Decision Index 0.4.1](https://huggingface.co/spaces/multimodalart/jev-decision-index) 榜单出分。Full score **53.96**，在 147 个 4.9B 以下的提交中排**第 4**（不计参考模型 Jev 1.14），与 Cerno 4B v1.2（53.88）并列总榜第 19 档；排在前面的三个 4B 模型都是全量微调，本模型是该档最高的 LoRA / 读出头方案。
- 📄 **2026-11-06**：系列技术报告 [arXiv:2610.23170](https://arxiv.org/abs/2610.23170) 更新到 v3，覆盖字母读出的纯决策模型与统一模型；本指针读出模型不在报告范围内。

## 结构

- **底座**：Qwen3.5-4B，LoRA（r 64）训练后合并为 BF16 单文件权重 `model.safetensors`。
- **指针读出**：每个候选项后放一个标记 `<|object_ref_end|>`，每个问题末尾放一个决策标记 `<|object_ref_start|>`；指针头（256 维）对“决策标记 × 候选标记”打分，softmax 后得到概率。
- **双向提示**：全注意力层在提示内部双向可见（线性注意力层保持因果），候选项能互相“看到”。
- **联合提问**：同一状态下的多个问题放在一条序列里，一次前向全部作答。
- **校准**：导出时在独立校准划分上拟合温度 T = 1.538（`decision_config.json`）。

## Decision Index 0.4.1 榜单

榜单按 Full score 排名：与 37 个公开基准考同类技能的私有题，加上一组新领域的私有任务，均由榜单方在 1 × NVIDIA RTX PRO 6000 上复测（2026-11-11 更新，共 297 个模型）。

| 指标 | 分数 |
|---|---:|
| **Full score** | **53.96** |
| 私有题：同类技能 | 56.32 |
| 私有题：新领域 | 39.18 |
| 公开题集 0.3 | 61.69 |
| 总榜名次 | 第 19 档（与 Cerno 4B v1.2 并列） |
| 4.9B 以下（147 个提交） | **第 4**（不计参考模型 Jev 1.14） |

同档前三：KnowLine 4B Gen5.5（55.99）、StartLux-Decision v1.1 4B（54.80）、Perplexity Decider Flash v1.2 (4B)（54.29），均为全量微调。新领域私有任务（39.18）是与公开题集分数拉开差距的主要原因。榜单方用本仓库的服务代码复跑提交结果，831 条抽查的决策一致率为 99.60%（门槛 98.5%）。

## Decision Index 0.3（公开题集）

| 领域 | 技能分（机会校正） |
|---|---:|
| Knowledge & Reasoning | 40.8 |
| Language Understanding | 63.2 |
| Retrieval & Classification | 70.6 |
| Tools & Automation | 85.2 |
| Arts & Human Taste | 50.7 |
| **公开指数** | **61.69**（raw 71.01） |

运行设置：`decision.decision_index_engine:ChineseDecisionEngine`，`debias="reverse"`（原顺序与逆序候选各算一次取平均），bf16，单卡单进程；4 个分片并行。NVIDIA A800 上单请求延迟中位数 64 ms、均值 147 ms、p95 386 ms。完整的 `results.jsonl`、`scores.json` 与 `environment.json` 见[评测结果数据集](https://huggingface.co/datasets/quincyqiang/chinese-decision-index-results/tree/main/runs/chinese-decision-pointer-4b)。

## 使用

```sh
git clone https://github.com/yanqiangmiffy/chinese-decision && cd chinese-decision && git checkout c16450b
pip install -e ".[torch]" && pip install -e <decision-index checkout>
python -m decision_index pipeline --edition 0.3 --engine decision.decision_index_engine:ChineseDecisionEngine \
  --model quincyqiang/chinese-decision-pointer-4b --option device=cuda --option debias='"reverse"' --compact
```

在代码里直接使用：

```python
from decision.modeling import DecisionModel
from decision.processing import DecisionProcessor

model = DecisionModel.from_pretrained("quincyqiang/chinese-decision-pointer-4b", device="cuda")
processor = DecisionProcessor.from_pretrained("quincyqiang/chinese-decision-pointer-4b")
# 见仓库 decision/decision_index_engine.py 与 examples/ 中的完整请求格式
```

## 训练

1. **从底座训练（v2）**：decision-mix-v2（76 个公开数据集的训练划分，约 112 万个问题）+ CJ-Dataset 中文决策数据，1 轮，CE + RLCD。
2. **分轮微调**：在 v2 之上继续训练，新增数据如下：
   - 自写生成器（标签由程序精确计算）：智能家居请求处理、因果阶梯问题、和弦识别、Python 输出预测、BIG-Bench 风格谜题（物体交换追踪、SVG 图形、行走回原点、真假话链、布尔表达式）；
   - 公开数据集的**训练划分**：VAST、ACOS、Habermas Machine 训练组、iSarcasmEval（不含英文任务 A）、MetaMathQA（GSM8K 改写题）。
3. **权重平均**：第 2 轮与第 4 轮微调结果做秩堆叠的 LoRA 精确平均（rank 128）后合并导出。

**去污染**：所有训练题都按 13 个词单元的片段与 Decision Index 0.3 公开题集比对，重合的题目被剔除；生成器只生成随机的新实例，不使用任何测试题。

## 许可与局限

- 权重沿用底座 Qwen3.5-4B 的 Apache-2.0 许可。训练数据按各来源自身的许可使用，其中部分为非商业或仅限研究用途；本模型不授予任何数据的再分发权利。
- 模型只做“在给定候选中选择”的判断，不生成文本，不弃权；候选和上下文由调用方提供。
- 需要多步推理的题（数学应用题、代码执行、国际象棋）仍明显弱于更大的模型。
- 微调后模型偏自信，依赖导出时拟合的温度来校准；修改温度会改变概率尺度。
