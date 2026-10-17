# 模型总览、结果口径与 Model Card

更新日期：2026-10-17。包含 0.8B、2B 及各自 Q8_0，四个版本均已公开发布并完成 CJ-Bench 全量评测；0.8B 沿用已有结果，2B 为本次新运行。

## 选择与交付

| 模型 | 基础模型 / 训练 | 格式 / 权重大小 | 默认温度 | 下载 | Model Card |
|---|---|---|---:|---|---|
| Chinese Decision 0.8B | Qwen3.5-0.8B / CE + LoRA | BF16 / 1.50 GB | 1.108133 | [HF](https://huggingface.co/quincyqiang/chinese-decision-0.8b) · [ModelScope](https://modelscope.cn/models/quincyqiang/chinese-decision-0.8b) | [0.8B](MODEL_CARD_0_8B.md) |
| Chinese Decision 0.8B Q8_0 | 同一 0.8B 合并权重 | GGUF / 0.812 GB | 1.108133 | [HF](https://huggingface.co/quincyqiang/chinese-decision-0.8b-gguf) · [ModelScope](https://modelscope.cn/models/quincyqiang/chinese-decision-0.8b-gguf) | [0.8B GGUF](MODEL_CARD_0_8B_GGUF.md) |
| Chinese Decision 2B | Qwen3.5-2B / CE + RLCD + LoRA | BF16 / 3.76 GB | 1.074872 | [HF](https://huggingface.co/quincyqiang/chinese-decision-2b) · [ModelScope](https://modelscope.cn/models/quincyqiang/chinese-decision-2b) | [2B](MODEL_CARD_2B.md) |
| Chinese Decision 2B Q8_0 | 同一 2B 合并权重 | GGUF / 2.01 GB | 1.074872 | [HF](https://huggingface.co/quincyqiang/chinese-decision-2b-gguf) · [ModelScope](https://modelscope.cn/models/quincyqiang/chinese-decision-2b-gguf) | [2B GGUF](MODEL_CARD_2B_GGUF.md) |

四份模型均使用 hmm-zh-v1、最多 16384 tokens 和 2–26 个候选，支持 noul / choice / score。参数规模使用基础模型名称，文件大小为十进制 GB；实际内存与上下文、后端有关。

2B 提高总体准确率和跨领域均值，0.8B 的文件和运行成本更小，部分法律判断与金融评分任务仍更好。Q8_0 降低文件/内存占用，速度需要按硬件与长度验证。

| 场景 | 建议 |
|---|---|
| GPU 服务，追求准确率 | 2B BF16（`scripts/serve_gpu.py`） |
| CPU / 无 PyTorch / 内存受限 | 2B Q8_0 或 0.8B Q8_0（`scripts/serve.py --backend gguf`） |
| 延迟、资源最敏感 | 0.8B（BF16 或 Q8_0） |
| 以法律判断、金融评分为主 | 先在自己的数据上对比 0.8B 与 2B，见下文[剩余短板](#剩余短板与复查) |

## 同一测试集：训练配置与部署配置分开

CJ-Bench 全量为 **307,900 条**（304,822 条硬标签、3,078 条软标签），通用/医疗/法律/金融分别为 100,000 / 200,000 / 4,300 / 3,600 条。全部保留原始候选顺序，无过滤、无截断。ACC/ECE 仅计硬标签，ECE 使用 15 个等宽分箱；分布 NLL 与 Brier 包含软标签；Score MAE 覆盖 38,746 条评分题，计算预测期望等级与目标期望等级的绝对差。

整体 micro 按硬标签样本数加权，四领域 macro 是四个领域的等权平均，专业平均仅对医疗、法律、金融等权。不同量不能互换，整体 ECE 也不是领域 ECE 的平均。选模和温度拟合只使用验证数据，未使用 CJ-Bench 标签。

| 模型 | Micro ACC (%) ↑ | Macro ACC (%) ↑ | ECE (%) ↓ | NLL ↓ | Brier ↓ | Score MAE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8B adapter / T=1 | 83.1088 | 75.7970 | 1.0656 | 0.432327 | 0.224075 | 0.578770 |
| 2B adapter / T=1 | 84.1097 | 77.1313 | 1.4914 | 0.419089 | 0.213828 | 0.560682 |
| 0.8B BF16 / T=1.108133 | 83.1052 | 75.7376 | 0.2706 | 0.431318 | 0.223833 | 0.584427 |
| 0.8B Q8_0 / T=1.108133 | 83.1233 | 75.8092 | 0.2696 | 0.431293 | 0.223815 | 0.583973 |
| 2B BF16 / T=1.074872 | 84.1107 | 77.1750 | 0.6361 | 0.416520 | 0.213295 | 0.564216 |
| 2B Q8_0 / T=1.074872 | 84.1094 | 77.3282 | 0.6534 | 0.416288 | 0.213189 | 0.563738 |

T=1 行是未合并 adapter；其余行为合并后的部署模型。不要把不同温度、合并与量化的联合差值称为单一温度或 RLCD 收益。0.8B Q8_0 全量使用 CUDA 后端，CPU 只做有限输入对照；2B 全量量化对照也在 CUDA 执行。

| 模型 | General ACC (%) | Medical ACC (%) | Legal ACC (%) | Finance ACC (%) | Professional avg. ACC (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.8B adapter / T=1 | 72.24 | 88.86 | 67.86 | 74.22 | 76.98 |
| 2B adapter / T=1 | 73.95 | 89.50 | 69.05 | 76.03 | 78.19 |
| 0.8B BF16 / T=1.108133 | 72.26 | 88.85 | 67.84 | 74.00 | 76.90 |
| 0.8B Q8_0 / T=1.108133 | 72.23 | 88.89 | 67.84 | 74.28 | 77.00 |
| 2B BF16 / T=1.074872 | 73.96 | 89.50 | 68.74 | 76.50 | 78.25 |
| 2B Q8_0 / T=1.074872 | 73.91 | 89.50 | 69.40 | 76.50 | 78.47 |

2B 相对 0.8B 的 T=1 adapter，micro 增加 1.0009 个百分点，macro 增加 1.3344 个百分点，专业平均增加 1.2110 个百分点。但 2B 整体 ECE 更高；部署版概率改善还受到合并/温度影响。

## 其他模型：作者报告值

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

来源：[Chinese-Jev v1 Table 1 / Table 2](https://arxiv.org/html/2609.36965v1)，2026-10-17 核对。外部结果没有在本项目重新复现；Chinese-Jev General + 三个 Specialist 为四个分别模型，Jev 为闭源 API。各列单位为 %，专业平均为医法金等权；不同硬件和网络延迟不作等价排名。

## Q8_0 量化与 CPU

两个 Q8_0 都由各自的 BF16 合并权重直接转换，保留原温度，在 CJ-Bench 全部 307,900 条上与同温度 BF16 逐条配对（llama.cpp CUDA 后端，无过滤、无截断；数据、源权重与 GGUF 指纹在评测后复核）：

| 配对比较（Q8_0 vs BF16） | 0.8B | 2B |
| --- | ---: | ---: |
| 权重大小（GGUF / safetensors） | 0.812 / 1.50 GB | 2.01 / 3.76 GB |
| 硬标签 ACC 变化 | +0.018 pp | −0.0013 pp |
| 首选项一致率 | 98.92% | 99.00% |
| 硬标签由对变错 / 由错变对 | 1,185 / 1,240 | 1,105 / 1,101 |
| 候选概率平均绝对差 | 0.0024 | 0.0023 |
| 每题最大概率差：95 分位 / 最大 | 0.0236 / 0.2269 | 0.0237 / 0.2752 |
| ECE 变化 | −0.001 pp | +0.017 pp |

整体指标几乎不变，但单条答案和概率会变，不能称为逐条无损。2B Q8_0 的四领域 macro 高出 0.15 个百分点，主要来自法律领域（4,300 条中净多对 28 条），不代表量化提升了能力。分领域、分题型与逐条差异见 [0.8B Q8_0](MODEL_CARD_0_8B_GGUF.md) 与 [2B Q8_0](MODEL_CARD_2B_GGUF.md) 模型卡。

为区分"格式 / 后端差异"和"量化差异"，2B 另在 1,200 条分层测试子集（4 领域 × 3 题型 × 100 条）上对比三条路径；子集领域权重不同于全量，准确率不能与上表直接比较：

| 2B 后端 | 测试条数 | ACC (%) | ECE (%) | NLL | Brier | Score MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Transformers BF16 | 1200 | 75.9228 | 2.5788 | 0.595550 | 0.329166 | 0.504446 |
| BF16 GGUF | 1200 | 75.8389 | 2.7841 | 0.596385 | 0.329873 | 0.506524 |
| Q8_0 GGUF | 1200 | 75.8389 | 2.4443 | 0.595557 | 0.329245 | 0.506174 |

子集上 Q8_0 相对 Transformers BF16 的首选项一致率为 98.75%，BF16 GGUF 相对 Q8_0 为 99.08%：仅更换后端带来的差异与量化差异处于同一量级。

2B 同组 25 条输入的延迟与内存（0.8B 见 [CPU 部署](CPU.md)）：

| 2B 后端 | 短输入 p50 (ms) | 全部输入 p50 (ms) | 全部 p95 (ms) | 进程内存观测最大值 (GiB) |
| --- | ---: | ---: | ---: | ---: |
| Transformers FP32 / CPU 16 threads | 545.29 | 601.34 | 11320.53 | 11.74 |
| Q8_0 / CPU 16 threads | 664.50 | 817.99 | 12884.58 | 3.34 |
| Transformers BF16 / GPU | 44.68 | 44.77 | 90.27 | 4.67 |
| Q8_0 / GPU | 20.39 | 23.60 | 274.74 | 3.02 |

服务器 25 条输入各测 3 次，短输入 92–255 tokens，全部 92–6521；共享 Xeon Gold 5318Y / A800。内存为进程 RSS，不是 GPU 显存；延迟不含分词/HTTP/网络。Q8_0 CPU 在本次测量降低内存但 p50 更慢。Windows i7-9700 的 2B Q8_0 短诊断输入 p50 为 1.351 秒，峰值 2.701 GiB；不能跨不同输入与服务器结果排名。

## 剩余短板与复查

- 法律 noul：0.8B T=1 为 85.80%，2B T=1 为 82.53%，2B 部署版为 82.07%，2B Q8_0 为 82.73%。
- 金融 score：0.8B T=1 为 83.08%，2B T=1 为 79.83%，2B 部署版为 80.33%，2B Q8_0 为 80.58%。
- 2B 商品分类来源（3,151 条）在 step3000 暂时崩落，最终 70.99% 仍低于 step4000 的 73.94%；Q8_0 与 BF16 交付版在该来源上完全相同（70.96%）。全局 ECE 低不代表各来源都校准良好。
- 训练目标、模型规模与 batch 共同变化，不能归因为 RLCD 单项收益；进度按实际样本数计算。

## 文档索引

| 文档 | 内容 |
| --- | --- |
| [README](../README.md#评测结果) | 项目介绍、两个尺寸的训练阶段历史与完整明细 |
| [0.8B 模型卡](MODEL_CARD_0_8B.md) · [0.8B Q8_0 模型卡](MODEL_CARD_0_8B_GGUF.md) | 0.8B 使用方法、应用案例、量化对照 |
| [2B 模型卡](MODEL_CARD_2B.md) · [2B Q8_0 模型卡](MODEL_CARD_2B_GGUF.md) | 2B 使用方法、全量结果、量化对照 |
| [2B 训练与交付记录](TRAINING_2B.md) | 2B 训练配置、各阶段结果、来源诊断、导出与校准 |
| [0.8B CPU 部署](CPU.md) · [2B CPU 部署](CPU_2B.md) | GGUF 转换、CPU 服务与检查记录 |

原始结果：各次评测的 full.json 与逐条预测在本机 runs/cj-bench（0.8B 为 latest-best-*、gguf-q8-20261015；2B 为 qwen35-2b-rlcd-*、gguf-2b-q8-20261017），汇总报告在 reports/cj-bench-2b-step*、reports/server-gguf-20261015 与 reports/gguf-2b-20261017。
