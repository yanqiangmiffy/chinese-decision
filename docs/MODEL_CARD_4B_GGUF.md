---
license: apache-2.0
language:
- zh
base_model:
- quincyqiang/chinese-decision-4b
base_model_relation: quantized
library_name: gguf
datasets:
- bbldCVer-hf/CJ-Dataset
pipeline_tag: zero-shot-classification
tags:
- gguf
- llama.cpp
- q8_0
- decision-making
- classification
- calibration
- chinese
- qwen3.5
- rlcd
tasks:
- zero-shot-classification
---

# Chinese Decision 4B GGUF Q8_0

**Chinese Decision 4B 的 llama.cpp 部署包，保留相同提示、候选概率计算与校准温度。**

> A 4.48 GB Q8_0 GGUF export of [Chinese Decision 4B](https://huggingface.co/quincyqiang/chinese-decision-4b) for llama.cpp, converted directly from the merged step-5446 weights and keeping their calibrated temperature. On the full CJ-Bench (307,900 decisions) it reaches 85.99% accuracy and 0.96% ECE, versus 85.97% and 1.00% for the BF16 release, with 99.19% top-choice agreement. It runs on CPU without PyTorch or a GPU. Like the original model, it is not a chat model: read the final-position logits of the candidate letters and apply softmax.

[代码](https://github.com/yanqiangmiffy/chinese-decision) · [模型总览](MODELS.md) · [Decision Studio](https://github.com/yanqiangmiffy/chinese-decision-studio) · [训练数据](https://huggingface.co/datasets/bbldCVer-hf/CJ-Dataset) · [测试集](https://huggingface.co/datasets/bbldCVer-hf/CJ-Bench)

[0.8B BF16](MODEL_CARD_0_8B.md) · [0.8B Q8_0](MODEL_CARD_0_8B_GGUF.md) · [2B BF16](MODEL_CARD_2B.md) · [2B Q8_0](MODEL_CARD_2B_GGUF.md) · [4B BF16](MODEL_CARD_4B.md) · [4B Q8_0](MODEL_CARD_4B_GGUF.md)

## 相关资源

| 资源 | 说明 | 链接 |
|---|---|---|
| 代码 | 训练、评测、GGUF 导出与 HTTP 服务（GPU / CPU） | [GitHub](https://github.com/yanqiangmiffy/chinese-decision) |
| 测试平台 | Decision Studio：决策实验台、场景库、鲁棒性实验室、评测中心与性能压测 | [GitHub](https://github.com/yanqiangmiffy/chinese-decision-studio) |
| 4B BF16 | 合并权重与校准温度，Transformers 推理；本仓库由它直接转换 | [Hugging Face](https://huggingface.co/quincyqiang/chinese-decision-4b) · [ModelScope](https://modelscope.cn/models/quincyqiang/chinese-decision-4b) |
| 4B GGUF Q8_0（本仓库） | llama.cpp 推理，CPU 即可部署，不需要 PyTorch | [Hugging Face](https://huggingface.co/quincyqiang/chinese-decision-4b-gguf) · [ModelScope](https://modelscope.cn/models/quincyqiang/chinese-decision-4b-gguf) |
| 2B GGUF Q8_0 | 2.01 GB，CPU 延迟约为 4B 的一半 | [Hugging Face](https://huggingface.co/quincyqiang/chinese-decision-2b-gguf) · [ModelScope](https://modelscope.cn/models/quincyqiang/chinese-decision-2b-gguf) |
| 0.8B GGUF Q8_0 | 0.812 GB，最小最快 | [Hugging Face](https://huggingface.co/quincyqiang/chinese-decision-0.8b-gguf) · [ModelScope](https://modelscope.cn/models/quincyqiang/chinese-decision-0.8b-gguf) |

## 简介

本仓库是 [Chinese Decision 4B](https://huggingface.co/quincyqiang/chinese-decision-4b) 的 GGUF Q8_0 导出，用于 [llama.cpp](https://github.com/ggml-org/llama.cpp) / [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) 推理。原版是一个中文**决策模型**：给定上下文和一组固定候选项（单题 2–26 个；类别更多的 `choice` 问题可用服务端的 `/v1/tournament` 分组淘汰），一次前向直接返回每个候选项的概率，不生成任何 token；支持 `noul`（是/否）、`choice`（单选）、`score`（有序等级）三种题型，覆盖通用、医疗、法律、金融四个领域。

- **无需 PyTorch 和 GPU**：`llama-cpp-python` 即可在 CPU 上运行，也可用 CUDA 版 llama.cpp 跑在 GPU 上。
- **体积约为原版的一半**：Q8_0 为 4.48 GB，原版 safetensors 为 8.41 GB。
- **精度基本不变**：CJ-Bench 全量 307,900 条上准确率 85.9908%（BF16 交付版 85.9731%），首选项一致率 99.19%。
- **系列中准确率最高的 CPU 版本**：整体 ACC 比 2B Q8_0 高 1.88 个百分点，医疗 / 法律 / 金融平均高 5.57 个百分点。
- **接口不变**：与 BF16 版共用提示编译器、温度和 HTTP 接口，可直接接入 [Decision Studio](https://github.com/yanqiangmiffy/chinese-decision-studio)。

## 交付状态与模型信息

2026-10-21 完成量化、文件校验、CPU / HTTP 检查与 CJ-Bench 全量对照。

| 项目 | 值 |
|---|---|
| 源模型 | Chinese Decision 4B BF16 合并模型，验证 best step 5446 |
| 基础模型 | Qwen/Qwen3.5-4B |
| 量化文件 | chinese-decision-4b-Q8_0.gguf |
| 大小 | **4,482,402,784 字节，约 4.48 GB / 4.17 GiB** |
| 相对源 safetensors | 体积减少约 46.7% |
| 温度 | **1.024757981300354**，沿用独立验证拟合值，量化后未重拟合 |
| 输入 / 候选项 | 最长 16,384 tokens / 单题 2–26 项（训练覆盖 2–16 项），类别更多用 `/v1/tournament` 分组淘汰 |
| 提示版本 | hmm-zh-v1，单次前向读取候选字母 logits |
| 实测运行库 | llama-cpp-python 0.3.36，支持 Qwen3.5 混合注意力架构 |
| 发布 | [Hugging Face](https://huggingface.co/quincyqiang/chinese-decision-4b-gguf) · [ModelScope](https://modelscope.cn/models/quincyqiang/chinese-decision-4b-gguf) |

Q8_0 SHA256：0cffa8f69f1e0a108cfd06641bde67c0a075a018da726ad6e2ec64eb0a9d4b0a。
源 safetensors SHA256：5abd1c94bc8ed883e1e7cec554b1cbe4b26ade4255b7c9ffc71c8cd152da77c0。

## 文件说明

| 文件 | 说明 |
|---|---|
| chinese-decision-4b-Q8_0.gguf | 推理权重 |
| tokenizer.json、tokenizer_config.json、chat_template.jinja、config.json | 原始 tokenizer / 提示编译所需元数据 |
| decision_config.json | 模型身份、温度、输入与候选上限 |
| conversion.json、delivery.json | 转换器和源权重指纹、交付校验 |
| training_config.json、training_summary.json、validation_report.json | **源 BF16 模型**的训练/验证元数据，不是量化测试成绩 |
| LICENSE、LICENSES/、THIRD_PARTY_NOTICES.md | 原始许可与归属 |

本仓库不需要原始 safetensors，也不包含训练数据。BF16 GGUF（8,424,393,184 字节）只作为服务器上的转换对照，不随本仓库发布。

## 快速开始

### 下载

只需要 Q8_0 文件和同目录的 tokenizer / 配置文件，不需要原始 safetensors：

```bash
hf download quincyqiang/chinese-decision-4b-gguf --local-dir models/chinese-decision-4b-gguf
# 或 ModelScope
modelscope download --model quincyqiang/chinese-decision-4b-gguf --local_dir models/chinese-decision-4b-gguf
```

### 安装与启动 HTTP 服务

在 [代码仓库](https://github.com/yanqiangmiffy/chinese-decision) 根目录安装 CPU 推理依赖。运行库必须支持 Qwen3.5；本项目实际验证版本为 llama-cpp-python 0.3.36。

```bash
pip install -e ".[gguf]"
python scripts/serve.py --backend gguf \
  --base models/chinese-decision-4b-gguf \
  --gguf models/chinese-decision-4b-gguf/chinese-decision-4b-Q8_0.gguf \
  --threads 8 --context-size 16384 --host 127.0.0.1 --port 8791
```

Windows PowerShell，从项目根运行：

```powershell
$env:CUDA_VISIBLE_DEVICES = ""
$env:TOKENIZERS_PARALLELISM = "false"
.venv\Scripts\python.exe scripts/serve.py `
  --backend gguf `
  --base models/chinese-decision-4b-gguf `
  --gguf models/chinese-decision-4b-gguf/chinese-decision-4b-Q8_0.gguf `
  --threads 8 --context-size 16384 --host 127.0.0.1 --port 8791
```

```bash
curl http://127.0.0.1:8791/health
curl http://127.0.0.1:8791/v1/systemone -H 'Content-Type: application/json' --data-binary @examples/request.json
python scripts/verify_export.py --url http://127.0.0.1:8791
```

health 应显示 model=chinese-decision-4b、step=5446、backend=gguf、device=cpu、temperature=1.024757981300354。GGUF 目录不含 safetensors，因此验证已有服务时使用 --url。

### Python 调用

以下代码在项目环境内运行，通过与服务相同的编译器、tokenizer 校验和每题状态清理读取概率，不依赖 PyTorch：

```python
import json
from pathlib import Path
from decision.inference import GGUFBackend, candidate_probabilities
from decision.schema import compile_prompt, options_for

base = Path("models/chinese-decision-4b-gguf")
config = json.loads((base / "decision_config.json").read_text(encoding="utf-8"))
request = json.loads(Path("examples/request.json").read_text(encoding="utf-8"))
backend = GGUFBackend(base, base / "chinese-decision-4b-Q8_0.gguf", threads=8, context_size=16384)
try:
    for name, question in request["questions"].items():
        options = options_for(question)
        prompt = compile_prompt(backend.tokenizer, request["state"], question["instructions"], options)
        ids = backend.tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) > min(config["max_length"], backend.max_length):
            raise ValueError("input exceeds context limit")
        backend.validate_tokens(prompt, ids)
        # logits() clears hybrid attention/recurrent state for each question.
        probs = candidate_probabilities(backend.logits(ids, len(options)), config["temperature"])
        answer = {"probabilities": {key: p for (key, _), p in zip(options, probs)}}
        if question["type"] == "score":
            answer["score"] = sum(i * p for i, p in enumerate(probs))
        elif question["type"] == "noul":
            answer["noul"] = probs[1]
        else:
            answer["choice"] = options[max(range(len(probs)), key=probs.__getitem__)][0]
        print(name, answer)
finally:
    backend.close()
```

| 接口 | 用途 |
|---|---|
| `POST /v1/systemone` | 决策；可追加 `?debug=1` 查看候选 logits |
| `POST /api/alpha/decisions` | 兼容 hmm 的同一决策路由 |
| `POST /v1/batch` | 多请求批处理，单个请求失败不影响其他请求 |
| `POST /v1/compile` | 编译输入并计数 tokens |
| `POST /v1/tournament` | 类别更多的 `choice` 问题：分组淘汰后决赛，每组不超过 16 个候选 |
| `GET /health`、`GET /v1/info` | 模型身份、step、温度及运行信息 |

每个请求最多 64 个问题，每题 2–26 个候选（类别更多时用 `/v1/tournament`，示例 [examples/tournament.json](https://github.com/yanqiangmiffy/chinese-decision/blob/main/examples/tournament.json) 有 41 个客服意图），最长 16,384 tokens；超长或不合法输入返回 422，不静默截断。`score` 输出从 0 开始的等级期望值，不是最大概率等级；`confidence = (K*p_max-1)/(K-1)` 衡量分布尖锐程度，不等于经验准确率。

## 量化质量：CJ-Bench 全量对照

在 CJ-Bench 全量测试集上，对比 4B BF16 合并权重（Transformers）与本仓库的 Q8_0 GGUF（llama.cpp CUDA）。两者使用相同的输入 token（共 68,515,700 个）、候选顺序和部署温度 T = 1.024758，307,900 条全部覆盖（硬标签 304,822、软标签 3,078），没有过滤或截断，也没有在测试集上拟合温度或阈值。全量于 2026-10-21 20:33（北京时间）完成，源模型、GGUF 与数据指纹在评测后复核通过。

| 指标 | chinese-decision-4b（BF16） | chinese-decision-4b-gguf（Q8_0，本仓库） | 变化（Q8−BF16） |
| --- | ---: | ---: | ---: |
| 整体 micro ACC (%) ↑ | 85.9731 | 85.9908 | +0.0177 pp |
| 四领域 macro ACC (%) ↑ | 81.9239 | 81.9521 | +0.0282 pp |
| 医法金等权 ACC (%) ↑ | 84.0033 | 84.0385 | +0.0352 pp |
| ECE (15 bins, %) ↓ | 0.9970 | 0.9568 | −0.0402 pp |
| 分布 NLL ↓ | 0.370641 | 0.370469 | −0.000172 |
| Hard NLL ↓ | 0.364202 | 0.364038 | −0.000164 |
| Brier ↓ | 0.190859 | 0.190742 | −0.000117 |
| Score MAE ↓ | 0.525270 | 0.525313 | +0.000043 |

### 分领域与分题型

| 领域 | 条数 | BF16 ACC (%) | Q8 ACC (%) | ΔACC (pp) | BF16 ECE (%) | Q8 ECE (%) | BF16 NLL | Q8 NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 通用 | 100,000 | 75.6856 | 75.6928 | +0.0072 | 2.9129 | 2.8725 | 0.621271 | 0.620947 |
| 医疗 | 200,000 | 91.1895 | 91.2110 | +0.0215 | 0.2480 | 0.2268 | 0.241017 | 0.240859 |
| 法律 | 4,300 | 77.2093 | 77.3488 | +0.1395 | 2.0195 | 1.3783 | 0.528906 | 0.530872 |
| 金融 | 3,600 | 83.6111 | 83.5556 | −0.0556 | 1.6454 | 1.7856 | 0.421052 | 0.421744 |

| 题型 | 条数 | BF16 ACC (%) | Q8 ACC (%) | BF16 ECE (%) | Q8 ECE (%) | BF16 NLL | Q8 NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `noul` | 134,577 | 95.4665 | 95.4613 | 0.4326 | 0.4184 | 0.107485 | 0.107399 |
| `choice` | 134,577 | 83.2505 | 83.3062 | 1.2599 | 1.1827 | 0.474925 | 0.474825 |
| `score` | 38,746 | 60.4267 | 60.3875 | 2.1804 | 2.1359 | 0.922455 | 0.921737 |

**领域 × 题型准确率（Q8_0，%）**

| 领域 | `noul` | `choice` | `score` |
|---|---:|---:|---:|
| 通用 | 88.54 | 78.64 | 58.29 |
| 医疗 | 98.02 | 85.09 | 67.83 |
| 法律 | 89.07 | 74.53 | 67.08 |
| 金融 | 85.83 | 76.92 | 87.92 |

### 逐条变化与来源诊断

全量 top-1 一致率 **99.19%**（305,400 / 307,900），共有 **2,500 条**首选项改变。硬标签 **948 条由对变错、1,002 条由错变对**，净多对 **54 条**。平均候选概率绝对差 **0.001865**（约 0.19 个百分点）；逐题最大概率差的中位数为 **0.000752**、p95 为 **0.020506**，全量最大为 **0.167899**（16.79 个百分点）。整体成绩接近不意味着每条概率或答案完全相同。

| 分组 | top-1 一致率 | 由对变错 / 由错变对 | 逐题最大概率差 p95 |
| --- | ---: | ---: | ---: |
| 通用 | 98.49% | 566 / 573 | 0.022866 |
| 医疗 | 99.56% | 335 / 378 | 0.017822 |
| 法律 | 98.51% | 25 / 31 | 0.031706 |
| 金融 | 98.67% | 22 / 20 | 0.028992 |
| `noul` | 99.78% | 151 / 144 | 0.010279 |
| `choice` | 99.18% | 388 / 463 | 0.023147 |
| `score` | 97.14% | 409 / 395 | 0.025405 |

法律准确率增加 0.14 个百分点、ECE 改善 0.64，但这只对应 4,300 条中净多对 6 条；金融准确率下降 0.06、ECE 升高 0.14。`score` 题型的一致率最低（97.14%），整体 Score MAE 基本不变。不能据此宣称量化普遍提高精度。

商品分类 mep3m_product_catalog 的 3,151 条准确率 64.5827% → 64.6144%（多对 1 条），ECE 仍约 22.6%，量化没有改善源模型在该来源上的短板。USTS 来源共有 3,200 条，但只有 122 条硬标签：ACC 68.8525% → 63.9344%（少对 6 条），不能把这个变化当成对全部 3,200 条的统计。法律 `noul` 准确率 88.8667% → 89.0667%，ECE 3.5973% → 3.3179%；金融 `score` 准确率 88.1667% → 87.9167%。

原始全量报告与逐条对比汇总存于 reports/release-4b-5446-20261021：q8-full/full.json、full-quantized-comparison.json、summary.json、evaluation_analysis.json；逐条预测与服务器同步在 runs/cj-bench/release-4b-5446-20261021/q8-full，复查不需要重新运行模型。

### 1200 条分层 CJ-Bench 测试子集

为区分"格式与后端差异"和"量化差异"，额外导出未量化的 BF16 GGUF 作为对照（不发布）。4 个领域 × 3 种题型，每组 100 条，seed 42；1,192 条硬标签与 8 条软标签。三路使用同样输入、候选顺序和 T = 1.024757981300354；15-bin ECE。该测试子集与训练选模使用的 1200 条验证集不同，也不具有全量测试集的领域权重。

| 4B 后端 | 测试条数 | ACC (%) | ECE (%) | NLL | Brier | Score MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Transformers BF16 | 1200 | 80.4530 | 3.4379 | 0.484243 | 0.272322 | 0.426471 |
| BF16 GGUF | 1200 | 80.8725 | 3.8746 | 0.483633 | 0.271973 | 0.426534 |
| Q8_0 GGUF | 1200 | 80.4530 | 3.6900 | 0.482511 | 0.271181 | 0.425485 |

| 配对比较 | top-1 一致率 | 由对变错 / 由错变对 | 平均概率绝对差 | 最大概率差 |
| --- | ---: | ---: | ---: | ---: |
| BF16 GGUF vs Transformers BF16（只换后端） | 99.17%（1190/1200） | 2 / 7 | 0.002453 | 0.038587 |
| Q8_0 GGUF vs BF16 GGUF（只做量化） | 99.17%（1190/1200） | 6 / 1 | 0.002471 | 0.096816 |
| Q8_0 GGUF vs Transformers BF16（端到端） | 98.67%（1184/1200） | 6 / 6 | 0.003536 | 0.119542 |

只换后端与只做量化带来的首选项变化处于同一量级。端到端差异包含格式、后端、量化与批次算术，不能声称完全无损。

### 全量对比口径

CJ-Bench 全量为 **307,900 条**（304,822 条硬标签、3,078 条软标签），通用/医疗/法律/金融分别为 100,000 / 200,000 / 4,300 / 3,600 条。全部保留原始候选顺序，无过滤、无截断。ACC/ECE 仅计硬标签，ECE 使用 15 个等宽分箱；分布 NLL 与 Brier 包含软标签；Score MAE 覆盖 38,746 条评分题，计算预测期望等级与目标期望等级的绝对差。

整体 micro 按硬标签样本数加权，四领域 macro 是四个领域的等权平均，专业平均仅对医疗、法律、金融等权。选模和温度拟合只使用验证数据，未使用 CJ-Bench 标签。全量质量评测在 llama.cpp CUDA 上执行，CPU 只完成实际接口、诊断输入与 25 条测试输入的延迟测量，不能表述为 CPU 全量测试。

与 0.8B、2B 及外部模型的完整对比见 [4B BF16 模型卡](MODEL_CARD_4B.md) 与 [模型总览](MODELS.md)。

## 推理效率

### Windows 本机

Intel Core i7-9700，8 核/8 线程，32 GB RAM；CPU 8 线程，运行容量 16384 tokens。15 条诊断输入各测 3 次，其中 12 条短输入小于 100 tokens：

| 指标 | 4B Q8_0 | 对照：2B Q8_0 |
|---|---:|---:|
| 短输入 p50 | 2.550 秒 | 1.351 秒 |
| 短输入 p95 | 3.139 秒 | 1.961 秒 |
| 852-token 输入中位延迟 | 29.268 秒 | 14.293 秒 |
| 诊断进程峰值内存 | 5.365 GiB | 2.701 GiB |
| 交错输入后重复概率最大差 | 0 | 0 |
| HTTP 三题型、双路由、批量、无效与超长输入 | 通过 | 通过 |

延迟仅含预先分词后的前向和概率计算，不含分词、HTTP 或网络；诊断输入没有质量标签，不是业务准确率评测。临时 HTTP 测试服务已关闭。

### Linux 共享服务器

同一组 25 条 CJ-Bench 输入（92–6521 tokens），每条预热后测 3 次；短输入为 16 条、92–255 tokens。CPU 为 Xeon Gold 5318Y，GPU 为 A800 80GB：

| 4B 后端 | 短输入 p50 (ms) | 全部输入 p50 (ms) | 全部 p95 (ms) | 进程内存观测最大值 (GiB) |
| --- | ---: | ---: | ---: | ---: |
| Q8_0 / CPU 16 threads | 1695.09 | 2004.01 | 31481.25 | 6.14 |
| Transformers BF16 / GPU | 59.94 | 60.06 | 198.99 | 9.01 |
| Q8_0 / GPU | 39.66 | 47.29 | 569.17 | 5.32 |

进程内存为 RSS 观测值，不含 GPU 显存；表内取所记录当前 RSS 与峰值采样中的较大值。本次没有测量 4B 的 Transformers FP32 / CPU 基线。Q8_0 在 GPU 上短输入比 BF16 更快，长输入的 p95 更高；CPU 上长输入延迟可达数十秒，适合短文本或离线批处理。共享服务器存在其他任务，结果仅适用于该次测量。

## 转换来源与局限

转换器 SHA256 为 e9a1da876330bbce9687541ab31736542a01b4ac43c6686126514a50f122fb7f，参数 --outtype q8_0 --no-mtp，源模型未改写；BF16 GGUF 对照 SHA256 为 930c9acb573afa82e4bd5aded4996ecc42aefc15ad241ae28d8c7fed4a2a4e6f。为兼容现有转换脚本，4B BF16 导出保存为单个 safetensors 文件。

- 继承源 4B 的短板：商品分类来源准确率偏低、通用领域 ECE 高于 2B 和 0.8B；候选顺序保持一致。
- Qwen3.5 含循环状态，每道独立题必须清空状态，不能跨题复用前缀缓存。
- 更低比特格式未在本次交付中提供，不能套用 Q8_0 检查结论。
- 固定 logits 的温度不改变 argmax，量化及后端则可能改变；量化后没有重新拟合温度。
- 这是决策模型，不能用 Ollama / llama-cli 的聊天生成结果作为验证。

训练方法、数据与源模型全量结果见 [4B BF16 模型卡](MODEL_CARD_4B.md)；训练过程中各 checkpoint 的结果见 [训练过程 checkpoint 评测](TRAINING_HISTORY.md)。

## 归属、许可与引用

模型权重保留基础 Qwen3.5-4B 的 Apache-2.0 LICENSE。项目代码以 MIT 为主，RLCD 改编部分保留 Apache-2.0；原文与声明见 LICENSES/ 和 THIRD_PARTY_NOTICES.md。数据许可按来源分别适用，模型交付不授予数据再分发权利。方法参考 [hmm](https://github.com/n4ze3m/hmm)，RLCD 与数据参考 [Chinese-Jev](https://github.com/gulucaptain/Chinese-Jev)；GGUF 转换使用 [llama.cpp](https://github.com/ggml-org/llama.cpp)。使用数据或基线时请同时引用 [Chinese-Jev 论文](https://arxiv.org/html/2609.36965v1)。

```bibtex
@misc{chinese-decision-2026,
  title = {Chinese Decision: A Lightweight Chinese Probabilistic Decision Model},
  author = {yanqiangmiffy},
  year = {2026},
  howpublished = {\url{https://github.com/yanqiangmiffy/chinese-decision}}
}
```
