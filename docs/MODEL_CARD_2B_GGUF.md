---
license: apache-2.0
language:
- zh
base_model:
- quincyqiang/chinese-decision-2b
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

# Chinese Decision 2B GGUF Q8_0

**Chinese Decision 2B 的 CPU 部署包，保留相同提示、候选概率计算与校准温度。**

> A 2.01 GB Q8_0 GGUF export of [Chinese Decision 2B](https://huggingface.co/quincyqiang/chinese-decision-2b) for llama.cpp, converted directly from the merged step-4994 weights and keeping their calibrated temperature. On the full CJ-Bench (307,900 decisions) it reaches 84.11% accuracy and 0.65% ECE, versus 84.11% and 0.64% for the BF16 release, with 99.00% top-choice agreement. It runs on CPU without PyTorch or a GPU. Like the original model, it is not a chat model: read the final-position logits of the candidate letters and apply softmax.

[代码](https://github.com/yanqiangmiffy/chinese-decision) · [模型总览](MODELS.md) · [Decision Studio](https://github.com/yanqiangmiffy/chinese-decision-studio) · [训练数据](https://huggingface.co/datasets/bbldCVer-hf/CJ-Dataset) · [测试集](https://huggingface.co/datasets/bbldCVer-hf/CJ-Bench)

[0.8B BF16](MODEL_CARD_0_8B.md) · [0.8B Q8_0](MODEL_CARD_0_8B_GGUF.md) · [2B BF16](MODEL_CARD_2B.md) · [2B Q8_0](MODEL_CARD_2B_GGUF.md)

## 相关资源

| 资源 | 说明 | 链接 |
|---|---|---|
| 代码 | 训练、评测、GGUF 导出与 HTTP 服务（GPU / CPU） | [GitHub](https://github.com/yanqiangmiffy/chinese-decision) |
| 测试平台 | Decision Studio：决策实验台、场景库、鲁棒性实验室、评测中心与性能压测 | [GitHub](https://github.com/yanqiangmiffy/chinese-decision-studio) |
| 2B BF16 | 合并权重与校准温度，Transformers 推理；本仓库由它直接转换 | [Hugging Face](https://huggingface.co/quincyqiang/chinese-decision-2b) · [ModelScope](https://modelscope.cn/models/quincyqiang/chinese-decision-2b) |
| 2B GGUF Q8_0（本仓库） | llama.cpp 推理，CPU 即可部署，不需要 PyTorch | [Hugging Face](https://huggingface.co/quincyqiang/chinese-decision-2b-gguf) · [ModelScope](https://modelscope.cn/models/quincyqiang/chinese-decision-2b-gguf) |
| 0.8B GGUF Q8_0 | 0.812 GB，更小更快 | [Hugging Face](https://huggingface.co/quincyqiang/chinese-decision-0.8b-gguf) · [ModelScope](https://modelscope.cn/models/quincyqiang/chinese-decision-0.8b-gguf) |

## 简介

本仓库是 [Chinese Decision 2B](https://huggingface.co/quincyqiang/chinese-decision-2b) 的 GGUF Q8_0 导出，用于 [llama.cpp](https://github.com/ggml-org/llama.cpp) / [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) 推理。原版是一个中文**决策模型**：给定上下文和 2–26 个固定候选项，一次前向直接返回每个候选项的概率，不生成任何 token；支持 `noul`（是/否）、`choice`（单选）、`score`（有序等级）三种题型，覆盖通用、医疗、法律、金融四个领域。

- **无需 PyTorch 和 GPU**：`llama-cpp-python` 即可在 CPU 上运行，也可用 CUDA 版 llama.cpp 跑在 GPU 上。
- **体积约为原版的一半**：Q8_0 为 2.01 GB，原版 safetensors 为 3.76 GB。
- **内存占用更低**：同一台服务器上，CPU 推理进程内存观测最大值 3.34 GiB，Transformers FP32 为 11.74 GiB。
- **精度基本不变**：CJ-Bench 全量 307,900 条上准确率 84.1094%（BF16 交付版 84.1107%），首选项一致率 99.00%。
- **接口不变**：与 BF16 版共用提示编译器、温度和 HTTP 接口，可直接接入 [Decision Studio](https://github.com/yanqiangmiffy/chinese-decision-studio)。

## 交付状态与模型信息

2026-10-17 完成量化、文件校验、CPU / HTTP 检查与 CJ-Bench 全量对照，并在 Hugging Face 与 ModelScope 公开发布。本次转换、CPU 检查及新评测只运行 2B；0.8B 仅引用既有报告。

| 项目 | 值 |
|---|---|
| 源模型 | Chinese Decision 2B BF16 合并模型，验证 best step 4994 |
| 基础模型 | Qwen/Qwen3.5-2B |
| 量化文件 | chinese-decision-2b-Q8_0.gguf |
| 大小 | **2,012,012,800 字节，约 2.01 GB / 1.87 GiB** |
| 相对源 safetensors | 体积减少约 46.5% |
| 温度 | **1.074872374534607**，沿用独立验证拟合值，量化后未重拟合 |
| 输入 / 候选项 | 最长 16,384 tokens / 2–26 项 |
| 提示版本 | hmm-zh-v1，单次前向读取候选字母 logits |
| 实测运行库 | llama-cpp-python 0.3.36，支持 Qwen3.5 混合注意力架构 |
| 发布 | [Hugging Face](https://huggingface.co/quincyqiang/chinese-decision-2b-gguf) · [ModelScope](https://modelscope.cn/models/quincyqiang/chinese-decision-2b-gguf) |

Q8_0 SHA256：4f902f73781f87cc8388617bc7e2e99eb14fdeb8551f7ebb48ffbde69394cc02。
源 safetensors SHA256：a8fadd326b19a278f8c26216ca481b607369baca33962aac0d8fda603c830068。

## 文件说明

| 文件 | 说明 |
|---|---|
| chinese-decision-2b-Q8_0.gguf | CPU 推理权重 |
| tokenizer、chat template、config.json | 原始 tokenizer / 提示编译所需元数据 |
| decision_config.json | 模型身份、温度、输入与候选上限 |
| conversion.json、delivery.json | 转换器和源权重指纹、交付校验 |
| training_summary.json、validation_report.json | **源 BF16 模型**的训练/验证元数据，不是量化测试成绩 |
| LICENSE、LICENSES/、THIRD_PARTY_NOTICES.md | 原始许可与归属 |

本机 CPU 包不需要原始 safetensors，也不包含训练数据。BF16 GGUF（3,775,709,440 字节）是服务器保留的转换对照，不属于本机 Q8_0 必需文件。

## 快速开始

### 下载

只需要 Q8_0 文件和同目录的 tokenizer / 配置文件，不需要原始 safetensors：

```bash
hf download quincyqiang/chinese-decision-2b-gguf --local-dir models/chinese-decision-2b-gguf
# 或 ModelScope
modelscope download --model quincyqiang/chinese-decision-2b-gguf --local_dir models/chinese-decision-2b-gguf
```

### 安装与启动 HTTP 服务

在 [代码仓库](https://github.com/yanqiangmiffy/chinese-decision) 根目录安装 CPU 推理依赖。运行库必须支持 Qwen3.5；本项目实际验证版本为 llama-cpp-python 0.3.36。

```bash
pip install -e ".[gguf]"
python scripts/serve.py --backend gguf \
  --base models/chinese-decision-2b-gguf \
  --gguf models/chinese-decision-2b-gguf/chinese-decision-2b-Q8_0.gguf \
  --threads 8 --context-size 16384 --host 127.0.0.1 --port 8790
```

本机 Windows 已有可用环境，从项目根运行：

```powershell
$env:CUDA_VISIBLE_DEVICES = ""
$env:TOKENIZERS_PARALLELISM = "false"
.venv\Scripts\python.exe scripts/serve.py `
  --backend gguf `
  --base models/chinese-decision-2b-gguf `
  --gguf models/chinese-decision-2b-gguf/chinese-decision-2b-Q8_0.gguf `
  --threads 8 --context-size 16384 --host 127.0.0.1 --port 8790
```

```bash
curl http://127.0.0.1:8790/health
curl http://127.0.0.1:8790/v1/systemone -H 'Content-Type: application/json' --data-binary @examples/request.json
python scripts/verify_export.py --url http://127.0.0.1:8790
```

health 应显示 model=chinese-decision-2b、step=4994、backend=gguf、device=cpu、temperature=1.074872374534607。GGUF 目录不含 safetensors，因此验证已有服务时使用 --url。

### Python 调用

以下代码在项目环境内运行，通过与服务相同的编译器、tokenizer 校验和每题状态清理读取概率，不依赖 PyTorch：

```python
import json
from pathlib import Path
from decision.inference import GGUFBackend, candidate_probabilities
from decision.schema import compile_prompt, options_for

base = Path("models/chinese-decision-2b-gguf")
config = json.loads((base / "decision_config.json").read_text(encoding="utf-8"))
request = json.loads(Path("examples/request.json").read_text(encoding="utf-8"))
backend = GGUFBackend(base, base / "chinese-decision-2b-Q8_0.gguf", threads=8, context_size=16384)
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
| `GET /health`、`GET /v1/info` | 模型身份、step、温度及运行信息 |

每个请求最多 64 个问题，每题 2–26 个候选，最长 16,384 tokens；超长或不合法输入返回 422，不静默截断。`score` 输出从 0 开始的等级期望值，不是最大概率等级；`confidence = (K*p_max-1)/(K-1)` 衡量分布尖锐程度，不等于经验准确率。

## 量化质量：CJ-Bench 全量对照

在 CJ-Bench 全量测试集上，对比 2B BF16 合并权重（Transformers）与本仓库的 Q8_0 GGUF（llama.cpp CUDA）。两者使用相同的输入 token（共 68,515,700 个）、候选顺序和部署温度 T = 1.074872，307,900 条全部覆盖（硬标签 304,822、软标签 3,078），没有过滤或截断，也没有在测试集上拟合温度或阈值。全量于 2026-10-17 21:34:34（北京时间）完成，源模型、GGUF 与数据指纹在评测后复核通过。

| 指标 | chinese-decision-2b（BF16） | chinese-decision-2b-gguf（Q8_0，本仓库） | 变化（Q8−BF16） |
| --- | ---: | ---: | ---: |
| 整体 micro ACC (%) ↑ | 84.1107 | 84.1094 | -0.0013 pp |
| 四领域 macro ACC (%) ↑ | 77.1750 | 77.3282 | +0.1532 pp |
| 医法金等权 ACC (%) ↑ | 78.2476 | 78.4663 | +0.2187 pp |
| ECE (15 bins, %) ↓ | 0.6361 | 0.6534 | +0.0173 pp |
| 分布 NLL ↓ | 0.416520 | 0.416288 | -0.000232 |
| Hard NLL ↓ | 0.410254 | 0.410017 | -0.000236 |
| Brier ↓ | 0.213295 | 0.213189 | -0.000106 |
| Score MAE ↓ | 0.564216 | 0.563738 | -0.000478 |

### 分领域与分题型

| 领域 | 条数 | BF16 ACC (%) | Q8 ACC (%) | ΔACC (pp) | BF16 ECE (%) | Q8 ECE (%) | BF16 NLL | Q8 NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 通用 | 100,000 | 73.9574 | 73.9141 | -0.0433 | 1.9259 | 1.9809 | 0.662951 | 0.662286 |
| 医疗 | 200,000 | 89.4985 | 89.5035 | +0.0050 | 0.2412 | 0.2482 | 0.284226 | 0.284240 |
| 法律 | 4,300 | 68.7442 | 69.3953 | +0.6512 | 3.3741 | 2.9244 | 0.703494 | 0.701638 |
| 金融 | 3,600 | 76.5000 | 76.5000 | +0.0000 | 2.1299 | 1.4687 | 0.578118 | 0.578179 |

| 题型 | 条数 | BF16 ACC (%) | Q8 ACC (%) | BF16 ECE (%) | Q8 ECE (%) | BF16 NLL | Q8 NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `noul` | 134,577 | 95.1084 | 95.0928 | 0.3352 | 0.2992 | 0.114612 | 0.114498 |
| `choice` | 134,577 | 80.1222 | 80.1563 | 0.8967 | 0.8514 | 0.559377 | 0.559318 |
| `score` | 38,746 | 57.6651 | 57.5838 | 1.6469 | 1.5513 | 0.968959 | 0.967715 |

### 逐条变化与来源诊断

全量 top-1 一致率 **99.00%**（304,826 / 307,900），共有 **3,074 条**首选项改变。硬标签 **1,105 条由对变错、1,101 条由错变对**，净少对 **4 条**。平均候选概率绝对差 **0.002326**（约 0.2326 个百分点）；逐题最大概率差的 p95 为 **0.023659**，全量最大为 **0.275168**（27.52 个百分点）。整体成绩接近不意味着每条概率或答案完全相同。

Q8_0 保留了整体表现：micro ACC 从 84.1107% 到 84.1094%，净少答对 4 条；NLL、Brier 和整体 Score MAE 略降，整体 ECE 从 0.6361% 小幅升至 0.6534%。四领域 macro 从 77.1750% 到 77.3282%，主要来自法律领域的变化；不能据此宣称量化普遍提高精度。

通用准确率下降 0.0433 个百分点，医疗增加 0.0050，法律增加 0.6512，金融不变。法律 ECE 改善 0.4497 个百分点，金融改善 0.6612；通用和医疗 ECE 略升。总体 score ACC 从 57.6651% 到 57.5838%，尽管整体评分 MAE 略好，金融 score 的 MAE 却从 0.402933 升到 0.404192。

商品分类 mep3m_product_catalog 的 3,151 条准确率保持 70.9616%，NLL 从 1.512522 到 1.508412，没有再次出现训练 step3000 的崩落。USTS 来源共有 3,200 条，但只有 122 条硬标签：ACC 61.4754%→57.3770%（少对 5 条），ECE 10.8889%→13.4536%；不能把这个变化当成对全部 3,200 条硬分类的统计。法律 noul 的准确率 82.0667%→82.7333%，但其 ECE 2.6117%→3.2733%，也不能用领域总体 ECE 代替题型结论。

原始全量报告、逐条对比汇总、最大概率差样例与完成条件核对存于 reports/gguf-2b-20261017：server/q8-full/full.json、comparison.json、summary.json、completion_verification.json、analysis.md；逐条预测（full_predictions.jsonl.gz）与服务器同步在 runs/cj-bench/gguf-2b-q8-20261017/q8-full，复查不需要重新运行模型。0.8B Q8_0 的同口径结果（一致率 98.92%）见 [0.8B GGUF 模型卡](MODEL_CARD_0_8B_GGUF.md)，两者对照见 [模型总览](MODELS.md#q8_0-量化与-cpu)。

### 1200 条分层 CJ-Bench 测试子集

为区分"格式与后端差异"和"量化差异"，额外导出未量化的 BF16 GGUF 作为对照（未发布）。4 个领域 × 3 种题型，每组 100 条，seed 42；1,192 条硬标签与 8 条软标签，包含 400 条评分题。三路使用同样输入、候选顺序和 T=1.074872374534607；15-bin ECE。该测试子集与训练选模使用的 1200 条验证集不同，也不具有全量测试集的领域权重。

| 2B 后端 | 测试条数 | ACC (%) | ECE (%) | NLL | Brier | Score MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Transformers BF16 | 1200 | 75.9228 | 2.5788 | 0.595550 | 0.329166 | 0.504446 |
| BF16 GGUF | 1200 | 75.8389 | 2.7841 | 0.596385 | 0.329873 | 0.506524 |
| Q8_0 GGUF | 1200 | 75.8389 | 2.4443 | 0.595557 | 0.329245 | 0.506174 |

Q8_0 相对 Transformers BF16 的 top-1 一致率为 **98.75%**（1185/1200），平均候选概率绝对差 **0.004257**、最大 **0.073770**；6 条硬标签由对变错、5 条由错变对。BF16 GGUF 相对 Q8_0 的 top-1 一致率为 **99.08%**，准确率相同，但单条答案与概率仍会变化。端到端差异包含格式、后端、量化与批次算术，不能声称完全无损。

### 全量对比口径

CJ-Bench 全量为 **307,900 条**（304,822 条硬标签、3,078 条软标签），通用/医疗/法律/金融分别为 100,000 / 200,000 / 4,300 / 3,600 条。全部保留原始候选顺序，无过滤、无截断。ACC/ECE 仅计硬标签，ECE 使用 15 个等宽分箱；分布 NLL 与 Brier 包含软标签；Score MAE 覆盖 38,746 条评分题，计算预测期望等级与目标期望等级的绝对差。

整体 micro 按硬标签样本数加权，四领域 macro 是四个领域的等权平均，专业平均仅对医疗、法律、金融等权。不同量不能互换，整体 ECE 也不是领域 ECE 的平均。选模和温度拟合只使用验证数据，未使用 CJ-Bench 标签。

固定任务 runs/cj-bench/gguf-2b-q8-20261017；对照为 runs/cj-bench/qwen35-2b-rlcd-merged-calibrated-20261017-175001。全量质量评测在 llama.cpp CUDA 上执行，CPU 只完成实际接口、诊断输入与 25 条测试输入的后端对照，不能表述为 CPU 全量测试。

0.8B 与其他模型的完整成绩和论文来源见 [模型总览](MODELS.md)。0.8B Q8_0 已有独立全量结果，不能据此推断 2B 量化结果。

## 推理效率

### Windows 本机

Intel Core i7-9700，8 核/8 线程，32 GB RAM；CPU 8 线程，运行容量 16384 tokens。15 条诊断输入各测 3 次，12 条短输入小于 100 tokens：

| 指标 | 2B Q8_0 |
|---|---:|
| 短输入 p50 | 1.351 秒 |
| 短输入 p95 | 1.961 秒 |
| 852-token 输入中位延迟 | 14.293 秒 |
| 诊断进程峰值内存 | 2.701 GiB |
| 交错输入后重复概率最大差 | 0 |
| HTTP 三题型、双路由、批量、无效与超长输入 | 通过 |

延迟仅含预先分词后的前向和概率计算，不含分词、HTTP 或网络；诊断输入没有质量标签，不是业务准确率评测。临时 HTTP 测试服务已关闭。

### Linux 共享服务器

同一组 25 条 CJ-Bench 输入（92–6521 tokens），每条预热后测 3 次；短输入为 16 条、92–255 tokens。CPU 为 Xeon Gold 5318Y，GPU 为 A800 80GB，CPU 16 线程：

| 2B 后端 | 短输入 p50 (ms) | 全部输入 p50 (ms) | 全部 p95 (ms) | 进程内存观测最大值 (GiB) |
| --- | ---: | ---: | ---: | ---: |
| Transformers FP32 / CPU 16 threads | 545.29 | 601.34 | 11320.53 | 11.74 |
| Q8_0 / CPU 16 threads | 664.50 | 817.99 | 12884.58 | 3.34 |
| Transformers BF16 / GPU | 44.68 | 44.77 | 90.27 | 4.67 |
| Q8_0 / GPU | 20.39 | 23.60 | 274.74 | 3.02 |

进程内存为 RSS 观测值，不含 GPU 显存；表内取所记录当前 RSS 与峰值采样中的较大值。Q8_0 在这个 CPU 环境降低内存，但短输入与整体 p50 比 FP32 更慢；GPU 上短输入更快，整体 p95 更高。量化收益不能概括为所有输入更快。共享服务器存在其他任务，结果仅适用于该次测量。

## 转换来源与局限

转换器 SHA256 为 e9a1da876330bbce9687541ab31736542a01b4ac43c6686126514a50f122fb7f，参数 --outtype q8_0 --no-mtp，源模型未改写。BF16 GGUF SHA256 为 9c930f5274f4dcd88ddbd8efd360d567df1c3cef796813ec37650ee5df5539e2。转换的可选 Git 版本记录曾因源码包的失效 Git 指针报错；校验两份 GGUF 和源权重后已补齐元数据，conversion_recovery.json 为 passed，未重新转换权重。

- 继承源 2B 的法律判断、金融评分及商品分类波动；候选顺序保持一致。
- Qwen3.5 含循环状态，每道独立题必须清空状态，不能跨题复用前缀缓存。
- 更低比特格式未在本次交付中提供，不能套用 Q8_0 检查结论。
- 固定 logits 的温度不改变 argmax，量化及后端则可能改变；量化后没有重新拟合温度。
- 这是决策模型，不能用 Ollama / llama-cli 的聊天生成结果作为验证。

更完整的运行记录、启动与复现见 [2B CPU 文档](CPU_2B.md)，本机证据在 reports/gguf-2b-20261017。训练方法、数据与源模型全量结果见 [2B BF16 模型卡](MODEL_CARD_2B.md) 与 [训练交付记录](TRAINING_2B.md)。

## 归属、许可与引用

模型权重保留基础 Qwen3.5-2B 的 Apache-2.0 LICENSE。项目代码以 MIT 为主，RLCD 改编部分保留 Apache-2.0；原文与声明见 LICENSES/ 和 THIRD_PARTY_NOTICES.md。数据许可按来源分别适用，模型交付不授予数据再分发权利。方法参考 [hmm](https://github.com/n4ze3m/hmm)，RLCD 与数据参考 [Chinese-Jev](https://github.com/gulucaptain/Chinese-Jev)；GGUF 转换使用 [llama.cpp](https://github.com/ggml-org/llama.cpp)。使用数据或基线时请同时引用 [Chinese-Jev 论文](https://arxiv.org/html/2609.36965v1)。

```bibtex
@misc{chinese-decision-2026,
  title = {Chinese Decision: A Lightweight Chinese Probabilistic Decision Model},
  author = {yanqiangmiffy},
  year = {2026},
  howpublished = {\url{https://github.com/yanqiangmiffy/chinese-decision}}
}
```
