# Chinese Decision 2B：GGUF Q8_0 与 CPU 部署

更新日期：2026-10-17。Q8_0 已转换、校验并发布到 [Hugging Face](https://huggingface.co/quincyqiang/chinese-decision-2b-gguf) / [ModelScope](https://modelscope.cn/models/quincyqiang/chinese-decision-2b-gguf)，实际 CPU 推理及 HTTP 检查通过，CJ-Bench 全量 307,900 条质量对照已完成。本次只运行 2B 模型；0.8B 的 CPU 部署见 [CPU.md](CPU.md)。

## 本次交付

源模型为已合并的 Chinese Decision 2B，验证集选出的 step 4994；基础模型 Qwen3.5-2B，训练 13,048,000 条决策。保留独立验证划分拟合的温度 **T = 1.074872374534607**，不在测试集上重新拟合。

| 文件 | 大小（字节） | 位置 |
|---|---:|---|
| 原始 BF16 safetensors | 3,763,692,048 | 服务器 models/chinese-decision-2b |
| BF16 GGUF 转换对照 | 3,775,709,440 | 服务器 models/chinese-decision-2b-gguf |
| **Q8_0 GGUF** | **2,012,012,800** | **本机和服务器 models/chinese-decision-2b-gguf** |

Q8_0 文件约 **2.01 GB / 1.87 GiB**，比原始 safetensors 小约 **46.5%**。运行内存还包括上下文缓存、计算缓冲区、tokenizer 和运行库，不能用权重文件大小代替实际内存。

Q8_0 SHA256：

```text
4f902f73781f87cc8388617bc7e2e99eb14fdeb8551f7ebb48ffbde69394cc02
```

源权重 SHA256 为 a8fadd326b19a278f8c26216ca481b607369baca33962aac0d8fda603c830068；BF16 GGUF SHA256 为 9c930f5274f4dcd88ddbd8efd360d567df1c3cef796813ec37650ee5df5539e2。

本机目录包含 Q8_0、tokenizer、chat template、decision_config.json、训练及校准元数据、许可证、conversion.json 和 delivery.json，CPU 推理不需要原始 safetensors。BF16 GGUF 对照文件保留在服务器，没有复制到本机。

## 本机 CPU 启动

在项目根运行以下 PowerShell 命令，使用现有支持 Qwen3.5 的 llama-cpp-python 0.3.36 CPU 环境：

```powershell
$env:CUDA_VISIBLE_DEVICES = ""
$env:TOKENIZERS_PARALLELISM = "false"
.venv\Scripts\python.exe scripts/serve.py `
  --backend gguf `
  --base models/chinese-decision-2b-gguf `
  --gguf models/chinese-decision-2b-gguf/chinese-decision-2b-Q8_0.gguf `
  --threads 8 --context-size 16384 `
  --host 127.0.0.1 --port 8790
```

两条决策路由为 POST /v1/systemone 和 POST /api/alpha/decisions；同时提供 /v1/batch、/v1/info 等接口。GET /health 应显示模型 chinese-decision-2b、backend gguf、device cpu、step 4994 和上述温度。服务读取候选字母的末位 logits，进行候选项 softmax，不生成文本。

新终端中验证运行中的服务：

```powershell
.venv\Scripts\python.exe scripts/verify_export.py --url http://127.0.0.1:8790
```

这个命令验证已有 HTTP 服务。GGUF 目录没有 safetensors，不使用 verify_export.py 的 --model 模式。

## 验证口径

<!-- CPU_RESULTS -->
本机：Windows、Intel Core i7-9700（8 核 / 8 线程）、32 GB RAM，推理线程 8，运行容量 16384 tokens，llama-cpp-python 0.3.36 CPU 版本。GPU 对此过程不可见。

| 检查 | 结果 |
|---|---|
| 诊断输入 / 重复次数 | 15 条 × 3 次 |
| 短输入定义 / 测量数 | 小于 100 tokens，12 条输入 / 36 次 |
| 短输入中位延迟 | 1.351 秒 |
| 最长诊断输入 | 852 tokens，中位 14.293 秒 |
| 进程峰值内存 | 2.701 GiB |
| 不同输入交错后重复概率最大差 | 0 |
| HTTP 三题型、两路由、批量接口 | 通过，单条/批量概率一致 |
| 无效输入 / 超长输入 | 返回 422；不静默截断 |
| 模型识别 | chinese-decision-2b，step 4994，backend gguf，device cpu |

诊断延迟包括预先分词后的单次前向和候选概率计算，不含分词、HTTP 或网络；每道题清空独立序列状态。15 条输入不是准确率或校准评测集，不代表完整业务请求的长度分布。实际临时 HTTP 服务检查结束后已关闭，不占用常驻端口。

原始记录为 reports/gguf-2b-20261017/local-cpu-q8.json、local-api.json 及对应日志。
<!-- /CPU_RESULTS -->

服务器质量对照仅使用这一次 2B step4994：

- Transformers BF16、BF16 GGUF、Q8_0 GGUF 在相同 1200 条分层测试输入上对照，用于观察后端转换与量化差异。
- 使用相同 25 条输入测量 2B 的 CPU FP32、CPU Q8_0、GPU BF16、GPU Q8_0 请求延迟和概率差异；CPU 测量为共享服务器环境。
- Q8_0 再覆盖 CJ-Bench 全部 307,900 条，与已完成的同一 2B 合并模型评测对照。两者使用同一温度、候选顺序和数据版本。

全量基线固定为 runs/cj-bench/qwen35-2b-rlcd-merged-calibrated-20261017-175001；本次任务固定为 runs/cj-bench/gguf-2b-q8-20261017，使用 GPU 0。其他训练和服务不作修改或中断。

<!-- QUALITY_RESULTS -->

### 全量 CJ-Bench 对照（CUDA）

Q8_0 在 CJ-Bench 全部 307,900 条上与同温度 2B BF16 交付版逐条配对（硬标签 304,822、软标签 3,078，无过滤、无截断，相同候选顺序与温度；源权重、GGUF 与数据指纹在评测后复核通过）。全量于 2026-10-17 21:34:34 完成，在 llama.cpp CUDA 上测得；CPU 只做了上文的本机诊断、HTTP 检查与下节 25 条输入对照，不能表述为 CPU 全量测试。

| 指标 | 2B BF16 / T=1.074872 | 2B Q8_0 / 同温度 | 变化 |
| --- | ---: | ---: | ---: |
| 整体 micro ACC (%) ↑ | 84.1107 | 84.1094 | −0.0013 pp |
| 四领域 macro ACC (%) ↑ | 77.1750 | 77.3282 | +0.1532 pp |
| ECE (15 bins, %) ↓ | 0.6361 | 0.6534 | +0.0173 pp |
| 分布 NLL ↓ | 0.416520 | 0.416288 | −0.000232 |
| Brier ↓ | 0.213295 | 0.213189 | −0.000106 |
| Score MAE ↓ | 0.564216 | 0.563738 | −0.000478 |
| 首选项一致率 | — | 99.00% | 1,105 条对→错，1,101 条错→对 |

整体几乎不变，但单条答案和概率会变；macro 的提高主要来自法律领域（4,300 条），不代表量化提升了能力。分领域、分题型、来源诊断与最大概率差见 [2B Q8_0 模型卡](MODEL_CARD_2B_GGUF.md#量化质量cj-bench-全量对照)。

### 1,200 条测试子集的后端对照

1,200 条分层 **CJ-Bench 测试子集** 对照已完成，硬标签 1,192 条、软标签 8 条；这不是训练选模验证集。三路使用同一温度和样本，均无过滤或截断。

| 2B 后端 | ACC (%) | ECE (15 bins, %) | 分布 NLL | Brier |
|---|---:|---:|---:|---:|
| 2B Transformers BF16 | 75.9228 | 2.5788 | 0.595550 | 0.329166 |
| 2B BF16 GGUF | 75.8389 | 2.7841 | 0.596385 | 0.329873 |
| 2B Q8_0 GGUF | 75.8389 | 2.4443 | 0.595557 | 0.329245 |

相对 Transformers BF16，Q8_0 的首选项一致率为 **98.75%**，平均候选概率绝对差为 **0.004257**，最大候选概率差为 **0.073770**；6 条硬标签由对变错、5 条由错变对。BF16 GGUF 与 Q8_0 的首选项一致率为 **99.08%**，整体准确率相同，但单条概率和答案仍有变化。该对照不能证明全量质量不变。

这组1200条结果用于识别后端转换也会改变输出；不代表全量的领域权重。端到端差异包括量化、后端和批次算术，不在测试标签上拟合温度或阈值。原始报告、成功条件、逐条对比及最大差样例见本机 reports/gguf-2b-20261017/analysis.md、comparison.json、summary.json 和 completion_verification.json。

<!-- /QUALITY_RESULTS -->

完整本机记录位于 reports/gguf-2b-20261017。训练与 BF16 导出来源见 [2B 训练与交付记录](TRAINING_2B.md)。

## 服务器 CPU / GPU 同组输入测量

25 条输入各预热后测 3 次，92–6521 tokens；其中短输入为 92–255 tokens。共享 Xeon Gold 5318Y / A800；耗时不含分词、HTTP、网络，内存是进程 RSS 观测值，不是 GPU 显存。

| 2B 后端 | 短输入 p50 (ms) | 全部输入 p50 (ms) | 全部 p95 (ms) | 进程内存观测最大值 (GiB) |
| --- | ---: | ---: | ---: | ---: |
| Transformers FP32 / CPU 16 threads | 545.29 | 601.34 | 11320.53 | 11.74 |
| Q8_0 / CPU 16 threads | 664.50 | 817.99 | 12884.58 | 3.34 |
| Transformers BF16 / GPU | 44.68 | 44.77 | 90.27 | 4.67 |
| Q8_0 / GPU | 20.39 | 23.60 | 274.74 | 3.02 |

Q8_0 在服务器 CPU 上减小内存，但 p50 比 FP32 更慢；不能将文件缩小等同于所有长度推理加速。完整模型卡与模型选型见 [2B Q8_0](MODEL_CARD_2B_GGUF.md)、[模型总览](MODELS.md)。

## 转换复现

转换使用与既有流程相同的固定 converter，SHA256 为 e9a1da876330bbce9687541ab31736542a01b4ac43c6686126514a50f122fb7f。服务器源包位于 local/gguf-runtime-0.3.36/llama-cpp-python-0.3.36/vendor/llama.cpp。--no-mtp 与源 checkpoint 的实际权重对应。

复现时使用新的输出目录：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python scripts/convert_gguf.py \
  --base models/chinese-decision-2b \
  --out models/chinese-decision-2b-gguf-reexport \
  --llama-cpp local/gguf-runtime-0.3.36/llama-cpp-python-0.3.36/vendor/llama.cpp \
  --formats bf16 q8_0
```

本次两个文件转换完成后，源码包中的失效 Git 指针曾令可选版本记录步骤失败。已将 Git 版本读取改为可选，保留 converter 的文件指纹；在检查文件头、体积、SHA256 和源权重指纹后补齐 conversion.json，未重写权重。原始日志和 conversion_recovery.json 保留该过程。
