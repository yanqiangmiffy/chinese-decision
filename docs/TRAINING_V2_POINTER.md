# 第二版训练：指针读出 + 双向提示 + 多题一条序列（2026-11-03）

本文对应"直接从第二期开始"的决定：从 Qwen3.5 基座重新训练，第一步就用最终结构，不再做只换读出层的中间版本。设计依据见 [候选读出调研](CANDIDATE_READOUT_SURVEY.md)，数据来源见 [训练数据调研](DECISION_INDEX_TRAINING_DATA.md)。

## 1. 模型结构

| 部分 | 做法 | 位置 |
|---|---|---|
| 读出 | **指针头**：每个候选行末尾一个候选标记 `<|object_ref_end|>`，每道题末尾一个决策标记 `<|object_ref_start|>`。<br>`logit_k = <W_c·LN(h_k), W_q·LN(h_decide)>/√256 + <w, W_c·LN(h_k)>`。<br>候选数只受上下文长度限制（上限常量 4096），不再有 26 个字母的限制。 | `decision/modeling.py: PointerHead` |
| 注意力 | **提示双向**：全注意力层（Qwen3.5 每 4 层 1 层）里，从"待判断的上下文"到最后一个决策标记之间的 token 互相可见；之后的部分、以及线性注意力（Gated DeltaNet）层保持因果。<br>做法是给 `Qwen3_5TextModel` 传入按层类型区分的 mask 字典，不改 transformers。 | `decision/modeling.py: prompt_masks` |
| 多题 | **一条序列答一个请求的所有问题**：状态只写一次，后面依次是每道题的说明、候选和决策标记。一个请求不论几道题，都只算一次前向。 | `DecisionProcessor.pointer_ids / encode_request` |
| 训练时打乱 | 题目顺序打乱；choice 和 noul 的候选顺序打乱；score 档位保持顺序。 | `encode_request` |
| 损失 | 每道题一行，沿用现有交叉熵 + RLCD（score 题用有序 RPS）。 | `scripts/train.py` |

提示格式（`prompt_version = pointer-joint-zh-v1`）：

```text
<|im_start|>user
待判断的上下文：{state}            ← 双向区间从这里开始
问题1：{instructions}
候选项：
A: key — 描述<|object_ref_end|>
...（超过 26 个接 AA、AB…）
选出最符合的候选项。<|object_ref_start|>
问题2：... <|object_ref_start|>     ← 双向区间到最后一个决策标记为止
<|im_end|>
<|im_start|>assistant ...
```

标记位置在拼接 token 时直接记录，不靠在文本里搜索；用户文本中如果出现这两个标记字符串，会被删掉。

## 2. 已做的验证

| 验证 | 结果 |
|---|---|
| 单元测试（CPU，tiny Qwen3.5） | 164 项全部通过。包括：<br>• 26 个以上候选；<br>• 多题一行一题且不受 padding 影响；<br>• **双向 mask 确实让前面的状态 token 读到后面候选的内容，因果模式下读不到**；<br>• 存盘、重载、续训 checkpoint；<br>• 引擎一次读完 77 个选项。 |
| 端到端（CPU，真实 tokenizer） | 混合数据 → 多题请求 → batch → 损失和梯度都有限值；含 151 个候选的题能正常训练。 |
| 156 双卡 0.8B 冒烟 | DDP + bf16 + gradient checkpointing + RLCD 跑完 20 步，验证准确率 19% → 44%，峰值显存 7.3 GB。<br>中途 kill 后用 `--resume` 从第 20 步续到第 40 步，准确率 58%。 |
| 导出 | `finalize.py` 合并 LoRA + 指针头 → 拟合温度 → 重载核对 → 用 `serve.py` 走 HTTP 校验，全部通过（choice / noul / score 都能答）。 |
| 榜单引擎 | 150 条分层样本全部成功；联合读法 + 反序去偏，0.8B 用时 15.9 秒。 |

冒烟中顺带修掉的问题：
- `train.py` 用 `--rlcd` 时，损失结果覆盖了输出目录变量 `out`，第一次存 checkpoint 就会报错。这是旧代码里已有的 bug。
- 温度拟合原来假设最多 26 个候选。
- 多题数据的 manifest 在 Windows 上写成了反斜杠路径。
- 导出的 prompt_version 写错。

## 3. 训练数据（decision-mix-v1）

- 由 `scripts/build_decision_mix.py` 从 `G:/pretrained_models/datasets/decision-training-catalog` 生成。63 个来源，分 10 个领域：typed、evidence、risk、intent、nli、commonsense、tools、retrieval、preference。
- **只用 train 划分**。每条都用训练读取器校验过，并与 Decision Index 0.3 的 140,620 条请求做 13 个单位（中文按单字）的重叠去重；出现在 20 行以上的模板片段视为通用措辞，不算泄漏。
- 每个来源有上限，按固定种子抽样；同一状态的问题合并成一行（最多 8 题）。验证集按组哈希另抽约 3%，每个来源最多 300 题，同一组不会同时出现在训练集和验证集。
- `build_report.json` 逐来源记录读取、去重删除、保留的训练和验证题数。
- 标签只用数据集自带的人工或程序标签，以及标注者分布（UNLI、Measuring Hate Speech）。合成决策集的软标签只取非 Jev 教师的平均。**不使用任何 Jev 输出**。
- 措辞与榜单任务对齐：RAGTruth、HoVer、ESCI、When2Call、Humicroedit、GSM8K 的问题和候选写法与官方构建脚本一致，但不含任何测试样本。

## 4. 训练命令（4 卡，156）

```bash
cd /data/users/yanqiang/chinese-decision-v2
export PYTHONPATH=$PWD NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=2,3,4,7
/data/users/yanqiang/chinese-decision/.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 scripts/train.py \
  --base models/Qwen3.5-4B --out runs/pointer-4b-v1 --readout pointer --rlcd \
  --data datasets/decision-mix-v1/raw datasets/CJ-Dataset \
  --prepared-data datasets/decision-mix-v1/prepared datasets/CJ-mixed \
  --domains typed evidence risk intent nli commonsense tools retrieval preference general medical legal finance \
  --domain-fraction general:choice=0.105 general:noul=0.105 general:score=0.03 medical=0.1 \
  --global-batch-size 1024 --token-budget 262144 --lr 1e-4 --head-lr 1e-3 --lora-r 64 --epochs 1
```

**CJ 的比例（按题型抽样）**：
- 通用：选择题、是/否题各约 35 万，打分题约 10 万（榜单没有打分题，只保留这项能力）；
- 医疗：约 30 万；
- 法律、金融：全取。

合计约 115 万，与 100 万题的混合数据约 1:1。混合数据 decision-mix-v1 的终版为 1,000,815 题（947,493 行），已修复去重、幻觉标签平衡和钓鱼邮件捷径三个问题，见 [数据审阅](DECISION_MIX_V1_REVIEW.md)。

**导出**：

```bash
python scripts/finalize.py --run runs/pointer-4b-v1 --out models/chinese-decision-pointer-4b
```

**评测**：Decision Index 用 `--option debias="reverse"`。这一版训练时已经打乱候选顺序，可以对比加与不加去偏，看是否还需要。

## 5. 已知限制

- GGUF / llama.cpp 读不了自定义指针头，这一版不出 GGUF。
- `serve_gpu.py` 和 `serve_unified.py` 仍是字母读出。指针模型的 HTTP 服务目前用 `scripts/serve.py`（逐题读）或 `DecisionPipeline`。
- 双向 mask 是稠密的 `[B,1,T,T]`。现有训练在有 padding 时本来也会生成同样大小的 mask，所以显存不比原来多；但不能走 flash 内核，长序列会慢一些。
