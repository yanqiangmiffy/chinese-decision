# 决策模型扩展：文本分类与信息抽取（可行性实验）

日期：2026-10-23。分支：`feat/classification-extraction`。机器：156（单张 A800）。

## 结论

同一个 0.8B 决策模型可以同时承担文本分类和实体抽取，一起微调也不会拖累任何一项。

| 任务 | 测试集 | 结果 |
|---|---|---|
| 实体抽取 | CLUENER（官方 dev，1,343 句） | F1 **78.7**，实体 ECE 0.033 |
| 实体抽取 | 人民日报 NER（1,000 句） | F1 **89.1**，实体 ECE 0.017 |
| 新分类任务 | DuEE 事件大类，9 选 1（1,406 句） | 准确率 90.9% → **98.6%**，NLL 0.270 → 0.074 |

上表来自联合训练：CLUENER 只用 4,000 句，人民日报只用 3,000 句，训练 2 轮，耗时约 10 分钟。

还有两个问题：

1. **原有决策能力有遗忘，校准也被破坏。** 即使加入了回放数据，Ifeng 新闻分类准确率仍从 86.5% 降到 79.0%，NLL 从 0.35 升到 0.92。只训练分类的对照组同样下降，所以原因在于小数据微调的配方，而不是抽取任务本身。
2. **从未见过的实体类型抽不出来。** Resume 中的国籍、学历、专业、民族这几类 F1 都是 0。只有与训练类型相近的人名、组织机构、职称能抽出来。

## 设计

```text
                     ┌─ 最后位置的字母 logits ─→ 候选项分布（分类、原有决策）
共享骨干 + LoRA ──────┤
                     └─ 末层隐状态 ─→ 跨度头 ─→ 每个候选跨度在 [none, 类型1..M] 上的分布（抽取）
```

- **分类**直接复用现有决策格式：状态、问题、候选项，读出字母分布，无需新参数。
- **抽取**沿用 CalOn 的类型化决策：对每个候选跨度，在“不是实体”和各类型之间做一次 softmax。跨度向量由首 token、尾 token、下一个 token、均值和宽度组成；类型向量从输入中的“类型名：描述”池化得到。代码见 [decision/modeling.py](../decision/modeling.py)（`SpanHead`）与 [decision/processing.py](../decision/processing.py)（编码与解码）。
- **回声布局。** CalOn 用双向编码器，而 Qwen3.5 是因果模型且包含线性注意力层。因此文本先按自然分词读一遍，再逐字重复一遍，跨度从重复的那份上取。

  这样做有两个好处。其一，自然分词会切开 9–14% 的实体边界（如“是三”“在西”“集团的”），逐字副本让每个字符边界都成为 token 边界。其二，重复副本中的每个位置都已看过整句，相当于给因果模型补上了右侧上下文。
- **损失。** 两个头都使用 [decision/rlcd_loss.py](../decision/rlcd_loss.py) 的 `RLCDLoss`，本实验只开 CE。none 跨度的权重乘以 0.2；推理时给 none 的 logit 加上 log(1/0.2)，把先验还原回去。

## 数据

由 [scripts/build_multitask_data.py](../scripts/build_multitask_data.py) 构建，随机种子 42。

| 用途 | 来源 | 训练 / 开发 / 测试 |
|---|---|---|
| 抽取 | CLUENER2020（10 类细粒度） | 4,000 / 300 / 1,343（官方 dev 作测试；官方 test 无标签） |
| 抽取 | 人民日报 NER（人名 / 地名 / 机构名） | 3,000 / 300 / 1,000 |
| 抽取，零样本 | Resume NER（8 类，其中 5 类未在训练中出现） | 仅测试，477 |
| 分类，新任务 | DuEE 句子，限单一事件大类，9 选 1 | 3,000 / 300 / 1,406 |
| 分类，回放 / 保持 | CJ-Dataset：Ifeng 新闻、ChnSentiCorp、MASSIVE 意图 | 每源回放 600；每源测试最多 400 |
| 保持 | CJ general 验证集中的其他来源 | 仅测试，600 |

## 设置

- **初始化：** `models/chinese-decision-0.8b`（已合并的决策模型），在其上新加 LoRA r=32。
- **学习率：** LoRA 1e-4，跨度头 5e-4，cosine 调度。
- **批次与轮数：** 每轮把两种任务的 batch 按数据量混合后打乱，共训练 2 轮；联合训练 2,352 步。
- **批大小：** 决策 16，抽取 8。
- **跨度：** 最大宽度 20 个字。
- **对照组：** 只训分类（`--tasks decision`）、只训抽取（`--tasks extraction`）、去掉回声（`--no-echo`），以及未经微调的基座模型（`--eval-only`）。

## 结果

### 实体抽取（测试集，精确匹配“起止位置 + 类型”）

| 配置 | CLUENER P / R / F1 | 人民日报 P / R / F1 | 实体 ECE（CLUENER / 人民日报） |
|---|---|---|---|
| 联合训练 | 82.1 / 75.6 / **78.7** | 92.7 / 85.7 / **89.1** | 0.033 / 0.017 |
| 只训抽取 | 80.0 / 78.3 / **79.1** | 90.8 / 86.6 / 88.6 | 0.055 / 0.022 |
| 只训抽取，无回声 | 76.9 / 66.6 / 71.4 | 90.4 / 77.4 / 83.4 | 0.023 / 0.074 |

- **联合 vs 单任务：** 两者相差在 ±0.5 F1 以内，加入分类没有拖累抽取。
- **回声布局：** 带来 CLUENER +7.7、人民日报 +5.2 F1，提升主要来自召回。
- **无法表示的实体：** CLUENER 有 12 个 gold 实体无法表示（跨度超过 20 字，或边界落在英文单词内部）。
- **CLUENER 各类 F1（联合）：** 人名 87、游戏 85、政府 83、书名 81、影视 80、公司 79、职位 79、组织 76、景点 71、地址 64。
- **人民日报各类 F1（联合）：** 人名 95、地名 90、机构名 82。

### 零样本：Resume（训练中没有这套类型）

| 配置 | P / R / F1 | 人名 | 组织机构 | 职称 | 国籍 / 学历 / 专业 / 民族 / 地点 |
|---|---|---|---|---|---|
| 联合训练 | 79.4 / 61.9 / 69.6 | 100 | 78 | 69 | 全部为 0 |
| 只训抽取 | 82.7 / 64.2 / 72.3 | 100 | 81 | 71 | 全部为 0 |
| 无回声 | 79.2 / 38.6 / 51.9 | 97 | 64 | 42 | 全部为 0 |

总体约 70 的 F1 全部来自与训练类型相近的人名、组织机构、职位/职称。训练类型只有 13 个，跨度头学到的是针对这 13 类的检测器，而不是“读取类型描述再去匹配”。CalOn 的零样本能力来自数千种类型名的训练数据。

### 分类与原有能力保持（准确率 % / NLL / ECE）

| 配置 | DuEE 事件大类（新） | ChnSentiCorp | Ifeng 新闻 | MASSIVE | 其他 CJ 决策 |
|---|---|---|---|---|---|
| 基座（未微调） | 90.9 / 0.270 / 0.022 | 93.8 / 0.246 / 0.060 | 86.5 / 0.346 / 0.023 | 97.8 / 0.083 / 0.024 | 72.4 / 0.691 / 0.063 |
| 联合训练 | **98.6 / 0.074 / 0.011** | 92.0 / 0.339 / 0.065 | 79.0 / 0.917 / 0.173 | 97.0 / 0.092 / 0.015 | 69.7 / 0.923 / 0.139 |
| 只训分类 | 98.0 / 0.099 / 0.016 | 92.9 / 0.433 / 0.066 | 82.5 / 0.854 / 0.148 | 97.8 / 0.132 / 0.022 | 71.2 / 0.921 / 0.135 |
| 只训抽取 | 88.9 / 0.451 / 0.136 | 92.9 / 0.251 / 0.049 | 82.5 / 0.428 / 0.073 | 97.2 / 0.156 / 0.090 | 71.7 / 0.706 / 0.051 |

- **新分类任务：** 决策模型零样本就有 90.9%；微调后达到 98.6%，NLL 和 ECE 同时下降。联合训练略好于只训分类。
- **遗忘：** 主要出现在含决策训练的两组。Ifeng 和其他 CJ 决策的 NLL 从 0.35–0.69 升到 0.85–0.92，ECE 升到 0.14–0.17，说明模型变得过度自信。
  - 原始训练用 1e-4 的学习率配合 1,024 的 batch，这里是 1e-4 配合 16，单样本的更新强度高得多；600 条回放也挡不住这种过拟合。
  - 只训抽取对旧决策的损伤较小，但 DuEE 准确率和 MASSIVE 的校准也有所下降。
- **样本量提醒：** 其他 CJ 决策 n=600、Ifeng n=400，准确率的标准误约 2 个百分点；NLL 的变化远大于噪声。

## 下一步

1. **控制遗忘：**
   - 降低学习率（2e-5）或增大 batch；
   - 回放改为从全部 CJ 来源均匀抽样；
   - 在回放样本上加 KL 锚定，参考分布用 PEFT 的 `disable_adapter()` 直接取得，无需第二份模型；
   - 训练后重新拟合温度。
2. **零样本类型：** 加入多类型数据（ie_collections 里的 B2NERD 有数百种中文实体类型），并在训练中打乱、替换或匿名化类型名，迫使跨度头读取类型描述。
3. **扩大规模：** 使用 CLUENER 和人民日报全量数据，并加入 MSRA、Weibo；再接入关系抽取（DuIE）与事件论元（DuEE）。CalOn 的关系头结构可以沿用。
4. **部署：** 跨度头不在 GGUF 中，抽取目前只能走 PyTorch（`DecisionPipeline`）。

## 意图 + 字段抽取（槽位填充）

目标格式：`{"query": "Q: 打开空调", "response": {"device_control": {"device": "空调", "function": "打开"}}}`。决策头先在 6 个意图中选一个，跨度头再按该意图的字段抽取。代码见 [decision/slots.py](../decision/slots.py)。

**Schema**（[examples/slot_filling/schema.json](../examples/slot_filling/schema.json)）

| 意图 | 字段 |
|---|---|
| device_control | device、function、location、attribute、value |
| weather_query | city、date |
| music_play | singer、song |
| navigation | destination |
| alarm_set | time、event |
| chitchat | 无字段 |

标注规范：字段值必须是原句中的逐字片段；function 只标动词及其补语，不含“一下”“了”。

**数据**

- **训练：** 用 156 上的 Qwen3-32B 合成 3,600 条，脚本见 [scripts/synthesize_slot_data.py](../scripts/synthesize_slot_data.py)。共调用 272 次，丢弃对不齐的 319 条、使用 schema 外字段的 305 条、重复的 136 条。
- **测试：** 79 句、170 个字段，见 [examples/slot_filling/test_handwritten.jsonl](../examples/slot_filling/test_handwritten.jsonl)。这些句子由 Claude 逐句手写，不经过 Qwen 生成，但也不是真实用户日志。
- **训练集划分：** 打乱后先留出 200 条作开发集，再取前 N 条（100 / 500 / 2000，嵌套）加入上文的联合训练数据。其余配置与 `mt-joint` 完全相同。

**端到端结果**（79 句）

| 合成数据量 | 意图准确率 | 字段 P / R / F1 | 整句完全正确 |
|---|---|---|---|
| 0（即 `mt-joint`） | 98.7% | 100 / 8.2 / 15.2 | 13.9% |
| 100 | 100% | 94.8 / 64.7 / 76.9 | 40.5% |
| 500 | 100% | 98.2 / 94.7 / **96.4** | **89.9%** |
| 2000 | 100% | 95.9 / 95.3 / 95.6 | 88.6% |

- **样例：** “打开空调”在 N ≥ 100 时都输出 `{"device_control": {"function": "打开", "device": "空调"}}`。N=500 时两个字段的概率分别为 0.9995 和 0.995。
- **数据量：** 500 条已经饱和，2000 条没有进一步提升（测试集小，1 句约等于 1.3 个百分点）。
- **原有任务不受影响：** CLUENER F1 78.1–78.7，人民日报 87.1–89.4，DuEE 98.0–98.4%；CJ 保持集的结果与 N=0 处在同一水平。

**剩余错误几乎都在字段边界上**

- `关了` 被抽成 function，而规范要求 `关`；同类还有 `加热` vs `加热到`、`开始` vs `开始清扫`，以及“导航回家”中的 `家` 没有抽出来。
- 原因不是合成标注违反规范（1,789 个 function 标签中只有 3 个以 “了” 结尾），而是 Qwen 很少写出“把X关了”“开一下”这类口语说法（3,600 条中只有 11 条）。
- event 标注不一致：合成数据里 19% 的 event 带“叫我 / 提醒我”前缀，导致预测出“叫我起床”而不是“起床”。

所以继续加同一来源的数据帮助不大，更有效的做法有三种：

- 在合成提示中加入口语化的短动词句式；
- 对合成标签做规范化，例如去掉 event 的“提醒我”前缀；
- 补几十条真实用户话语的人工标注，并用真实日志构造测试集。

```bash
# 156：合成（约 15 分钟）→ 构建混合数据 → 训练 → 端到端评测
python scripts/synthesize_slot_data.py --out datasets/slots/pool.jsonl --exclude examples/slot_filling/test_handwritten.jsonl
python scripts/build_slot_mix.py --n 500 --out datasets/multitask-slots-500
python scripts/train_multitask.py --data datasets/multitask-slots-500 --out runs/mt-slots-500
python scripts/evaluate_slots.py --run runs/mt-slots-500
```

```python
from decision import DecisionPipeline, SlotFillingPipeline
from decision.data.slots import load_schema

filler = SlotFillingPipeline(DecisionPipeline.from_run("models/chinese-decision-0.8b", "runs/mt-slots-500"),
                             load_schema("examples/slot_filling/schema.json"))
result = filler("打开空调")
{"query": "Q: 打开空调", "response": {result["intent"]: result["slots"]}}
```

## 复现

```bash
# 本地：构建数据（需要 pandas；扫描 CJ 训练集约 5 分钟）
python scripts/build_multitask_data.py --out datasets/multitask-mini

# 156：PYTHONPATH 必须指向本目录，否则会导入旧项目以 editable 方式安装的 decision 包
export PYTHONPATH=$PWD
CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/train_multitask.py --out runs/mt-joint
CUDA_VISIBLE_DEVICES=2 .venv/bin/python scripts/train_multitask.py --out runs/mt-base --eval-only
.venv/bin/python scripts/evaluate_extraction.py --run runs/mt-joint \
    --files datasets/multitask-mini/extraction/resume_zero_shot.test.jsonl
```

推理：

```python
from decision import DecisionPipeline

pipe = DecisionPipeline.from_run("models/chinese-decision-0.8b", "runs/mt-joint")
pipe.extract("华为公司董事长梁华在深圳会见了腾讯游戏负责人。",
                  {"人名": "人物的姓名", "公司": "企业名称", "地址": "具体地点", "职位": "职务与头衔"})
# [{"text": "华为公司", "label": "公司", "start": 0, "end": 4, "score": 0.96}, ...]
```

156 上的实验目录：`/data/users/yanqiang/chinese-decision-mt`，包含 `runs/mt-*`（adapter、`span_head.pt`、各项指标）。本地只保存了指标副本，位于 `runs/mt-156/`。
