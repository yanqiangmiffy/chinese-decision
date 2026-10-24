# 统一模型：一份权重同时做分类和抽取，并修复联合训练的遗忘

日期：2026-10-24。分支：`feat/classification-extraction`。

代码位置：

- 单文件模型：[decision/modeling.py](../decision/modeling.py)（`DecisionModel.from_pretrained` / `save_pretrained`）、[decision/pipeline.py](../decision/pipeline.py)（`DecisionPipeline`）
- 训练：[scripts/train_multitask_ddp.py](../scripts/train_multitask_ddp.py)
- 回放与独立验证集：[scripts/build_replay_data.py](../scripts/build_replay_data.py)

## 三个问题的结论

### 1. 抽取和分类能否放在同一个模型结构里

能。而且现在两者已经共享了几乎全部参数：同一个底座，同一个 LoRA。两个任务只在最后的读出层不同。

```text
                         ┌─ 最后位置的字母 logits（LM head）─→ 候选项分布（分类，原有决策）
底座 + LoRA（训练后合并）──┤
                         └─ 末层隐状态 ─→ 跨度头（约 3M 参数）─→ 每个候选跨度在 [none, 类型…] 上的分布（抽取）
```

**真正独立的部分**只有跨度头。它是一个很小的模块：两个 MLP、一个 none 向量和一个宽度嵌入，0.8B 上约 3M 参数，不到总量的 0.4%。

**之前出现两份文件的原因**是训练产物没有合并：`adapter/` 保存 LoRA，`span_head.pt` 保存跨度头，加载时还需要另外提供底座。

**改造后的发布形态**是一个标准的模型目录：

```text
model/
  model.safetensors        底座（LoRA 已合并）+ span_head.* 张量，同一份权重；大模型自动分片，仍是一份权重
  decision_config.json     temperature、prompt_version、max_length，以及 extraction: {head, none_shift, echo}
  tokenizer 文件、training_summary.json
```

**两种加载方式：**

- `AutoModelForCausalLM.from_pretrained(model/)` 照常可用于分类。`span_head.*` 会被报告为未使用的权重，不影响结果。
- 本项目的 `DecisionPipeline.from_pretrained(model/)` 只加载一次，就能同时提供两种读出。

**已验证的往返测试**（`tests/test_unified.py`）：

- 合并、保存、重新加载后，logits 与带 adapter 时一致；
- 跨度头的参数逐项一致；
- 目录里只有一个 safetensors 文件。

**另一种可选方案**是完全不加新参数：在逐字回显的每个位置读取 LM head 的字母 logits，做 BIO 标注（大写字母表示某类型的开头，小写表示内部）。

- 好处：模型就是一个纯标准的 HF 模型，转成 GGUF 后也能做抽取。
- 代价：无法表示嵌套实体和多标签跨度（CMeEE 等需要），而且抽取效果要从头验证。

这个方案可以作为后续实验，这次没有采用。

### 2. 不想要两份权重

训练过程中的 checkpoint 仍然拆成 adapter 和跨度头两部分，这样续训时可以恢复优化器状态。这些只是内部文件。

训练结束时，`train_multitask_ddp.py` 会做以下几步：

1. 把选中的 checkpoint 合并，导出成上面那样的单个目录 `--out/model/`；
2. 温度在独立的校准集上拟合，写入 `decision_config.json`；
3. 导出前检查合并前后的决策是否一致。

**对外只发布 `model/` 一个目录。**

### 3. 预测时怎么知道该输出分类还是抽取

由请求决定，而不是由预测结果决定。

两种任务的输入本身就不同：

| 任务 | 请求里有什么 | 拼成的提示 | 读出 |
|---|---|---|---|
| 分类 | `state` + `questions`（每题带候选项） | “待判断的上下文 / 问题 / 候选项 / 只返回字母” | 最后位置的候选字母分布 |
| 抽取 | `text` + `types`（类型名和说明） | “从文本中找出下列类型的实体 / 类型列表 / 文本 / 逐字定位” | 回显文本上每个跨度的类型分布 |

任何输入其实都能同时算出两个读出的数值，但不属于该请求的那个读出没有意义，也不能拿两者的置信度做比较。所以“看哪个输出更可信就用哪个”行不通。

`DecisionPipeline` 按请求字段路由（`pipe(request)`）：

- 带 `questions`（或 `question`）的请求走分类；
- 带 `types` 的请求走抽取；
- 两者都带，或两者都没有，会直接报错。

```python
from decision import DecisionPipeline

pipe = DecisionPipeline.from_pretrained("runs/cj-ie-0.8b-v2/model")
pipe({"state": "会议改到明天下午三点。",
               "questions": {"q": {"type": "choice", "instructions": "这条消息通知了什么变化？",
                                   "criteria": {"A": "会议时间", "B": "会议地点", "C": "参会人员"}}}})
# {"task": "decision", "answers": {"q": {"A": 0.97, "B": 0.02, "C": 0.01}}}
pipe({"text": "华为公司董事长梁华在深圳会见了腾讯游戏负责人。",
               "types": {"人名": "人物的姓名", "公司": "企业名称", "地址": "具体地点"}})
# {"task": "extraction", "entities": [{"text": "华为公司", "label": "公司", "start": 0, "end": 4, "score": ...}, ...]}
```

**“根据预测来选输出”真正适用的是级联：**先用分类得到结果，再据此决定抽取什么。

- **意图 + 槽位**（`SlotFillingPipeline`）：先判断意图，再只抽这个意图的槽位。
- **事件**：先判断句子里有哪些事件类型（`ee_type`），再对每个类型抽取触发词和论元（`ee_argument`）。

如果入口是一句自然语言，而不是结构化请求，可以先加一道“任务路由”选择题，例如“这个请求要分类、抽取还是别的”。这道题本身也由同一个模型的分类读出回答，仍然是同一份权重。

## 遗忘：发生了什么

0.8B 联合训练（CJ-IE，27,514 步，1 epoch，只用 CE，没有回放）在全量 CJ-Bench（307,900 条，T=1）上的结果：

| | 原模型 | 联合训练后 | 变化 |
|---|---:|---:|---:|
| 全部 | 83.11 | 71.02 | −12.09 |
| 通用 | 72.26 | 59.59 | −12.67 |
| 医疗 | 88.85 | 77.53 | −11.32 |
| 法律 | 67.84 | 39.98 | −27.86 |
| 金融 | 74.00 | 53.81 | −20.19 |

校准也明显变差：hard NLL 从 0.43 升到 0.73，法律的 ECE 从 0.057 升到 0.243。

4B 的情况类似。到 checkpoint-15000 时，QUICK 旧能力（1,626 条）从 86.21 降到 78.20，所以已在 21,100 步时停止。

原因有四个，可以同时成立：

1. **没有回放。** 训练数据只有 CJ-IE。旧任务只用来评估，不参与训练。
2. **单条样本的更新强度高了一个数量级以上。** 原模型训练时用 lr 1e-4，全局 batch 1,024 条决策。联合训练同样是 lr 1e-4，但每步只有 4 个 micro-batch，约 30–60 条。
3. **抽取的梯度也会改动共享参数。** 跨度损失回传到共享 LoRA，会改变分类依赖的隐状态。
4. **选 checkpoint 时没有旧能力指标。** 只看新任务的 dev，一路训练到最后。

## 这次的改法

| 机制 | 实现 | 默认值 |
|---|---|---|
| 旧任务回放 | `build_replay_data.py` 从 CJ-Dataset 训练集抽样。每个领域有固定预算，领域内各来源按样本数的平方根分配，避免大来源淹没小来源 | 通用 20 万、医疗 10 万、法律和金融全量（各约 2.4 万，正是掉得最多的两个领域） |
| 向原模型蒸馏 | 回放批次在 CE/RLCD 之外再加 `KL(原模型 ‖ 当前模型)`。原模型就是关闭 LoRA 的同一个模型（`disable_adapter()`），不需要加载第二份 | `--kl-weight 1.0` |
| RLCD | 两个头默认都开 | `--no-rlcd` 可以关闭 |
| 降低单步强度 | 梯度累积：每次更新平均 4 卡 × 8 个 micro-batch，约 400 条样本。LoRA 梯度在全部 micro-batch 上平均，跨度头的梯度只在抽取 micro-batch 上平均 | `--accumulate 8` |
| 独立验证 | 选模型和拟合温度都只用 CJ-Dataset 的验证集，从不读 CJ-Bench：哈希分区 1–4 按（领域，题型）分层，每层 300 条，作为 `cj_dev`；分区 0 每层 200 条，用于导出时拟合温度 | |
| 选择 checkpoint | 训练前先在 step 0 评一次，作为基线；之后每 500 步保存并评估一次。在 `cj_dev` 准确率下降不超过 1 个百分点的 checkpoint 中，选新任务分数最高的（新任务决策准确率与抽取宏平均 F1 的均值）。如果没有满足条件的，就选下降最小的 | `--max-retention-drop 1.0` |
| 单文件导出 | 合并选中的 checkpoint，写出 `model/` | `--export model` |

**CJ-Bench 只用于最终的回归评测**，训练和选模型过程中都不读它。

**为什么这次不用“分类和抽取的适配器隔离”。** 隔离能保证分类完全不变，但它的代价有三点：

- 实际上就是两套权重，推理时还要按任务切换 adapter，和“一份权重”的要求冲突；
- CJ-IE 里 61 万条新的抽取类决策（关系判断、实体类型、事件类型）也走分类读出。隔离后这些任务要么学不到，要么又回到共享参数上；
- 在单模型里用回放和 KL 约束，同样能控制遗忘。

如果这次训练后 `cj_dev` 仍然掉得多，退路是：冻结底座，只训练跨度头，必要时在跨度头里加几层小网络。这样分类结果与原模型逐位一致，抽取能力会弱一些，但仍然只有一份权重。

## 复现

```bash
# 在 156 上：回放和独立验证集（读取 CJ-mixed 与 CJ-Dataset validation）
python scripts/build_replay_data.py --out datasets/cj-replay
# 训练（4 卡；这台机器上 4 卡必须设置 NCCL_P2P_DISABLE=1）
bash runs/cj-ie-0.8b-v2/launch.sh
```
