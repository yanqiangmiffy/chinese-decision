---
license: apache-2.0
language:
- zh
base_model:
- Qwen/Qwen3.5-4B
base_model_relation: finetune
library_name: transformers
pipeline_tag: zero-shot-classification
datasets:
- bbldCVer-hf/CJ-Dataset
tags:
- decision
- classification
- information-extraction
- named-entity-recognition
- relation-extraction
- event-extraction
- chinese
- qwen3.5
frameworks:
- Pytorch
tasks:
- zero-shot-classification
- named-entity-recognition
---

<div align="center">

# chinese-decision-unified-4b

**一份权重，两种读出：中文概率决策 + 类型化信息抽取（4B 版）**

[GitHub 代码](https://github.com/yanqiangmiffy/chinese-decision) ·
[测试平台 Decision Studio](https://github.com/yanqiangmiffy/chinese-decision-studio) ·
[chinese-decision-unified-0.8b](https://huggingface.co/quincyqiang/chinese-decision-unified-0.8b) ·
[chinese-decision-unified-2b](https://huggingface.co/quincyqiang/chinese-decision-unified-2b) ·
[基础模型 Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) ·
[训练数据 CJ-Dataset](https://huggingface.co/datasets/bbldCVer-hf/CJ-Dataset) ·
[评测集 CJ-Bench](https://huggingface.co/datasets/bbldCVer-hf/CJ-Bench)

</div>

> The 4B member of the unified Chinese Decision family: one weight file, two readouts. Given a context and candidates it returns a probability per candidate (`noul` / `choice` / `score` decisions); given a text and entity types (name + description) it returns typed character spans with positions and probabilities; one request may carry both. Trained from raw Qwen3.5-4B with exactly the recipe of the [chinese-decision-unified-0.8b](https://huggingface.co/quincyqiang/chinese-decision-unified-0.8b) and [chinese-decision-unified-2b](https://huggingface.co/quincyqiang/chinese-decision-unified-2b). On the full CJ-Bench (307,900 items, T = 1) it reaches 86.86% (chinese-decision-unified-0.8b: 83.46%, chinese-decision-unified-2b: 85.00%, chinese-decision-4b: 85.97%); on the CJ-IE test set its span-extraction micro F1 is 68.9 (chinese-decision-unified-2b: 66.0, chinese-decision-unified-0.8b: 63.1).

![统一模型：共享的 Qwen3.5 骨干，决策读出（候选字母 logits）和抽取读出（类型化跨度头）](../assets/chinese-decision-unified-acl.png)

## 模型系列

这是统一模型系列的 **4B 版本**，和 **0.8B**、**chinese-decision-unified-2b**用同一套训练配方、同一种模型结构、同一套 CJ-IE 数据，只有底座大小不同。想了解模型结构、请求怎么选择任务、类型怎么写，0.8B 的模型卡写得更详细：[Hugging Face](https://huggingface.co/quincyqiang/chinese-decision-unified-0.8b) · [ModelScope](https://www.modelscope.cn/models/quincyqiang/chinese-decision-unified-0.8b)。

| | chinese-decision-unified-0.8b | chinese-decision-unified-2b | **chinese-decision-unified-4b（本模型）** |
|---|---|---|---|
| 仓库 | [HF](https://huggingface.co/quincyqiang/chinese-decision-unified-0.8b) · [ModelScope](https://www.modelscope.cn/models/quincyqiang/chinese-decision-unified-0.8b) | [HF](https://huggingface.co/quincyqiang/chinese-decision-unified-2b) · [ModelScope](https://www.modelscope.cn/models/quincyqiang/chinese-decision-unified-2b) | [HF](https://huggingface.co/quincyqiang/chinese-decision-unified-4b) · [ModelScope](https://www.modelscope.cn/models/quincyqiang/chinese-decision-unified-4b) |
| 底座 | Qwen3.5-0.8B | Qwen3.5-2B | Qwen3.5-4B |
| 权重文件 | 单个 `model.safetensors`，约 1.5 GB | 单个 `model.safetensors`，约 3.8 GB | 单个 `model.safetensors`，约 8.4 GB |
| 跨度头 | 约 3.4M 参数 | 约 6.6M 参数 | 约 8.2M 参数 |
| CJ-Bench 全量准确率（T = 1） | 83.46% | 85.00% | **86.86%** |
| CJ-IE 测试集决策准确率 | 90.65% | 91.30% | **91.89%** |
| CJ-IE 测试集抽取 micro F1 | 63.1 | 66.0 | **68.9** |
| 训练耗时（4 × A800） | 约 17 小时 | 约 26 小时 | 约 83 小时 |

只做决策、不需要抽取时，还有纯决策版：[chinese-decision-0.8b](MODEL_CARD_0_8B.md) · [chinese-decision-2b](MODEL_CARD_2B.md) · [chinese-decision-4b](MODEL_CARD_4B.md)。**统一 4B 的决策准确率反而高于chinese-decision-4b**（86.86% 对 85.97%），所以需要 4B 精度时，直接用统一版即可，决策和抽取一份权重都有。怎么选：对延迟和显存敏感用 chinese-decision-unified-0.8b；想要更高的精度又不想付 4B 的成本用 chinese-decision-unified-2b；追求最高的决策和抽取精度、尤其是法律和金融用本模型。

## 简介

统一模型在同一份权重里提供两种读出，由请求的形状决定走哪一种：

- 请求带 `questions` → **决策**：读最后位置上候选字母 `A`–`Z` 的 logits，softmax 后即为概率。
- 请求带 `types` → **抽取**：把原文逐字回显一遍，在每个字符位置上给跨度分类，合成带位置和概率的实体。

两者共用同一个骨干和同一组 LoRA 参数（训练后合并），只有最后的读出层不同；都是一次前向，**不生成文本**。事件抽取和关系抽取是在这两种读出之上的组合：先用是/否问题判断事件类型、再抽触发词和论元；给定主体、抽客体。模型结构见上方的架构图和[统一模型设计说明](UNIFIED_MODEL.md)。

## 模型信息

| 项目 | 内容 |
|---|---|
| 基础模型 | [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)（文本主干） |
| 参数量 | 约 4.21B 骨干（含词表嵌入）+ 约 8.2M 跨度头，BF16（跨度头为 FP32），单个 `model.safetensors`（8.44 GB） |
| 微调方式 | LoRA（r = 32）训练后合并；跨度头随机初始化，一起训练 |
| 决策训练数据 | [CJ-Dataset](https://huggingface.co/datasets/bbldCVer-hf/CJ-Dataset) 13,048,000 条 + CJ-IE 决策 611,354 条 |
| 抽取训练数据 | CJ-IE 跨度行 573,294 条（32 个 NER 来源、DuIE / CMeIE 关系、DuEE / FewFC 事件论元），详见 [CJ-IE 数据说明](IE_DATASET.md) |
| 发布检查点 | step 14,504（1 个 epoch 的最后一步），按“决策与抽取联合分数”选出 |
| 校准温度 | T = 1.0986（见 `decision_config.json`，只作用于决策读出） |
| 最大输入 | 决策 16,384 tokens；抽取训练时文本按 ≤ 300 字切窗 |
| 候选项数 | 决策单题 2–26（训练数据覆盖 2–16） |
| 抽取 | 回显布局；跨度最宽 20 个字；none 先验校正 `none_shift = log(1/0.2)`（输出的概率用它，保持校准）；判定阈值 `decision_shift = 0.75`（决定哪些跨度算实体，在 CJ-IE dev 上按 F1 选出） |
| 提示版本 | `hmm-zh-v1` |
| 语言 | 中文 |

**文件说明**

| 文件 | 说明 |
|---|---|
| `model.safetensors` | 合并后的骨干权重 + `span_head.*`，同一份权重 |
| `config.json`、`generation_config.json` | 模型配置 |
| `tokenizer.json`、`tokenizer_config.json`、`chat_template.jinja` | Qwen3.5 分词器与聊天模板 |
| `decision_config.json` | 校准温度、提示版本、最大长度，以及 `extraction`（跨度头结构、`none_shift`、`decision_shift`、是否回显） |
| `training_summary.json` | 训练配置、选中的检查点及其验证分数 |
| `cj_bench_report.json` | CJ-Bench 全量评测的完整指标（T = 1，分领域、题型、来源） |
| `cj_ie_test_report.json` | CJ-IE 测试集的完整指标：决策逐来源，抽取逐来源和逐类型（含实体 ECE） |
| `LICENSE` | 基础模型 Qwen3.5-4B 的 Apache-2.0 许可证原文 |
| `assets/` | 本说明中的架构图 |

## 快速开始

### 安装依赖

决策读出可以直接用 `transformers`；抽取读出需要本项目的 `decision` 包（跨度头的代码在里面）：

```bash
pip install "chinese-decision[torch] @ git+https://github.com/yanqiangmiffy/chinese-decision.git@main"
```

推荐同时安装 `flash-linear-attention`，它为 Qwen3.5 的线性注意力层提供高效 kernel：

```bash
pip install "flash-linear-attention>=0.4"
```

### 下载模型

Hugging Face：

```bash
hf download quincyqiang/chinese-decision-unified-4b --local-dir chinese-decision-unified-4b
```

ModelScope：

```bash
modelscope download --model quincyqiang/chinese-decision-unified-4b --local_dir chinese-decision-unified-4b
```

### 调用方式

模型可以本地加载（`DecisionPipeline`），也可以起一个 HTTP 服务调用。两种方式的请求格式相同，**请求里带什么字段，就做什么任务**：

| 请求带 | 任务 | 返回 |
|---|---|---|
| `state` + `questions`（或单个 `question`） | 决策 | `answers`：每个问题的候选概率 |
| `text` + `types` | 抽取 | `entities`：带位置和概率的实体 |
| 两组都带 | 对同一段输入既决策又抽取 | `answers` + `entities` |

两组都带时 `text` 可以省略，默认取 `state`。输入较短时（一个问题时大约 1,000 字以内，问题越多这个长度越短），两种读出在**同一次前向**里完成：A800 上一个这样的请求约 65 ms，分开发两次约 188 ms（决策 120 ms + 抽取 68 ms）。输入更长时自动拆成两次前向，结果不变。

### 本地加载

**分开调用**：决策和抽取各发一个请求。

```python
from decision import DecisionPipeline

pipe = DecisionPipeline.from_pretrained("chinese-decision-unified-4b", device="cuda")  # 没有 GPU 时用 "cpu"（需要约 10 GB 内存）

# 带 questions → 决策
pipe({
    "state": "客户留言：付款后一直没有收到发票，我已经催了三次，今天必须拿到。",
    "questions": {
        "urgent": {"type": "noul", "instructions": "客户是否表达了紧迫性？"},
        "department": {"type": "choice", "instructions": "应该由哪个部门处理？",
                       "criteria": {"billing": "支付、发票及退款", "technical": "故障和系统问题", "sales": "购买咨询与升级"}},
    },
})

# 带 types → 抽取：每个类型是 “名称: 说明”
pipe({
    "text": "收货地址：浙江省杭州市西湖区文三路90号东部软件园科技大厦A座1203室，联系人王小明。",
    "types": {
        "省份": "省级行政区划,省、自治区、直辖市,不含港澳台",
        "城市": "地级行政区划,地级市、地区、自治州等",
        "区县": "县级行政区划,市辖区、县级市、县等",
        "道路": "有正式名称的道路,包括隧道、高架、街、弄、巷等. 步行街、商业街、统一标注为道路,除非地图上明显表明是 兴趣点类别(只有一栋楼);xx 路段、xx 路南段等整体 为道路",
        "门牌号": "路号,如'000号'或'000弄',通常位于道路名称之后",
    },
    "threshold": 0.3,   # 可选：只保留概率不低于它的实体
})
```

输出（A800 上运行，概率保留 4 位）：

```text
{'task': 'decision', 'answers': {
    'urgent':     {'false': 0.0981, 'true': 0.9019},
    'department': {'billing': 0.9803, 'technical': 0.0087, 'sales': 0.0110}}}

{'task': 'extraction', 'entities': [
    {'text': '浙江省', 'label': '省份',   'start': 5,  'end': 8,  'score': 1.0},
    {'text': '杭州市', 'label': '城市',   'start': 8,  'end': 11, 'score': 0.9893},
    {'text': '西湖区', 'label': '区县',   'start': 11, 'end': 14, 'score': 0.9868},
    {'text': '文三路', 'label': '道路',   'start': 14, 'end': 17, 'score': 0.9981},
    {'text': '90号',  'label': '门牌号', 'start': 17, 'end': 20, 'score': 0.9903}]}
```

**一次请求同时决策和抽取**：`questions` 和 `types` 放进同一个请求。

```python
result = pipe({
    "state": "客户留言：我叫张伟，上周在杭州的苏宁易购门店买了空调，付款后一直没有收到发票，已经催了三次，今天必须拿到。",
    "questions": {
        "urgent": {"type": "noul", "instructions": "客户是否表达了紧迫性？"},
        "department": {"type": "choice", "instructions": "应该由哪个部门处理？",
                       "criteria": {"billing": "支付、发票及退款", "technical": "故障和系统问题", "sales": "购买咨询与升级"}},
    },
    # 省略 text：从 state 里抽取
    "types": {"人名": "人物的姓名", "地名": "地名或地理政治实体，如省、市、区县", "公司": "企业名称"},
    "threshold": 0.3,
})

department = result["answers"]["department"]
print(max(department, key=department.get))                     # billing
print([(e["label"], e["text"]) for e in result["entities"]])   # [('人名', '张伟'), ('地名', '杭州'), ('公司', '苏宁易购')]
```

输出（A800 上运行，概率保留 4 位）：

```text
{'task': 'decision+extraction',
 'answers': {
    'urgent':     {'false': 0.1086, 'true': 0.8914},
    'department': {'billing': 0.9579, 'technical': 0.0068, 'sales': 0.0353}},
 'entities': [
    {'text': '张伟',     'label': '人名', 'start': 7,  'end': 9,  'score': 0.9257},
    {'text': '杭州',     'label': '地名', 'start': 13, 'end': 15, 'score': 0.6372},
    {'text': '苏宁易购', 'label': '公司', 'start': 16, 'end': 20, 'score': 0.7845}]}
```

`answers` 是每个候选项的概率，已按 `decision_config.json` 的温度校准；本地调用不附带 `choice` 字段，取概率最大的候选即可。`start` / `end` 是字符下标（左闭右开）。`threshold` 可选，只保留概率不低于它的实体。两组字段都没有会报 `ValueError`。

也可以不拼请求，直接调方法：

```python
pipe.decide(state, questions)                                          # {问题名: {候选: 概率}}
pipe.extract(text, types, threshold=0.3)                               # [实体, ...]
pipe.decide_and_extract(state, questions, text, types, threshold=0.3)  # (决策概率, 实体)
```

只做决策时，`AutoModelForCausalLM.from_pretrained(path)` 也能加载本模型，`span_head.*` 会被报告为未使用的权重，不影响结果。抽取读出不在 GGUF / llama.cpp 里，只能走 PyTorch。

### HTTP 服务

项目仓库的 `scripts/serve_unified.py` 把同一份权重做成一个服务（`--name` 是响应里的模型名）：

```bash
python scripts/serve_unified.py --model chinese-decision-unified-4b --name chinese-decision-unified-4b --port 8011
```

| 接口 | 说明 |
|---|---|
| `POST /v1/predict` | 推荐。按请求字段选择任务；两组都带时一次返回 `answers` 和 `entities` |
| `POST /v1/decide` | 只决策，只读 `state` 和 `questions`；请求格式与 hmm / System One 相同 |
| `POST /v1/extract` | 只抽取：`{"text", "types": {名称: 说明}, "threshold"?}` |
| `GET /v1/info` | 模型名、温度、`can_extract`、GPU、运行统计 |

`/v1/decide` 和 `/v1/extract` 会忽略不属于自己的字段，要同时做两件事，请发到 `/v1/predict`。

**一次请求同时决策和抽取**：

```bash
curl http://127.0.0.1:8011/v1/predict -H 'Content-Type: application/json' -d '{
  "state": "客户留言：我叫张伟，上周在杭州的苏宁易购门店买了空调，付款后一直没有收到发票，已经催了三次，今天必须拿到。",
  "questions": {
    "urgent": {"type": "noul", "instructions": "客户是否表达了紧迫性？"},
    "department": {"type": "choice", "instructions": "应该由哪个部门处理？",
                   "criteria": {"billing": "支付、发票及退款", "technical": "故障和系统问题", "sales": "购买咨询与升级"}}
  },
  "types": {"人名": "人物的姓名", "地名": "地名或地理政治实体，如省、市、区县", "公司": "企业名称"},
  "threshold": 0.3
}'
```

返回（A800 上运行，概率保留 4 位）：

```json
{
  "task": "decision+extraction",
  "model": "chinese-decision-unified-4b",
  "answers": {
    "urgent": {"type": "noul", "probabilities": {"false": 0.1086, "true": 0.8914}, "confidence": 0.7828, "noul": 0.8914},
    "department": {"type": "choice", "probabilities": {"billing": 0.9579, "technical": 0.0068, "sales": 0.0353},
                   "confidence": 0.9368, "choice": "billing"}
  },
  "text": "客户留言：我叫张伟，上周在杭州的苏宁易购门店买了空调，付款后一直没有收到发票，已经催了三次，今天必须拿到。",
  "entities": [
    {"text": "张伟", "label": "人名", "start": 7, "end": 9, "score": 0.9257},
    {"text": "杭州", "label": "地名", "start": 13, "end": 15, "score": 0.6372},
    {"text": "苏宁易购", "label": "公司", "start": 16, "end": 20, "score": 0.7845}
  ],
  "latency_ms": 69.0
}
```

HTTP 的决策答案比本地调用多几个字段：`choice` 是概率最大的候选（`choice` 题）；`noul` 是“是”的概率（`noul` 题）；`score` 是按概率加权的分数，附带 `legend`（`score` 题）。`confidence` = (k × 最大概率 − 1) / (k − 1)，k 是候选数：均匀分布为 0，完全确定为 1。它衡量的是分布有多集中，不是准确率。

同样的请求用 Python：

```python
import requests

body = {
    "state": "客户留言：我叫张伟，上周在杭州的苏宁易购门店买了空调，付款后一直没有收到发票，已经催了三次，今天必须拿到。",
    "questions": {
        "urgent": {"type": "noul", "instructions": "客户是否表达了紧迫性？"},
        "department": {"type": "choice", "instructions": "应该由哪个部门处理？",
                       "criteria": {"billing": "支付、发票及退款", "technical": "故障和系统问题", "sales": "购买咨询与升级"}},
    },
    "types": {"人名": "人物的姓名", "地名": "地名或地理政治实体，如省、市、区县", "公司": "企业名称"},
    "threshold": 0.3,
}
out = requests.post("http://127.0.0.1:8011/v1/predict", json=body, timeout=30).json()

print(out["answers"]["department"]["choice"])                 # billing
print(out["answers"]["urgent"]["noul"])                       # 0.8914
print([(e["label"], e["text"]) for e in out["entities"]])     # [('人名', '张伟'), ('地名', '杭州'), ('公司', '苏宁易购')]
```

**只决策或只抽取**：去掉另一组字段即可，发到 `/v1/predict` 或对应的专用接口都行。

```bash
# 只决策
curl http://127.0.0.1:8011/v1/decide -H 'Content-Type: application/json' -d '{
  "state": "客户留言：付款后一直没有收到发票，我已经催了三次，今天必须拿到。",
  "questions": {"department": {"type": "choice", "instructions": "应该由哪个部门处理？",
                               "criteria": {"billing": "支付、发票及退款", "technical": "故障和系统问题", "sales": "购买咨询与升级"}}}
}'

# 只抽取
curl http://127.0.0.1:8011/v1/extract -H 'Content-Type: application/json' -d '{
  "text": "华为公司董事长梁华在深圳会见了腾讯游戏负责人。",
  "types": {"人名": "人物的姓名", "公司": "企业名称"},
  "threshold": 0.3
}'
```

这个服务同时兼容 `scripts/serve_gpu.py` 的决策接口：`POST /v1/systemone`（`?debug=1` 附带 logits）、`/api/alpha/decisions`、`/v1/batch`、`/v1/compile`、`/v1/tournament`，以及 `/health`、`/v1/info`（没有训练进度和热更新）。所以**只部署它一个**就够了：测试平台 [Decision Studio](https://github.com/yanqiangmiffy/chinese-decision-studio) 把 `DECISION_BASE_URL` 指向它，实验台、场景库、评测、压测、鲁棒性和“抽取”模式都能用，不需要再部署纯决策服务。

## 类型怎么写

**类型的“说明”比类型名重要得多。** 训练时每个类型都带一句来自语料标注规范的定义，模型学到的是“按定义找”，而不是“按名字猜”。

- **用训练时的说明。** 召回最高；可以从 Decision Studio 实验台的“换成训练时的类型组”开始，名称保持原样、说明按业务微调。
- **只写类型名或写一句很短的话**，相近类型分不开，召回明显下降。至少写一句定义，包括边界（什么不算）。
- **没有定义的类型，CJ-IE 测试里 F1 接近 0。** `military_ner` 的 4 个类型名称和说明相同，0.8B 和 2B 都一个没预测，4B 的 F1 也只有 3.0（召回 1.5%）。
- **训练里没出现过的新类型**，基本抽不出来，或很不稳。上线前一定在自己的数据上测。
- **关系**：主体固定，类型名写成“{主体}的{关系}”，说明写客体类型，抽出的跨度就是客体。**事件**：先对每个事件类型问一个是/否问题，再对过门槛的事件用“触发词 + 角色”抽取论元。

Studio 里的 39 条案例和两个流水线场景（每条都记录了成功和失败）是在 0.8B 上逐条测的，2B 和 4B 没有逐条重测；大致规律以 0.8B 模型卡的说明为准，4B 的整体抽取指标见下文。

## 评测结果

### 决策：CJ-Bench 全量

在 CJ-Bench 全量测试集（307,900 条）上评测本仓库发布的合并权重（Transformers，BF16）。**概率使用 T = 1（不做温度校准）**，候选项保持原始顺序；温度不改变选出的答案，所以准确率不受影响。chinese-decision-4b 一列的 ACC 是发布权重的结果（与 README 一致，ACC 不受温度影响）；它的 ECE 和 NLL 取自同一模型在 T = 1 下的评测，和其余三列口径一致。

| 切分 | 样本数 | chinese-decision-unified-0.8b | chinese-decision-unified-2b | chinese-decision-4b | **chinese-decision-unified-4b** | 比 chinese-decision-unified-2b | 比chinese-decision-4b |
|---|---:|---:|---:|---:|---:|---:|---:|
| **整体（micro）** | 307,900 | 83.46 | 85.00 | 85.97 | **86.86** | +1.85 | +0.89 |
| 四领域等权平均（macro） | — | 76.17 | 79.12 | 81.92 | **83.85** | +4.72 | +1.93 |
| 通用 | 100,000 | 73.86 | 75.43 | 75.69 | **77.30** | +1.88 | +1.61 |
| 医疗 | 200,000 | 88.61 | 90.03 | 91.19 | **91.64** | +1.61 | +0.45 |
| 法律 | 4,300 | 67.88 | 73.42 | 77.21 | **82.44** | +9.02 | +5.23 |
| 金融 | 3,600 | 74.31 | 77.61 | 83.61 | **84.00** | +6.39 | +0.39 |
| `noul` | 134,577 | 94.88 | 95.24 | 95.47 | **95.65** | +0.40 | +0.18 |
| `choice` | 134,577 | 79.01 | 81.68 | 83.25 | **84.77** | +3.08 | +1.52 |
| `score` | 38,746 | 57.17 | 58.91 | 60.43 | **61.60** | +2.69 | +1.17 |

校准（T = 1，15 个等宽分箱，ECE 单位 %）与 hard NLL：

| 切分 | chinese-decision-unified-0.8b ECE | chinese-decision-unified-2b ECE | chinese-decision-4b ECE | **chinese-decision-unified-4b ECE** | chinese-decision-unified-4b hard NLL（chinese-decision-unified-0.8b / chinese-decision-unified-2b / chinese-decision-4b） |
|---|---:|---:|---:|---:|---|
| 整体 | 1.1 | **0.9** | 1.3 | 1.0 | 0.342（0.417 / 0.381 / 0.365） |
| 通用 | 2.8 | **2.5** | 3.3 | 2.8 | 0.568（0.638 / 0.605 / 0.611） |
| 医疗 | 0.4 | 0.3 | 0.3 | **0.2** | 0.229（0.300 / 0.266 / 0.241） |
| 法律 | 2.3 | 3.3 | **1.7** | 2.5 | 0.412（0.715 / 0.597 / 0.529） |
| 金融 | **1.1** | 2.2 | 1.4 | 1.5 | 0.408（0.605 / 0.526 / 0.421） |
| `noul` / `choice` / `score` | 0.7 / 1.3 / 2.3 | 0.7 / 1.0 / **2.0** | **0.5** / 1.6 / 2.7 | 0.7 / **0.9** / 2.4 | 0.106 / 0.429 / 0.899 |

整体（T = 1）：分布 NLL（含软标签）0.3483、Brier 0.1799，`score` 题期望等级 MAE 0.5144；完整指标见 `cj_bench_report.json`。

> 发布的 T = 1.0986 在 CJ-Dataset 的独立校准划分上拟合（1,984 条硬标签样本）。它在 CJ-Bench 上的整体指标没有重新跑，上面的 ECE 都是 T = 1。法律和金融样本少（4,300 / 3,600 条），这两个领域的 ECE 波动较大；本模型在这两个领域的 ECE 并不比chinese-decision-4b 或 chinese-decision-unified-0.8b更好。ACC 和 ECE 只在硬标签样本上计算。与 Jev、Chinese-Jev 等外部基线的对比见 [chinese-decision-0.8b 模型卡](MODEL_CARD_0_8B.md#cj-bench-测试集)。

### 决策：CJ-IE 测试集（抽取相关的分类任务）

CJ-IE 的测试集由各语料的官方 dev / test 构成，做过去泄漏，**是内部评测集，不是公开基准**。决策任务共 84,894 条，整体准确率 **91.89%**（chinese-decision-unified-2b：91.30%，chinese-decision-unified-0.8b：90.65%），ECE 0.84%，hard NLL 0.236：

| 任务 | 内容 | 样本数 | chinese-decision-unified-4b ACC (%) | chinese-decision-unified-2b | chinese-decision-unified-0.8b |
|---|---|---:|---:|---:|---:|
| `doc_event` | 公告是否披露了某类股权事件 | 1,942 | 99.79 | 99.74 | 99.74 |
| `doc_field` | 某事件的某个字段是哪个提及 | 2,756 | 97.79 | 97.50 | 97.13 |
| `ee_type` | 这句话是否描述了某事件类型 | 13,604 | 92.66 | 93.69 | 92.82 |
| `ner_typing` | 标出的实体属于哪个类型 | 20,312 | 94.39 | 93.55 | 92.68 |
| `re_pair` | 给定主体和客体，选关系或“无关系” | 40,417 | 90.47 | 89.57 | 88.77 |
| `ee_category` | 事件大类（DuEE 9 选 1、FewFC 10 类 + 无） | 5,863 | 85.42 | 83.75 | 84.94 |

旧能力保持（CJ 老来源 3,032 条）：92.81%（chinese-decision-unified-2b：91.88%，chinese-decision-unified-0.8b：91.51%）。

### 抽取：CJ-IE 测试集

**精确匹配**（起止位置 + 类型都对才算对），35 个来源共 232,291 个标注跨度：

| | 精确率 | 召回率 | F1 | chinese-decision-unified-2b F1 | chinese-decision-unified-0.8b F1 |
|---|---:|---:|---:|---:|---:|
| micro | 78.9 | 61.1 | **68.9** | 66.0 | 63.1 |
| 来源宏平均（35 个） | — | — | **69.8** | 66.3 | 62.5 |
| NER（31 个来源） | — | — | 71.0 | 67.4 | 64.0 |
| 关系（DuIE、CMeIE） | — | — | 56.9 | 55.4 | 47.2 |
| 事件论元（DuEE、FewFC） | — | — | 64.0 | 59.2 | 54.5 |

> **判定阈值（2026-10-31 更新）。** 上表和下面的逐来源表都按发布时的解码方式计算：用 `none_shift = log 5` 判定哪些跨度算实体。这个值让输出的概率保持校准，但判定门槛偏高、召回偏低。现在 `decision_config.json` 增加了 `decision_shift = 0.75`（在 CJ-IE dev 上按 F1 选出，test 只报一次）：概率仍按 `none_shift` 计算，是否输出某个实体按 `decision_shift` 判定。默认推理下，CJ-IE test 的 micro P / R / F1 从 79.0 / 61.0 / 68.8 变为 **72.5 / 68.7 / 70.6**，实体 ECE 从 3.2% 变为 4.2%。想要原来的高精确率，可以在加载后把 `config.extraction.decision_shift` 设成 `None`。详见[扫描报告](NONE_SHIFT_SWEEP.md)。

35 个来源里有 33 个比 chinese-decision-unified-2b好、全部 35 个都比 chinese-decision-unified-0.8b好，对 2B 涨幅最大的是 Weibo（+11.0）、CLUENER（+8.2）、cail2022（+7.6）、wanchuang（+7.1）、multiconer22（+7.0）；CMeIE（−1.2）、IMCS21（−0.5）略低。相对 2B，精确率涨了 3.5、召回率涨了 2.5。模型仍然偏保守：精确率明显高于召回率。

| 来源 | 内容 | P | R | F1 | chinese-decision-unified-2b F1 | chinese-decision-unified-0.8b F1 |
|---|---|---:|---:|---:|---:|---:|
| Resume | 简历实体 | 93.7 | 88.3 | **90.9** | 87.3 | 88.1 |
| IMCS21 | 医患对话医学实体 | 90.3 | 82.5 | **86.3** | 86.7 | 85.7 |
| CCKS2021 地址 | 16 种地址要素 | 91.2 | 86.6 | **88.9** | 86.4 | 85.8 |
| NLPCC2018 | NER | 90.5 | 86.9 | **88.7** | 85.2 | 82.9 |
| 人民日报 | 人名 / 地名 / 机构名 | 93.8 | 79.7 | **86.2** | 84.3 | 80.8 |
| MSRA | 人名 / 地名 / 机构名 | 93.9 | 73.8 | **82.6** | 82.2 | 76.0 |
| zh-OntoNotes | NER | 83.6 | 77.6 | **80.5** | 77.9 | 77.3 |
| WikiANN | 自动标注人地机构 | 79.0 | 77.2 | 78.1 | 76.2 | 73.8 |
| DuIE | 关系（客体） | 79.6 | 77.4 | 78.5 | 74.3 | 66.2 |
| CLUENER | 10 类细粒度 | 77.1 | 65.3 | 70.7 | 62.5 | 55.7 |
| DuEE | 事件论元 | 73.5 | 63.8 | 68.3 | 62.2 | 56.3 |
| CMeEE-V2 | 医学实体 | 77.5 | 52.4 | 62.6 | 58.5 | 58.6 |
| FewFC | 金融事件论元 | 68.8 | 52.8 | 59.8 | 56.1 | 52.6 |
| Weibo | 社交媒体 | 77.1 | 54.1 | 63.6 | 52.6 | 37.2 |
| MultiCoNER2 | 33 类细粒度 | 53.3 | 34.6 | 42.0 | 40.0 | 34.7 |
| CMeIE | 医学关系（客体） | 64.1 | 24.4 | 35.4 | 36.5 | 28.2 |
| military_ner | 军事领域，类型无说明 | 58.8 | 1.5 | 3.0 | 0.0 | 0.0 |

按类型看，样本数不少于 400 的 127 个类型里，F1 ≥ 80 的有 41 个（2B 是 35 个），50 到 80 的有 60 个，低于 50 的有 26 个（2B 是 30 个），已经没有 F1 为 0 的类型（2B 还有 4 个）。最稳的类型是 CCKS2021 地址的省份（99.2）、CCKS2021 地址的城市（98.0）、CCKS2021 地址的门牌号（97.1）、wanchuang 的药物成分（96.8）。跨度超过 20 字或落在英文单词里、模型表示不了的实体只占 0.59%（1,371 个），不是瓶颈。

> 说明：类型名称和定义在不同语料上统一成了 B2NERD 的标准化中文标签，测试时每个类型都给了训练里见过的标准说明，所以这是**最好情况**，换成自己写的说明或全新类型会更低；与各语料的公开榜单数字不能直接比较。完整的逐来源、逐类型指标（含实体 ECE）见 `cj_ie_test_report.json`，数据构成见 [CJ-IE 数据说明](IE_DATASET.md)。

## 训练细节

配方与 0.8B、chinese-decision-unified-2b相同：同样的数据、损失、学习率、批次累积和模型选择。每个 micro-batch 的 token 上限与 0.8B 一致（24,576），2B 因为显存把它降为一半；4B 的激活更大，所以改为开启梯度检查点，显存每张卡只占约 26 GB。

| 项目 | 设置 |
|---|---|
| 初始化 | 原始 Qwen3.5-4B（不是决策模型）；跨度头随机初始化 |
| 微调方式 | LoRA，r = 32；LoRA 学习率 1e-4，跨度头学习率 5e-4，前 100 步预热 |
| 损失函数 | 两个读出都用 CE + RLCD（权重 1.0，4 个采样，σ 从 0.4 降到 0.1）；抽取里 none 跨度的权重为 0.2 |
| 批次 | 每次更新累积 8 个 micro-batch × 4 卡；决策 micro-batch 32 条、抽取 16 行；每个 micro-batch 最多 24,576 tokens；**梯度检查点** |
| 训练量 | 1 个 epoch，14,504 步，4 × A800 80GB，约 82.6 小时 |
| 抽取布局 | 回显：文本先按自然分词读一遍，再逐字重复一遍，跨度从重复的那份上取；跨度最宽 20 字 |
| 模型选择 | 每 1,000 步在独立开发集上评估：决策准确率与抽取 F1 的均值；开发集来自 CJ-Dataset validation 和 CJ-IE dev，**从不读取 CJ-Bench**；选中最后一步（开发集：CJ 准确率 82.89%、hard NLL 0.411、IE 决策 93.44%、抽取 F1 70.68） |
| 温度校准 | 在 CJ-Dataset validation 的独立划分上，以硬标签交叉熵拟合单一温度 T = 1.0986 |
| 导出 | 合并 LoRA，写出单个 `model.safetensors`；导出时记录合并前后 logits 的最大差（0.0625，BF16 合并误差） |

开发集上的曲线在第 12,000 步之后基本走平（综合分 0.818、0.817、0.817、0.821），说明这个训练量下已经收敛。**训练前（第 0 步）的原始 Qwen3.5-4B** 在 CJ 开发集上准确率为 60.2%，抽取 F1 为 0，作为基线。训练、评测与导出脚本见 [GitHub 仓库](https://github.com/yanqiangmiffy/chinese-decision)，为什么从原始底座训练见 [统一模型与遗忘修复](UNIFIED_MODEL.md)。

## 局限性

- **抽取召回偏低**：CJ-IE 测试集上 micro F1 68.9，精确率 78.9、召回率 61.1。边界要求精确匹配，实体边界的细小差异也算错。
- **说明的写法影响很大**：只写类型名、没有定义的类型，或训练里没出现过的新类型，效果会明显下降，甚至几乎抽不出（`military_ner`）。上线前请在自己的数据上用自己的类型测一遍。
- **某些领域和关系较弱**：多类型细粒度语料（MultiCoNER2，F1 42.0）、医学关系（CMeIE，召回 24.4%，F1 比 2B 低 1.2）；事件里带方向的角色会串（0.8B 上的观察）；`ee_type`（这句话是否描述了某事件类型）比 2B 低 1.0 个点。
- **通用模型不如专项模型**：同一份 CLUENER 官方 dev（1,343 句），之前只用 4,000 句 CLUENER 专门训练的实验 F1 是 78.7，统一 4B 是 70.7。只需要一套固定标签的 NER 时，专项模型会更准；统一模型的价值在于一份权重覆盖多种任务和多套类型。
- **不支持意图 + 槽位填充**：训练里没有槽位数据。意图用决策读出，槽位用抽取读出，需要自己级联并自己评测。
- **抽取表示能力有限**：跨度最宽 20 个字；边界落在英文单词内部的实体表示不了；文本较长时请自己按句切窗（训练时 ≤ 300 字）。
- **只能用 PyTorch 抽取**：GGUF / llama.cpp 版本没有跨度头，只有决策读出。
- **最大也最慢**：约 4.2B 参数，8.4 GB 权重，本地 CPU 加载需要约 10 GB 内存，训练约 83 小时；对延迟或显存敏感时，0.8B 或 chinese-decision-unified-2b是更轻的选择。
- **`score` 题型仍然最弱**（CJ-Bench 61.60%）；法律和金融的 ECE（T = 1）不如chinese-decision-4b。
- **`confidence` 不是准确率**，温度只在 CJ-Dataset 的校准划分上拟合；按概率阈值自动执行前，请先在自己的数据上验证。
- **CJ-IE 测试集是内部评测集**，类型已经标准化，数字不能和各语料的公开榜单直接比较。
- **标签含义来自上游任务**，请勿将输出作为医疗、法律或金融决策的唯一依据；**不用于文本生成**，`generate()` 的输出没有意义；仅支持中文。

## 许可证

- **模型权重**：Apache-2.0，继承自基础模型 Qwen3.5-4B（`LICENSE` 为其原文）。
- **训练数据**：CJ-Dataset、CJ-Bench 以及 CJ-IE 所用的 NER / 关系 / 事件语料，许可按数据来源分别适用，其中部分来源为非商业许可或许可不明。本模型不授予任何数据层面的额外权利，商业使用前请自行评估。CJ-IE 本身没有随模型发布，构建方法见 [CJ-IE 数据说明](IE_DATASET.md)。
- **代码**：推理、训练与评测代码以 MIT 许可证发布，见 [GitHub 仓库](https://github.com/yanqiangmiffy/chinese-decision)。

## 致谢

- [Chinese-Jev](https://arxiv.org/abs/2609.36965)：CJ-Dataset、CJ-Bench 与评测口径。
- [n4ze3m/hmm](https://github.com/n4ze3m/hmm)：选项字母决策目标、候选项格式与 System One 响应约定。
- B2NERD 等中文 NER / 关系 / 事件语料：抽取训练数据的来源。
- [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)：基础模型。

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
