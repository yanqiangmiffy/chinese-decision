---
pretty_name: CJ-IE
license: other
license_name: mixed-per-source
language:
- zh
task_categories:
- token-classification
- text-classification
task_ids:
- named-entity-recognition
- multi-class-classification
size_categories:
- 1M<n<10M
tags:
- information-extraction
- named-entity-recognition
- relation-extraction
- event-extraction
- chinese
- multi-task
- decision
configs:
- config_name: extraction
  data_files:
  - split: train
    path: extraction/*.train.jsonl
  - split: validation
    path: extraction/*.dev.jsonl
  - split: test
    path: extraction/*.test.jsonl
- config_name: structured
  data_files:
  - split: train
    path: structured/*.train.jsonl
  - split: validation
    path: structured/*.dev.jsonl
  - split: test
    path: structured/*.test.jsonl
---

<div align="center">

# CJ-IE：中文信息抽取与抽取相关决策数据

**把公开的中文 NER、关系和事件语料整理成一种格式：类型带定义的跨度抽取，加上同一批语料派生的选择题 / 判断题**

[GitHub 代码](https://github.com/yanqiangmiffy/chinese-decision) ·
[构建说明 IE_DATASET.md](https://github.com/yanqiangmiffy/chinese-decision/blob/main/docs/IE_DATASET.md) ·
[用它训练的模型 Chinese Decision Unified 0.8B](https://huggingface.co/quincyqiang/chinese-decision-unified-0.8b)

</div>

> **English summary.** CJ-IE is a Chinese information-extraction collection built from 39 public NER, relation and event corpora. It has three parts: `extraction` (span rows whose entity types carry a name and a definition, 669,982 rows), `decision` (multiple-choice and yes/no questions derived from the same corpora, in the CJ-Dataset format, 708,802 rows) and `structured` (full entity / relation / event annotations kept for end-to-end evaluation, 142,192 rows). It was used to train the extraction readout of [Chinese Decision Unified 0.8B](https://huggingface.co/quincyqiang/chinese-decision-unified-0.8b). **CJ-IE is a compilation of third-party corpora with different licenses, some of them non-commercial or restrictive (see [License](#许可与使用限制)); it has no single license, and you must follow the terms of every source you use.**

## 数据集简介

CJ-IE 由 [Chinese Decision](https://github.com/yanqiangmiffy/chinese-decision) 项目整理，用来训练 [Chinese Decision Unified 0.8B](https://github.com/yanqiangmiffy/chinese-decision/blob/main/docs/MODEL_CARD_UNIFIED_0_8B.md) 的抽取读出，同时给它提供“与抽取相关的分类任务”。它做的事情是**格式和类型体系的统一**，而不是新标注：

- **统一的跨度格式**：每一行是 `文本 + 类型表（名称和定义）+ 实体跨度`。类型的定义随行给出，所以同一个模型可以“按定义找”，而不是记住某个语料的标签集。
- **统一的类型名称与定义**：所有 NER 来源统一使用 [B2NERD](https://huggingface.co/datasets/Umean/B2NERD) 标准化后的中文标签及其定义，“人名”在不同语料中含义一致；标注规范的差异由定义体现。
- **关系和事件也写成跨度**：关系把“主体 + 谓词”写进类型名，跨度是客体；事件把“触发词 + 论元角色”作为类型，跨度是触发词和论元。
- **派生的决策题**：同一批语料派生出选择题和是 / 否判断（实体类型、关系判断、事件类型、事件大类、公告字段），格式与 CJ-Dataset 相同。
- **严格的划分与去泄漏**：按文本稳定哈希划分，同一文本的所有视图在同一划分里；跨语料重叠的记录已丢弃。

| 子集 | 内容 | train | dev | test | 合计 |
|---|---|---:|---:|---:|---:|
| `extraction` | 跨度行（NER / 关系客体 / 事件论元） | 573,294 | 11,647 | 85,041 | 669,982 |
| `decision` | 派生的选择 / 判断题 | 611,354 | 12,554 | 84,894 | 708,802 |
| `structured` | 完整的实体 / 关系 / 事件标注（用于端到端评测） | 125,859 | 1,295 | 15,038 | 142,192 |

跨度行里的实体数：train 1,621,219，dev 42,198，test 232,291。三个子集合计 1,520,976 行，约 2.2 GB（JSONL，UTF-8）。

> `structured` 与前两个子集来自同一批原始记录，用于端到端评测（例如完整的三元组、事件结构），不建议与 `extraction` 混在一起训练。

## 数据结构

```text
CJ-IE/
├── extraction/{task}_{source}.{train,dev,test}.jsonl   跨度行，每个来源一个文件（共 105 个文件）
├── decision/{train,dev,test}.cases.jsonl               决策题，各来源随机交错
├── structured/{source}.{train,dev,test}.jsonl          DuIE / CMeIE / DuEE / FewFC 的完整结构
├── manifest.json                                       构建参数、逐文件计数、过滤统计、各来源类型表
└── README.md                                           本数据卡
```

### `extraction`：跨度行

```json
{"id": "ner/msra/train/train-4166",
 "source": "ner_msra",
 "text": "下半场后半段，中国队表现稍有起色，但同成熟的俄罗斯队比起来终究要逊一筹。",
 "types": [
   {"key": "LOC", "name": "地名或地理政治实体", "description": "文本中独立出现的具体地理位置或地理政治实体,包括国家、城市、街道、建筑物、自然地标,地区等"},
   {"key": "ORG", "name": "组织机构名", "description": "文本中独立出现的组织机构完整名称,包括政府部门、公司企业、非政府组织、新闻媒体、政党等,注意名称中连带的地名实体不单独提取"},
   {"key": "PER", "name": "人名", "description": "文本中的个人姓名,包括姓名和单独出现的姓或名,提取时不要带上尊称(如'先生')"}],
 "entities": [{"start": 7, "end": 10, "label": "ORG"}, {"start": 22, "end": 26, "label": "ORG"}],
 "_meta": {"source_family": "ner:msra", "task_id": "ner", "source": "msra", "dataset": "MSRA-NER",
           "quality": "gold", "domain": "news", "split": "train", "group_key": "msra:train-4166"}}
```

| 字段 | 说明 |
|---|---|
| `id`、`source` | 行标识和来源文件名（如 `ner_msra`） |
| `text` | 文本，已按 ≤ 300 字切窗 |
| `types` | 本行可用的类型表：`key`（`entities[].label` 引用它）、`name`（显示名称）、`description`（定义） |
| `entities` | 实体跨度：`start` / `end` 是**字符下标，左闭右开**，`label` 对应 `types[].key`；可以为空（负样本） |
| `_meta` | `source_family`（`<任务>:<来源>`）、`task_id`、`dataset`、`quality`、`domain`、`split`、`group_key`（同一原始记录的所有视图共享，用于防泄漏），事件行还有 `event_type` |

三类任务的跨度行：

| `task_id` | 来源 | 一行是什么 |
|---|---|---|
| `ner` | 32 个 NER 来源 | 一段文本 + 该语料的全部类型及定义 |
| `re_object` | DuIE、CMeIE | **每个主体一行**。类型名写成“{主体}的{谓词}”，`description` 写客体的类型，跨度是客体；主体只出现在类型文本里，模型必须读类型才知道要抽什么 |
| `ee_argument` | DuEE、FewFC | **每个（句子，事件类型）一行**。类型是“触发词”和该事件的论元角色。另有 25% 的句子附加一行“未出现的事件类型”，不含任何跨度 |

关系行的例子（谓词在类型名里，客体是跨度）：

```json
{"id": "re_duie/train/train-121746#0",
 "text": "维克多·雨果《巴黎圣母院》《悲惨世界》《九三年》巴尔扎克《高老头》《欧也妮·葛朗台》莫泊桑《羊脂球》",
 "types": [{"key": "作者", "name": "九三年的作者", "description": "人物"}],
 "entities": [{"start": 0, "end": 6, "label": "作者"}]}
```

事件行的例子：

```json
{"id": "ee_duee/train/train-8440#1",
 "text": "昨天上午9时30分许，普陀区真北路梅川路路口发生一起交通事故：一辆搅拌车在路口撞击一辆非机动车，造成一名阿婆不幸身亡。",
 "types": [
   {"key": "trigger", "name": "触发词", "description": "表示“车祸”事件发生的核心词"},
   {"key": "受伤人数", "name": "受伤人数", "description": "“车祸”事件的受伤人数"},
   {"key": "地点", "name": "地点", "description": "“车祸”事件的地点"},
   {"key": "时间", "name": "时间", "description": "“车祸”事件的时间"},
   {"key": "死亡人数", "name": "死亡人数", "description": "“车祸”事件的死亡人数"}],
 "entities": [{"start": 0, "end": 10, "label": "时间"}, {"start": 11, "end": 22, "label": "地点"},
              {"start": 39, "end": 41, "label": "trigger"}, {"start": 50, "end": 52, "label": "死亡人数"}]}
```

### `decision`：派生的决策题

（这个子集没有配成 `datasets` 的 config，见[怎么加载](#怎么加载)。）格式与 [CJ-Dataset](https://huggingface.co/datasets/bbldCVer-hf/CJ-Dataset) 相同：`state`（题干里的上下文）、`questions`（`noul` 是 / 否，或 `choice` 单选，候选项**不超过 16 个**，也是模型训练时见过的上限）、`gold`（标准答案）、`_meta`。多个正确答案时，`gold` 写成均分的概率分布。

```json
{"id": "ee_type/fewfc/test/test_trans-a56d54842465366ea23514309b937ffa#签署合同",
 "state": {"句子": "其中中国银联在全国范围内与近200个城市近8000个停车场合作;建设银行深圳分行在深圳市签约了2500余家停车场开展无感支付业务。"},
 "questions": {"q": {"type": "noul", "instructions": "这句话是否描述了“签署合同”事件（合作双方在合同等书面文件上签字或盖章确立合作关系）？"}},
 "gold": {"q": "true"},
 "_meta": {"source_family": "ee_type:fewfc", "task_id": "ee_type", "source": "fewfc", "dataset": "FewFC",
           "quality": "gold", "domain": "finance", "split": "test", "event_type": "签署合同"}}
```

| `task_id` | 题型 | 来源 | 内容 | train / dev / test |
|---|---|---|---|---:|
| `re_pair` | choice | DuIE、CMeIE、FinRE、SanWen | 给定主体和客体，选关系或“无关系”。DuIE / CMeIE 的负例是同句中类型允许、但没有标注关系的有序实体对，数量为正例的一半 | 345,513 / 5,233 / 40,417 |
| `ner_typing` | choice | 32 个 NER 来源 | 文本中用【】标出一个实体，从该语料的类型中选择（不超过 16 个） | 184,660 / 4,728 / 20,312 |
| `ee_type` | noul | DuEE、FewFC | 这句话是否描述了某事件类型；每句包括它的全部事件类型，外加一个未出现的类型 | 30,812 / 915 / 13,604 |
| `ee_category` | choice | DuEE、FewFC | DuEE：只含一个大类的句子，9 选 1；FewFC：只含一种事件的句子，10 类加“无上述事件” | 13,339 / 404 / 5,863 |
| `doc_field` | choice | ChFinAnn | 某事件的某个字段是哪一个提及（候选为同类提及），或“公告未提及”。只在该类事件的各条记录取值一致时提问 | 21,372 / 743 / 2,756 |
| `doc_event` | noul | ChFinAnn | 公告是否披露了某类股权事件（5 类） | 15,658 / 531 / 1,942 |

### `structured`：完整结构

保留原语料的完整标注，用于端到端评测，不用于直接训练。`duie.*` 和 `cmeie.*` 是 `entities` + `relations`（`subject` / `object` 是 `entities` 的下标）；`duee.*` 和 `fewfc.*` 是 `events`（每个事件有 `type` 和带角色的 `mentions`）。

```json
{"id": "duie/test/validation-18033", "source": "duie",
 "text": "饭间，毛泽东特意向周恩来介绍说，“这是王观澜的爱人徐明清",
 "entities": [{"start": 19, "end": 22, "text": "王观澜", "labels": ["人物"]}, {"start": 25, "end": 28, "text": "徐明清", "labels": ["人物"]}],
 "relations": [{"subject": 0, "predicate": "妻子", "object": 1}, {"subject": 1, "predicate": "丈夫", "object": 0}]}
```

## 怎么加载

下载整个数据集（约 2.2 GB）：

```bash
hf download quincyqiang/CJ-IE --repo-type dataset --local-dir CJ-IE
modelscope download --dataset quincyqiang/CJ-IE --local_dir CJ-IE
```

`extraction` 和 `structured` 可以直接用 `datasets` 读取，各自带 `train` / `validation` / `test`：

```python
from datasets import load_dataset

ner_train = load_dataset("quincyqiang/CJ-IE", "extraction", split="train")
structured_test = load_dataset("quincyqiang/CJ-IE", "structured", split="test")
```

`decision` **没有配成 `datasets` 的 config**：它沿用 CJ-Dataset 的格式，`gold` 里的值在单个正确答案时是字符串，在多个正确答案时是概率字典，同一列里类型不一致，`datasets` 的 JSON 读取器无法统一。请直接读 JSONL：

```python
import json

with open("CJ-IE/decision/test.cases.jsonl", encoding="utf-8") as f:
    for line in f:
        case = json.loads(line)
        if case["_meta"]["task_id"] == "re_pair":      # 用 _meta 按任务或来源过滤
            print(case["questions"]["q"]["instructions"], case["gold"]["q"])
            break
```

`extraction` 同样可以直接读文件，并用类型表把标签还原成名称：

```python
with open("CJ-IE/extraction/ner_msra.test.jsonl", encoding="utf-8") as f:
    row = json.loads(next(f))
types = {t["key"]: t["name"] for t in row["types"]}
for e in row["entities"]:
    print(row["text"][e["start"]:e["end"]], types[e["label"]])
```

> `extraction` 各来源的 `_meta` 字段不完全相同（例如事件行多一个 `event_type`），`datasets` 会把它们合并成一个较宽的结构。需要逐来源处理时，直接读文件更省事。

## 构建方法

构建脚本是 [`scripts/build_ie_data.py`](https://github.com/yanqiangmiffy/chinese-decision/blob/main/scripts/build_ie_data.py)，转换代码在 [`decision/data/ie/`](https://github.com/yanqiangmiffy/chinese-decision/tree/main/decision/data/ie)，设计细节见 [IE_DATASET.md](https://github.com/yanqiangmiffy/chinese-decision/blob/main/docs/IE_DATASET.md)。本数据集由默认参数构建（随机种子 42），`manifest.json` 记录了完整的构建参数、逐文件计数和过滤统计。

- **切窗**：文本按 ≤ 300 字切窗；公告类长文本最多取 3,000 字。
- **采样上限**：每个来源每个划分有行数上限（默认 train 50,000 / dev 500 / test 5,000；DuIE train 100,000；ChFinAnn 按公告条数 train 8,000 / dev 300 / test 1,000），避免个别大来源淹没小来源。`ner_typing` 每个来源 train 10,000 / dev 200 / test 1,000。
- **类型名称与定义**：NER 来源统一使用 B2NERD 标准化后的中文标签。独立下载的语料（MSRA、CLUENER、Resume、Weibo、CMeEE-V2、Boson、MultiCoNER2）把原始标签映射到 B2NERD 中同一语料的标签，在训练文本上与 B2NERD 逐条对照，99% 以上的实体一致。显示名称经过整理（如“组织机构名->公司”写作“公司”），没有定义的编号变体（如“品牌名2”）并入基础标签；WikiANN 和 MultiNERD 不在 B2NERD 中，使用本项目编写的中文名称和定义。
- **实体定位**：B2NERD 的实体只给出类型和名称，没有位置。处理方式是名称的每次出现都标为实体，但跳过与先放置的更长实体重叠的位置；只在更长实体内部出现的名称，按嵌套实体放在第一次出现处；名称在文本中找不到的行整行丢弃（本次为 0 行）。CMeIE 的主体和客体也按同样方法定位。
- **负例**：`re_pair` 的负例数量为正例的一半；`ee_argument` 有 25% 的句子附加一行未出现的事件类型（无跨度）；`doc_field` 有 20% 的“公告未提及”。
- **去重与去泄漏**：构建日志统计丢弃了 180,190 条重复记录和 12,896 条与 held-out 划分重叠的记录。主要是 MSRA train 中属于人民日报 dev / test 的 4,955 条，以及 CMeIE / CMeEE-V2 之间共享的医学段落（分别为 2,506 条和 916 条）；另有 59 个训练窗口与 held-out 窗口文本相同，也被丢弃。构建后核对：train、dev、test 两两之间规范化后完全相同的文本为 0（跨度行和决策都已检查）。
- **划分**：使用文本的稳定哈希，重跑结果相同；公开 test 没有标签的语料（CLUENER、CMeEE-V2、CMeIE、DuIE、DuEE）用官方 dev 作为 test，dev 从 train 切出 3%。PeopleDaily-NER 只保留 dev / test（它的全部文本都包含在 MSRA 中）；Microbiota 只有 99 句，全部作为训练集。

## 来源

共 39 个来源（32 个 NER、2 个关系、2 个事件，另有 FinRE、SanWen 和 ChFinAnn 只提供决策题）。质量标签：`gold` 为人工标注，`silver` 为自动标注（WikiANN、MultiNERD），`distant` 为远程监督（ChFinAnn）。

| 来源 | 原始数据集 | 领域 | 质量 | 任务 |
|---|---|---|---|---|
| `msra` | MSRA-NER | news | gold | `ner`、`ner_typing` |
| `people_daily` | PeopleDaily-NER | news | gold | `ner`、`ner_typing` |
| `cluener` | CLUENER2020 | general | gold | `ner`、`ner_typing` |
| `resume` | Resume-NER | general | gold | `ner`、`ner_typing` |
| `weibo` | Weibo-NER | social | gold | `ner`、`ner_typing` |
| `cmeee_v2` | CMeEE-V2 | medical | gold | `ner`、`ner_typing` |
| `boson` | Boson-NER | news | gold | `ner`、`ner_typing` |
| `multiconer2` | MultiCoNER2-zh | general | gold | `ner`、`ner_typing` |
| `wikiann` | WikiANN-zh | general | silver | `ner`、`ner_typing` |
| `multinerd` | MultiNERD-zh | general | silver | `ner`、`ner_typing` |
| `bank` | B2NERD / bank | finance | gold | `ner`、`ner_typing` |
| `cail2021` | B2NERD / CAIL2021 | legal | gold | `ner`、`ner_typing` |
| `cail2022` | B2NERD / CAIL2022 | legal | gold | `ner`、`ner_typing` |
| `ccks2017` | B2NERD / CCKS2017 | medical | gold | `ner`、`ner_typing` |
| `ccks2019` | B2NERD / CCKS2019 | medical | gold | `ner`、`ner_typing` |
| `ccks2020` | B2NERD / CCKS2020 | medical | gold | `ner`、`ner_typing` |
| `ccks2021address` | B2NERD / CCKS2021address | general | gold | `ner`、`ner_typing` |
| `chineseliterature_ner` | B2NERD / chineseliterature-ner | general | gold | `ner`、`ner_typing` |
| `fned` | B2NERD / FNED | general | gold | `ner`、`ner_typing` |
| `gaiic2022` | B2NERD / gaiic2022 | ecommerce | gold | `ner`、`ner_typing` |
| `imcs21` | B2NERD / imcs21 | medical | gold | `ner`、`ner_typing` |
| `math_high` | B2NERD / math-high | education | gold | `ner`、`ner_typing` |
| `microbiota` | B2NERD / Microbiota | medical | gold | `ner`、`ner_typing` |
| `military_ner` | B2NERD / military-ner | military | gold | `ner`、`ner_typing` |
| `multiconer22` | B2NERD / multiconer22 | general | gold | `ner`、`ner_typing` |
| `nlpcc2018` | B2NERD / nlpcc2018 | dialogue | gold | `ner`、`ner_typing` |
| `peopledaily2014` | B2NERD / peopledaily2014 | general | gold | `ner`、`ner_typing` |
| `ruijinmcc` | B2NERD / ruijinmcc | medical | gold | `ner`、`ner_typing` |
| `taobao` | B2NERD / Taobao | ecommerce | gold | `ner`、`ner_typing` |
| `wanchuang` | B2NERD / wanchuang | medical | gold | `ner`、`ner_typing` |
| `youku` | B2NERD / youku | general | gold | `ner`、`ner_typing` |
| `zh_ontonotes` | B2NERD / zh-ontonotes | general | gold | `ner`、`ner_typing` |
| `duie` | DuIE-HF | general | gold | `re_object`、`re_pair` |
| `cmeie` | CMeIE | medical | gold | `re_object`、`re_pair` |
| `finre` | FinRE | finance | gold | `re_pair` |
| `sanwen` | SanWen-RE | literature | gold | `re_pair` |
| `duee` | DuEE-HF | news | gold | `ee_argument`、`ee_category`、`ee_type` |
| `fewfc` | FewFC | finance | gold | `ee_argument`、`ee_category`、`ee_type` |
| `chfinann` | ChFinAnn | finance | distant | `doc_event`、`doc_field` |

<details>
<summary><b>逐来源的跨度行数 / 实体数</b></summary>

| 文件 | train 行 / 实体 | dev 行 / 实体 | test 行 / 实体 |
|---|---:|---:|---:|
| `ee_duee` | 15,085 / 40,155 | 465 / 1,224 | 1,974 / 5,250 |
| `ee_fewfc` | 4,878 / 19,665 | 126 / 516 | 6,869 / 24,834 |
| `ner_bank` | 9,217 / 15,385 | 267 / 422 | 501 / 772 |
| `ner_boson` | 2,334 / 19,018 | 67 / 600 | 183 / 1,315 |
| `ner_cail2021` | 4,004 / 20,603 | 111 / 575 | 1,045 / 5,427 |
| `ner_cail2022` | 417 / 4,207 | 10 / 103 | 108 / 1,159 |
| `ner_ccks2017` | 2,785 / 53,438 | 76 / 1,555 | 305 / 5,611 |
| `ner_ccks2019` | 2,261 / 21,619 | 64 / 628 | 171 / 1,638 |
| `ner_ccks2020` | 2,752 / 22,290 | 108 / 908 | 177 / 1,496 |
| `ner_ccks2021address` | 8,518 / 41,298 | 270 / 1,310 | 1,969 / 9,828 |
| `ner_chineseliterature_ner` | 23,266 / 131,344 | 487 / 2,859 | 2,839 / 16,455 |
| `ner_cluener` | 10,429 / 23,248 | 312 / 699 | 1,343 / 3,072 |
| `ner_cmeee_v2` | 13,742 / 74,309 | 420 / 2,575 | 5,040 / 26,081 |
| `ner_fned` | 9,664 / 207,694 | 323 / 6,797 | 482 / 10,611 |
| `ner_gaiic2022` | 6,221 / 93,723 | 192 / 2,973 | 356 / 5,356 |
| `ner_imcs21` | 50,000 / 42,163 | 497 / 409 | 5,000 / 4,108 |
| `ner_math_high` | 1,298 / 7,219 | 47 / 244 | 258 / 1,514 |
| `ner_microbiota` | 98 / 162 | — | — |
| `ner_military_ner` | 313 / 2,736 | 10 / 81 | 80 / 647 |
| `ner_msra` | 39,999 / 64,275 | 443 / 643 | 4,323 / 6,160 |
| `ner_multiconer2` | 9,124 / 13,774 | 498 / 763 | 5,000 / 6,899 |
| `ner_multiconer22` | 15,220 / 23,879 | 500 / 822 | 5,000 / 5,707 |
| `ner_multinerd` | 50,181 / 66,740 | 490 / 639 | 5,028 / 6,545 |
| `ner_nlpcc2018` | 13,390 / 9,149 | 420 / 272 | 728 / 505 |
| `ner_people_daily` | — | 501 / 846 | 4,641 / 7,707 |
| `ner_peopledaily2014` | 50,692 / 136,279 | 505 / 1,245 | 5,084 / 13,760 |
| `ner_resume` | 3,798 / 13,393 | 462 / 1,496 | 476 / 1,630 |
| `ner_ruijinmcc` | 25,808 / 124,467 | 547 / 2,534 | 2,999 / 14,774 |
| `ner_taobao` | 3,916 / 19,278 | 467 / 2,298 | 957 / 4,677 |
| `ner_wanchuang` | 2,276 / 37,315 | 93 / 1,547 | 137 / 2,222 |
| `ner_weibo` | 1,170 / 1,601 | 253 / 374 | 262 / 410 |
| `ner_wikiann` | 17,817 / 22,577 | 490 / 621 | 5,003 / 6,234 |
| `ner_youku` | 7,579 / 12,393 | 489 / 796 | 990 / 1,573 |
| `ner_zh_ontonotes` | 14,481 / 12,783 | 499 / 772 | 4,311 / 7,359 |
| `re_cmeie` | 14,602 / 34,517 | 461 / 1,123 | 4,626 / 11,522 |
| `re_duie` | 135,959 / 188,523 | 677 / 929 | 6,776 / 9,433 |

</details>

<details>
<summary><b>逐来源的决策题数</b></summary>

| source_family | train | dev | test |
|---|---:|---:|---:|
| `doc_event:chfinann` | 15,658 | 531 | 1,942 |
| `doc_field:chfinann` | 21,372 | 743 | 2,756 |
| `ee_category:duee` | 10,866 | 347 | 1,406 |
| `ee_category:fewfc` | 2,473 | 57 | 4,457 |
| `ee_type:duee` | 23,532 | 733 | 3,045 |
| `ee_type:fewfc` | 7,280 | 182 | 10,559 |
| `ner_typing:bank` | 8,243 | 200 | 445 |
| `ner_typing:boson` | 1,588 | 49 | 129 |
| `ner_typing:cail2021` | 3,899 | 108 | 1,000 |
| `ner_typing:cail2022` | 387 | 9 | 100 |
| `ner_typing:ccks2017` | 1,765 | 51 | 201 |
| `ner_typing:ccks2019` | 1,210 | 32 | 92 |
| `ner_typing:ccks2020` | 1,292 | 51 | 85 |
| `ner_typing:ccks2021address` | 8,517 | 200 | 1,000 |
| `ner_typing:chineseliterature_ner` | 10,000 | 200 | 1,000 |
| `ner_typing:cluener` | 10,000 | 200 | 1,000 |
| `ner_typing:cmeee_v2` | 10,000 | 200 | 1,000 |
| `ner_typing:fned` | 9,662 | 200 | 480 |
| `ner_typing:gaiic2022` | 6,221 | 192 | 356 |
| `ner_typing:imcs21` | 10,000 | 200 | 1,000 |
| `ner_typing:math_high` | 1,298 | 47 | 258 |
| `ner_typing:microbiota` | 60 | 0 | 0 |
| `ner_typing:military_ner` | 310 | 10 | 80 |
| `ner_typing:msra` | 10,000 | 200 | 1,000 |
| `ner_typing:multiconer2` | 9,124 | 200 | 1,000 |
| `ner_typing:multiconer22` | 10,000 | 200 | 1,000 |
| `ner_typing:multinerd` | 10,000 | 200 | 1,000 |
| `ner_typing:nlpcc2018` | 8,473 | 200 | 466 |
| `ner_typing:people_daily` | 0 | 200 | 1,000 |
| `ner_typing:peopledaily2014` | 10,000 | 200 | 1,000 |
| `ner_typing:resume` | 3,699 | 200 | 461 |
| `ner_typing:ruijinmcc` | 10,000 | 200 | 1,000 |
| `ner_typing:taobao` | 3,910 | 200 | 957 |
| `ner_typing:wanchuang` | 1,082 | 40 | 63 |
| `ner_typing:weibo` | 617 | 139 | 159 |
| `ner_typing:wikiann` | 10,000 | 200 | 1,000 |
| `ner_typing:youku` | 7,540 | 200 | 980 |
| `ner_typing:zh_ontonotes` | 5,763 | 200 | 1,000 |
| `re_pair:cmeie` | 66,187 | 2,177 | 22,317 |
| `re_pair:duie` | 250,252 | 1,219 | 12,538 |
| `re_pair:finre` | 12,268 | 998 | 3,358 |
| `re_pair:sanwen` | 16,806 | 839 | 2,204 |

</details>

## 用途

- **训练和评测抽取模型**：尤其适合“类型带定义”的开放类型抽取。用它训练的 [Chinese Decision Unified 0.8B](https://github.com/yanqiangmiffy/chinese-decision/blob/main/docs/MODEL_CARD_UNIFIED_0_8B.md) 在 CJ-IE 测试集上的抽取 micro F1 为 63.1（精确率 73.3，召回率 55.5，精确匹配），决策题整体准确率 90.65%。
- **多任务训练**：`extraction` 对应跨度头，`decision` 对应决策头，两者可以在同一个批次流里混合。
- **研究 NER / 关系 / 事件在统一格式下的迁移**。

不适用于：医疗、法律、金融等场景下的自动决策；也不应把它当作各语料官方榜单的替代品（见下）。

## 局限与偏差

- **不是新标注，也不是公开基准。** CJ-IE 的 test 由各语料的官方 dev / test 构成并做过去泄漏，是**内部评测集**。类型名称和定义已经统一成 B2NERD 的标准化中文标签，所以数字不能与各语料的公开榜单直接比较。
- **标注质量不一。** WikiANN、MultiNERD 是自动标注（`silver`），ChFinAnn 是远程监督（`distant`），其余为人工标注，但各语料的标注规范和一致性不同。
- **B2NERD 来源的实体位置是推断出来的。** 名称的每次出现都被标为实体，同名的非实体出现也会被标上，会引入假阳性。
- **来源规模悬殊。** DuIE 的 train 有 135,959 行，而 Microbiota 只有 98 行；`micro` 指标会被大来源主导，请同时看逐来源指标。
- **领域和文体偏窄。** 新闻、医学、金融、电商、社交媒体、简历等比较多，法律、军事、教育很少；`military_ner` 的类型没有单独定义（名称和说明相同），用它训练的模型一个都没抽出来。
- **文本窗口 ≤ 300 字**。用它训练的 Unified 0.8B 的跨度最宽 20 个字，更长的实体、边界落在英文单词内部的实体都表示不了。
- **关系的客体抽取依赖主体由外部提供**（`re_object` 的主体只写在类型文本里）。
- **含个人信息。** 语料来自新闻、微博、简历、医患对话等真实文本，可能含姓名、机构、地址等信息；请遵守各来源的使用条款，不要用于识别或追踪个人。
- **标签含义来自上游任务。** 请勿将在此数据上训练的模型输出作为医疗、法律或金融决策的唯一依据。

## 许可与使用限制

**CJ-IE 是多个第三方语料的整理和转换，没有单一的许可证。** 我们不对任何来源重新授权，也不授予任何额外的再分发权限；构建代码以 MIT 许可证发布，但这不改变各来源数据的条款。**使用 CJ-IE 即表示你需要遵守你所使用的每个来源的许可条款。** 下表是我们在本地来源目录里能找到的许可信息：

- “镜像元数据”指我们下载的 Hugging Face 社区镜像自己标注的许可，**不能自动替代原始数据的使用条款**；
- “未核实”表示我们没有找到或没有核对原始发布方的条款，请在使用前自行确认。

| 来源 | 我们找到的许可信息 |
|---|---|
| FinRE、SanWen-RE | MIT（THUNLP Chinese_NRE，LICENSE 文件） |
| ChFinAnn | MIT（Doc2EDAG，LICENSE 文件） |
| MultiCoNER2-zh | CC BY 4.0（数据集 README） |
| FewFC | CC BY-SA 4.0（数据集 README 与 LICENSE 文件） |
| MultiNERD-zh | CC BY-NC-SA 4.0，仅限非商业研究用途（数据集 README） |
| Boson-NER | BosonNLP 数据使用许可证（license.txt）：仅供科研；禁止向第三方提供数据；商业用途需付费 |
| CLUENER2020 | 镜像元数据 Apache-2.0；原始条款未核实 |
| Resume-NER | 镜像元数据 Apache-2.0；原始条款未核实 |
| MSRA-NER | 镜像元数据未填写（另一个变体标注 Apache-2.0）；未核实 |
| Weibo-NER | 镜像元数据未填写（清洗变体标注 Apache-2.0）；未核实 |
| PeopleDaily-NER | 未找到许可说明 |
| WikiANN-zh | 镜像元数据未填写；自动标注自维基百科；未核实 |
| CMeEE-V2、CMeIE（CBLUE） | 镜像元数据 MIT；CBLUE 原始条款未核实 |
| DuIE、DuEE | 镜像元数据 MIT；百度原始发布条款未核实 |
| B2NERD 收录的来源：`bank`、`cail2021`、`cail2022`、`ccks2017`、`ccks2019`、`ccks2020`、`ccks2021address`、`chineseliterature_ner`、`fned`、`gaiic2022`、`imcs21`、`math_high`、`microbiota`、`military_ner`、`multiconer22`、`nlpcc2018`、`peopledaily2014`、`ruijinmcc`、`taobao`、`wanchuang`、`youku` | B2NERD 元数据标注 MIT，但其说明要求使用者自行确保拥有原始数据集的相应许可；各原始语料的条款未核实 |
| `zh_ontonotes`（B2NERD） | B2NERD 元数据标注 MIT，但其说明要求使用者自行确保拥有原始数据集的相应许可；各原始语料的条款未核实；名称对应 OntoNotes 中文部分，该语料由 LDC 发布，再分发条件需核实 |

需要特别注意的几点：

1. **Boson（`ner_boson`、`ner_typing:boson`）**：BosonNLP 数据使用许可证仅供科研使用，**禁止将数据提供给第三方**，直接用于商业目的需要付费。
2. **MultiNERD（`ner_multinerd`、`ner_typing:multinerd`）**：CC BY-NC-SA 4.0，仅限非商业研究，且衍生数据需以相同方式共享。
3. **FewFC**：CC BY-SA 4.0，需署名并以相同方式共享。
4. **WikiANN、PeopleDaily-NER、MSRA**：我们没有找到明确的许可说明。**zh-OntoNotes** 对应的 OntoNotes 语料由 LDC 发布，再分发条件需要核实。请自行核实后再使用或再分发。

如果你需要在商业场景或产品中使用，请先逐个来源确认条款，必要时只使用许可明确的来源（过滤时用 `_meta.source`）。如果你是某个来源的权利人，并希望将它从 CJ-IE 中移除，请在 [GitHub](https://github.com/yanqiangmiffy/chinese-decision/issues) 上联系我们。

## 引用

如果使用了 CJ-IE，请引用本项目，并同时引用你所使用的原始语料。

```bibtex
@misc{cjie2026,
  title        = {CJ-IE: Chinese Information Extraction Data with Typed Definitions},
  author       = {Yan, Qiang},
  year         = {2026},
  howpublished = {\url{https://github.com/yanqiangmiffy/chinese-decision}},
  note         = {Built for Chinese Decision Unified 0.8B}
}
```

使用其中的决策题（格式与 CJ-Dataset 相同）并与 CJ-Bench 对照时，请同时引用 Chinese-Jev：

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

Chinese Decision 系列的论文是 [arXiv:2610.23170](https://arxiv.org/abs/2610.23170)；CJ-IE 和统一版是在该论文之后整理和发布的，不在论文中。

## 致谢

- [B2NERD](https://huggingface.co/datasets/Umean/B2NERD)：NER 标签体系和类型定义的来源。
- 上表中的各个语料及其作者：MSRA、人民日报、CLUENER、Resume、Weibo、CMeEE-V2 / CMeIE（CBLUE）、Boson、MultiCoNER、MultiNERD、WikiANN、DuIE / DuEE、FewFC、FinRE / SanWen（THUNLP Chinese_NRE）、ChFinAnn（Doc2EDAG），以及 B2NERD 收录的各中文 NER 语料。
- [Chinese-Jev](https://arxiv.org/abs/2609.36965)：CJ-Dataset 的决策题格式。
