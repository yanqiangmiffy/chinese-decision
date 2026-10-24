# CJ-IE：中文信息抽取的大规模多任务数据

日期：2026-10-23。分支：`feat/classification-extraction`。构建脚本：[scripts/build_ie_data.py](../scripts/build_ie_data.py)，转换代码：[decision/data/ie/](../decision/data/ie/)。

CJ-IE 是多任务训练里信息抽取的那一部分。它把 `ie_collections` 中的中文 NER、关系和事件数据整理成两种头都能直接读取的格式，并划分好训练集、验证集和测试集：

- **跨度头**（[decision/modeling.py](../decision/modeling.py) 的 `SpanHead`，输入由 `DecisionProcessor.encode_extraction` 编码）：读取 `{"text", "types", "entities"}` 跨度行；
- **决策头**：读取 CJ-Dataset 格式的决策（`state / questions / gold / _meta`）。

目录布局与 [MULTITASK_EXTRACTION.md](MULTITASK_EXTRACTION.md) 中的 `multitask-mini` 相同，`scripts/train_multitask.py --data datasets/cj-ie` 可以直接读取。

## 规模

默认参数的一次构建（`datasets/cj-ie`，共 2.2 GB，约 4 分钟）：

| 头 | task_id | train | dev | test |
|---|---|---:|---:|---:|
| 跨度 | `ner`（32 个来源） | 402,770 | 9,918 | 64,796 |
| 跨度 | `re_object`（DuIE、CMeIE） | 150,561 | 1,138 | 11,402 |
| 跨度 | `ee_argument`（DuEE、FewFC） | 19,963 | 591 | 8,843 |
| 决策 | `re_pair` | 345,513 | 5,233 | 40,417 |
| 决策 | `ner_typing` | 184,660 | 4,728 | 20,312 |
| 决策 | `ee_type` | 30,812 | 915 | 13,604 |
| 决策 | `ee_category` | 13,339 | 404 | 5,863 |
| 决策 | `doc_field`（ChFinAnn） | 21,372 | 743 | 2,756 |
| 决策 | `doc_event`（ChFinAnn） | 15,658 | 531 | 1,942 |
| | **跨度行合计** | **573,294** | **11,647** | **85,041** |
| | **决策合计** | **611,354** | **12,554** | **84,894** |

逐来源的计数见构建目录中生成的 `README.md` 和 `manifest.json`。

**去泄漏丢弃的记录。** 主要是 MSRA train 中属于人民日报 dev/test 的 4,955 条，以及 CMeIE / CMeEE-V2 之间共享的医学段落（分别为 2,506 条和 916 条）。另有 59 个训练窗口与 held-out 窗口文本相同，也被丢弃。

**构建后的独立核对。**

- train、dev、test 两两之间，规范化后完全相同的文本为 0（跨度行和决策都已检查）；
- 每个跨度文件抽样的行都能通过 `ExtractionEncoder` 编码，并组成 batch；
- 每个划分抽样 3,000 条决策，均未超出长度上限；
- 所有决策写出前都经过 `examples_for` 校验。

## 输出

```text
datasets/cj-ie/
  extraction/{task}_{source}.{train,dev,test}.jsonl   跨度行，每个来源一个文件
  decision/{train,dev,test}.cases.jsonl               CJ 决策，各来源随机交错
  structured/{source}.{train,dev,test}.jsonl          DuIE/CMeIE/DuEE/FewFC 的完整结构，用于端到端评测
  manifest.json                                       参数、逐文件计数、过滤统计、各来源类型表
  README.md                                           本次构建的计数表
```

### 跨度行

```json
{"id": "ner/cluener/train/train-5732", "source": "ner_cluener", "text": "……中国中医研究院针灸研究所开展的检测也表明：",
 "types": [{"key": "organization", "name": "其它组织机构名", "description": "中文或英文的，除政府和公司以外的其它组织团体的名字……"}, …],
 "entities": [{"start": 20, "end": 31, "label": "organization"}],
 "_meta": {"source_family": "ner:cluener", "task_id": "ner", "dataset": "CLUENER2020", "quality": "gold", "domain": "general", "split": "train", "group_key": "cluener:train-5732"}}
```

偏移是字符下标，左闭右开。`label` 对应 `types[].key`。编码器只读取 `text / types / entities`，`_meta` 用于按来源统计和配比。

### 决策

与 CJ-Dataset 相同：`choice` 的候选项不超过 16 个（CJ-Dataset 的最大值，也是模型训练时见过的上限）；多个正确答案时，`gold` 写成均分的概率分布。`_meta.source_family` 的格式是 `<task>:<source>`，`train_multitask.py` 按它分组计算指标。

## 任务视图

同一条原始记录的所有视图都落在同一个划分里。

| task_id | 头 | 来源 | 内容 |
|---|---|---|---|
| `ner` | 跨度 | 32 个 NER 来源 | 每行给出该语料的全部类型及其定义 |
| `ner_typing` | 决策 | 同上 | 文本中用【】标出一个实体，从该语料的类型中选择（不超过 16 个） |
| `re_object` | 跨度 | DuIE、CMeIE | 每个主体一行。类型名写成“<主体>的<谓词>”，跨度是客体；主体只出现在类型文本里，跨度头必须读类型才知道要抽什么 |
| `re_pair` | 决策 | DuIE、CMeIE、FinRE、SanWen | 给定主体和客体，选择谓词或“无关系”。DuIE/CMeIE 的负例是同句中类型允许、但没有标注关系的有序实体对，数量为正例的一半 |
| `ee_argument` | 跨度 | DuEE、FewFC | 每个（句子，事件类型）一行，类型为“触发词”和该事件的论元角色。另有 25% 的句子附加一行未出现的事件类型，不含任何跨度 |
| `ee_type` | 决策（noul） | DuEE、FewFC | 这句话是否描述了某事件类型；每句包括它的全部事件类型，外加一个未出现的类型（DuEE 优先选同一大类） |
| `ee_category` | 决策 | DuEE、FewFC | DuEE：只含一个大类的句子，9 选 1；FewFC：只含一种事件的句子，10 类加“无上述事件” |
| `doc_event` | 决策（noul） | ChFinAnn | 公告是否披露了某类股权事件（5 类） |
| `doc_field` | 决策 | ChFinAnn | 某事件的某个字段是哪一个提及（候选为同类提及），或“公告未提及”。只在该类事件的各条记录取值一致时提问 |

推理时的组合方式：

- **NER**：跨度头。
- **关系**：先用 NER 或已知实体找到主体，再用 `re_object` 抽取客体；`re_pair` 用于复核候选实体对。
- **事件**：先用 `ee_type` 判断事件类型，再对每个类型用 `ee_argument` 抽取触发词和论元。

## 类型名称与定义

所有 NER 来源统一使用 B2NERD 标准化后的中文标签及其定义，因此“人名”在不同语料中的含义一致，标注规范的差异由定义体现，例如 MSRA 的“组织机构名”与人民日报 2014 的“组织机构名(非标准)”。

- **独立语料的映射。** 独立下载的语料（MSRA、CLUENER、Resume、Weibo、CMeEE-V2、Boson、MultiCoNER2）把原始标签映射到 B2NERD 中同一语料的标签。在训练文本上与 B2NERD 逐条对照，99% 以上的实体一致。
- **名称整理。** 显示名称经过整理：“组织机构名->公司”写作“公司”，“地名->其它”写作“其它地名”。括号内的说明（如“带币种”）移到定义末尾；整理后同名的标签保留原名。
- **编号变体。** 没有定义的编号变体（如 gaiic2022 的“品牌名2”）并入基础标签，与 B2NERD 官方精简版的做法一致。
- **自写定义。** WikiANN 和 MultiNERD 不在 B2NERD 中，使用本仓库编写的中文名称和定义。

B2NERD 的实体只给出类型和名称，没有位置。处理方式：

- 名称的每次出现都标为实体，但跳过与先放置的更长实体重叠的位置；
- 只在更长实体内部出现的名称，按嵌套实体放在第一次出现处；
- 名称在文本中找不到的行整行丢弃（本次为 0 行）。

CMeIE 的主体和客体也按同样方法定位。

## 来源与划分

| 来源 | 划分处理 |
|---|---|
| MSRA | 官方 train/test；dev 从 train 切出 3% |
| PeopleDaily-NER | **只保留 dev/test**：它的全部 27,818 条文本都包含在 MSRA 中。保留它是为了与此前 mini 实验的人民日报指标对照 |
| CLUENER、CMeEE-V2、CMeIE、DuIE、DuEE | 公开 test 没有标签，官方 dev 作为 test；dev 从 train 切出 3% |
| Resume、Weibo、WikiANN、MultiNERD、MultiCoNER2、ChFinAnn、FinRE、SanWen | 官方 train/dev/test |
| FewFC（`rearranged/`） | train_base + train_trans 作为训练集，test_base + test_trans 作为测试集；dev 从训练集切出 3% |
| Boson | 无划分，按文本哈希切出 dev 3% / test 7% |
| B2NERD 子集 | 有官方划分的沿用；缺 dev 时从 train 切出 3%，缺 test 时切出 5% |
| Microbiota | 只有 99 句，全部作为训练集 |

切分使用文本的稳定哈希，重跑结果相同，同一文本的各个视图也一定在同一划分里。

**去泄漏。** 比较前先规范化文本（NFKC、去空白、小写），再整体比较，并按 12 字以上的句子逐句比较。

- **规则：** dev 与任何来源的 test 重叠时丢弃；train 与任何来源的 dev 或 test 重叠时丢弃。
- **重叠的定义：** 文本相同；held-out 中的单句文本出现在该记录里；或该记录一半以上的句子出现在 held-out 记录中。
- **例外：** 在 20 条以上记录中出现的句子不计入，例如公告免责声明和病历体检模板。

**跨来源重复。** 训练集内部的跨来源重复保留，同一句话在两种标注规范下出现，正好能说明类型定义的作用。完全被其他来源包含的子集则不收录：

- CCIR2021、finance-sina：包含于 zh-ontonotes；
- CCKS2018：797 条中有 777 条与 CCKS2017 相同。

**采样上限。** 每个来源、每个划分最多保留 train 50,000 / dev 500 / test 5,000 条记录，随机抽取（种子 42）。

- DuIE 的 train 上限为 100,000；
- ChFinAnn 的上限为 8,000 / 300 / 1,000，因为公告长达约 2k token；
- `ner_typing` 决策每个来源最多 train 10,000 / dev 200 / test 1,000 条；
- NER 文本超过 300 字时按句末切成窗口，切点不会落在实体内部。

**质量标记。** `_meta.quality` 取以下值，配比时可以据此降权：

- `gold`：人工标注；
- `silver`：WikiANN 和 MultiNERD 是自动标注；
- `distant`：ChFinAnn 来自远程监督。

## 没有收录的数据

- **IPRE。** 这份 HF 转换版的标签是包级的远程监督，例如一份演员表被标成“生父”。用于句子级决策会教模型从共现推断关系，因此没有收录。
- **其他未收录项。** 英文数据（DocRED、Re-DocRED、GUIDEX、InstructUIE 镜像、InstructIE），繁体中文事实核验（CFEVER），以及 FIE2025 公开样例（事实性推理，不属于抽取）。

## 尚未接入的新数据（2026-10-23/24 下载）

`ie_collections` 在 CJ-IE 构建之后新增了三批共 30 个目录。原始说明见各批 README，下面按“能喂给哪个头”整理。这些数据都**还没有进入 CJ-IE**：接入前需要按原文分组去重，并与现有来源（MSRA、DuIE、DuEE、CMeIE 等被多个集合复用）做跨集合检查。

| 用途 | 数据 | 能直接用的监督 | 主要注意事项 |
|---|---|---|---|
| 跨度头：类型以定义为条件 | IEPile（中文 103 万条指令）、YAYI-UIE（中文 54 万）、InstructIE-ZJUNLP（17 万，12 个主题）、KnowLM-KE（NER 7.8 万 / RE 45.6 万 / EE 1.9 万） | 文本、候选 schema（含难负 schema）和字符串答案 | 只有字符串、没有偏移，需要定位；大量复用 DuIE、MSRA 等原文；输出协议不统一（JSON、元组、NAN） |
| 跨度头：事件 / 法律 / 金融 | LEVEN（法律事件 108 类，含 29.7 万个显式负候选）、DocFEE（文档级金融事件 1.8 万篇）、LexRel（民事法律关系 + 主客体，带定义） | LEVEN 有字符位置；DocFEE、LexRel 只有字符串 | DocFEE 有 1 万多个规范化角色值无法在原文中定位，不能伪造跨度；LexRel 两个训练版本大量复用同一批案件 |
| 跨度头：观点 | DiaASQ-zh（对话四元组）、Chinese-Review-ASTE（繁体，弱标注 6.4 万，人工标注 300） | DiaASQ 有 token 位置 | ASTE 的问题里直接写出了方面词，人工 gold 只有 300 条 |
| 决策头：意图分类 | MASSIVE-zh（60 个意图）、SMP2017（31 类）、CMID（医疗）、MInDS-14（银行）、CFLUE 银行意图（只有 500 条评测）、xSID-zh | 意图标签，可以直接做成 choice 题 | 应该与 slot 数据配合使用，按意图条件化 |
| 决策头 + 跨度头：意图 + 槽位 | CAIS、FewJoint、ECSA（只有槽位）、MAC-SLU（车载，多意图）、XingIntent（schema 条件化）、SmartHome（社区数据） | BIO 槽位，或可定位的槽值 | MAC-SLU 有弱标注和空帧；SmartHome 的测试集全部出现在训练集里；ECSA 是远程监督 |
| 上下文与澄清 | CrossWOZ、RiSAWOZ、BiToD（多轮状态）、DialogUSR（复合指令拆分）、Chinese-Ambiguous-Reference（是否需要澄清） | 对话行为、状态，以及“是否追问”这类判断 | 必须保留上下文，并按整段对话切分；状态值可能是归一化后的，无法定位 |
| 不适合 | MME-RealWorld-CN（需要图像）、Title2Event（偏移为空，1 万多个论元不是原文的连续子串） | | |

建议的接入顺序：

1. **LEVEN 和 LexRel：** 法律是这次遗忘最严重的领域，它们又带显式负例和类型定义；
2. **MASSIVE、CAIS、FewJoint：** 补意图与槽位，可以接到 `SlotFillingPipeline`（[decision/pipeline.py](../decision/pipeline.py)）的级联用法上；
3. **IEPile、YAYI：** 先只取能定位的样本，用来扩充类型名的多样性。

## 训练时的注意事项

1. **类型定义占用大量 token。** 对短句、多类型的来源（MultiCoNER2 有 33 类，还有 gaiic2022、CLUENER），类型定义约占每行 token 的 70–80%。
   - 跨度行平均 100–900 token，最长约 1,100；
   - 决策最长约 1,800 token（ChFinAnn）。
2. **训练脚本。** `train_multitask.py` 会把全部数据读入内存并每轮重新编码，只适合 mini 规模。CJ-IE 用 [scripts/train_multitask_ddp.py](../scripts/train_multitask_ddp.py)：按 batch 惰性编码、多卡训练，并支持旧任务回放（见 [UNIFIED_MODEL.md](UNIFIED_MODEL.md)）。各来源条数差别很大，如需调整配比，可按 `_meta.source_family` 设置权重，或对大来源降采样。
3. **与 CJ-Dataset 可能重叠。** CJ-Dataset 的金融领域包含 FinRE，FinRE 的测试结果可能受此影响。

## 复现

```bash
# 需要 pyarrow（Anaconda Python 即可），全部来源约 4 分钟
python scripts/build_ie_data.py --out datasets/cj-ie --overwrite
# 冒烟：每个来源每个划分 200 条
python scripts/build_ie_data.py --out tmp/cj-ie-smoke --limit 200 --overwrite
```
