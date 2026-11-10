# 决策模型训练数据清单

2026-11-02 核查，面向 chinese-decision-4b 的第二轮训练。下表收录 **81 个候选条目，覆盖 8 个方向**。按可单独取样的训练任务计数，例如 CLUE 下的 AFQMC 和 IFLYTEK 单列。合集中的原始样本会重复，训练规模应按原始来源去重后统计。

## 第一轮优先取哪些

项目模型卡记录，**仅决策 4B** 用了 13,048,000 条 CJ 训练数据，其中通用 1,000 万、医疗 300 万、金融与法律各 2.4 万。**Unified 4B** 还使用了 CJ-IE 决策和跨度数据。附件报告的去偏后公开分分别为 41.12 和 39.29。仅决策版在 RAGTruth 为 14.2，PhishNChips 为 50.5，报告观察到模型倾向预测“无幻觉”和“正常”。这是补数据的依据。数据覆盖、标签极性、决策阈值和多候选推理各自贡献多少，仍需分开验证。[仅决策模型卡](MODEL_CARD_4B.md) · [Unified 模型卡](MODEL_CARD_UNIFIED_4B.md)

| 顺序 | 首批来源 | 为什么优先 | 在独立留出集上看什么 |
|---|---|---|---|
| 1 | RAGTruth train + RAGBench train | 直接补“回答超出证据”的人标正例和跨域样本。先跑人标版，再单独测加入模型标注的增益 | 幻觉类召回、精确率、无幻觉误报率，按 QA、摘要、数据转文本分开 |
| 2 | Nazario + 经抽检的 SpamAssassin hard/easy ham，SpaPhish 先留作西语迁移检验 | 英文风险邮件与正常邮件都要有。加入合法密码重置、账单催缴等难负例，另留独立来源的英文邮件 | 钓鱼召回、正常邮件误报率、换模板/域名后的结果。训练和验证移除来源名与泄漏标签的邮件头 |
| 3 | BANKING77 train + CLINC150 train/oos_train + MASSIVE 中英 train | 同时覆盖近邻意图、域外拒识和中英语义。现有引擎会将大候选集分组再决赛 | 全 77/151 候选的宏 F1、正确类别首轮保留率、OOS 召回。单组与全候选分别测 |
| 4 | Tasksource default/train 的可追溯子集 + WinoGrande、HellaSwag 或 WANLI 的 train | 给英文常识与推断加覆盖。先择少数来源、按源限量，并与直接下载的原语料去重 | 未见来源的准确率，保留 CJ-Bench 作为中文能力回归评测 |
| 5 | When2Call 的 SFT/偏好训练集 + ToolACE，按需加入 ToolRet-Training-20w | 分开训练“该不该调用”“选哪个工具”“哪个工具更相关” | 工具不足时追问/拒绝的召回率，近名工具混淆率 |
| 6 | OCNLI、ChiFraud、T2Ranking、程序生成决策 | 中文推断、中文反诈、中文检索和规则决策可分别作为跨域实验 | 用未见文档、网站或规则族留出，并对照只回放 CJ 的结果 |

**两项直接核查。** 本项目重建 0.3 时固定的 RAGTruth 原文件有 15,090 条 train 回答，其中 6,721 条有幻觉标注，8,369 条无标注片段。用官方判定 `len(labels)>0` 计数，训练回答与 test 的 `source_id` 交叉为 0。该核查仅针对来源 ID，跨库同文仍需另查。PhishNChips 固定版本的 `core_emails.csv` 共 2,000 封，构建器打乱后遍历了全部 2,000 封。**整份核心文件应作为评测集隔离。** API-Bank 的 `lv1-lv2-samples` 也被官方直接用来出题，工具训练优先用独立训练语料。[RAGTruth 固定版本](https://github.com/ParticleMedia/RAGTruth/tree/c103204b9ce28d6bbad859304bf30de72b8ed8fe) · [榜单的 RAGTruth 和 PhishNChips 构建代码](https://github.com/apolinario/decision-index/blob/25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d/decision_index/suite/build/adapters_added.py) · [API-Bank 构建代码](https://github.com/apolinario/decision-index/blob/25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d/decision_index/suite/build/adapters_selection.py)

**规模口径也会改变选库决定。** 本次核查的 tasksource 修订中，`default/train` 为 999,832 行，`full/train` 为 2,550,012 行，另有提取、视觉、历史筛选版及留出数据。规划文本决策训练预算时要选定配置，按父例计数。RAGTruth 上述数量为原始训练行，实际入库时还可按 `quality` 单列截断或异常拒答的样本。[Tasksource 固定修订](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions/blob/8df2e5c9a33f19a2c2b8febf291c3acb0a91a8a8/README.md) · [RAGTruth 字段说明](https://github.com/ParticleMedia/RAGTruth)

## 榜单模型实际披露的来源

下表按作者模型卡或论文记录。阶段样本、唯一记录和问题数采用作者各自的单位。只把作者明确列出的内容归给对应模型。

| 模型与版本 | 作者披露的数据与做法 | 一手来源 |
|---|---|---|
| Verdin-4B v1.1 | 179,197 条记录、306,594 道题。新增的 tasksource 部分从 560 个来源抽 62,193 条，每源至多 128 条。还列出 RAGTruth、When2Call、ToolACE、ANLI、BANKING77、CLINC、Nazario 与 SpamAssassin 等。作者特别披露，4,643 条 tasksource 记录缺原始上游划分。 | [固定版模型卡与逐源表](https://huggingface.co/victorbennett/Verdin-4B-v1.1/blob/45830d216bb7e45f43f0a482c131a7cafa6d189f/README.md) |
| Amber 4B | 两次训练各用 136,848 条 tasksource 记录，来自 547 个来源，每源至多 400 条。还列出 WinoGrande、HellaSwag、ANLI、NLI4CT、When2Call、GSM8K train 与 MetaMathQA 子集。第二次训练打乱候选顺序，并调整二分类中的多数类占比。 | [Amber 4B 模型卡](https://huggingface.co/MLukauskas/amber-4b) |
| Cerno 4B | 公开指令/分类混合，另列 BANKING77、CLINC、ANLI、RAGTruth、ContractNLI、Amazon ESCI、When2Call 等任务族，按 GSM8K、BBH、CRUXEval、CLadder 的形式用程序生成新题。作者还报告剔除训练同源任务族后的成绩。其关于任务族的披露不能替代对具体训练文件的核查。 | [固定版模型卡](https://huggingface.co/apertixai/cerno-4b/blob/ed4b5c59b542ee96c227ee874776c973980c640b/README.md) |
| JPT-4B | 32,835 条记录、49,221 道题，训练时使用两份打乱选项的副本。作者列出 Banking77、MASSIVE、ContractNLI、TAT-QA、Mind2Web、ToolACE，以及小游戏轨迹和按代码算答案的规则题。适合参考其在较小数据量下覆盖多种状态的做法。 | [固定版模型卡](https://huggingface.co/kirp/jpt-4b/blob/78312f855b8bebf83ae7e9e8f05b5a0f73f519a9/README.md) |
| Perplexity Decider Flash v1.2 | 4B 在 27B 决策模型的训练请求上学习完整教师概率分布。作者明确肯定 tasksource 的贡献。另一个版本 v1.1 的 27B 发布了逐源训练清单，可用来发现数据源，两个版本的采样数字要分开看。 | [Flash v1.2 模型卡](https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.2) · [27B v1.1 训练清单](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b/blob/main/training/data-manifest.json) |
| KnowLine-4B Gen5 | 作者明确列出 GSM8K、WinoGrande XL、ACOS、Amazon ESCI 的上游训练数据和同形式合成题，游戏决策用模拟器或引擎打标签，训练及回放语料按题目和来源段落去重。 | [固定版模型卡](https://huggingface.co/PelaAI/KnowLine-4B-Gen5/blob/a8e0cd664f88bd2d2561dc69de74fdc48c641da5/README.md) |
| StartLux 4B Instant v1.1 | 两阶段分别约 270 万和 170 万个决策样例。论文称第一阶段 Lux-Release 汇集分类、NLI、QA、安全、偏好、工具和社区类型化数据，第二阶段 Lux-Rounds 扩展上游 train、可执行程序题和教师标签，并回放部分第一阶段数据。两阶段数量包含回放。 | [论文 §2.3](https://arxiv.org/html/2610.17309v1) · [模型卡](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-4B-Instant) |

## 如何读下面的 81 项

训练入口一栏标出官方 train、明确的训练文件或需要自建留出的样本池。数量按选定配置和核查版本记录，部分条目刻意使用标签或文档组作为计数单位。许可栏摘录发布者标注，复合语料、教师输出、邮件、网页和图文应保留原始来源条件。“见上游”表示本次未确认统一许可。适合训练的是按这些条件筛出的样本。

### 1. 可直接转为 choice / noul / score 的混合训练集（8 项）

| 编号 | 数据集 | 可取的训练部分 | 转成哪种决策、怎样分组 | 许可或取用条件 | 发布来源 |
|---:|---|---|---|---|---|
| 01 | Tasksource typed decisions | default/train，999,832 行，full/train 2,550,012 行 | 按 source、group_id 和原始 source_index 抽样。先保留原始 train 可定位的英文监督，default、full 与 filtered-full 有重合。 | 逐源。看 license、license_use | [Hugging Face](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions) |
| 02 | Procedural typed decisions | all/train，或只选一个任务配置的 train | 规则可验算，覆盖路径、可达性、计划、约束、检索。按规则族留出，能单测状态理解。 | Apache-2.0 | [Hugging Face](https://huggingface.co/datasets/tasksource/procedural-typed-decisions) |
| 03 | Synthetic typed decisions | train，6,666 个状态、21,190 个问题 | 邮件、工单、语气、紧急度与路由。保留同一工作流的问题关联，软标签由模型生成和复核。 | Apache-2.0，模型标签 | [Hugging Face](https://huggingface.co/datasets/tasksource/synthetic-typed-decisions) |
| 04 | Open-Jev v1.1 | 公开投影的 train，147,139 条 | 保留 target 分布。训练集中 82,045 条来自 WANLI，若单独加 WANLI，要按原始样本去重。 | 逐源，WANLI 为 CC BY 4.0 | [Hugging Face](https://huggingface.co/datasets/ZefanCai/Open-Jev-v1.1) |
| 05 | JevEmbed-Data | train，1,601,157 道题 | 直接映射 request_json 与 answers_json，按 source_repo/source_row 采样。含 Open-Jev、程序题、HelpSteer 等衍生数据。 | 逐行 license 和来源附加条件 | [Hugging Face](https://huggingface.co/datasets/HIT-TMG/JevEmbed-Data) |
| 06 | SN92 generalist | 训练分片，109,547 个合成样本 | 多语言策略、结构化选项、反事实对。按 ancestry 的模板、翻译和原例分组。 | 卡标 CC BY 4.0，附来源清单 | [Hugging Face](https://huggingface.co/datasets/decisionmodels/sn92-generalist-train-v1) |
| 07 | Calibrated typed decisions | train，16,995 题 | 比较硬标签和软分布两种目标。区分 GoEmotions 的人类投票与其他来源的校准教师概率。 | 卡标 CC BY 4.0，追溯上游 | [Hugging Face](https://huggingface.co/datasets/m-lagnajit/calibrated-typed-decisions) |
| 08 | Typed decisions synth | data/train.jsonl | 多领域合成状态，一例可有多问，小比例补概率监督，单独记录教师来源。 | 卡标 MIT，合成标签 | [Hugging Face](https://huggingface.co/datasets/n4ze3m/typed-decisions-synth) |

### 2. 有无证据、幻觉、长文和结构化事实（10 项）

| 编号 | 数据集 | 可取的训练部分 | 转成哪种决策、怎样分组 | 许可或取用条件 | 发布来源 |
|---:|---|---|---|---|---|
| 09 | RAGTruth | response.jsonl 的 split=train，15,090 条 | 与榜单同源。按 source_id 接 source_info.jsonl，用 prompt、response 作输入，labels 非空即有幻觉。按原文分组。 | 上下文逐源，MS MARCO、Yelp 有额外条件 | [GitHub](https://github.com/ParticleMedia/RAGTruth) |
| 10 | RAGBench | 各配置 train，取 adherence_score 有效的行 | 外部迁移。documents＋response，adherence_score=false 转为存在不支持陈述，过滤无判定项并保留标注模型身份。 | 卡标 CC BY 4.0，核上游语料 | [Hugging Face](https://huggingface.co/datasets/galileo-ai/ragbench) |
| 11 | FaithDial | data/train.json | 用知识和对话历史判断回复是否有据。BEGIN 标签描述原 Wizard 回复，分别处理 original_response 和修正后的 response。 | 卡标 MIT，核上游 Wizard of Wikipedia | [Hugging Face](https://huggingface.co/datasets/McGill-NLP/FaithDial) |
| 12 | FEVER | v1.0/train，按原 claim 去重 | 证据＋陈述判支持、反驳或证据不足。从原证据索引取维基句子，避免把缺证据标成反驳。 | 卡标 CC BY-SA 3.0 / GPL-3.0，按资产区分 | [Hugging Face](https://huggingface.co/datasets/fever/fever) |
| 13 | SciFact | claims/train，配 corpus，按 claim 与论文划分 | 科研摘要中的支持或反驳，带证据句。同一 claim 可以有多行证据，不能按行随机拆。 | 卡标 CC BY-NC 2.0 | [Hugging Face](https://huggingface.co/datasets/allenai/scifact) |
| 14 | TabFact | 作者的训练表 ID 列表及其陈述 | 对表格判真假，训练数字、列名、比较关系。按整表隔离。 | 见原仓库及维基表格条款 | [GitHub](https://github.com/wenhuchen/Table-Fact-Checking) |
| 15 | SQuAD 2.0 | train，130,319 条 | 段落＋问题判能否回答。无答案是信息不足，保持与“回答内含虚构”分开的题干。 | CC BY-SA 4.0 | [Hugging Face](https://huggingface.co/datasets/rajpurkar/squad_v2) |
| 16 | BoolQ | train，9,427 条 | 段落＋问题判是或否。加入答案确实为“否”的有据例，纠正把否定等同于不确定的倾向。 | CC BY-SA 3.0 | [Hugging Face](https://huggingface.co/datasets/google/boolq) |
| 17 | ContractNLI | 原版 train.json，按合同 ID 留出 | 与榜单同源。合同全文＋条款假设判蕴含、矛盾、未提及，同一合同的不同假设在同一划分。 | 依项目数据许可与原合同 | [GitHub](https://github.com/stanfordnlp/contract-nli) |
| 18 | HoVer | 官方 train claims，配证据文档 | 与榜单同源。多跳证据判支持或不支持，榜单取 dev。再检查训练与评测的维基证据段落重合。 | 数据 CC BY-SA 4.0，保留维基署名 | [GitHub](https://github.com/hover-nlp/hover) |

### 3. 钓鱼、欺诈与带标签的风险样本（7 项）

| 编号 | 数据集 | 可取的训练部分 | 转成哪种决策、怎样分组 | 许可或取用条件 | 发布来源 |
|---:|---|---|---|---|---|
| 19 | Nazario phishing corpus | 分年 mbox，自建时间和活动分组 | 英文真实钓鱼正例池。抽检误收，将 MIME 转成统一正文，清除可直接泄漏答案的邮件头。 | 作者目录 LICENSE.txt，CC BY 4.0 | [作者数据页](https://monkey.org/~jose/phishing/) |
| 20 | SpamAssassin public corpus | easy_ham、hard_ham，按邮件线程留出 | 英文正常邮件候选，人工抽核非钓鱼，重点取像警告、推广或有链接的正常邮件。spam 不能一律当作钓鱼。 | 按原邮件权利与语料 README，使用前核准 | [作者数据页](https://spamassassin.apache.org/old/publiccorpus/) |
| 21 | Phishing Pot | 原始邮件样本，自建训练/留出 | 补充英文攻击模板。按相似正文和活动去重，来源标记不进模型输入。 | 核仓库许可及第三方邮件权利 | [GitHub](https://github.com/rf-peixoto/phishing_pot) |
| 22 | SpaPhish | 原作者发布的有标签语料，按时间或模板自划 | 真实西班牙语邮件，同时有合法和钓鱼两类。论文报告 731 封钓鱼、664 封合法，适合跨语言验证来源捷径。 | CC BY 4.0 | [作者数据页](https://data.mendeley.com/datasets/hz2d6gz7pc/5) |
| 23 | ChiFraud | 作者中文网页文本训练数据 | 按原风险类别与正常类做 choice，可辅助中文诈骗词变形和诱导意图。其对象是网页文本。 | 仓库与论文未作统一商用认定 | [GitHub](https://github.com/xuemingxxx/ChiFraud) |
| 24 | TeleAntiFraud | binary_classification.zip 的 train.json，4,000 通话 | 中文电话反诈。原始任务含音频，文本模型需要与通话 ID 对齐的转写或自行转写，并去掉答案提示。 | HF 受控访问，按数据使用条件 | [Hugging Face](https://huggingface.co/datasets/JimmyMa99/TeleAntiFraud) |
| 25 | PhiUSIIL Phishing URL | 公开带标签表，自划，按域名和站点家族分组 | 辅助 URL/网页风险判断。只向模型提供部署时可见的 URL 或属性，和整封邮件检测分别统计。 | UCI 数据页 CC BY 4.0 | [作者数据页](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset) |

### 4. 多意图、近义类别、拒识与多轮状态（6 项）

| 编号 | 数据集 | 可取的训练部分 | 转成哪种决策、怎样分组 | 许可或取用条件 | 发布来源 |
|---:|---|---|---|---|---|
| 26 | BANKING77 | train，10,003 条，77 类 | 与榜单同源。用混淆意图作难负例，例如转账未到和退款未到，留出集同时测全 77 类。 | CC BY 4.0 | [Hugging Face](https://huggingface.co/datasets/PolyAI/banking77) |
| 27 | CLINC150 + OOS | data_full.json 的 train＋oos_train，共 15,100 条 | 与榜单同源。150 类域内加域外，显式保留 OOS，让已知相邻意图与真正域外样本都出现。 | CC BY 3.0 | [GitHub](https://github.com/clinc/oos-eval) |
| 28 | MASSIVE | 优先 en-US/train 和 zh-CN/train | 60 类意图、跨语言并行请求。相同原始 id 的多语言版本同组，也可用已标注槽位生成条件判断。 | CC BY 4.0 | [Hugging Face](https://huggingface.co/datasets/AmazonScience/massive) |
| 29 | Bitext customer support | 发布的客服训练语料，自留模板组 | 用 instruction 预测 intent 或 category，示范回复与标签不进输入。与银行意图组合能增加客服场景。 | CDLA-Sharing-1.0 | [Hugging Face](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) |
| 30 | MultiWOZ | 固定版本的官方训练对话 | 用截至当前轮的历史预测对话行为、槽位是否齐全或下一步操作。按完整对话划分。 | 按所选版本及其数据许可 | [GitHub](https://github.com/budzianowski/multiwoz) |
| 31 | CrossWOZ | 中文跨域任务对话 train | 查酒店、餐厅或交通时判断该查询、追问还是切换服务。按对话 ID 划分，只给当前轮可见信息。 | 见原仓库与数据说明 | [GitHub](https://github.com/thu-coai/CrossWOZ) |

### 5. 自然语言推理、常识和因果选择（16 项）

| 编号 | 数据集 | 可取的训练部分 | 转成哪种决策、怎样分组 | 许可或取用条件 | 发布来源 |
|---:|---|---|---|---|---|
| 32 | ANLI | train_r1、train_r2、train_r3，共 162,865 条 | 与榜单同源。用对抗反例练三分类，分轮报告，避免只靠否定词判断矛盾。 | CC BY-NC 4.0 | [Hugging Face](https://huggingface.co/datasets/facebook/anli) |
| 33 | MultiNLI | train，392,702 条 | 多文体的蕴含、矛盾、中立。与 WANLI、翻译 NLI 合用时查原始 premise。 | 多种条款，按原文来源 | [Hugging Face](https://huggingface.co/datasets/nyu-mll/multi_nli) |
| 34 | SNLI | train，550,152 行，过滤无共识标签 | 基础三分类，可测单看假设的捷径。与 UNLI 重复的父样本同组。 | CC BY-SA 4.0 | [Hugging Face](https://huggingface.co/datasets/stanfordnlp/snli) |
| 35 | WANLI | 作者 train.jsonl | 人机协作构造、人工复核的 NLI 困难例，与 Open-Jev v1.1 的 WANLI 部分共用血缘。 | CC BY 4.0 | [Hugging Face](https://huggingface.co/datasets/alisawuffles/WANLI) |
| 36 | PAWS | labeled_final/train，49,401 条 | 判断两句话是否同义。高词汇重合的反例适合练实体与论元位置。 | 按原任务及文本来源 | [Hugging Face](https://huggingface.co/datasets/google-research-datasets/paws) |
| 37 | PAWS-X | 各语言 train，包括 zh | 用中英比较保持语义等价，训练翻译和英语 PAWS 按父例归并。 | 按原任务及文本来源 | [Hugging Face](https://huggingface.co/datasets/google-research-datasets/paws-x) |
| 38 | WinoGrande | winogrande_xl/train，40,398 条 | 与榜单同源，榜单取 validation。相近实体二选一，不同训练规模配置有嵌套。 | 以 WinoGrande 上游条款为准 | [Hugging Face](https://huggingface.co/datasets/allenai/winogrande) |
| 39 | HellaSwag | train，39,905 条 | 与榜单同源，榜单取 validation。真实事件前缀选四个结尾，按活动和上下文控制重复。 | 核原始视频描述及来源条款 | [Hugging Face](https://huggingface.co/datasets/Rowan/hellaswag) |
| 40 | SocialIQA | train，33,410 条 | 第三方外部常识。给社交场景选择动机、反应或影响，能补语用推断。 | 见上游数据许可 | [Hugging Face](https://huggingface.co/datasets/allenai/social_i_qa) |
| 41 | CommonsenseQA | train，9,741 条 | 在相近概念中选五个之一。可与 HellaSwag 区分短问题知识与事件延续。 | 卡标 MIT | [Hugging Face](https://huggingface.co/datasets/tau/commonsense_qa) |
| 42 | WIQA | train，29,808 条 | 让模型判断过程受到扰动后结果增加、减少或无影响，适合补因果方向。 | 见上游数据许可 | [Hugging Face](https://huggingface.co/datasets/allenai/wiqa) |
| 43 | e-CARE | 原始训练划分 | 两个事件候选，选择合理的原因或结果。训练时可用因果解释验标签，答题时只给问题信息。 | 见原仓库 | [GitHub](https://github.com/Waste-Wood/e-CARE) |
| 44 | LogiQA 2.0 | 训练文件，固定中文或英文任务 | 逻辑阅读选择，可训练必要条件、反驳和隐含前提。中英对应题按原题分组。 | 见原仓库及试题来源 | [GitHub](https://github.com/csitfun/LogiQA2.0) |
| 45 | QuaRTz | train，2,696 条 | 根据一段定性关系和新情境选结论，适合构造方向相反的配对例。 | CC BY 4.0 | [Hugging Face](https://huggingface.co/datasets/allenai/quartz) |
| 46 | QASC | train，8,134 条 | 结合两条事实选择科学问题答案。可对比提供完整证据与缺一条证据的任务。 | CC BY 4.0 | [Hugging Face](https://huggingface.co/datasets/allenai/qasc) |
| 47 | SciQ | train，11,679 条 | 带支持段和干扰项的科学选择题，作知识保持的小规模对照组。 | CC BY-NC 3.0 | [Hugging Face](https://huggingface.co/datasets/allenai/sciq) |

### 6. 检索、重排、工具路由与网页操作（10 项）

| 编号 | 数据集 | 可取的训练部分 | 转成哪种决策、怎样分组 | 许可或取用条件 | 发布来源 |
|---:|---|---|---|---|---|
| 48 | When2Call | train_sft 15,000 条，train_pref 9,000 条 | 与榜单同源。提取选择策略或 chosen/rejected，另保留 BFCL 原例 ID 去重。两配置可能表达同一问题。 | CC BY 4.0 | [Hugging Face](https://huggingface.co/datasets/nvidia/When2Call) |
| 49 | ToolACE | 发布的训练对话 data.json | 通过已验证的调用记录抽取应选工具，加入名字近似但参数不符的候选。多工具场景允许多个正确选择。 | Apache-2.0，合成对话 | [Hugging Face](https://huggingface.co/datasets/Team-ACE/ToolACE) |
| 50 | xLAM function-calling 60k | xlam_function_calling_60k.json，受控访问 | 把任务、可用工具和真实调用改成工具选择或参数合法性判断，按工具家族留出。 | 卡标 CC BY 4.0，接受访问条件 | [Hugging Face](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) |
| 51 | ToolRet-Training-20w | train，208,826 条 | query 已配 positive 与 negative，可直接做成对或列表排序，和榜单 ToolRet 评测查询另行比对。 | HF 卡未标统一许可，核上游来源 | [Hugging Face](https://huggingface.co/datasets/mangopy/ToolRet-Training-20w) |
| 52 | Home Assistant Requests | 训练请求，按场景或设备模板留出 | 把设备状态、用户请求和候选服务组成选择题，训练缺少参数时先追问。 | 见数据卡及其来源 | [Hugging Face](https://huggingface.co/datasets/acon96/Home-Assistant-Requests) |
| 53 | Mind2Web | 官方 train 的网页演示轨迹 | 状态和历史动作预测下一网页元素或操作，评测按网站或任务模板留出，避免只认元素名。 | 按项目数据使用条件 | [GitHub](https://github.com/OSU-NLP-Group/Mind2Web) |
| 54 | Amazon ESCI | examples 的 split=train，按 product_id 与 locale 连接产品 | 与榜单同源。判断精确匹配、替代、配套或无关，查询、商品族去重。 | 数据说明 Apache-2.0 | [GitHub](https://github.com/amazon-science/esci-data) |
| 55 | MIRACL | 有标注的语言 train qrels，包括中文 | 查询＋文章判相关或比较两篇文章。未标注不能直接当强负例。 | 卡标 Apache-2.0，核维基文档条款 | [Hugging Face](https://huggingface.co/datasets/miracl/miracl) |
| 56 | T2Ranking | queries.train.tsv、qrels.train.tsv、训练难负例 | 中文检索。用标注等级做 score 或选择最相关段落，区分人工等级与挖掘负例。 | Apache-2.0 | [GitHub](https://github.com/THUIR/T2Ranking) |
| 57 | MS MARCO Passage Ranking | 官方训练查询、qrels、triples | 英文大规模重排。与含 MS MARCO 的 RAGTruth 等来源比较 query 和 passage。 | 按 Microsoft 数据条款，研究用途另审 | [GitHub](https://github.com/microsoft/MSMARCO-Passage-Ranking) |

### 7. 软标签、偏好、内容安全与艺术判断（12 项）

| 编号 | 数据集 | 可取的训练部分 | 转成哪种决策、怎样分组 | 许可或取用条件 | 发布来源 |
|---:|---|---|---|---|---|
| 58 | UNLI | train.jsonl，区别于 validation/test | 由人给出假设成立概率，适合训练 noul 概率，与普通 NLI 硬标签配对做校准对照。与 SNLI 追溯父例。 | 镜像卡标 MIT，保留底层 SNLI 条款 | [Hugging Face](https://huggingface.co/datasets/Zhengping/UNLI) |
| 59 | HelpSteer2 | 训练文件 train.jsonl.gz | 人给回复打多个属性分，按属性单独构造有序 score，一条 prompt 的多个答案同组。 | CC BY 4.0 | [Hugging Face](https://huggingface.co/datasets/nvidia/HelpSteer2) |
| 60 | Stanford Human Preferences | 各版块 train.json | 给定同一讨论上下文二选一，原偏好来自社区投票。控制长度、发帖时间和版块捷径。 | 依原 Reddit 文本与项目说明 | [Hugging Face](https://huggingface.co/datasets/stanfordnlp/SHP) |
| 61 | Anthropic HH-RLHF | helpful / harmless 的 train.jsonl.gz | 同一对话比较 chosen/rejected。保留帮助性与无害性任务区别，做位置对称样本。 | 卡标 MIT | [Hugging Face](https://huggingface.co/datasets/Anthropic/hh-rlhf) |
| 62 | UltraFeedback | 作者发布的偏好训练池，按 prompt 自留 | 多回答、分维度教师反馈可做 score 或 pairwise，记录教师来源并过滤来源不清的记录。 | 卡标 MIT，另核教师及上游源 | [Hugging Face](https://huggingface.co/datasets/openbmb/UltraFeedback) |
| 63 | GoEmotions | simplified/train，需要投票时用其训练 ID 回连 raw | 多标签情绪拆成多个可同时为真的 noul。多人票数保留为分布，不把整张 raw 表混入。 | Apache-2.0 | [Hugging Face](https://huggingface.co/datasets/google-research-datasets/go_emotions) |
| 64 | Measuring Hate Speech | 发布的 train 标注表，按 comment_id 自留 | 多人对同一文本给多维序数标注，可学主观判断分歧。按评论计权，避免重复人数扭曲分布。 | CC BY 4.0 | [Hugging Face](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech) |
| 65 | WildGuardMix | train/wildguard_train.parquet | 区分请求有害、回复有害、是否拒绝三个标签，作为安全判断辅助域。 | 卡标 ODC-BY，核训练来源 | [Hugging Face](https://huggingface.co/datasets/allenai/wildguardmix) |
| 66 | Aegis Safety 2.0 | 官方 train 文件 | 按指定策略判安全或风险类别。与钓鱼邮件采用独立任务标签和独立验证集。 | CC BY 4.0 | [Hugging Face](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0) |
| 67 | BeaverTails | 例如 round0/30k/train，避免重复并入大版本 | 给定问答判是否安全及危害类别，与 tasksource 的 BeaverTails 分支去重。 | CC BY-NC 4.0 | [Hugging Face](https://huggingface.co/datasets/PKU-Alignment/BeaverTails) |
| 68 | Humicroedit | 官方训练部分，按原新闻标题分组 | 与榜单同源。用人工幽默评分或两版标题的相对偏好，保留平局或接近票。 | 依原数据及新闻文本条款 | [作者数据页](https://www.cs.rochester.edu/u/nhossain/humicroedit.html) |
| 69 | New Yorker Caption Contest | matching 配置的 train，排除评测 contest_number | 与榜单同源。用公开的文字描述配标题，跨配置仍按同一场比赛隔离。 | 卡标 CC BY 4.0，图文权利分别核 | [Hugging Face](https://huggingface.co/datasets/jmhessel/newyorker_caption_contest) |

### 8. 中文任务、表格数值和可执行决策（12 项）

| 编号 | 数据集 | 可取的训练部分 | 转成哪种决策、怎样分组 | 许可或取用条件 | 发布来源 |
|---:|---|---|---|---|---|
| 70 | OCNLI | 原生中文 train，约 5 万对 | 中文蕴含、矛盾和中立，训练集中仅部分有五人票，可单独保留软标签。 | 见作者数据说明 | [GitHub](https://github.com/CLUEbenchmark/OCNLI) |
| 71 | AFQMC | afqmc/train，34,334 条 | 中文金融短句语义等价，适合近义意图和细微条件差异。与 CJ 金融源做文本去重。 | CLUE 分任务核许可 | [Hugging Face](https://huggingface.co/datasets/clue/clue) |
| 72 | IFLYTEK | iflytek/train，12,133 条 | 长应用描述选多类标签，训练相似类别成组竞争，完整类别集合用于独立验证。 | CLUE 分任务核许可 | [Hugging Face](https://huggingface.co/datasets/clue/clue) |
| 73 | TNEWS | tnews/train，53,360 条 | 中文短标题主题分类。与客服意图分别计权，避免主题词替代意图。 | CLUE 分任务核许可 | [Hugging Face](https://huggingface.co/datasets/clue/clue) |
| 74 | C³ | c3/train，HF 配置 11,869 条 | 中文对话与短文阅读选择，加入符合常识但不符合文本的干扰项。 | CLUE 分任务核许可 | [Hugging Face](https://huggingface.co/datasets/clue/clue) |
| 75 | KUAKE-QIC | CBLUE 的 KUAKE-QIC 训练划分 | 中文医疗搜索意图，适合保持 CJ 医疗能力并训练问诊、诊断、治疗等近邻标签。 | 按 CBLUE 各任务授权 | [GitHub](https://github.com/CBLUEbenchmark/CBLUE) |
| 76 | FinQA | 官方 train，保留证据与程序 | 给真实财报的表格和文本选择数值或操作。可用程序核对答案，按报告分组。 | 核数据页及原财报条款 | [GitHub](https://github.com/czyssrs/FinQA) |
| 77 | TAT-QA | 官方 train，以表格/报告为组 | 给财报表格与段落，选数值、单位或证据跨度。不同数值干扰项应由同量纲生成。 | 按项目及原财报条款 | [GitHub](https://github.com/NExTplusplus/TAT-QA) |
| 78 | GSM8K | main/train，7,473 题 | 与榜单同源。只在训练池内生成数值干扰项，保留原题 ID，不加入榜单替换文件。 | MIT | [Hugging Face](https://huggingface.co/datasets/openai/gsm8k) |
| 79 | MBPP | full/train，374 道独立题 | 用训练题的函数与测试断言生成输出选择，过滤非确定性结果，排除 test 与 prompt。 | CC BY 4.0 | [Hugging Face](https://huggingface.co/datasets/google-research-datasets/mbpp) |
| 80 | CodeContests | data/train，按竞赛题分组 | 用题面、测试用例、正确/错误解构造判断。选可核验的执行结果，排除相同题目的跨平台副本。 | 卡标 CC BY 4.0，保留原题来源 | [Hugging Face](https://huggingface.co/datasets/deepmind/code_contests) |
| 81 | DeepMind Mathematics Dataset | 使用生成器的训练模块与新种子 | 生成日期以外的代数、比较、概率、单位等可验算题，按表达式结构和难度留出。 | 生成代码及数据分别按原项目说明 | [GitHub](https://github.com/google-deepmind/mathematics_dataset) |

## 组第一版训练集时的四个决定

1. **先建两份对照集。** 一份按上游来源分层，一份整源留出，例如留下独立邮件来源、完整语言或未见任务族。中文 CJ 验证和英文任务验证分开看。模型选择用这些留出集，公开 0.3 题集用来做最后的横向比较。
2. **沿着原始来源去重。** 每条保留 repo、revision、原始 split、父样本/文档 ID、任务类型、语言、标签来源、许可。先排除与榜单固定文件相交的父组，再做规范化文本、足够长的词片段和近重复核查。中文增加字符片段匹配。翻译、改写、反转选项、同文多个回答及多工具版本都跟随父组。13 个英文词的匹配只是可选筛查手段，模板和短句命中需要复核。[官方题集构成](https://github.com/apolinario/decision-index/blob/25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d/docs/suite.md)
3. **控制合集的重复计权。** Tasksource 的 default、full 和 filtered-full 属于同一发布，Open-Jev v1.1 的 WANLI 与独立 WANLI 有重合，JevEmbed 的来源表含程序化 Jev、Open-Jev、HelpSteer/HelpSteer2 等，校准集也有 GoEmotions、MASSIVE 派生题。先确定每个源的唯一训练入口，再按原始内容去重。训练标签、来源名、答案解释、标注 span 只留在监督侧。[Tasksource 数据卡](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions) · [JevEmbed 来源审计](https://huggingface.co/datasets/HIT-TMG/JevEmbed-Data/blob/main/docs/PROCESSING_REPORT.md) · [Open-Jev 来源](https://huggingface.co/datasets/ZefanCai/Open-Jev-v1.1)
4. **首轮用相同训练预算比较。** 保留相同的 CJ 回放、基座和优化设置，先比较“只补风险数据”和“只补英文广度”，再看两者合并。风险任务按正负类报告召回、精确率与误报率，并在保留真实类比例的集合上校准。多类任务并行测单组选择和当前淘汰赛，候选重排时同步重映射金标和概率。独立测试可以分清数据覆盖、置信度偏移与首轮淘汰带来的错误。

这份清单核查了发布页、划分、数据卡和所列模型的作者披露，并对本机固定版 RAGTruth、PhishNChips 的文件口径做了实查。其他候选整库与本机评测题的逐条去重、许可证逐来源复核，以及每个来源的实际训练增益，均为入库或实验阶段的待验项。
