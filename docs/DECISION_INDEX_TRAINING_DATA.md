# Decision Index 训练数据调研（2026-11-02）

## 1. 官方有没有训练集？

**没有。** 官方仓库 [apolinario/decision-index](https://github.com/apolinario/decision-index) 只提供评测题集，以及重建题集的脚本：

- 公开题集来自 43 个上游数据集的 **test 或 validation 划分**，另有 1 个官方自建基准（Home appliances）。
- WinoGrande 和 HellaSwag 用的是 validation，其余基本都是 test。
- 私有测试（正式分 Full 的 100%）不公开，也不提供给任何人。

官方规则中和训练数据相关的有三条：

- **可以用上游的 train 划分训练**，但最好在模型卡里写明。多个上榜模型都这样做了，例如 Cerno、Verdin、Amber。
- **私有测试一旦发现训练数据重叠**，那一项就不计分，榜单会标注，但不说是哪一项。
- **BBH 带 BIG-bench canary，官方明确写了 "never train on it"**。可以按 BBH 的格式自己生成题目，Amber 和 Cerno 就是这样做的，但不能用 BBH 原题。

## 2. 头部 4B 模型用了什么数据（来自它们的模型卡）

| 模型 | 公开指数 | 训练数据 |
|---|---:|---|
| StartLux 4B | 67.8 | 两阶段：约 270 万 + 170 万条决策样例；4 次训练结果做权重平均；卡上承认按公开分选过训练配方 |
| Senova-4B v2 | 66.9 | 约 180 万条，带选项级软标签；卡上承认训练源里有来自测试集的数据 |
| KnowLine 4B | 66.8 | 自建数据，迭代训练；Decision Index 格式数据做过去重；不使用 Jev 输出作为标签 |
| Verdin-4B v1.1 | 61.0 | 17.9 万条记录（30.7 万道题）：各基准 train 划分（每个来源最多 1,000 条）+ 560 个 tasksource 来源 + 公开人工标注数据 + 中文数据 + 安全类数据 + 程序生成题 |
| Perplexity Decider Flash 4B | 60.1 | 从自家 27B 决策模型**蒸馏**整个概率分布；"大部分提升来自 tasksource 数据" |
| Amber 4B / Amber Hybrid | 52.1 / 59.9 | `tasksource/tasksource-jev-typed-decisions` 13.7 万条 + 各基准 train 划分 + GSM8K/MetaMathQA + 选择题知识集；**训练时打乱选项顺序** |
| Cerno 4B | 57.6 | 公开指令/分类混合数据 + 18 个基准的 train/dev 划分 + 仿 GSM8K/BBH/CRUXEval/CLadder 格式的程序生成题 |
| Standard One 3B | 58.1 | 主要是自建合成题；**修过选项位置和标签比例的漂移** |

**共性**：

1. 都用了**英文的类型化决策数据**，主要是 tasksource。
2. 大多用了**各基准的 train 划分**。
3. 都按 13 词重叠之类的规则和测试集做过**去重**。
4. 多数专门处理过**位置和标签偏差**。

## 3. 按我们的弱项，对应可用的训练来源

下表主要整理自 Amber 0.4B 和 Verdin 卡里的训练数据表。

| 基准（我们的分数 → 4B 中位数） | 可用训练来源 | 许可 |
|---|---|---|
| RAGTruth（14 → 72） | `ParticleMedia/RAGTruth` train 划分 | MS MARCO 上下文仅限研究；Yelp 上下文禁止再分发 |
| PhishNChips（50 → 62） | `AreLit/PhishNChips` 中未出题的邮件；另有安全、钓鱼类数据集 | 查看各自条款 |
| BANKING77（72 → 82） | `mteb/banking77` 或 `legacy-datasets/banking77` train | CC BY 4.0 |
| CLINC150（79 → 88） | `clinc/oos-eval` train + validation（含 out-of-scope） | CC BY 3.0 |
| WinoGrande（68 → 75） | `allenai/winogrande` train | 数据集条款 |
| HellaSwag（80 → 85） | `Rowan/hellaswag` train | 数据集条款 |
| ANLI（54 → 60） | `facebook/anli` train_r1 至 train_r3 | **CC BY-NC 4.0（非商用）** |
| NLI4CT、ContractNLI、VAST、ACOS、iSarcasmEval | 各自的 train/dev 划分 | 数据集条款 |
| When2Call（59 → 72） | `nvidia/When2Call` 训练偏好对 | CC BY 4.0 |
| API-Bank、BFCL、ToolRet | API-Bank lv1/lv2 训练对话；`mangopy/ToolRet-Training-20w`（20.9 万条）；`Team-ACE/ToolACE` | 数据集条款 |
| HoVer（64 → 65） | HoVer train 声明 + 官方 Wikipedia 库 | CC BY-SA 4.0 |
| Amazon ESCI（38 → 46） | ESCI 训练对 | Apache-2.0 |
| Humicroedit、New Yorker、cfcolor、Habermas | 各自的 train 划分（cfcolor 只用训练调色板） | 数据集条款 |
| BPoMP | `Yorth/limerick-topic` 打油诗，仿 BPoMP 方式扰动 | 数据集条款 |
| POP909 | POP909-CL 中官方没抽到的 115 首歌，可做 12 个移调 | 仅限研究 |
| Home appliances（33 → 23，我们已高于中位数） | `acon96/Home-Assistant-Requests` + 自己生成的家居场景 | MIT |
| CLadder、BBH、CRUXEval | **自己按格式生成**；不能用原题，BBH 有 canary | — |
| GSM8K | `openai/gsm8k` train + `meta-math/MetaMathQA` | MIT |
| 知识类（MMLU、ARC、GPQA） | ARC、OpenBookQA、SciQ、QASC、CommonsenseQA 的 train；MMLU dev | 数据集条款 |

## 4. 通用的 Jev 风格训练集（Hugging Face）

| 数据集 | 规模 | 许可 | 说明 |
|---|---|---|---|
| [tasksource/tasksource-jev-typed-decisions](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions) | **631 万行**，144 GB | other（按各来源条款） | **最重要**。数百个 tasksource 任务转成 choice/noul/score 格式，标签来自原始标注。Amber、Verdin、Perplexity 都用了 |
| [tasksource/procedural-typed-decisions](https://huggingface.co/datasets/tasksource/procedural-typed-decisions) | 457 MB | Apache-2.0 | 程序生成，一个状态配多道题，答案可计算 |
| [tasksource/synthetic-typed-decisions](https://huggingface.co/datasets/tasksource/synthetic-typed-decisions) | 16 MB | Apache-2.0 | LLM 写的邮件、聊天、报告等，判断意图、语气、紧急度 |
| [ZefanCai/Open-Jev-v1.1](https://huggingface.co/datasets/ZefanCai/Open-Jev-v1.1) | 32.7 万条（训练 14.7 万） | other | 含**中文**；Open-Jev 27B 的训练混合（可再分发部分） |
| [ZefanCai/Open-Jev](https://huggingface.co/datasets/ZefanCai/Open-Jev) | 87 MB | CC0 | 12 个数据配置，含中文、土耳其语 |
| [decisionmodels/sn92-generalist-train-v1](https://huggingface.co/datasets/decisionmodels/sn92-generalist-train-v1) | 11 万条 | CC BY 4.0 | 合成数据，27 种语言（含中文），10 个类别 |
| [HIT-TMG/JevEmbed-Data](https://huggingface.co/datasets/HIT-TMG/JevEmbed-Data) | 160 万训练 + 6.6 万测试 | Apache-2.0 / CC0 / CC BY 4.0 | 决策格式，可直接解析 |
| [kshitijthakkar/nirnaya-jevstyle-tasksource-v2](https://huggingface.co/datasets/kshitijthakkar/nirnaya-jevstyle-tasksource-v2) | 26 万条 | Apache-2.0 | 上面 procedural 数据的 ChatML 转换版 |
| [n4ze3m/typed-decisions-synth](https://huggingface.co/datasets/n4ze3m/typed-decisions-synth) | 7,414 个案例 / 2.6 万题 | MIT | 带软标签，149 个领域 |
| [m-lagnajit/calibrated-typed-decisions](https://huggingface.co/datasets/m-lagnajit/calibrated-typed-decisions) | 2.2 万条 | CC BY 4.0 | 人工金标 + 校准软标签 |
| [samatv256/jev-decisions-v1](https://huggingface.co/datasets/samatv256/jev-decisions-v1) | 1,200 万条，22 GB | CC BY 4.0 | 智能体决策：工具选择、路由 |
| [JevAgentHarness/AgentDecision-train](https://huggingface.co/datasets/JevAgentHarness/AgentDecision-train) | 4 MB | CC BY-NC 4.0 | 智能体下一步决策（非商用） |
| [mangopy/ToolRet-Training-20w](https://huggingface.co/datasets/mangopy/ToolRet-Training-20w) | 20.9 万条 | 未标注 | 工具检索训练集 |

**用 Jev 输出做标签的蒸馏数据，需谨慎使用**：

- 例如 `SargeDev/jev-distill-corpus-v3`（74 万行）、`MaziyarPanahi/Jev-RAG-Claim-Verification-130K`、`MaziyarPanahi/Jev-Safety-Refusal-Decisions-150K`、`gyanbhambhani/oss-jev-distill`、`jaeyong2/jev-decision-claude-distill`。
- 这些数据的标签或参考答案来自 Jev 1.13 或 Claude 的输出。
- TypeSafe 和 Anthropic 的服务条款可能限制用输出训练竞品模型。KnowLine 就特意声明没有使用 Jev 输出。使用前需确认条款。

## 5. 开源训练代码（GitHub）

- [sileod/tasksource](https://github.com/sileod/tasksource)：tasksource 数据集合与预处理框架，BSD-3-Clause。
- [Mapika/decider](https://github.com/Mapika/decider)：Qwen3.5 微调的决策模型系列，Apache-2.0。Cerno 等沿用了它的提示模板和标签读取方式。
- [jaredpalmer/kev](https://github.com/jaredpalmer/kev)：基于 Qwen3.5/3.8 的 Jev 风格模型，可自己训练，Apache-2.0。Verdin 用了它的生成器。
- [TianyuCodings/NanoJev](https://github.com/TianyuCodings/NanoJev)：Jev 的小型复现，含端到端训练流程，MIT。
- [bespokelabsai/nimble](https://github.com/bespokelabsai/nimble)：决策数据整理与评测。
- [AnotiaWang/awesome-decision-models](https://github.com/AnotiaWang/awesome-decision-models)：决策模型资源汇总。

## 6. 对我们的建议

1. **先补最大的两个缺口**：
   - RAGTruth train；
   - 钓鱼、安全类正负平衡数据，例如 Verdin F9 列出的 BeaverTails、Aegis 2.0、Nemotron-Safety-Guard（含中文）。
2. **加入 tasksource-jev-typed-decisions 的子集**：每个来源取几百条，像 Amber 一样每源最多 400 条。这是头部模型的共同底座，能整体提升英文 NLI、常识和分类能力。
3. **加入各基准的 train 划分**：BANKING77、CLINC150、WinoGrande、HellaSwag、When2Call、API-Bank 等，每源上限约 1,000 条，避免过拟合公开题集。
4. **训练时打乱选项顺序**：二选一的题再加入正反两种顺序，从根上消除位置偏差。之后可以去掉推理时的去偏，节省一半计算。
5. **候选数扩到 26 以上，并加入多子问题的样例**：改善 BANKING77、CLINC150 和 ACOS 这类结构化任务。
6. **严格去重**：对照 0.3 题集的全部 140,620 条请求做 13 词重叠检查，在模型卡里声明用了哪些 train 划分。
7. **留意非商用许可**：ANLI、AgentDecision-train 是非商用，RAGTruth 部分上下文仅限研究，加入训练前要确认许可是否允许。
8. **按私有分而不是公开分选模型**：官方正式分看私有测试。只追公开分的模型，正式分往往掉得多，例如 StartLux 从 67.8 掉到 54.8，Senova 从 66.9 掉到 51.0。可以像 KnowLine 和 Verdin 那样，自建"同技能"和"新领域"两类留出集来挑模型。
