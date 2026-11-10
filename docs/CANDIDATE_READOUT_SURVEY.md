# 决策模型的候选读出方式：公开做法调研与从头训练方案（2026-11-03）

背景：现有模型在最后一个位置读 A–Z 26 个字母的 logits（hmm 读法），超过 26 个候选要靠分组淘汰。下一轮打算从 Qwen3.5-4B 基座重新训练，借这个机会换掉读出层。

这次调研了两类资料：

- Decision Index 榜单前列约 40 个模型的模型卡；
- 此前的 [可变候选数设计调研](VARIABLE_CANDIDATE_DESIGN_RESEARCH.md)，里面有 Chinese-Jev、GLiClass、Qwen3-Reranker 等的源码分析。

## 1. 榜单模型实际怎么做

| 做法 | 单次上限 | 代表模型（公开指数） | 说明 |
|---|---|---|---|
| **A–Z 字母读出**（LM 头 logits） | 26 | StartLux 4B（67.8）、KnowLine 4B（66.8）、Jeffan 4B（64.9）、H2O Lightning、Standard One | 和我们现在一样；超过 26 个靠分组淘汰。StartLux 是每组 25 个、每组取前 3 进决赛，淘汰项共分 0.001 的概率 |
| **扩展编码 + 专用读出矩阵** | 255 | Perplexity Decider v1.2（"answer codes are single tokens, readout head has 255 rows"）、Sieve 31B v3（A…Z 后接 AA、AB…，共 255 个）、Fastino GLiDE（"255-way decision readout"）、Cerno 4B、Mica 4B、Amber 4B（A–Z 后接两字母编码，读出矩阵用编码 token 的词向量初始化） | **头部最常见**。读出层是一个 255×hidden 的小矩阵，只为已出现的编码取值后做 softmax |
| **指针头**（候选位置打分） | 无字母上限，只受上下文限制 | Verdin-4B v1.1（61.0）："A pointer head (dim 256) scores each option's end token against the `decide` token" | 每个候选末尾的隐状态，和最后的 decide 位置做点积 |
| 同一序列多题读出 | — | Perplexity Decider v1.2：每道题选项后放一个 marker（`<|fim_suffix|>`），一条序列答完所有问题 | 多题请求的成本约等于一题，题之间还能互相参考 |

榜单前列模型的另外两个共同点，与读出方式相互独立，但都和效果直接相关：

- **全注意力层改为双向**：StartLux（prompt 内双向）、Verdin（state 内按 8,192 分块双向）、CoCo（全注意力层去掉因果 mask）、Sieve。Qwen3.5 每 4 层里有 1 层全注意力，只放开这些层，线性注意力层保持因果。必须从训练第一步就带这个 mask。
- **消除位置偏差**：Sieve 和 H2O 推理时读两种顺序取平均；Amber、JPT 训练时打乱选项顺序。

## 2. 研究与开源界的做法

| 做法 | 是否有上限 | 代价 | 代表 |
|---|---|---|---|
| 字母或扩展编码读出 | 有（26 / 255） | 一次前向 | MMLU 式读出、上面的榜单模型 |
| 逐选项似然（选项文本的 log-prob 求和） | 无 | 长选项计算重，长度偏差大 | lm-eval-harness；榜单官方 `transformers` 引擎用选项 key 的 log-prob |
| 候选标记 + 共享打分头 | 无，只受上下文限制 | 一次前向；新头需要训练 | Chinese-Jev（`[MASK]` 位置 + 共享 MLP）、GLiClass/GLiNER（标签 token 表示与文本匹配）、Verdin 指针头 |
| 逐候选二分类（pointwise） | 无 | K 次前向 | Qwen3-Reranker、NLI 零样本分类 |
| 标签向量双塔 | 无 | 一次编码 + 向量匹配，交互弱 | SetFit、检索式分类 |
| 分组淘汰 | 无（外层编排） | 多轮，正确项可能被淘汰；概率是合成的 | StartLux、我们现在的 `/v1/tournament` |

## 3. 三个方案对比（针对本项目从头训练）

| | A. 扩展编码（255 行读出矩阵） | B. 指针头（候选标记打分） | C. 保持 A–Z + 分组淘汰 |
|---|---|---|---|
| 单次上限 | 255 | **无**（只受 16k 上下文限制） | 26 |
| 改动量 | 小：提示词换编码，读出层换成 255×D 矩阵 | 中：提示词插入标记、记录位置，新增打分头 | 无 |
| 与 LM 先验的衔接 | 好：读出行用编码 token 的词向量初始化，开局就有意义 | 新头随机初始化，必须训练（从头训练本来就要训） | — |
| 候选语义 | 模型要把编码和候选对上，标签越多越难 | 每个候选的分数直接来自它自己的表示，**与编号无关** | 同 A |
| 位置偏差 | 有字母偏好，需打乱顺序或读两遍 | 没有字母，只剩位置偏差，训练打乱即可 | 有 |
| 多题一条序列 | 可以（每题一个 marker） | 可以（每题的 decide 标记 + 选项标记） | 可以 |
| 榜单验证 | Perplexity、Sieve、GLiDE、Cerno、Mica 都在用 | Verdin（4B 第 6，61.0） | StartLux 4B（67.8） |

**结论：采用 B（指针头）作为主读出，同时保留 A 的编码作为提示里的候选编号。**

- 候选仍写成 `A: key — 描述` 这样的行，超过 26 个后接 AA、AB…，便于模型和人对照。但分数来自每行末尾标记位置的隐状态，不靠编号 token。这样单次上限只受上下文限制，CLINC150 的 151 类、IFLYTEK 的 119 类都能一次读完。
- 打分方式：`score_i = (W_c h_i) · (W_q h_q) / √d + b`，h_i 是候选 i 末尾标记的隐状态，h_q 是最后 decide 位置的隐状态。h_q 看过全部候选，负责比较；h_i 只看到它之前的内容。这与 Verdin 相同，维度 256。可选再加一层小 MLP。
- `noul` 也当作两个候选（否/是）读，`score` 的档位按顺序当候选，有序损失沿用现有 RLCD 里的 RPS。
- 训练时打乱 choice 和 noul 的选项顺序，档位不打乱。

**分两期上线，降低风险：**

1. **第一期**：指针头 + 因果注意力 + 打乱顺序 + 新数据。改动集中在读出层，能直接和现有字母读出做对照实验。可选一个小 A/B：同一批数据，字母读出 vs 指针头，在 0.8B 上各训一轮比较。
2. **第二期**：全注意力层开 state 内双向，多题一条序列读出。这两项都要求从第一步就按新 mask 训练，所以等第一期稳定后再整体重训。

## 4. 对现有代码的影响

- `decision/configuration.py`：新增 `readout: "letters" | "pointer"`、标记 token、头的维度；`max_options` 对 pointer 不再限制为 26。
- `decision/processing.py`：pointer 布局的提示词逐段编码，记录每个候选标记和 decide 标记的位置。标记用 `<|object_ref_end|>` 这类单 token 特殊符号，按位置记录，不靠搜索，所以用户文本里即使出现同样的字符串也不会读错。
- `decision/modeling.py`：新增 `PointerHead`；`score_decisions` 按 readout 分派，输出仍是 `[B, K_max]`、无效位置填 −1e4。这样训练、评估、温度拟合、服务端多数调用不用改。权重以 `pointer_head.*` 和 span head 一样存进同一个 safetensors。
- 训练脚本：头的梯度需要跨卡同步（参考 `train_multitask_ddp.py` 的手动同步）；放开 CJ 专用的数据量检查，改为按 manifest 读取。
- 推理：Transformers 后端和 Decision Index 引擎走 pointer 时，跳过分组淘汰；GGUF/llama.cpp 读不到自定义头，第一期 GGUF 版继续用字母模型。
