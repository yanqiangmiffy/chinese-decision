# 可变候选数决策模型：开源实现调研与本项目改造建议

调研日期：2026-10-18。对象：StartLux-Decision、hmm、Chinese-Jev、GLiClass、Qwen3-Reranker、Transformers 的 NLI 零样本分类流程，以及本项目 chinese-decision。

本文围绕三个问题：如何超过 26 个候选、能否复用已有权重而不训练、怎样获得有明确含义的概率。实现结论来自固定提交的源码；文中另行标注工程推断与尚未验证的效果。没有加载外部模型权重，没有进行中文准确率、校准或吞吐基准测试。

## 1. 结论与建议

1. **StartLux 没有消除单次读取的 26 项上限。** 它仍读取 A–Z 的词表输出；大候选 `choice` 通过分组初选、决选支持。当前代码另将 `score` 限定为 2–10 档。因此，接口接受大候选集合，不代表模型在一次前向中直接计算了全部类别的联合分布。详见第 3 节。
2. **真正不绑定字母表的方案已经有开源实现。** Chinese-Jev 从候选标记位置提取表示，用共享 MLP 输出一个分数；GLiClass 将输入中的标签表示与文本表示匹配。候选维度来自输入，仍受序列预算和实现配置约束。详见第 4 节。
3. **不训练当前权重也能支持更多候选。** 可采用分组决选，或复用 `noul` 对每个候选做二分类。后者与 Qwen3-Reranker、NLI 零样本分类的输入组织方式有共同点，但业务效果需要另行验证。
4. **新增随机初始化的共享评分头需要训练；等价改写已有输出头是例外。** 将二分类输出的两行权重相减，可以得到数学上等价的单标量评分头；这不会凭空增加联合比较候选的能力。详见第 5 节。
5. **对当前项目，建议先验证不训练的推理层方案，再决定是否训练新架构。** 分别比较逐候选二分类、分组决选以及一个现成的动态标签模型，测量质量、延迟与概率含义，不直接启动全量重训。

## 2. 先区分三种“没有类别数量限制”

| 说法 | 实际含义 | 能否据此判断性能或校准可靠 |
|---|---|---|
| 接口可接收任意长度的候选列表 | 服务可以分批、分组、检索后再决选 | 不能；需要看内部处理及淘汰过程 |
| 输出结构不固定类别数 | 共享评分层对 K 个输入候选产生 K 个分数 | 不能；K 可能超出训练分布，输入也可能被截断 |
| 新增类别不需要业务微调 | 使用预训练的标签理解、NLI 或相关性模型处理新标签 | 不能；标签描述、语言、领域都会影响结果 |

“不需要训练”也要区分：**保留当前模型权重**、**换用别人已经训练好的权重**、**冻结骨干但训练新头**。前两种不需要自己做梯度训练，第三种仍然需要训练。

## 3. StartLux-Decision：字母读出 + 大候选淘汰流程

### 3.1 单次读出仍是 A–Z

在提交 `83fe6c7d61a687a6e17f118b4f7d8f429f5d3a01` 中，`prompt.py` 定义 `LETTERS` 为大写字母、`MAX_OPTIONS = 26`、`MAX_LEVELS = 10`。校验路径分别约束单次候选数和评分档位数；`score` 的限制比本项目还小。[源码：常量与校验](https://github.com/StartLuxLabs/StartLux-Decision/blob/83fe6c7d61a687a6e17f118b4f7d8f429f5d3a01/startlux_decide/prompt.py#L26-L103)

Transformers 后端取最后输入位置的隐藏表示，再乘以语言模型输出层中候选字母对应的权重行。它没有在每个候选位置增加一个共享候选 MLP。随后按题型温度，对本题字母分数做 softmax。[源码：`_readout`](https://github.com/StartLuxLabs/StartLux-Decision/blob/83fe6c7d61a687a6e17f118b4f7d8f429f5d3a01/startlux_decide/backends/transformers.py#L95-L109)、[源码：概率归一化](https://github.com/StartLuxLabs/StartLux-Decision/blob/83fe6c7d61a687a6e17f118b4f7d8f429f5d3a01/startlux_decide/readout.py#L18-L23)

### 3.2 超过 26 个 choice：初选与决选

默认流程为：

1. 将候选按原顺序划为近似等大的组，每组最多 25 项。
2. 每组分别调用模型，保留得分最高的 3 项。
3. 如果入围总数超过 26，再按初选中的组内概率排序，截取 26 项。
4. 对入围项再调用一次模型决选。

`allow_none` 开启时，决选还要给“以上皆非”预留一个字母位置，最多容纳 25 个真实候选。该控制针对 `choice`，不把 `score` 扩展成大档位淘汰流程。[源码：准备分组与运行决选](https://github.com/StartLuxLabs/StartLux-Decision/blob/83fe6c7d61a687a6e17f118b4f7d8f429f5d3a01/startlux_decide/pipeline.py#L219-L234)、[源码：运行路径](https://github.com/StartLuxLabs/StartLux-Decision/blob/83fe6c7d61a687a6e17f118b4f7d8f429f5d3a01/startlux_decide/pipeline.py#L311-L365)

例如 100 个候选、默认配置、不启用思考和“以上皆非”时：4 组各 25 项 → 每组保留 3 项 → 12 项决选，共 **5 个问题级模型评估**。初选可以并发，5 次评估不等于必须串行等待 5 次；也不等于只计算了一条输入。[配置与接口说明](https://github.com/StartLuxLabs/StartLux-Decision/blob/83fe6c7d61a687a6e17f118b4f7d8f429f5d3a01/docs/api.md#L45-L55)

### 3.3 全类别概率有一部分来自人工分配

默认 `residual = 0.001`：入围项分配决选概率乘以 0.999；所有未入围项共同分享 0.001，按各自初选概率的相对比例分配。最终再归一化。[源码：`knockout_combine`](https://github.com/StartLuxLabs/StartLux-Decision/blob/83fe6c7d61a687a6e17f118b4f7d8f429f5d3a01/startlux_decide/readout.py#L124-L132)

**本文分析：** 这个输出可以覆盖所有原始类别并使概率和为 1，但不是对全体 K 类做一次联合评分后得到的原始分布。`residual` 是配置值；组内 softmax 的分母也不同，跨组比较初选概率可能产生偏差。正确类别如果初选被淘汰，就没有机会参加决选。在默认设置下，它所得的合成概率不超过 0.001，单样本 NLL 至少约为 6.91 nats。该数值是由残余质量上界推导出的性质，不是实测成绩。

**可借鉴之处：** 大候选请求的编排、并发初选、候选 ID 映射、调用次数与输入 token 统计。迁移时应明确标注输出为“淘汰流程合成分布”，并测量入围召回率、换组稳定性和校准。

### 3.4 双向注意力不等于动态候选评分头

Instant 模型的全注意力层允许同一输入内双向可见；Gated DeltaNet 线性注意力及短因果卷积仍保留因果计算。AutoThink 使用因果路径，并可在不确定时生成思考后重新读出。因此，“使用双向注意力”和“使用候选位置共享打分”是两个独立设计维度。[源码：注意力实现说明](https://github.com/StartLuxLabs/StartLux-Decision/blob/83fe6c7d61a687a6e17f118b4f7d8f429f5d3a01/startlux_decide/backends/bidirectional.py#L1-L19)、[后端说明](https://github.com/StartLuxLabs/StartLux-Decision/blob/83fe6c7d61a687a6e17f118b4f7d8f429f5d3a01/startlux_decide/backends/transformers.py#L1-L12)

源码说明这些 Instant 权重按该注意力方式训练；不能据此推断，把本项目的因果模型直接切换注意力后就能获得相同效果。该参考实现还明确检查 Transformers 5.8.x；本项目依赖范围不能代表环境恰好满足其要求。[源码：版本检查](https://github.com/StartLuxLabs/StartLux-Decision/blob/83fe6c7d61a687a6e17f118b4f7d8f429f5d3a01/startlux_decide/backends/bidirectional.py#L33-L45)

同一快照下的旧版 `v1/startlux_decision/model.py` 也保留字母读出和大候选淘汰实现。本文不把它与当前引擎的 AutoThink 能力混为一谈。[旧版源码](https://github.com/StartLuxLabs/StartLux-Decision/blob/83fe6c7d61a687a6e17f118b4f7d8f429f5d3a01/v1/startlux_decision/model.py#L618-L640)

## 4. 其他开源方案怎样设计

### 4.1 hmm：与当前项目同类的字母读出基线

`training/train.py` 定义 A–Z，检查每个字母必须是单 token，从最后位置取这些字母的 logits，并屏蔽无效候选。这种实现简单，能复用语言模型的词表输出层，但本身没有动态候选位置评分机制。适合用作现有方案的对照，不能作为天然超过 26 项的证据。[hmm 训练源码](https://github.com/n4ze3m/hmm/blob/78d8d4a62a41857ce67d903aadf504580feea832/training/train.py#L100-L157)

### 4.2 Chinese-Jev：候选标记 + 双向编码器 + 共享 MLP

它在输入中给每个候选安排一个 `[MASK]` 标记。模型采用双向编码器，可叠加带题型信息的 Transformer 层，然后收集标记位置的表示，使用同一组 MLP 参数逐候选输出标量。候选维度由标记数量决定，无效位置通过 mask 排除。这里的 `[MASK]` 是候选定位标记，不能把它理解成逐个生成类别名称。[序列布局](https://github.com/gulucaptain/Chinese-Jev/blob/aed6193be5697488c8d0c6a77cc0c7e297329e61/src/chinese_jev/sequence.py#L1-L11)、[模型与评分头](https://github.com/gulucaptain/Chinese-Jev/blob/aed6193be5697488c8d0c6a77cc0c7e297329e61/src/chinese_jev/modeling.py#L22-L55)

这正是此前讨论的“同一序列内，在候选位置提取表示，再用共享层评分”。把它移植到本项目需要训练新头；Qwen 的因果结构也不能直接当作双向编码器使用。

没有字母表限制仍不代表输入无限。该项目存在 `head_max_len`、单候选 token 截断和总长度预算；数据编码会检查题头是否容纳候选，避免候选描述退化或标记丢失。因此部署时必须检查候选完整性，而不是只放宽一个 K 的校验。[序列预算实现](https://github.com/gulucaptain/Chinese-Jev/blob/aed6193be5697488c8d0c6a77cc0c7e297329e61/src/chinese_jev/sequence.py#L124-L146)、[编码时的预算检查](https://github.com/gulucaptain/Chinese-Jev/blob/aed6193be5697488c8d0c6a77cc0c7e297329e61/src/chinese_jev/encoding.py#L58-L107)

### 4.3 GLiClass：动态标签表示与文本表示匹配

GLiClass 的统一编码路径识别输入中的类别标记，提取类别表示，并与文本表示一起送入共享 scorer。`scorers.py` 中可见输出单标量的 MLP，以及其他匹配结构；模型不要求给每个业务类别单独建立一个输出节点。[标签表示提取](https://github.com/knowledgator/gliclass/blob/6a00f8067648f25aa0f4f477d167c5f391cb4165/gliclass/model.py#L176-L236)、[共享 scorer](https://github.com/knowledgator/gliclass/blob/6a00f8067648f25aa0f4f477d167c5f391cb4165/gliclass/scorers.py#L73-L96)

需要特别核对版本配置：当前快照里仍能看到默认 `max_num_classes=25`，但 `max_labels_alloc` 默认为 `dynamic`，pipeline 会按实际标签数计算本批次分配量。数字 25 不能单独证明存在 A–Z 式固定输出上限；使用旧 checkpoint 或固定分配配置时，也不能直接假设能处理任意标签数。[配置定义](https://github.com/knowledgator/gliclass/blob/6a00f8067648f25aa0f4f477d167c5f391cb4165/gliclass/config.py#L35-L53)、[动态分配逻辑](https://github.com/knowledgator/gliclass/blob/6a00f8067648f25aa0f4f477d167c5f391cb4165/gliclass/pipeline.py#L379-L384)

输出语义也明确分开：单标签对候选 logits 做 softmax，多标签对每项做 sigmoid。最新源码还包括支持 Qwen3/Qwen3.5 的 `decoder-kv` 路径及文本缓存接口，是“复用文本计算，再处理标签”的研究方向；这套 scorer 自身仍有需要学习的参数，不能无训练套在本项目权重上。[概率后处理](https://github.com/knowledgator/gliclass/blob/6a00f8067648f25aa0f4f477d167c5f391cb4165/gliclass/pipeline.py#L282-L311)、[decoder-kv 模型](https://github.com/knowledgator/gliclass/blob/6a00f8067648f25aa0f4f477d167c5f391cb4165/gliclass/model.py#L981-L1072)

使用别人训练好的 GLiClass checkpoint 可以不自行训练，但属于更换或新增模型。本文未验证某一 checkpoint 在中文决策、101 档 score 或大 K 下的效果，也不直接采用 README 的速度宣传作为本项目性能预测。

### 4.4 Qwen3-Reranker：每个候选做一次 yes/no 判断

官方 Transformers 示例把 query 与 document 配对，读取末位置 `yes`、`no` 两个 token 的 logits，做二类归一化后取 yes 分数。多个候选可以批量送入，数量不绑定 A–Z；上下文会在多个配对中重复，不能把一个 batch 的调用理解成一条联合序列的计算成本。[官方示例](https://github.com/QwenLM/Qwen3-Embedding/blob/44548aa5f0a0aed1c76d64e19afe47727a325b8f/examples/qwen3_reranker_transformers.py#L35-L87)

对本项目的启发是：把“候选是否符合问题和上下文”转换为现有 `noul`，复用已训练的 A/B 二分类能力，无需增加随机评分头。若改用 Qwen3-Reranker 的权重，则是采用一个已经训练好的相关性模型。检索相关性分数不自动等价于互斥类别的业务后验概率；中文分类效果、拒识与校准仍需独立测量。

### 4.5 Transformers NLI 零样本分类：固定三类头，动态业务类别

该 pipeline 将每个业务标签转成一条 hypothesis，与原文构成 premise/hypothesis 配对。底层 NLI 模型预测的是蕴含、矛盾等固定关系，外层业务类别由输入提供，所以业务类别数可以变化。[配对构建与模型要求](https://github.com/huggingface/transformers/blob/d4804f9d76b8c071b993d813e9e87067b780cbee/src/transformers/pipelines/zero_shot_classification.py#L12-L48)

单标签路径在各候选的 entailment logits 之间归一化；多标签路径对每个候选的 entailment/contradiction 分数独立归一化。它不是简单把一组已经得到的 yes 概率相加归一化。使用时需要已经训练好的 NLI 模型，不能只给当前 Qwen 换个 pipeline 名称。[源码：`postprocess`](https://github.com/huggingface/transformers/blob/d4804f9d76b8c071b993d813e9e87067b780cbee/src/transformers/pipelines/zero_shot_classification.py#L212-L230)

## 5. “共享评分头必须训练”的准确边界

### 5.1 新能力：需要补充训练

以下改造改变了模型的计算或任务含义：随机初始化一个 MLP；把候选末位置直接送入新头；让一个序列内的全部候选联合交互；从因果计算改成双向计算。不能因为骨干已有权重，就认为新增部分已经学会评分。

可以先冻结骨干只训练小头，再评估是否需要解冻 LoRA；这是比从头训练便宜的实验路径，但训练数据、效果和资源费用仍需实测。

### 5.2 等价改写：可以不训练，但没有增加能力

以下是根据线性词表输出与二类 softmax 作出的**数学推导**。设同一个输入的末位置表示为 h：

```text
z_false = W_false · h + b_false
z_true  = W_true  · h + b_true

s = (W_true - W_false) · h + (b_true - b_false)
P(true) = sigmoid(s)
```

因此，可以用旧输出层的两行差初始化一个单标量头，保持与原二分类完全相同的函数。若原模型使用温度 T，则为 `sigmoid(s / T)`。本项目 `noul` 的候选顺序是 false、true，映射为 A、B，所以这里对应 `W_B - W_A`。部署时还要保留相同提示、分词、末位置选择、LoRA 合并状态和注意力方式，并检查精度与量化误差。[本项目候选顺序](https://github.com/yanqiangmiffy/chinese-decision/blob/718ea62efd599070bf4b8f5b6d3b7c6822db9b30/decision/schema.py#L28-L34)

这只把“每个候选做一次二分类”的输出实现改写了：仍需构造 K 个判断输入，仍没有得到“一条输入内对 K 个候选联合评分”的新能力。为这个等价变换自行增加专用头，可能还需要额外适配导出和推理引擎；第一版直接复用原有二分类输出更简单。

## 6. 方案对照：训练、计算与概率含义

下表的“无需训练”指采用该流程时是否必须自行做梯度训练，不表示原作者的模型没有训练过。

| 方案 | 超过 26 的方式 | 自行训练要求 | 主要计算 | 输出的含义 |
|---|---|---|---|---|
| 当前 chinese-decision / hmm | 原生不支持 | 原样使用无需 | 每题一条序列 | 本题字母候选上的分布 |
| StartLux 大候选 choice | 分组初选 + 决选 | 编排可不训练 | 多个组序列 + 决选序列 | 决选与残余质量合成分布 |
| 当前模型逐候选 noul | K 个二分类问题 | 无需改权重 | K 个配对序列，可分批 | 每项独立的二分类概率 |
| Qwen3-Reranker | K 个 query/document 配对 | 使用现成权重无需 | K 个配对序列，可分批 | 各配对的 yes/no 相关性分数 |
| NLI 零样本分类 | K 个文本/标签假设配对 | 使用现成 NLI 权重无需 | K 个配对序列 | 单标签归一化或独立关系概率 |
| Chinese-Jev 候选标记头 | K 个候选标记与共享 MLP | 迁移到本项目需训练；用现成权重另评估 | 候选放入同一序列 | 有效候选上的分布 |
| GLiClass | 标签表示与文本表示匹配 | 现成权重可直接评估；迁移其新头需训练 | 随选用的编码架构变化 | softmax 单标签或 sigmoid 多标签 |

对所有方案都应区分：**分数可用于排序、归一化后和为 1、经过目标业务校准**，这三件事不等价。更换候选集合、候选数量或提示后，原有温度未必仍适用。

## 7. 面向当前项目的实施顺序

### 阶段 A：不训练，先建立大候选基线

保留现有 0.8B/2B 权重与单题 26 项编译器，在外层新增两种可选策略：

1. **binary-rank：** 每个业务候选转换为现有 `noul`，得到独立分数；如只需单选，可把前若干项再送入一次 `choice` 决选。二分类分数按候选完整返回，决选概率只声明针对入围集合。
2. **tournament：** 借鉴 StartLux 的分组初选与决选，记录每轮候选映射、被淘汰项及成本。若要求返回全 K 项分布，应显式标注合成规则和残余质量参数，不能沿用原生联合分布的含义。

服务可使用 `candidate_strategy`、`probability_semantics`、`num_model_evaluations`、`finalists` 等字段说明过程。这些是本文建议的新增字段，不是当前项目已经提供的功能。候选 ID 必须全程保持一致，不能把每组的 A/B/C 当成全局业务 ID。

尤其要注意：本项目在 `schema.py` 就会拒绝 >26 项。外层编排必须在该检查之前拆分大问题，再将每个子问题送入已有编译器；不能先调用旧 `options_for` 再尝试分组。[本项目校验](https://github.com/yanqiangmiffy/chinese-decision/blob/718ea62efd599070bf4b8f5b6d3b7c6822db9b30/decision/schema.py#L22-L38)

### 阶段 B：对比现成动态标签模型

以 GLiClass、中文适配的 NLI 模型或 Qwen3-Reranker 建立外部基线。这里只做推理评测，不要求先训练新模型。应检查选定 checkpoint 的中文覆盖、标签预算、许可证和输出语义，不能直接拿各项目不同数据集上的宣传成绩横向排序。

### 阶段 C：有收益证据后再训练候选位置头

若实际瓶颈是 K 次配对导致的成本，再开展同序列候选头或文本缓存方案。可以先冻结骨干训练小头，再与微调 LoRA 比较。对 Qwen 同序列路径，要处理候选的因果可见范围，或引入有明确定义的跨候选交互；不要只在候选末尾接一个随机 Linear 就宣称完成迁移。

新架构需要保存评分头、候选标记约定及配置。本项目当前 Transformers/GGUF 后端读取词表 logits；候选位置头或 decoder-kv scorer 需要额外适配加载、导出和推理。StartLux 的 GGUF 支持不能证明本项目任意新增评分头可直接由现有 GGUF 服务读取。[本项目输出路径](https://github.com/yanqiangmiffy/chinese-decision/blob/718ea62efd599070bf4b8f5b6d3b7c6822db9b30/decision/model.py#L27-L46)

## 8. 伪代码：三种方案的区别

以下为说明计算流程的伪代码，不是可以直接运行的实现。候选完整性校验、动态 batching、错误处理和超长输入拒绝需要在工程实现中补齐。

### 8.1 不训练：复用现有 noul，逐候选判断

```python
def binary_rank(state, question, candidates, model):
    jobs = []
    for candidate in candidates:
        jobs.append(make_noul_question(
            state=state,
            instructions=membership_question(question, candidate),
        ))

    # 多批调用已有服务；每个 job 都只有 false / true 两个候选。
    answers = infer_in_batches(model, jobs)
    scores = {c.id: a.probabilities["true"]
              for c, a in zip(candidates, answers)}
    return {
        "ranking": sort_descending(scores),
        "scores": scores,
        "probability_semantics": "independent_binary",
    }
```

这里没有添加随机头，也没有训练。不得仅因 `scores` 处于 0–1，就把它们当作一个互斥 K 类分布。进一步做决选时，应单独返回入围集合内的概率。

### 8.2 不训练：分组决选

```python
def tournament_choice(state, question, candidates, model):
    if len(candidates) <= 26:
        return original_choice(model, state, question, candidates)

    groups = balanced_groups(candidates, max_size=25)
    group_results = infer_groups(model, state, question, groups)
    finalists = keep_top_each_group(group_results, keep=3)

    # 对照 StartLux 时可使用其初选概率截取规则。
    # 业务改造应额外比较继续分轮与跨组截取的入围召回率。
    finalists = apply_explicit_finalist_policy(finalists, limit=26)
    final = original_choice(model, state, question, finalists)
    return {
        "choice": final.choice,
        "probabilities": final.probabilities,
        "probability_semantics": "conditional_on_finalists",
        "finalists": [c.id for c in finalists],
        "eliminated": ids_not_in(candidates, finalists),
    }
```

此处建议默认只返回真实参与决选的条件分布；若为了兼容旧接口而仿照 StartLux 给淘汰项补充残余质量，必须将语义改为合成分布，并返回 residual 参数。

### 8.3 需训练：同序列候选位置共享评分

```python
def forward_candidate_head(backbone, shared_scorer, examples):
    encoded = compile_with_candidate_end_markers(examples)
    assert_all_candidates_preserved(encoded, examples)

    hidden = backbone(
        input_ids=encoded.input_ids,
        attention_mask=encoded.attention_mask,
    ).last_hidden_state
    candidate_hidden = gather_positions(hidden, encoded.candidate_positions)
    logits = shared_scorer(candidate_hidden).squeeze(-1)
    logits = mask_invalid_candidates(logits, encoded.candidate_mask)
    return logits

def train_step(batch):
    logits = forward_candidate_head(backbone, shared_scorer, batch)
    loss = distribution_cross_entropy(logits, batch.target_distribution)
    loss.backward()
    optimizer.step()  # 更新新 scorer；是否同时更新 LoRA 由实验决定。
```

该示例约定标记在候选末尾，使因果骨干的标记能看到本候选内容。Chinese-Jev 的标记在候选前，是因为双向编码器可看到后文；不能直接照搬到因果 Qwen。即使用末尾标记，早期候选仍看不到后续候选；这段伪代码没有解决联合比较和顺序偏差，需要进一步设计与训练验证。

## 9. score 与大候选实验应单独设计

当前项目的 `score` 是有序档位分布的期望，不是自由生成任意实数。已有全量审计结果显示：当前训练集中所有题型最多 16 候选，`score` 自身只有 3–7 档。该结果来自 2026-10-16 的本地数据审计，本次没有重新扫描 1,300 万条数据。

因此，不应因为架构能分配 101 个输出位置，就宣称支持可靠的 0–100 每整数一档评分。建议区分：

- **少量明确档位映射到百分制：** 是尺度换算，原始档位分布仍只有 K 项。
- **101 个独立评分档位：** 需要验证细档位的可辨识性，并准备相应标注与有序损失。
- **连续回归：** 可单独设计连续输出头，但若是新头，通常仍需训练；它也不提供每个离散档位的概率。

不能直接套用 `choice` 的淘汰流程去计算全量 `score` 期望：淘汰档位及人工补充概率可能扭曲分数。对有序等级可考虑保留 RPS 或其他有序约束，但需要通过 MAE、校准和分布指标验证。

## 10. 建议的最小验证计划

| 维度 | 建议覆盖 | 核心问题 |
|---|---|---|
| 候选数量 | 8、16、26、32、64、128、256 | 候选增多后准确率与成本如何变化 |
| 候选难度 | 相近类别、长描述、重复近义标签、无正确类别 | 是否只是依靠容易的负例获得高分 |
| 位置与分组 | 换序、重新分组、不同 keep | 预测与入围召回率是否稳定 |
| 分类质量 | Top-1、Top-k、宏平均召回、淘汰前后召回 | 哪一阶段丢失正确类别 |
| 概率质量 | NLL、Brier、ECE，按 K 分桶 | 正确归一化后是否仍然失准 |
| 评分质量 | MAE、档位分布误差、RPS | 有序错误与百分制换算是否合理 |
| 性能 | p50/p95 延迟、吞吐、token 总量、峰值内存 | 并行是否只是掩盖了计算量增加 |

原始材料、候选构造和同源派生问题按组划分训练/校准/测试，避免泄漏。若只返回 Top-k 或决选分布，NLL/ECE 的评测集合与条件范围必须明确，不能只在入围成功的样本上报告全任务成绩。

本文做了源码常量核对和合成数值检查：检查分组容量、100 候选的默认评估次数、残余质量归一化及二分类输出头的等价公式。这些检查只验证实现理解与数学关系，不构成任何模型的实际质量或性能证据。

## 11. 来源版本与许可

| 项目 | 本次固定提交 | 核心阅读文件 |
|---|---|---|
| StartLux-Decision | `83fe6c7d61a687a6e17f118b4f7d8f429f5d3a01` | `prompt.py`、`readout.py`、`pipeline.py`、Transformers/双向后端、API 文档、旧版 model |
| hmm | `78d8d4a62a41857ce67d903aadf504580feea832` | `training/train.py`、LICENSE |
| Chinese-Jev | `aed6193be5697488c8d0c6a77cc0c7e297329e61` | `modeling.py`、`sequence.py`、`encoding.py`、LICENSE |
| GLiClass | `6a00f8067648f25aa0f4f477d167c5f391cb4165` | `model.py`、`scorers.py`、`pipeline.py`、`config.py`、LICENSE |
| Qwen3-Embedding / Reranker | `44548aa5f0a0aed1c76d64e19afe47727a325b8f` | `examples/qwen3_reranker_transformers.py`；另核对官方模型卡 |
| Transformers NLI pipeline | `d4804f9d76b8c071b993d813e9e87067b780cbee` | `src/transformers/pipelines/zero_shot_classification.py` |
| 本项目 | `718ea62efd599070bf4b8f5b6d3b7c6822db9b30` | `decision/schema.py`、`decision/model.py`、`decision/api.py` |

本文按调研时读取到的提交固定链接，不能将结论自动外推到之后的版本或所有历史 checkpoint。未复现各项目论文、README 中的质量或速度结果，也未确认 StartLux 完整训练配方可由当前推理仓库复现。

许可方面：hmm 代码为 MIT；Chinese-Jev 与 GLiClass 代码为 Apache-2.0。采用现成 checkpoint 时仍要核对该权重及底模各自的条件。[hmm LICENSE](https://github.com/n4ze3m/hmm/blob/78d8d4a62a41857ce67d903aadf504580feea832/LICENSE)、[Chinese-Jev LICENSE](https://github.com/gulucaptain/Chinese-Jev/blob/aed6193be5697488c8d0c6a77cc0c7e297329e61/LICENSE)、[GLiClass LICENSE](https://github.com/knowledgator/gliclass/blob/6a00f8067648f25aa0f4f477d167c5f391cb4165/LICENSE)

StartLux 仓库明确区分 **Apache-2.0 代码**与 **CC BY-NC 4.0 模型权重**；复用它的编排思路与直接部署它的权重是不同选择。官方 Qwen3-Reranker-0.6B 模型卡标注 Apache-2.0；不要据此推断所有第三方转换版权重或其他项目也采用同一许可。[StartLux 许可说明](https://github.com/StartLuxLabs/StartLux-Decision/blob/83fe6c7d61a687a6e17f118b4f7d8f429f5d3a01/README.md#license)、[Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)
