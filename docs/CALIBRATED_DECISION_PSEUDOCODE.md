# 校准决策模型：RLCD 训练伪代码

本文是独立设计，不是 TypeSafe Jev 内部算法的复现（其损失函数未公开）。损失函数已实现为可运行的参考代码 [decision/rlcd_loss.py](../decision/rlcd_loss.py)，只依赖 PyTorch。数据、模型前向、分布式与存储部分保留为伪代码。生产训练器 [decision/rlcd.py](../decision/rlcd.py) 保持不变。

## 1. 损失定义

设一道题的候选 logits 为 `z`，有效候选掩码为 `m`，目标分布为 `t`（硬标签即 one-hot）：

```text
p      = softmax(z)                                   # 只在有效候选上归一化
loss   = ce_weight * CE(t, p) + brier_weight * Brier(t, p) + rlcd_weight * L_rlcd
L_rlcd = -E_eps[ S(softmax(z + sigma * eps), t) ]

S(q, t) = sum_k t_k * log(q_k)                        # 对数评分
        + spherical_weight * <t, q> / ||q||_2         # 球面评分
        - rps_weight * RPS(q, t)                      # 排序概率评分，仅有序题
```

- `eps` 是有效候选上的标准高斯噪声，并投影到零和子空间。softmax 对平移不变，去掉的方向不会改变奖励，只会增加方差。
- `S` 是严格适当评分规则的非负组合，期望奖励在 `q = t` 处最大。软标签下 RPS 只差一个与 `q` 无关的常数，因此仍然适当。
- `sigma > 0` 时，平滑后目标的最优点只是接近 `t`。训练中应把 `sigma` 退火到较小值，最终概率再交给温度校准。
- RPS 用 `K-1` 个等级边界归一化，单候选题为 0。有序题的候选必须按等级自然顺序排列，不能打乱。

## 2. 两种梯度估计器与对照组

| 配置 | 梯度路径 | 用途 |
|---|---|---|
| `RLCDLossConfig(rlcd_weight=0)` | 只有 CE | 必做基线 |
| `RLCDLossConfig(estimator="reparam")` | 经 `z + sigma*eps` 直接反传 | 检查收益是否来自扰动评分目标 |
| `RLCDLossConfig(estimator="reinforce")` | 奖励停止梯度，经高斯 log-density 乘留一优势 | RL 形式，主方案 |
| `RLCDLossConfig.chinese_jev()` | reinforce，log 下限 −9.21，批内优势标准化 | 与 `decision/rlcd.py` 数值一致 |

REINFORCE 分支的核心只有四行：

```text
z~_g   = stop_gradient(z) + sigma * eps_g             # g = 1..G，高斯策略的动作
R_g    = S(softmax(z~_g), t)                          # 奖励，停止梯度
A_g    = R_g - mean_{j != g} R_j                      # 留一基线，梯度无偏
L_rlcd = -mean_g A_g * log N(z~_g; z, sigma^2 I)      # d/dz log N = (z~_g - z) / sigma^2
```

留一优势等于 `G/(G-1) * (R_g - mean_j R_j)`，因此标准化以后与组均值基线完全相同，这正是 `chinese_jev()` 预设能逐位对上生产实现的原因。优势标准化相当于自适应步长，梯度不再是原目标的无偏估计。`log_score_floor` 能限制奖励方差，但低于下限的区域不再严格适当。

对照实验的要求：

1. `reparam` 和 `reinforce` 使用同一评分、噪声分布、`sigma` 日程、样本数和 CE 权重，才能比较梯度估计方式。
2. REINFORCE surrogate 的数值不是目标值，训练曲线看 `out.reward`、`out.ce` 和验证指标。
3. 三组都经过同样的温度校准与阈值流程，再比较测试结果。

最小调用示例（不下载模型）：

```python
import torch
from decision.rlcd_loss import RLCDLoss, RLCDLossConfig

logits = torch.tensor([[1.2, 0.3, 0.0], [0.2, 0.7, -0.4]], requires_grad=True)
mask = torch.tensor([[True, True, False], [True, True, True]])
target = torch.tensor([[0.8, 0.2, 0.0], [0.0, 1.0, 0.0]])
ordinal = torch.tensor([False, True])

loss_fn = RLCDLoss(RLCDLossConfig(estimator="reinforce", num_samples=8))
out = loss_fn(logits, target, mask, ordinal, sigma=0.3)
out.loss.backward()
assert logits.grad[0, 2] == 0           # padding 候选没有梯度
print(out.loss.item(), out.ce.item(), out.reward.item())
```

## 3. 数据协议与划分

```python
DecisionRecord = {
    "case_id": "同一原始状态或文档的分组 ID",
    "state": "输入状态",
    "question": "明确的问题",
    "kind": "choice | noul | score",
    "options": [{"id": "稳定 ID", "text": "候选描述"}, ...],
    "target": [0.1, 0.7, 0.2],  # 分布之和为 1，硬标签为 one-hot
    "domain": "业务领域",
}

# 同一原始文档、其改写和其多道问题一起分组，避免跨集合泄漏。
train, dev_select, dev_temperature, dev_gate, test = grouped_split(records)
assert_disjoint_case_groups_and_near_duplicates(all_splits)

# train：参数训练；dev_select：选超参数和 checkpoint。
# dev_temperature：拟合温度；dev_gate：拟合接受阈值。
# test：模型、温度与阈值全部锁定以后，只作最终评价。
# 小数据可采用嵌套划分/交叉拟合；不能用 test 填补校准数据。

def collate(records):
    # choice 的选项可打乱，但 options、target 和稳定 ID 必须同步重排。
    # score 的有序等级不能打乱；noul 的候选语义固定。
    # 全部候选文本必须完整容纳，超预算样本显式分流，不静默删除候选。
    return Batch(
        model_inputs=encode_state_question_and_options(records),
        candidate_mask=valid_candidate_mask(records),   # [B, K] bool
        target=pad_target_with_zeros(records),          # [B, K]
        ordinal=tensor([r.kind == "score" for r in records]),  # [B] bool
        option_ids=stable_option_ids(records),
    )
```

软标签应有可解释的来源，例如标注者分布或经过验证的教师分布。缺失标注不等于负标签。`none` 只在题目允许“以上皆非”时作为语义候选；低置信度转人工属于模型之外的路由动作，两者不混用。

## 4. 模型只需暴露统一的打分接口

```python
def score_candidates(model, batch):
    # 可接本项目的字母 logits、候选标记上的共享 MLP，或 pointer head。
    logits = model.forward_candidate_scores(batch.model_inputs)  # [B, K]
    assert all_valid_candidate_scores_are_finite(logits)
    return logits

# 一个 state 的多个问题可以共享计算，但问题间必须保持约定的隔离。
# 混合注意力/递归骨干不能仅凭普通 attention mask 假定问题隔离成立。
# 训练与推理使用相同的输入格式、候选映射与截断规则。
```

## 5. 训练主循环

```python
from dataclasses import replace
from decision.rlcd_loss import RLCDLoss, RLCDLossConfig, linear_schedule

config = RLCDLossConfig(estimator="reinforce", num_samples=8, rlcd_weight=1.0, reduction="sum")
sigma_start, sigma_end = 0.4, 0.1
rlcd_ramp_steps = 200            # 先让 CE 稳定，再逐步打开 RLCD
kl_weight = 0.0                  # 可选：锚定初始模型
grad_clip, seed = 1.0, 42

set_all_seeds(seed)
model = build_model_and_optional_lora()
optimizer, scheduler, scaler = build_optimizer_scheduler_and_precision()
reference = frozen_copy(model).eval() if kl_weight else None
noise_rng = torch.Generator(device).manual_seed(seed + 1)  # 不消耗数据顺序/dropout 的随机流
best_score = +infinity

if resume:
    checkpoint = load_checkpoint()
    assert_same_data_model_prompt_candidate_mapping_and_loss_config(checkpoint)
    restore_model_optimizer_scheduler_scaler_rng_noise_rng_and_data_cursor(checkpoint)

for epoch in remaining_epochs:
    for window in next_microbatch_window(train, data_cursor):
        optimizer.zero_grad(set_to_none=True)
        model.train()
        # token 预算使 microbatch 大小不同，按全局真实决策数归一化。
        global_count = all_reduce_sum(real_decision_counts(window))

        sigma = linear_schedule(sigma_start, sigma_end, examples_seen / planned_examples)
        weight = config.rlcd_weight * min(1.0, optimizer_step / rlcd_ramp_steps)
        loss_fn = RLCDLoss(replace(config, rlcd_weight=weight))   # 无参数模块，可每步重建

        for microbatch in window:
            logits = score_candidates(model, microbatch)
            out = loss_fn(logits, microbatch.target, microbatch.candidate_mask,
                          microbatch.ordinal, sigma=sigma, generator=noise_rng)
            loss = out.loss                                       # 本 microbatch 的求和
            if kl_weight:
                with no_grad():
                    ref_logits = score_candidates(reference, microbatch)
                loss += kl_weight * sum_over_decisions(KL(p(logits) || p(ref_logits)))
            # DDP 默认对各 rank 梯度取平均，故乘 world_size 再除以全局决策数。
            backward_with_precision_scaler(loss * world_size / global_count)

        unscale_gradients_if_needed(optimizer)
        assert_all_gradients_finite()
        clip_grad_norm(model, grad_clip)
        optimizer_step_and_update_scaler(optimizer)
        scheduler.step()
        examples_seen += global_count
        optimizer_step += 1
        log(out.ce, out.brier, out.rlcd, out.reward, sigma, weight, gradient_norm, lr)

        if should_validate(optimizer_step):
            model.eval()
            rows = predict_without_noise_or_grad(model, dev_select)
            metrics = compute_raw_probability_metrics(rows)
            # 预先固定选模标准，例如验证 NLL，并把 Accuracy 设为验收约束。
            if meets_accuracy_constraint(metrics) and metrics.nll < best_score:
                atomic_save_best(model, config, model_and_data_fingerprints)
                best_score = metrics.nll
            model.train()
        if should_checkpoint(optimizer_step):
            atomic_save_checkpoint(model, reference, optimizer, scheduler, scaler,
                                   python_numpy_torch_cuda_rng, noise_rng, data_cursor,
                                   epoch, optimizer_step, examples_seen, best_score, config)

assert_best_checkpoint_exists()  # 没有满足约束的模型时，显式报告验收失败。
model = load_best_checkpoint().eval()
```

分布式实现还需保证各 rank 的同步步数相同，并处理没有真实样本的 rank。本项目的 `scripts/train.py` 已实现这些细节；接入时在现有循环里替换损失计算即可，不要换成上面的简化循环。

## 6. 校准、接受阈值与交付

下面三个函数是伪代码，不在参考模块中。

```python
def fit_temperature(logits, target, mask):
    # 在对数网格 T ∈ [0.1, 10] 上最小化 dev_temperature 的分布 NLL；命中边界时扩大网格。
    return argmin_T mean(CE(target, masked_softmax(logits / T, mask)))

def fit_threshold(probs, target, risk_budget=0.05, min_accepted=30):
    confidence, prediction = probs.max(-1)
    risk = 1 - target[prediction]          # 软标签下是预测类别的期望错误率
    # 按置信度降序计算前缀经验风险；只在并列置信度组的末尾切分。
    # 取“接受数 ≥ min_accepted 且前缀风险 ≤ risk_budget”中覆盖率最大的切点。
    # 没有可行切点时阈值为 +inf，即全部转人工。
    return threshold, coverage, empirical_risk

def decide(logits, mask, ordinal, T, threshold):
    probs = masked_softmax(logits / T, mask)   # 推理不加训练噪声
    top1_probability, top1_index = probs.max(-1)
    expected_level = sum_k(k * probs[k]) if ordinal else None
    return probs, top1_index, top1_probability, top1_probability >= threshold, expected_level

T = fit_temperature(*predict_without_noise_or_grad(model, dev_temperature))
gate = fit_threshold(softmax(predict(model, dev_gate).logits / T), dev_gate.target)
# 这只是 gate 集合上的经验结果，不是总体 5% 错误率的统计保证。

freeze(model_weights, candidate_protocol, T, gate.threshold)
test_rows = predict_without_noise_or_grad(model, test)
raw_metrics = evaluate(test_rows, temperature=1, threshold=None)
served_metrics = evaluate(test_rows, temperature=T, threshold=gate.threshold)
report_by_domain_question_type_candidate_count_and_source(raw_metrics, served_metrics)

export_bundle(weights=model, input_and_candidate_protocol=protocol, temperature=T,
              acceptance_rule={"top1_probability_gte": gate.threshold},
              calibration_data_fingerprints=hashes(dev_temperature, dev_gate),
              model_selection_and_test_reports=reports)
```

`T > 0` 不改变每题的 argmax，但会改变跨题的置信度排序。`top1_probability` 对应 top1 答案；有序题的 `expected_level` 是另一种输出，不能把 top1 的正确率解释直接套用到等级均值上。

评估至少包含：Accuracy、NLL、full-Brier、ECE、风险–覆盖曲线/AURC，以及冻结阈值下的实际覆盖率、错误率和拒绝率。软标签可评价分布交叉熵、Brier 和预测类别的期望错误率；需要硬标签准确率时保留独立的硬标签字段。

## 7. 参考模块之外的四条路线

这些路线的数据、采样或推理接口与主线不同，默认不与主损失相加，这里只给伪代码。

### 7.1 CalOn：类型选择加 none，奖励换成任务收益

```text
候选 = [none, 类型1, 类型2, ...]
L_prob = -log sum_{g in gold_set} p_g + beta/2 * ||p - t||^2   # 多 gold 集合似然，不等于 soft CE
z~_g = stop_gradient(z) + sigma * eps_g
a_g  = argmax z~_g
U_g  = +1 正确类型 / -1 错误类型或误抽取 / 0 选 none
A_g  = U_g - mean_{j != g} U_j
L    = L_prob - pg_weight * mean_g A_g * log N(z~_g; z, sigma^2 I)
```

结构上就是把 `RLCDLoss` 中的 `proper_score` 换成任务收益。任务收益不是适当评分规则，none 重加权、类别重采样也会改变训练先验，因此必须单独验证校准。论文报告模型的 `beta = 0`。

### 7.2 eve-rlcd：只有所选动作的对错

```text
p   = softmax(z)
a_g ~ p                                   # G 次真实、独立的交互
c_g = environment.reveal_correctness(a_g) # 只返回所选动作是否正确
R_g = c_g - stop_gradient(p[a_g])
A_g = R_g - mean_{j != g} R_j
L   = -mean_g A_g * log p[a_g]            # E[grad L] = grad(Brier / 2)
```

`stop_gradient` 是算法的一部分。必须从当前策略采样；旧日志不能直接当作 on-policy 数据。有完整标签时，直接计算 Brier 更自然。

### 7.3 OpenJev-RLCD：推理轨迹，分两阶段训练

```text
r_i ~ model(x)                            # 采样 M 条推理，到 "Answer:" 为止
u_i = p_model(· | x, r_i)                 # 直接读出答案分布，不再采样答案
R_i = 2 * u_i[y] - ||u_i||^2              # 负 Brier 的等价形式，每条轨迹单独评分

阶段一：L = -mean_i R_i                   # 只经答案分布反传，骨干并未冻结
阶段二：L = -mean_i R_i
          - 0.3 * mean_i stop_gradient(A_i) * log p(r_i | x)
          + KL(轨迹策略 || 阶段一模型)
```

给多条轨迹平均后的分布评分，会额外奖励轨迹间的分歧（`J_mixture = J_single + Var_r(u)`）。部署只查询一次时，应按单条轨迹评分。

### 7.4 rlcd-lite：答案之外再输出置信度

```text
a ~ answer_policy, c ~ confidence_policy  # c 是单独输出的置信度档位
R = 1[a = y] - (c - 1[a = y])^2
L = -mean A * (log p(a) + log p(c))
```

只奖励置信度平方误差时，模型可以学成“经常答错，但准确承认自己会错”，所以答对项不能省。单独输出的置信度与答案 softmax 的最大概率是两个信号，需要分别评价。

## 8. 验收与验证记录

1. 三组核心对照固定数据划分、模型初始化、样本顺序、样本预算、验证周期和校准流程，并分别报告训练成本。
2. padding 候选的概率与梯度为 0；变长候选、单候选题、硬/软标签都不产生 NaN。
3. REINFORCE 的采样、奖励和基线停止梯度；reparam 的采样保留梯度。
4. 只在校准集拟合温度，阈值锁定后报告独立测试集的风险；同时保留原始与校准后的结果。ECE 降低本身不能说明模型更有用。

验证记录（2026-10-23，CPU）：[tests/test_rlcd_loss.py](../tests/test_rlcd_loss.py) 的 6 项检查全部通过：

- 目标分布在含 RPS 的评分下得分最高；
- `rlcd_weight=0` 时与 `F.cross_entropy` 的软标签结果一致；
- 两种估计器下 padding 梯度为 0，且每题梯度之和为 0；
- 三种 reduction 一致；
- 10 万样本下 REINFORCE 与 reparam 梯度的余弦相似度大于 0.999；
- `chinese_jev()` 预设的损失与梯度和 `decision/rlcd.py` 加 CE 的结果一致（误差约 1e-6）。

`python examples/rlcd_loss.py` 会在自由 logits 的玩具任务上训练四种配置，确认都能稳定收敛到目标附近。在这个任务里 CE 本身已经最优，所以它不说明哪种方法在真实模型上校准得更好。

公开参考：

- [Chinese-Jev：高斯 logits 探索与混合适当评分](https://github.com/gulucaptain/Chinese-Jev/blob/f4138c028e0da30c40e7e32aa294a38b817b694d/src/chinese_jev/train.py)
- [CalOn：direct/pg 对照与任务收益](https://github.com/extraordinarylab/calon/blob/0e4e6587b14c179c034568a8ece7877ce6c70673/src/calon/model/losses.py)
- [OpenJev-RLCD：每条推理的概率评分](https://github.com/ZimmyGao/openjev-rlcd/blob/22db40ad3333f5cf0bc5139097da70e0f8c05018/openjev_rlcd/estimators.py)
- [eve-rlcd：bandit 反馈的 Brier 梯度估计](https://github.com/anthony-maio/eve-rlcd/blob/adb9a0457ef53d2dc0e106215315670f1f6da871/rlcd/rewards.py)
- [rlcd-lite：答案与置信度的联合奖励](https://github.com/arnabgho/rlcd-lite/blob/b0f981fab5e6ce53dd089eae71b141ab1099c31a/src/rlcd_lite/rewards.py)

参考代码按上文公式独立编写，没有搬用各仓库的训练器。
