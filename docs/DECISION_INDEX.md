# 本地运行 Jev Decision Index

本项目通过 `decision.decision_index_engine:ChineseDecisionEngine` 接入官方评测器，直接加载
4B 权重，不启动 HTTP 服务。适配器复用 `DecisionPipeline` 的概率读出、原有中文提示模板
和模型配置中的温度，每个问题单独前向。官方工具负责题集、断点续跑和评分。

截至 2026-10-31，榜单 0.4 仍要求公开 **0.3** 题集。这里产出的是 Public Index，
不是维护者私有题集算出的 Full 分。

参考：[官方仓库](https://github.com/apolinario/decision-index)、
[引擎接口](https://github.com/apolinario/decision-index/blob/main/docs/engines.md)、
[提交说明](https://github.com/apolinario/decision-index#submitting-a-model-to-the-leaderboard)。

## 1. 安装

以下命令都在 chinese-decision 项目根目录执行。在 Linux GPU 服务器上可以直接使用
原来跑模型的 Python 环境。先确认 PyTorch 能访问 GPU：

```sh
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

第二行必须是 `True`。本次检查发现这台 Windows 电脑的现有 `.venv` 是 `torch 2.11.0+cpu`，
即使 `nvidia-smi` 能看到 RTX 3090，这个 Python 环境也不能跑 CUDA。
可以在项目下建立单独的评测环境，避免改变原有依赖。Windows PowerShell 示例：

```powershell
python -m venv tmp/jev-venv
.\tmp\jev-venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

这是 [PyTorch 官方提供的 CUDA 12.8 安装源](https://pytorch.org/get-started/previous-versions/#v280)。
Linux 创建环境后用 `source tmp/jev-venv/bin/activate` 激活。
如果只想在已有 CPU 环境做极少量自检，可以用 `--option device=cpu`，但 4B 会以 float32
加载，权重本身约占 16.8 GB 内存，还需额外运行内存；不建议用它跑完整评测。

准备好环境后安装项目和官方工具：

```sh
python -m pip install -e ".[torch]"
git clone https://github.com/apolinario/decision-index.git tmp/decision-index
git -C tmp/decision-index checkout 25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d
python -m pip install -e "tmp/decision-index[rebuild]"
```

`tmp/decision-index` 已存在时跳过 clone。本次适配核对的是上述官方提交；
正式运行时记录两份代码的版本，升级评测工具后用新的输出目录。
保留官方源码目录：题集导入会使用其中的 `hub/` 清单与排除列表。

## 2. 先检查真实模型能否跑通（不下载官方题集）

仓库自带两个演示请求，一道 choice 和一道 noul。这不是官方评测，不计算榜单分数。

```sh
python -m decision_index run --engine decision.decision_index_engine:ChineseDecisionEngine --model models/chinese-decision-4b --option device=cuda --rows examples/decision_index_smoke.jsonl --compact --out runs/decision-index-4b-smoke
```

检查 `runs/decision-index-4b-smoke/results.jsonl`：两条记录应为 `status: "ok"`，
带答案和概率。初始化还会计算模型文件哈希，首次启动可能比单纯加载权重更慢。

`--model` 也可以填 Hub 模型 ID，例如 `quincyqiang/chinese-decision-4b`；
首次会下载权重。正式跑远程权重时使用 `--option revision=<模型提交哈希>` 固定版本。

## 3. 准备官方题集（只需一次）

本机已于 2026-11-01 完成公开 0.3 题集的下载、构建和官方校验。
最终目录为 `datasets/decision-index/suite-0.3`（约 83.5 MB），
包含 140,620 条评测请求，140,178 条可评分；已有该目录时可以直接进入下一节。
完整验证记录位于 `datasets/decision-index/verification.json`。

先登录 Hugging Face，并确保账号已取得 [HLE](https://huggingface.co/datasets/cais/hle)
等受限来源的访问权限。遇到权限错误，应先解决数据访问，不能跳过题目后声称完整运行。
官方估算约 7 GB 下载、17 GB 工作空间，模型与结果另计。

本机使用 `datasets/decision-index/prepare.py` 下载并重建题集，独立环境位于
`tmp/decision-index-env`。它处理 Windows 的换行、路径和文件排序差异，
保留 HLE 的官方下载来源记录，并兼容固定官方提交中已确认的
[SGD 冻结版本重建问题](https://github.com/apolinario/decision-index/issues/274)。
SGD 的历史题集包含未来对话，已不计入 0.3 主榜指数；这里只复刻官方冻结数据，
不能把该兼容行为用于未来修正版。具体记录见本地数据目录的 `README.md` 和
`windows-paths.patch`。不要用 `--no-verify` 绕过检查。

已有下载任务运行时不要重复启动。需要恢复下载或重建时，在项目根目录执行：

```powershell
& .\tmp\decision-index-env\Scripts\python.exe -X utf8 -u datasets/decision-index/prepare.py --workers 8
```

仅当 `download-status.json` 中 `complete` 为 `true` 且 `phase` 为 `ready`，
才表示 `datasets/decision-index/suite-0.3` 已完整构建并通过校验。
该最终目录可以整体复制到 Linux GPU 服务器；运行评测无需复制原始下载和中间文件。
在安装好官方工具的环境中，可以独立复核：

```sh
python -m decision_index suite verify --edition 0.3 --dir datasets/decision-index/suite-0.3
```

导入必须通过官方哈希验证。题集只保存在自己可访问的位置，不公开再分发。

## 4. 小样本检查

```sh
python -m decision_index suite sample --edition 0.3 --dir datasets/decision-index/suite-0.3 --n 100 --out datasets/decision-index/sample-100.jsonl.gz
python -m decision_index run --edition 0.3 --engine decision.decision_index_engine:ChineseDecisionEngine --model models/chinese-decision-4b --option device=cuda --rows datasets/decision-index/sample-100.jsonl.gz --compact --out runs/decision-index-4b-sample
```

采样保留关联请求的完整组，因此实际请求数可能超过 100。先查看 `status.json` 的状态计数，
再查看 `results.jsonl` 的答案、错误和耗时。小样本只用于检查功能与资源使用，
没有完成的正式题目会被评分器当作未答；不能将小样本的汇总分当作完整 Public Index。

## 5. 完整运行并自动出分

```sh
python -m decision_index pipeline --edition 0.3 --suite-dir datasets/decision-index/suite-0.3 --engine decision.decision_index_engine:ChineseDecisionEngine --model models/chinese-decision-4b --option device=cuda --compact --out runs/decision-index-4b-full
```

运行结束后查看：

- `scores.json`：汇总结果；`decision_index` 是公开指数，`complete` 表示是否有完整的结果记录，
  `counts` 区分成功、不支持和错误。`complete: true` 不等于每题都成功，要同时检查错误计数。
- `benchmark-summary.json`：各项任务的指标、覆盖率和错误数。
- `index.json`：公开指数、分领域和分任务分数。
- `results.jsonl`：逐请求答案、概率、状态与耗时。`--compact` 不保存题目正文，便于提交结果。
- `environment.json`：模型与推理设置、硬件、模型/配置/推理源码哈希。
- `status.json`：运行进度；其中的完成事件不是 `scores.json` 的全量完成标记。

用一条命令查看关键结果：

```sh
python -c "import json; from pathlib import Path; s=json.loads(Path('runs/decision-index-4b-full/scores.json').read_text(encoding='utf-8')); print(json.dumps({k:s[k] for k in ['edition','decision_index','complete','completed','counts','latency_ms']},ensure_ascii=False,indent=2))"
```

只重新打分、无需再次加载模型：

```sh
python -m decision_index score --edition 0.3 --suite-dir datasets/decision-index/suite-0.3 --results runs/decision-index-4b-full/results.jsonl
```

中断后，在**权重、代码、推理参数都没变**的情况下重跑原命令，官方工具会继续未完成请求并重试错误。
工具不会自动检查恢复前后模型是否一致；更换模型、温度、实现或容量时必须使用新输出目录，
不能混合不同设置的预测。

## 容量与分组决策

一题最多 26 个候选（A–Z）、16384 tokens。超过时适配器这样处理：

- **`choice` 候选超过 26 个**（BANKING77 77 类、CLINC150 151 类、POP909 129、ChessBench 最多 81、
  API-Bank 53，共 14,500 条请求）：走 `decision/serving/tournament.py` 的分组淘汰赛。候选按固定种子
  打乱后均分成每组不超过 16 个（训练见过的最大候选数），每组前 2 名晋级，直到决赛候选能放进一题。
  提示词、说明和候选描述都不改，不删候选。
- **`choice` 提示词超过 16384 tokens**：同样分组，组大小 16 → 8 → 4 → 3 逐步缩小直到每组放得下。
- **概率覆盖全部候选**：决赛候选保留决赛概率；被淘汰的候选取所在组的组内概率，并按该组晋级者
  在后面轮次获得的概率缩放，同时不超过任何晋级者。所以后面轮次的候选排名总在前面，胜者概率最大。
  官方评分只用 `choice` 和最大概率，结果满足官方校验（全部候选、合计为 1）。
- **`noul` 或分组后仍放不下**：整条请求在推理前拒绝，记为 `unsupported`（主要是 ToolRet 中 2.7 万
  tokens 的文档题，其中不少本就在官方排除名单里）。不截断。

同一请求的所有问题、以及每轮各组，合成一次批量前向（`batch_size=64`、`token_budget=65536`），
概率与逐题推理一致（单测核对）。可用 `--option group_size=16 --option keep=2 --option seed=0` 调整，
设置会写进 `environment.json`。

## 在 GPU 服务器上分片运行

`scripts/run_decision_index.py` 按 run_id 哈希把题集分成 N 片，每片在一张卡上调用官方 runner
（自带断点续跑），最后合并并用官方评分器出分：

```sh
export PYTHONPATH=$PWD:$PWD/kit   # 本项目 + 官方 decision-index 源码
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i nohup python scripts/run_decision_index.py run --suite-dir datasets/suite-0.3     --model models/chinese-decision-4b --out runs/4b-full --shards 4 --shard $i > runs/4b-full-shard$i.log 2>&1 &
done
# 全部结束后
python scripts/run_decision_index.py merge --suite-dir datasets/suite-0.3 --out runs/4b-full --shards 4 --engine-name chinese-decision-4b
```

`merge` 生成 `runs/4b-full/{results.jsonl,scores.json,benchmark-summary.json,index.json,environment.json}`，
格式与官方 `pipeline` 相同。进度看各分片的 `status.json`（日志有缓冲）。中断后重跑同一命令即可续跑。

这里测得的耗时是你自己的硬件结果，不能代替维护者在 RTX PRO 6000 上做的延迟资格测试
（中位数、平均、P80 都须 ≤ 1000 ms）。

## 跑完后上传并申请上榜

同步到榜单的流程是：**上传结果到自己的 HF Dataset → 向官方 GitHub 提交 PR →
维护者复算公开结果、运行私有题集和测延迟 → 维护者更新榜单**。
上传到 HF 不会自动出现在榜单；公开分也不能直接替代 Full 分。

1. **确认结果**：`scores.json` 中 `complete` 为 `true`，`counts` 里没有 `error`。不要手改任何结果文件。

2. **取回合并后的结果**（服务器访问不了 Hugging Face，在本机上传）。只取合并后的五个文件：

   ```sh
   for m in unified-4b 4b; do
     mkdir -p runs/decision-index/$m
     scp -i ~/.ssh/codex_mscapital_ed25519 -P 50000        "yanqiang@10.208.62.156:/data/users/yanqiang/decision-index-eval/runs/$m-full/{results.jsonl,scores.json,benchmark-summary.json,index.json,environment.json}"        runs/decision-index/$m/
   done
   ```

3. **发布推理代码**：把 `decision/decision_index_engine.py`、`decision/serving/tournament.py`、
   `scripts/run_decision_index.py` 提交并推送到公开仓库，记下 commit。`environment.json` 里的
   `source_sha256` 要和这个 commit 的文件一致，所以推送前不要再改这些文件。

4. **上传结果**（示例仓库名，确认后再运行；`hf auth login` 先登录）：

   ```sh
   python -c "from decision_index.pipeline import upload_run as u; r='quincyqiang/chinese-decision-index-results'; print(u(r, 'runs/decision-index/unified-4b', path_in_repo='runs/chinese-decision-unified-4b', private=False)); print(u(r, 'runs/decision-index/4b', path_in_repo='runs/chinese-decision-4b', private=False))"
   ```

   官方 `upload_run` 会把逐题结果压缩成 `results.jsonl.gz` 并上传评分和环境文件。不要上传题集。

5. **提交 PR**：fork [apolinario/decision-index](https://github.com/apolinario/decision-index)，在
   `submissions/README.md`（不存在则创建）里每个模型加一行，向官方仓库开 PR。例如：

   ```md
   | chinese-decision-unified-4b | [quincyqiang/chinese-decision-unified-4b](https://huggingface.co/quincyqiang/chinese-decision-unified-4b) | [scores.json](https://huggingface.co/datasets/quincyqiang/chinese-decision-index-results/blob/main/runs/chinese-decision-unified-4b/scores.json) | `decision.decision_index_engine:ChineseDecisionEngine` @ <commit> (decision-index 25c6ec7) | 1× A800 80GB per shard, bf16 | T=1.0986 (from decision_config.json), group_size=16, keep=2, seed=0; ≤26 options/prompt, 16384 tokens; >26 options via seeded group tournament, no truncation |
   ```

   PR 正文写清安装方式（`pip install` 本项目 + 官方 kit）、运行命令、容量限制和分组方法，
   并说明用了 4 个分片并行（每片单卡、单进程，逐请求计时）。

维护者复算逐题结果，并使用同一权重和推理实现跑私有题集、检查延迟，审核通过后才更新
Hugging Face Space 榜单。进度看 PR；公开规则没有承诺固定处理时间。
以后换权重或推理设置，要跑到新目录、上传到新路径，并在 PR 中说明。
