# GPU 部署

`scripts/serve_gpu.py` 是 GPU 上的生产推理服务：兼容 hmm / System One 接口，并补充了 [Decision Studio](https://github.com/yanqiangmiffy/chinese-decision-studio) 用到的批量、提示词编译、训练进度和热更新接口。CPU 部署（llama.cpp + GGUF）见 [docs/CPU.md](../../docs/CPU.md) 和 [deploy/gguf](../gguf/)，两者接口一致。

## 部署

在本机执行（需要能 SSH 到 GPU 服务器；服务器上已经有训练仓库和它的 `.venv`）：

```bash
cp deploy/gpu/deploy.env.example deploy/gpu/deploy.env   # 填 HOST、KEY、PROJECT 等，不会提交
deploy/gpu/deploy.sh
GPU=5 PORT=8013 deploy/gpu/deploy.sh                     # 换卡、换端口
```

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `HOST` | （必填） | SSH 目标，如 `user@gpu-server` |
| `SSH_PORT` | `22` | SSH 端口 |
| `KEY` | `~/.ssh/id_ed25519` | SSH 私钥 |
| `REMOTE` | `decision-service` | 服务器上的部署目录 |
| `PROJECT` | `~/chinese-decision` | 服务器上的训练仓库：提供 `.venv`、`models/Qwen3.5-0.8B`、`runs/cj-ddp` |
| `PYTHON` / `BASE` / `RUN_DIR` | 由 `PROJECT` 推出 | 单独覆盖 Python、基础模型、训练目录 |
| `GPU` / `PORT` | `2` / `8003` | 使用的 GPU 和端口 |

`deploy.sh` 做的事：

1. 用 `git archive HEAD` 把 `decision/` 包、`scripts/serve_gpu.py` 和本目录的服务器脚本同步到 `REMOTE`（只同步已提交的代码，部署前先提交）；
2. 第一次部署时生成 `.admin-token`（热更新用，权限 600）；
3. `snapshot.sh`：把训练目录的 `best/` 复制成 `snapshots/step-N`——训练会原地替换 `best/`，服务只加载不可变快照；
4. `start.sh`：在 tmux 会话 `decision` 里运行 `run.sh`，进程退出后 5 秒自动重启。

服务器上的部署目录：

```text
decision-service/
├── decision/ scripts/serve_gpu.py deploy/gpu/*.sh   # 每次部署覆盖
├── snapshots/step-N/                                # 不可变的 adapter 快照
├── logs/serve.log
└── .admin-token
```

服务器上常用命令：

```bash
tmux attach -t decision                       # 看实时日志（Ctrl-b d 退出）
tail -f ~/decision-service/logs/serve.log
cd ~/decision-service && GPU=2 PORT=8003 deploy/gpu/start.sh   # 重启
```

## 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/v1/systemone` | 决策接口；`?debug=1` 附带每个候选的 logits 和每题 token 数 |
| POST | `/api/alpha/decisions` | 同上，兼容 hmm 路由 |
| POST | `/v1/batch` | `{"requests": [...], "debug": bool}`，最多 512 个请求 / 4096 个决策，单个请求出错不影响其他 |
| POST | `/v1/compile` | 只编译提示词并计数 token |
| POST | `/v1/tournament` | 候选超过一题能放下的 `choice` 问题：分组淘汰后决赛（见主 README） |
| GET | `/v1/info` | 检查点、验证集指标、GPU、引擎配置、运行统计（延迟分位数、平均批大小、最近的前向） |
| GET | `/v1/training` | 训练目录的进度（只读） |
| POST | `/v1/admin/reload` | 快照最新的 best 并热加载，需要请求头 `X-Admin-Token` |
| GET | `/health` | 健康检查 |

## 动态批处理

所有前向都经过一个后台线程：

- 第一个请求到达后再等 `--max-wait-ms`（默认 4 ms），把这段时间内到达的请求合在一起；
- 按长度排序，按 `--token-budget`（默认 65,536，padding 后的 token 数）和 `--max-batch`（默认 64）切 micro-batch；
- 显存不足时自动退化成逐条前向；空闲 20 秒后释放缓存的显存。

在 A800 上的实测（工单请求，每个 3 个问题）：串行约 37 决策/秒、p50 81 ms；并发 32 时约 190 决策/秒、p50 约 470 ms。并发再升高时，吞吐反而下降——瓶颈是 Python 侧提示词编译与前向线程的 GIL 竞争。
