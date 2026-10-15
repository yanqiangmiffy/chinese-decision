#!/usr/bin/env bash
# 在 GPU 服务器上前台运行决策服务（scripts/serve_gpu.py）；崩溃后 5 秒自动重启。一般由 start.sh 放进 tmux 里跑。
#
#   GPU=2 PORT=8003 PROJECT=~/chinese-decision deploy/gpu/run.sh
#
# PROJECT 是服务器上的训练仓库，提供 Python 环境、基础模型和训练目录（只读）。
# 默认加载 snapshots/ 下最新的 adapter 快照（由 snapshot.sh 从训练目录的 best 复制而来）。
set -uo pipefail
cd "$(dirname "$0")/../.."

PROJECT=${PROJECT:-$HOME/chinese-decision}
PYTHON=${PYTHON:-$PROJECT/.venv/bin/python}
BASE=${BASE:-$PROJECT/models/Qwen3.5-0.8B}
RUN_DIR=${RUN_DIR:-$PROJECT/runs/cj-ddp}
PORT=${PORT:-8003}
export CUDA_VISIBLE_DEVICES=${GPU:-2}
export PYTHONPATH="$PWD"
export TOKENIZERS_PARALLELISM=false
[ -f .admin-token ] && export DECISION_ADMIN_TOKEN=$(cat .admin-token)
mkdir -p logs

while true; do
  ADAPTER=${ADAPTER:-$(ls -d snapshots/step-* 2>/dev/null | sort -V | tail -1)}
  echo "[run.sh] $(date '+%F %T') start: gpu=$CUDA_VISIBLE_DEVICES port=$PORT adapter=$ADAPTER" | tee -a logs/serve.log
  "$PYTHON" -u scripts/serve_gpu.py --base "$BASE" --adapter "$ADAPTER" --run-dir "$RUN_DIR" --snapshots snapshots \
    --port "$PORT" 2>&1 | tee -a logs/serve.log
  echo "[run.sh] $(date '+%F %T') exited with ${PIPESTATUS[0]}, restarting in 5s" | tee -a logs/serve.log
  unset ADAPTER
  sleep 5
done
