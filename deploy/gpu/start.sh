#!/usr/bin/env bash
# 在 tmux 会话 decision 里启动服务（已在运行则先停掉），并把 GPU / PORT / PROJECT / PYTHON / BASE / RUN_DIR
# 透传给 run.sh。查看日志：tmux attach -t decision 或 tail -f logs/serve.log
set -euo pipefail
cd "$(dirname "$0")/../.."
SESSION=${SESSION:-decision}
ENV="GPU=${GPU:-2} PORT=${PORT:-8003}"
for name in PROJECT PYTHON BASE RUN_DIR; do
  if [ -n "${!name:-}" ]; then ENV="$ENV $name=${!name}"; fi
done
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" "$ENV deploy/gpu/run.sh"
echo "started tmux session '$SESSION' ($ENV)"
