#!/usr/bin/env bash
# 从本机把 GPU 决策服务（scripts/serve_gpu.py）部署到 GPU 服务器并（重新）启动。
#
#   HOST=user@gpu-server deploy/gpu/deploy.sh
#   GPU=5 PORT=8013 HOST=user@gpu-server deploy/gpu/deploy.sh
#
# 连接参数可以写进 deploy/gpu/deploy.env（不提交，模板见 deploy.env.example）。
# 代码取自本仓库已提交的版本（git archive HEAD），服务和训练、评测用同一份 decision 包。
set -euo pipefail
cd "$(dirname "$0")/../.."
if [ -f deploy/gpu/deploy.env ]; then . deploy/gpu/deploy.env; fi
HOST=${HOST:?set HOST=user@gpu-server, or put it in deploy/gpu/deploy.env}
SSH_PORT=${SSH_PORT:-22}
KEY=${KEY:-$HOME/.ssh/id_ed25519}
REMOTE=${REMOTE:-decision-service}
GPU=${GPU:-2}
PORT=${PORT:-8003}
SSH=(ssh -i "$KEY" -o IdentitiesOnly=yes -p "$SSH_PORT" "$HOST")
# Settings that run.sh reads on the server; forward only the ones given here.
ENV="GPU=$GPU PORT=$PORT"
for name in PROJECT PYTHON BASE RUN_DIR; do
  if [ -n "${!name:-}" ]; then ENV="$ENV $name=${!name}"; fi
done

"${SSH[@]}" "mkdir -p $REMOTE/logs"
git -c core.autocrlf=false archive --format=tar HEAD decision scripts/serve_gpu.py \
  deploy/gpu/run.sh deploy/gpu/start.sh deploy/gpu/snapshot.sh \
  | "${SSH[@]}" "tar -xf - -C $REMOTE && chmod +x $REMOTE/deploy/gpu/*.sh"
"${SSH[@]}" "cd $REMOTE && ([ -f .admin-token ] || (umask 077 && head -c 18 /dev/urandom | base64 | tr -d '/+=' > .admin-token)) \
  && $ENV deploy/gpu/snapshot.sh && $ENV deploy/gpu/start.sh"
echo "deployed to $HOST:$REMOTE — health: curl http://${HOST#*@}:$PORT/health"
