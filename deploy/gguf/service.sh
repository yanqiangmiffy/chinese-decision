#!/usr/bin/env bash
# Start, stop or inspect the GGUF HTTP service (llama.cpp on CPU) in the background.
# Settings come from the environment: HOST, PORT, THREADS, CONTEXT_SIZE, MODEL.
# Start after a reboot with a crontab line such as:
#   @reboot sleep 30 && $HOME/chinese-decision/deploy/gguf/service.sh start >> $HOME/chinese-decision/logs/cron.log 2>&1
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODEL="${MODEL:-$ROOT/models/chinese-decision-0.8b-gguf}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8787}"
# One thread per physical core; adding hyper-threads made a 2 x 14-core Xeon about 10x slower.
THREADS="${THREADS:-$(lscpu -p=CORE,SOCKET | grep -v '^#' | sort -u | wc -l)}"
CONTEXT_SIZE="${CONTEXT_SIZE:-16384}"
PID_FILE="$ROOT/run/serve.pid"
LOG_FILE="$ROOT/logs/serve.log"

# Match the command line too: after a reboot the stale PID may belong to another process.
running() { [ -f "$PID_FILE" ] && ps -p "$(cat "$PID_FILE")" -o args= 2>/dev/null | grep -q "scripts/serve.py"; }

start() {
  if running; then echo "already running (pid $(cat "$PID_FILE"))"; return; fi
  mkdir -p "$ROOT/run" "$ROOT/logs"
  cd "$ROOT"
  PYTHONPATH="$ROOT" TOKENIZERS_PARALLELISM=false nohup .venv/bin/python -u scripts/serve.py \
    --backend gguf --base "$MODEL" --gguf "$MODEL/chinese-decision-0.8b-Q8_0.gguf" \
    --threads "$THREADS" --context-size "$CONTEXT_SIZE" --host "$HOST" --port "$PORT" \
    < /dev/null >> "$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"
  for _ in $(seq 120); do
    if curl -fsS "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
      echo "started (pid $(cat "$PID_FILE")) on $HOST:$PORT, threads $THREADS"; return
    fi
    if ! running; then echo "failed to start; last lines of $LOG_FILE:"; tail -20 "$LOG_FILE"; exit 1; fi
    sleep 1
  done
  echo "not healthy after 120 s; see $LOG_FILE"; exit 1
}

stop() {
  if running; then
    pid="$(cat "$PID_FILE")"
    kill "$pid"
    for _ in $(seq 20); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
    kill -0 "$pid" 2>/dev/null && kill -9 "$pid"
    echo "stopped (pid $pid)"
  else
    echo "not running"
  fi
  rm -f "$PID_FILE"
}

case "${1:-status}" in
  start) start ;;
  stop) stop ;;
  restart) stop; start ;;
  status)
    if running; then
      echo "running (pid $(cat "$PID_FILE"))"
      curl -fsS "http://127.0.0.1:$PORT/health"; echo
    else
      echo "not running"; exit 1
    fi ;;
  *) echo "usage: $0 {start|stop|restart|status}"; exit 2 ;;
esac
