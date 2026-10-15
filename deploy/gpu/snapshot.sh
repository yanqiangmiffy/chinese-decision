#!/usr/bin/env bash
# 把训练目录里的 best adapter 复制成不可变快照 snapshots/step-N。训练会原地替换 best/，
# 服务只加载快照，避免训练中途换掉正在使用的权重。
set -euo pipefail
cd "$(dirname "$0")/../.."
PROJECT=${PROJECT:-$HOME/chinese-decision}
SOURCE=${RUN_DIR:-$PROJECT/runs/cj-ddp}/best
STEP=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['step'])" "$SOURCE/selection.json")
TARGET=snapshots/step-$STEP
if [ -d "$TARGET" ]; then echo "$TARGET already exists"; exit 0; fi
mkdir -p snapshots
cp -r "$SOURCE" "$TARGET.partial"
AFTER=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['step'])" "$SOURCE/selection.json")
if [ "$AFTER" != "$STEP" ]; then rm -rf "$TARGET.partial"; echo "best changed while copying, retry"; exit 1; fi
mv "$TARGET.partial" "$TARGET"
echo "snapshot: $TARGET"
