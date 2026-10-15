#!/usr/bin/env bash
# Copy the serving code, these scripts and the Q8_0 GGUF model to a Linux host.
# Usage: deploy/gguf/push.sh user@host   (run from Git Bash or any POSIX shell)
# TARGET and SSH_KEY can also live in deploy/gguf/deploy.env (not committed; see deploy.env.example).
set -euo pipefail
cd "$(dirname "$0")/../.."
if [ -f deploy/gguf/deploy.env ]; then . deploy/gguf/deploy.env; fi
TARGET="${1:-${TARGET:-}}"
TARGET="${TARGET:?usage: deploy/gguf/push.sh user@host, or set TARGET in deploy/gguf/deploy.env}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_DIR="${REMOTE_DIR:-chinese-decision}"
MODEL="${MODEL:-models/chinese-decision-0.8b-gguf}"
GGUF=chinese-decision-0.8b-Q8_0.gguf
GGUF_SHA256=6a71cc35bd83f8b050499567faec59e9ae05778b75b1e2119d27d45b80f7e68b
SSH=(ssh -i "$SSH_KEY" -o IdentitiesOnly=yes "$TARGET")
REMOTE_MODEL="$REMOTE_DIR/models/chinese-decision-0.8b-gguf"

"${SSH[@]}" "mkdir -p $REMOTE_MODEL $REMOTE_DIR/logs $REMOTE_DIR/run"
# Serving code from the committed tree (LF line endings), plus the deploy scripts as they are now.
git -c core.autocrlf=false archive --format=tar HEAD decision scripts/serve.py examples pyproject.toml \
  LICENSE LICENSES THIRD_PARTY_NOTICES.md | "${SSH[@]}" "tar -xf - -C $REMOTE_DIR"
# Windows checkouts may carry CRLF, which bash on the host rejects.
tar -cf - deploy/gguf | "${SSH[@]}" "tar -xf - -C $REMOTE_DIR && sed -i 's/\r\$//' $REMOTE_DIR/deploy/gguf/*.sh && chmod +x $REMOTE_DIR/deploy/gguf/*.sh"

scp -i "$SSH_KEY" -o IdentitiesOnly=yes "$MODEL"/{LICENSE,README.md,chat_template.jinja,config.json,conversion.json,decision_config.json,tokenizer.json,tokenizer_config.json} \
  "$TARGET:$REMOTE_MODEL/"
# Optional release metadata (older exports lack it); -p keeps the mtime the server reports as updated_at.
for name in training_summary.json validation_report.json; do
  if [ -f "$MODEL/$name" ]; then scp -p -i "$SSH_KEY" -o IdentitiesOnly=yes "$MODEL/$name" "$TARGET:$REMOTE_MODEL/"; fi
done
if "${SSH[@]}" "echo '$GGUF_SHA256  $REMOTE_MODEL/$GGUF' | sha256sum -c --status - 2>/dev/null"; then
  echo "$GGUF already on $TARGET with the expected SHA256; not copied again"
else
  scp -i "$SSH_KEY" -o IdentitiesOnly=yes "$MODEL/$GGUF" "$TARGET:$REMOTE_MODEL/"
  "${SSH[@]}" "echo '$GGUF_SHA256  $REMOTE_MODEL/$GGUF' | sha256sum -c -"
fi
echo "next on $TARGET: cd $REMOTE_DIR && deploy/gguf/install.sh && deploy/gguf/service.sh start"
