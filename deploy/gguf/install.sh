#!/usr/bin/env bash
# Build the CPU-only serving environment for the Q8_0 GGUF model in $ROOT/.venv.
# No PyTorch: llama-cpp-python runs the model, transformers only supplies the tokenizer.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
JOBS="${JOBS:-16}"

cd "$ROOT"
[ -x .venv/bin/python ] || "$PYTHON" -m venv .venv
.venv/bin/python -m pip install -q -U pip
# llama-cpp-python is published as an sdist; compile llama.cpp for this CPU in parallel.
CMAKE_BUILD_PARALLEL_LEVEL="$JOBS" CMAKE_ARGS="-DGGML_NATIVE=ON" \
  .venv/bin/python -m pip install "llama-cpp-python==0.3.36"
.venv/bin/python -m pip install "transformers>=5.3,<6" fastapi uvicorn
# The package metadata asks for Python 3.11, so serving runs from source via PYTHONPATH
# instead of `pip install -e .`; check that the serving code compiles on this interpreter.
.venv/bin/python -m compileall -q decision scripts/serve.py
.venv/bin/python -c "import llama_cpp, transformers; print('llama-cpp-python', llama_cpp.__version__, '| transformers', transformers.__version__)"
