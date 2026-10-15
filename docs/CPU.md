# CPU deployment with Q8_0 GGUF

The GGUF backend preserves the existing `/v1/systemone` and `/api/alpha/decisions`
routes, three question types, prompt compiler, and candidate-only temperature
softmax. It evaluates the input and reads final-position logits; it does not
generate an answer token. `n_gpu_layers=0` forces CPU execution.

## Installation

Use Python 3.11 or later in an isolated environment:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[gguf]"
```

The tested llama-cpp-python version is 0.3.36. Its bundled llama.cpp must support
Qwen3.5. On Windows, a source installation requires Visual Studio C++ Build Tools
and a Windows SDK. Some CPU wheel indexes only provide older versions; those
versions may not load Qwen3.5. On the tested Windows machine, the package is built
for the local CPU with `GGML_CUDA=OFF` and `GGML_NATIVE=ON`.

The base package and GGUF extra do not require torch, PEFT or GPU attention
kernels. Training dependencies are available as `.[train]`, and Transformers
inference dependencies as `.[torch]`.

## Conversion

Get a current llama.cpp checkout and install its converter requirements in the
conversion environment. Conversion uses CPU PyTorch; a GPU is not required.
For this checkpoint use Transformers 5.19.0 or a compatible later version.

```powershell
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git local/tools/llama.cpp
.venv\Scripts\python.exe -m pip install -r local/tools/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
.venv\Scripts\python.exe -m pip install "transformers==5.19.0"
.venv\Scripts\python.exe scripts/convert_gguf.py --llama-cpp local/tools/llama.cpp
```

The script writes BF16 and Q8_0 GGUF files to
`models/chinese-decision-0.8b-gguf`, copies the tokenizer, chat template, decision
configuration and license, and records file sizes, SHA256 hashes and conversion
commands in `conversion.json`. It refuses to overwrite existing model files
unless explicitly invoked with `--overwrite`.

Both formats are exported directly from the source safetensors. `--no-mtp` is
intentional: the source config mentions an MTP layer but the merged checkpoint
has no MTP tensors. BF16 GGUF provides a baseline for separating conversion
differences from quantization differences. Export only Q8_0 with
`--formats q8_0` if the BF16 comparison file is unnecessary.

## Serving

```powershell
.venv\Scripts\python.exe scripts/serve.py --backend gguf --base models/chinese-decision-0.8b-gguf --gguf models/chinese-decision-0.8b-gguf/chinese-decision-0.8b-Q8_0.gguf --threads 8 --context-size 16384 --port 8787
```

The exported folder is self-contained; original safetensors are not needed for
serving. `--base` supplies tokenization and calibration configuration. The backend
checks all candidate-letter token IDs and checks source/GGUF tokenization of every
prompt. It resets the model sequence before every question and serializes model
evaluation because the llama.cpp context is mutable.

`--context-size` sets the runtime input capacity. The effective API limit is the
smaller of that capacity and `decision_config.json`'s maximum length. Requests
over that limit return HTTP 422 without truncation. Use 16384 to retain the
original model's full API input limit, with correspondingly higher memory use.

The original GPU backend remains the default. A compatibility CPU baseline is:

```powershell
.venv\Scripts\python.exe scripts/serve.py --base models/chinese-decision-0.8b --device cpu --threads 8
```

It loads FP32 parameters. GGUF keeps the source temperature (approximately
1.108); it has not been independently refitted after quantization. The existing
`calibrated` health field means the decision configuration is present, not that
the quantized model has passed a new calibration evaluation.

## Checks and reproducible benchmark

```powershell
.venv\Scripts\python.exe -m pip install httpx psutil
.venv\Scripts\python.exe scripts/verify_export.py --url http://127.0.0.1:8787
.venv\Scripts\python.exe scripts/benchmark_cpu.py --threads 8 --repeats 3
```

The benchmark runs Transformers FP32, BF16 GGUF and Q8_0 GGUF in separate
processes. It records input IDs, candidate logits/probabilities, repeated-request
stability, timings, load time and peak working set. Every decision starts from a
fresh sequence, so repeated prompts do not receive a prefix-cache speedup.
Reported model timings include evaluation and candidate softmax, and exclude
tokenization and HTTP. The API checks exercise both aliases and invalid schemas.

There are 15 diagnostic prompts, covering all question types, structured Unicode
input, 26 candidates and longer contexts. These comparisons measure output drift
and CPU performance, not labeled accuracy or calibration. The benchmark writes
`reports/cpu-benchmark.json` and separate per-backend JSON files. Results depend
on CPU, thread count, context length, build options and machine load.

## Local measurements (2026-10-15)

Windows, Intel Core i7-9700 (8 physical cores), 32 GB RAM, 8 inference threads,
2048-token runtime capacity, llama-cpp-python 0.3.36 with AVX2/FMA/OpenMP.
Each of 15 prompts was evaluated three times after warmup. The short-input
latency below is the median of 36 measurements over 12 prompts of 64–89 tokens;
it excludes HTTP and tokenization. Peak RSS is the process peak working set.

| Backend | Short-input p50 | Peak RSS |
|---|---:|---:|
| Transformers FP32 | 627 ms | 3.47 GiB |
| llama.cpp BF16 | 640 ms | 2.04 GiB |
| llama.cpp Q8_0 | 524 ms | 1.38 GiB |

Q8_0 is approximately 1.20 times as fast on these short prompts and uses about
60% less peak memory than Transformers FP32. It is not consistently faster on
longer inputs: the 852-token case took a median 6.65 s with Q8_0 versus 5.12 s
with Transformers FP32. A separate three-question thread comparison favored 8
threads (528 ms median) over 4 (683 ms) and 2 (1056 ms) on this machine.

All 15 top candidates matched across all three backends. The maximum absolute
candidate probability difference was 0.00204 (0.204 percentage points) between
Transformers FP32 and BF16 GGUF, and 0.03177 (3.177 percentage points) between
Transformers FP32 and Q8_0. All three backends returned identical probabilities
across repetitions interleaved with other questions. No temperature refitting
was performed. This small diagnostic set does not establish accuracy or
calibration preservation on production data.

The original safetensors file is 1,504,827,608 bytes. The exported BF16 GGUF is
1,516,745,056 bytes and Q8_0 is 811,843,936 bytes. Per-question outputs and raw
timings are in the locally generated `reports/cpu-benchmark.json`; thread tuning
results are in `reports/cpu-threads.json`.

The running CPU HTTP service was also checked with the original 16384-token
input capacity. Both API aliases returned identical probabilities; five invalid
request checks returned 422, and four concurrent requests remained consistent
with isolated requests. A five-request loopback measurement of the example
payload (three questions per request) had median latency 1.777 s, including
tokenization, three serial model evaluations and response handling. Increasing
the context capacity from 2048 to 16384 did not change example probabilities.
The raw HTTP results are in `reports/cpu-api.json`. All 34 automated tests passed.
