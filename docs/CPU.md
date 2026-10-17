# CPU deployment with Q8_0 GGUF (0.8B)

For the completed 2B model and its separate quantization/CPU checks, see
[Chinese Decision 2B CPU deployment](CPU_2B.md). The measurements below describe 0.8B. See the [model overview](MODELS.md) for all four packages and their separate evaluation states.

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

## Linux server measurements (2026-10-15)

Server: Intel Xeon Gold 5318Y, 48 physical cores / 96 logical CPUs, 512 GB RAM,
NVIDIA A800 80 GB PCIe. GPU tests use physical GPU 0. Other GPUs continue their
existing training workloads; these measurements describe this shared server.

The Linux CUDA runtime was built on the server from llama-cpp-python 0.3.36
source, using CUDA 12.8, architecture 80, and native AVX512/VNNI CPU support.
Its independent installation is `local/gguf-runtime-0.3.36/python` under the
server's project root.
The existing training Python environment was not modified. The deployment
weights were uploaded and verified against the local Q8_0 SHA256 checksum.

### Full CJ-Bench result

The full GPU0 comparison covers all 307900 released decisions: 304822 hard
targets and 3078 soft targets. Input tokens, candidate order, source checkpoint
and deployment temperature 1.10813307762146 match. There are no filtered or
truncated cases; total input tokens are 68515700 and the longest input is 6850
tokens. Dataset and weight checksums were verified again after evaluation.

| Metric | Transformers BF16 | Q8_0 GGUF | Change |
|---|---:|---:|---:|
| Hard accuracy | 83.1052% | 83.1233% | +0.0180 pp |
| Domain macro accuracy | 75.7376% | 75.8092% | +0.0716 pp |
| Distribution NLL | 0.431318 | 0.431293 | -0.000025 |
| ECE (15 bins) | 0.2706% | 0.2696% | -0.0010 pp |
| Brier | 0.223833 | 0.223815 | -0.000018 |
| Expected score MAE | 0.584427 | 0.583973 | -0.000453 |

Accuracy and ECE use hard targets; NLL and Brier include soft targets. Score
MAE uses score questions. Lower NLL/ECE/Brier/MAE is better.

| Domain | Decisions | Reference accuracy | Q8 accuracy | Change |
|---|---:|---:|---:|---:|
| general | 100000 | 72.2602% | 72.2333% | -0.0268 pp |
| medical | 200000 | 88.8530% | 88.8885% | +0.0355 pp |
| legal | 4300 | 67.8372% | 67.8372% | +0.0000 pp |
| finance | 3600 | 74.0000% | 74.2778% | +0.2778 pp |

Per-domain calibration gives a more qualified picture than pooled ECE. ECE
increases slightly in each domain even though pooled ECE decreases slightly;
opposite bin errors can cancel when domains are pooled.

| Domain | Reference ECE | Q8 ECE | Change |
|---|---:|---:|---:|
| general | 1.6827% | 1.7092% | +0.0265 pp |
| medical | 1.1195% | 1.1389% | +0.0194 pp |
| legal | 4.0525% | 4.0561% | +0.0036 pp |
| finance | 2.4489% | 2.6043% | +0.1554 pp |
| Four-domain mean | 2.3259% | 2.3771% | +0.0512 pp |

The last row is the arithmetic mean of four domain ECE values, distinct from
pooled ECE. Accuracy preservation does not establish unchanged calibration.

Across all 307900 decisions, top-choice agreement is 98.9227%.
Among hard targets, 1185 previously correct answers become incorrect and
1240 previously incorrect answers become correct. Mean absolute candidate
probability difference is 0.002412 (0.241 percentage points); the 95th
percentile of each question's maximum difference is 0.023567, and the largest
difference is 0.226881. Similar aggregate accuracy therefore coexists with
changes to some individual answers and confidence values.

The largest-change example is a medical choice case. Both backends choose the
same incorrect option C, whose probability rises from 0.521165 to 0.747657;
the correct option E falls from 0.465571 to 0.238690. Its ID and complete
candidate distributions are retained in `comparison.json`.

This is an end-to-end comparison of the deployed weight formats and backends.
The separate BF16 GGUF control below covers 1200 balanced inputs. Full CPU
accuracy/calibration was not measured; CPU checks cover 25 inputs. The
comparison does not fit temperatures or thresholds on test labels.

The full-pass elapsed times include concurrent GPU work and temporary pauses
for isolated latency measurements. Batch sizes also differ, so full-pass
throughput is not used as a direct speed comparison. See the request latency
measurements below.

Raw reports: `reports/server-gguf-20261015/summary.json`,
`reference/full.json`, `q8-b8192/full.json` and `comparison.json`. Server run:
`runs/cj-bench/gguf-q8-20261015/` under the server's project root.
Per-question compressed predictions remain on the server; their SHA256
checksums and the paired largest-change examples are in the reports.

### Independent request latency

Latency uses the same 25 CJ-Bench inputs for every backend: two seeded quick-set
inputs from each of the 12 domain/type strata, plus the longest quick-set input.
Each shape is warmed once, then measured three times. Inputs have 92–6521 tokens;
the short subset contains 16 inputs / 48 measurements with 92–255 tokens.
Timings cover pretokenized independent inference and candidate probabilities,
excluding tokenization, HTTP and network. Every request clears hybrid state.
GPU batch-one measurements run with the full-evaluation workers temporarily
stopped. Server CPU measurements use the same 16384-token runtime capacity as
the service. The CPU FP32 reference uses the official Torch implementations
instead of the installed CUDA-only optional convolution/FLA wrappers.

| Backend | Threads | Short p50 | All 25 inputs p50 | 6521-token input median | Process peak RSS |
|---|---:|---:|---:|---:|---:|
| Transformers CPU FP32 | 16 | 319.8 ms | 371.3 ms | 17.05 s | 7.44 GiB |
| llama.cpp Q8_0 CPU | 8 | 555.7 ms | 697.0 ms | 23.20 s | 2.20 GiB |
| llama.cpp Q8_0 CPU | 16 | 347.2 ms | 427.0 ms | 13.95 s | 2.19 GiB |
| Transformers GPU BF16 | 8 | 44.3 ms | 44.4 ms | 93.4 ms | — |
| llama.cpp Q8_0 GPU | 8 | 15.6 ms | 17.9 ms | 369.6 ms | — |

On this sample, 16 CPU threads outperform 8 for Q8. Relative to CPU FP32 at
16 threads, Q8 short latency is 8.55% higher, while the longest input is about
1.22 times as fast and process peak RSS is about 70.5% lower. Peak RSS includes
the benchmark harness and its Torch/tokenizer imports. The GPU Q8 path is about
2.84 times as fast on short inputs, but about 3.96 times as slow on the longest
input. These are sample measurements; the balanced sample and added longest
input do not represent the full benchmark's input-length distribution.

All 25 top candidates match across CPU FP32, CPU Q8 and GPU Q8 in these
batch-one checks. The maximum absolute candidate probability difference is
0.02480 between CPU FP32 and CPU Q8, and 0.02760 between CPU Q8 and GPU Q8.
Repeated outputs are identical for each backend. Q8 CPU outputs at 8 and 16
threads are also identical. Full CJ-Bench accuracy is measured separately on
GPU; these 25 inputs do not establish identical full CPU accuracy/calibration.

Raw latency reports are under the locally generated
`reports/server-gguf-20261015/` and the server run directory
`runs/cj-bench/gguf-q8-20261015/` (under the server's project root).

### Reusing the server installation

The run keeps its evaluation code separate from the active training checkout.
Use its frozen reference inputs and a fresh output directory to repeat the Q8
evaluation. The system C++ library preload below addresses this server's older
Conda `libstdc++`; it applies only to the launched process.

```bash
ROOT=/path/to/chinese-decision   # project root on the server
JOB="$ROOT/runs/cj-bench/gguf-q8-20261015"
cd "$ROOT"
export PYTHONPATH="$ROOT/local/gguf-runtime-0.3.36/python:$JOB/code-used:$ROOT"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"

CUDA_VISIBLE_DEVICES=0 .venv/bin/python "$JOB/code-used/scripts/evaluate_gguf.py" \
  --reference "$JOB/reference" --out "$JOB/q8-repeat" \
  --parallel 32 --token-budget 8192 --pool-size 32768 --threads 4

CUDA_VISIBLE_DEVICES='' .venv/bin/python "$JOB/code-used/scripts/benchmark_gguf_server.py" \
  --backend gguf-cpu --reference "$JOB/reference" \
  --out "$JOB/cpu-repeat.json" --threads 16 --repeats 3
```

The evaluator reads final-position candidate logits directly: it does not sample
or generate text. It verifies dataset/model checksums, preserves candidate order,
uses the existing temperature 1.10813307762146, and rejects overlength inputs.
Native multi-sequence batches contain equal true input lengths with distinct
sequence IDs and no padding. The runtime clears all attention/recurrent state
between independent batches. CPU serving retains the separate batch-one path.

### BF16 GGUF control on 1200 balanced inputs

The GPU control uses 100 fixed seed-42 inputs from each domain/type stratum,
1192 hard targets and 8 soft targets. All three paths use the same source
weights, input tokens, option ordering and deployment temperature. The balanced
sample has different domain weights from the 307900-decision full benchmark;
its overall accuracy is not directly comparable to the full overall accuracy.

| GPU backend | Accuracy (hard) | Distribution NLL | ECE (15 bins) | Brier |
|---|---:|---:|---:|---:|
| Transformers BF16 | 73.0705% | 0.605766 | 3.4345% | 0.337966 |
| llama.cpp BF16 GGUF | 73.4060% | 0.605027 | 2.5705% | 0.337455 |
| llama.cpp Q8_0 GGUF | 73.5738% | 0.604997 | 2.3818% | 0.337432 |

BF16 GGUF versus Q8 has 1184/1200 matching top candidates (98.67%); five hard
answers become incorrect and seven become correct. Mean absolute probability
difference per candidate is 0.003516, the 95th percentile of each question's
maximum difference is 0.024303, and the largest difference is 0.127219. Thus,
similar aggregate accuracy does not mean identical probability outputs.

The Transformers-to-BF16-GGUF comparison also changes some answers: top-choice
agreement is 98.17%, mean absolute probability difference is 0.002826 and the
maximum is 0.049478. Backend arithmetic and batching contribute to end-to-end
differences alongside weight quantization. A separate GPU check of 27 inputs
found identical repeated batch logits, all top choices matching batch-one
results, and maximum batch/batch-one probability difference 0.016778 for Q8.
These diagnostic counts are reported as checks, not an assertion of exact
CPU/GPU or batch-size equivalence on the entire dataset. No temperatures or
thresholds were fitted on benchmark labels.
