# Third-Party Notices

## n4ze3m/hmm

The option-letter decision objective, LoRA workflow, candidate formatting, and
System One response conventions in this project are adapted from
[n4ze3m/hmm](https://github.com/n4ze3m/hmm), revision
`78d8d4a62a41857ce67d903aadf504580feea832`, under the following license:

```text
MIT License

Copyright (c) 2026 n4ze3m

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## gulucaptain/Chinese-Jev

The optional `--rlcd` training term of `scripts/train.py` is adapted from
`src/chinese_jev/modeling.py` (`proper_reward`) and `src/chinese_jev/train.py`
(`decision_loss`) of [gulucaptain/Chinese-Jev](https://github.com/gulucaptain/Chinese-Jev),
revision `aed6193be5697488c8d0c6a77cc0c7e297329e61`, by the Chinese-Jev authors,
licensed under the Apache License, Version 2.0; a copy is in
[`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt). The adapted term is kept as the
frozen reference `legacy_rlcd_term` in `tests/test_rlcd_loss.py`: it reads this project's
option-letter logits and batch items, returns only the RLCD term, and defaults the spherical
weight to the value Chinese-Jev trains with. Training now computes it with
`RLCDLossConfig.chinese_jev()` in `decision/losses.py`, which that test holds equal to it.

## Qwen3.5-0.8B and Qwen3.5-2B

Base models: [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) and
[Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B), licensed under Apache-2.0.
Model weights are not distributed with this repository; exported
model directories keep the original `LICENSE` file.

## CJ-Dataset and CJ-Bench

Training data: [bbldCVer-hf/CJ-Dataset](https://huggingface.co/datasets/bbldCVer-hf/CJ-Dataset).
Benchmark: [bbldCVer-hf/CJ-Bench](https://huggingface.co/datasets/bbldCVer-hf/CJ-Bench).
Data files are not distributed with this repository. Their licenses are
source-specific (some sources are non-commercial); see each dataset's README and
SOURCES.md. This project grants no redistribution rights for the data.
