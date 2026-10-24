"""CJ-Bench with the deployment Q8 weights and padding-free llama.cpp sequences."""
import argparse
import gzip
import json
import math
import os
import statistics
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

from decision.data.cj import DOMAINS
from decision.serving.gguf_batch import GGUFSequenceBatch
from decision.metrics import BenchmarkMetrics
from decision.processing import PROMPT_VERSION
from decision.utils import log, sha256, write_json
from scripts.evaluate import encode_pool, read_records


def log_probabilities(values, temperature):
    scaled = [v / temperature for v in values]
    maximum = max(scaled)
    normalizer = maximum + math.log(sum(math.exp(v - maximum) for v in scaled))
    return [v - normalizer for v in scaled]


def equal_length_batches(pool, capacity):
    buckets = defaultdict(list)
    for row in pool:
        buckets[len(row['input_ids'])].append(row)
    for length in sorted(buckets):
        bucket = buckets[length]
        for start in range(0, len(bucket), capacity):
            yield bucket[start:start + capacity]


def validate_engine(engine, rows, tokenizer, records, temperature):
    # Check complete rendered prompts, including Chinese, chat special tokens and options.
    for record, row in zip(records[:24], rows[:24]):
        prompt, _ = record['example'].render(tokenizer)
        if engine.tokenize(prompt) != row['input_ids']:
            raise ValueError('native and HF prompt tokenization mismatch')
    chosen = []
    for batch in equal_length_batches(rows, min(engine.parallel, 4)):
        if len(batch) >= 2:
            chosen.append(batch)
        if len(chosen) >= 8:
            break
    if not chosen:
        chosen = [[rows[0], rows[0]]]
    maximum, total, count, agreement = 0., 0., 0, 0
    for batch in chosen:
        together = engine.logits(batch)
        repeated = engine.logits(batch)
        if max(abs(a-b) for u,v in zip(together,repeated) for a,b in zip(u,v)) > 1e-6:
            raise ValueError('independent repeated batches changed logits')
        alone = [engine.logits([row])[0] for row in batch]
        for first, second in zip(together, alone):
            p = [math.exp(v) for v in log_probabilities(first, temperature)]
            q = [math.exp(v) for v in log_probabilities(second, temperature)]
            delta = max(abs(a-b) for a,b in zip(p,q))
            maximum = max(maximum, delta)
            total += sum(abs(a-b) for a,b in zip(p,q))
            count += len(p)
            agreement += int(p.index(max(p)) == q.index(max(q)))
    # A large change means the execution path must be investigated before full evaluation.
    if maximum > .02:
        raise ValueError(f'batch/single probability deviation too large: {maximum}')
    return {'prompt_tokenizer_checks': min(24, len(rows)), 'batches': len(chosen),
            'rows': sum(map(len, chosen)), 'top1_agreement_count': agreement,
            'max_probability_delta': maximum, 'mean_absolute_probability_delta': total / count,
            'repeated_logits_identical_within': 1e-6,
            'note': 'Batch/single numerical difference is reported separately from weight quantization.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--model', default='models/chinese-decision-0.8b-gguf')
    parser.add_argument('--gguf', default='models/chinese-decision-0.8b-gguf/chinese-decision-0.8b-Q8_0.gguf')
    parser.add_argument('--data', default='datasets/CJ-Bench')
    parser.add_argument('--parallel', type=int, default=32)
    parser.add_argument('--token-budget', type=int, default=2048)
    parser.add_argument('--pool-size', type=int, default=32768)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--quick-only', action='store_true')
    args = parser.parse_args()
    out, reference, model, data = map(Path, (args.out, args.reference, args.model, args.data))
    out.mkdir(parents=True, exist_ok=True)
    if (out/'quick.json').exists() or (out/'full.json').exists():
        raise ValueError('use a fresh output directory; existing reports are not overwritten')
    import torch
    import llama_cpp
    from transformers import AutoTokenizer
    torch.set_num_threads(args.threads)
    config = json.loads((model/'decision_config.json').read_text())
    ref = json.loads((reference/'evaluation_config.json').read_text())
    if config['prompt_version'] != PROMPT_VERSION or config['temperature'] != ref['temperature']:
        raise ValueError('reference/deployment prompt or temperature mismatch')
    conversion = json.loads((model/'conversion.json').read_text())
    temperature, max_length = config['temperature'], config['max_length']
    manifest = json.loads((data/'MANIFEST.json').read_text())
    if sha256(data/'MANIFEST.json') != ref['manifest_sha256']:
        raise ValueError('reference/dataset manifest mismatch')
    for domain, entry in manifest['domains'].items():
        path = data/entry['path']
        if path.stat().st_size != entry['bytes'] or sha256(path) != entry['sha256']:
            raise ValueError(f'benchmark file hash mismatch: {domain}')
    tokenizer = AutoTokenizer.from_pretrained(str(model))
    rows = torch.load(reference/'quick_tokens.pt', map_location='cpu', weights_only=False)
    selected = json.loads((reference/'quick_selected_ids.json').read_text())
    by_key = {(r['domain'],r['line'],r['case_id']): r for r in selected[:24]}
    records = []
    found = {}
    for domain in DOMAINS:
        if not any(k[0] == domain for k in by_key):
            continue
        for record in read_records(data, domain):
            key = (domain,record['line'],record['example'].case_id)
            if key in by_key:
                found[key] = record
            if len(found) == len(by_key):
                break
    records = [found[(r['domain'],r['line'],r['case_id'])] for r in selected[:24]]
    metadata = {'backend':'llama.cpp Q8_0', 'llama_cpp_python':llama_cpp.__version__,
                'runtime_module':llama_cpp.__file__, 'model':str(model), 'gguf':str(args.gguf),
                'gguf_sha256':sha256(args.gguf), 'conversion':conversion,
                'temperature':temperature, 'prompt_version':PROMPT_VERSION,
                'max_length':max_length, 'length_policy':'fail on overflow; no padding or truncation',
                'candidate_order':'original; no permutation', 'calibration':'existing validation temperature; no test fitting',
                'manifest_sha256':ref['manifest_sha256'], 'verified_files':ref['verified_files'],
                'selected_checkpoint':ref['selected_checkpoint'], 'reference':str(reference),
                'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'), 'gpu':torch.cuda.get_device_name(0),
                'flash_attention':False, 'arguments':vars(args),
                'evaluator_sha256':sha256(__file__),
                'batch_code_sha256':sha256(Path(__file__).parents[1]/'decision/gguf_batch.py')}
    write_json(out/'evaluation_config.json', metadata)
    log('gguf_start', **metadata)
    engine = GGUFSequenceBatch(args.gguf, tokenizer, parallel=args.parallel, context_size=max_length,
                               token_budget=args.token_budget, threads=args.threads, gpu_layers=999)
    try:
        validation = validate_engine(engine, rows, tokenizer, records, temperature)
        write_json(out/'batch_validation.json', validation)
        log('batch_validation', **validation)

        def evaluate(phase, pools, expected):
            metrics, counts, types, hard = BenchmarkMetrics(15), Counter(), defaultdict(Counter), Counter()
            n = tokens = batches = 0
            lengths = []
            started = last = time.monotonic()
            path = out/f'{phase}_predictions.jsonl.gz'
            with gzip.open(path, 'wt', encoding='utf-8', compresslevel=1) as stream:
                for pool in pools:
                    for batch in equal_length_batches(pool,args.parallel):
                        values = engine.logits(batch)
                        for row, logits in zip(batch,values):
                            logp = log_probabilities(logits,temperature)
                            metrics.add(row,logp)
                            result = {k:v for k,v in row.items() if k != 'input_ids'}
                            result.update(log_probabilities=logp, probabilities=[math.exp(v) for v in logp])
                            stream.write(json.dumps(result,ensure_ascii=False)+'\n')
                            counts[row['domain']] += 1
                            types[row['domain']][row['kind']] += 1
                            hard[row['domain']] += int(row['hard'])
                            n += 1
                            length = len(row['input_ids'])
                            lengths.append(length)
                            tokens += length
                        batches += 1
                        now = time.monotonic()
                        if now-last >= 30 or n == expected:
                            elapsed = now-started
                            progress = {'status':'running','phase':phase,'evaluated':n,'expected':expected,
                                        'percent':100*n/expected,'seconds':elapsed,'decisions_per_second':n/elapsed,
                                        'estimated_remaining_minutes':(expected-n)*elapsed/max(n,1)/60,
                                        'by_domain':dict(counts), 'time':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
                            stream.flush()
                            write_json(out/'progress.json',progress)
                            log('gguf_progress',**progress)
                            last = now
            elapsed = time.monotonic()-started
            if n != expected:
                raise ValueError(f'coverage mismatch: {n} != {expected}')
            if phase == 'full':
                for domain,entry in manifest['domains'].items():
                    if (counts[domain] != entry['decisions'] or dict(types[domain]) != entry['question_types']
                            or hard[domain] != entry['targets']['hard']):
                        raise ValueError(f'domain/type/hard coverage mismatch: {domain}')
            report = metadata | {'status':'completed','phase':phase,'metrics':metrics.result(),
                                 'coverage':{'evaluated':n,'expected':expected,'filtered':0,'truncated':0,
                                             'by_domain':dict(counts)},
                                 'timing':{'seconds':elapsed,'decisions_per_second':n/elapsed,'input_tokens':tokens,
                                           'max_input_tokens':max(lengths),'mean_input_tokens':statistics.mean(lengths),
                                           'batches':batches,'padding_tokens':0,
                                           'note':'Batched throughput includes input processing and output writing; GPU may be shared with reference.'},
                                 'predictions':str(path),'predictions_sha256':sha256(path),
                                 'completed_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
            write_json(out/f'{phase}.json',report)
            log('gguf_completed',phase=phase,all=report['metrics']['groups']['all'],seconds=elapsed)

        evaluate('quick',[rows],len(rows))
        if not args.quick_only:
            def pools():
                for domain in DOMAINS:
                    pool = []
                    for record in read_records(data,domain):
                        pool.append(record)
                        if len(pool) >= args.pool_size:
                            yield encode_pool(pool,tokenizer,max_length)
                            pool = []
                    if pool:
                        yield encode_pool(pool,tokenizer,max_length)
            evaluate('full',pools(),manifest['total_decisions'])
        write_json(out/'progress.json',{'status':'completed','time':time.strftime('%Y-%m-%dT%H:%M:%S%z')})
    finally:
        engine.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if '--out' in sys.argv:
            out = Path(sys.argv[sys.argv.index('--out')+1])
            out.mkdir(parents=True,exist_ok=True)
            write_json(out/'progress.json',{'status':'failed','error':str(exc)})
        traceback.print_exc()
        sys.exit(1)
