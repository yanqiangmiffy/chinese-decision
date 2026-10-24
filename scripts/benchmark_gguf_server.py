"""Isolated batch-one latency on identical stratified CJ-Bench inputs."""
import argparse
import json
import os
import resource
import statistics
import subprocess
import time
from collections import Counter
from pathlib import Path

from decision.serving.backends import candidate_probabilities
from decision.processing import LETTERS
from decision.utils import log, write_json


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values)-1)*fraction
    lower = int(position)
    return values[lower]+(values[min(lower+1,len(values)-1)]-values[lower])*(position-lower)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=['transformers','transformers-cpu','gguf-gpu','gguf-cpu'],required=True)
    parser.add_argument('--reference',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--per-stratum',type=int,default=2)
    parser.add_argument('--model',default='models/chinese-decision-0.8b')
    parser.add_argument('--gguf',default='models/chinese-decision-0.8b-gguf/chinese-decision-0.8b-Q8_0.gguf')
    args=parser.parse_args()
    if args.backend.endswith('-cpu') and os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU benchmark requires CUDA_VISIBLE_DEVICES to be empty')
    import torch
    import psutil
    from transformers import AutoTokenizer
    torch.set_num_threads(args.threads)
    rows=torch.load(Path(args.reference)/'quick_tokens.pt',map_location='cpu',weights_only=False)
    selected=[]
    counts=Counter()
    for row in rows:
        key=(row['domain'],row['kind'])
        if counts[key] < args.per_stratum:
            selected.append(row)
            counts[key] += 1
    longest=max(rows,key=lambda r:len(r['input_ids']))
    if longest not in selected:
        selected.append(longest)
    temperature=json.loads((Path(args.model)/'decision_config.json').read_text())['temperature']
    start=time.perf_counter()
    if args.backend.startswith('transformers'):
        from decision.serving.backends import TransformersBackend
        engine=TransformersBackend(args.model,device='cpu' if args.backend.endswith('-cpu') else 'cuda',threads=args.threads)
        def infer(row):
            return engine.logits(row['input_ids'],len(row['target']))
        runtime={'torch':torch.__version__,'dtype':str(next(engine.model.parameters()).dtype),'device':engine.device}
    else:
        import llama_cpp
        tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
        gpu=args.backend=='gguf-gpu'
        engine=llama_cpp.Llama(model_path=args.gguf,n_gpu_layers=999 if gpu else 0,
                               offload_kqv=gpu,op_offload=gpu,n_ctx=16384,
                               n_threads=args.threads,n_threads_batch=args.threads,n_batch=512,
                               flash_attn=False,logits_all=False,verbose=False)
        letters=[tokenizer.encode(x,add_special_tokens=False)[0] for x in LETTERS]
        def infer(row):
            engine.reset()
            engine.eval(row['input_ids'])
            pointer=llama_cpp.llama_get_logits_ith(engine._ctx.ctx,-1)
            return [float(pointer[i]) for i in letters[:len(row['target'])]]
        runtime={'llama_cpp_python':llama_cpp.__version__}
    load_seconds=time.perf_counter()-start
    cases=[]
    for row in selected:
        # Shape-specific warm-up. Every infer resets all hybrid state; no prefix cache reuse.
        infer(row)
        samples=[]
        predictions=[]
        for _ in range(args.repeats):
            start=time.perf_counter()
            probabilities=candidate_probabilities(infer(row),temperature)
            samples.append((time.perf_counter()-start)*1000)
            predictions.append(probabilities)
        repeat_delta=max(abs(p-q) for values in predictions for p,q in zip(values,predictions[0]))
        case={k:v for k,v in row.items() if k!='input_ids'}
        case.update(input_tokens=len(row['input_ids']),samples_ms=samples,probabilities=predictions[0],
                    repeat_max_probability_delta=repeat_delta)
        cases.append(case)
        log('latency_case',backend=args.backend,domain=row['domain'],kind=row['kind'],
            tokens=len(row['input_ids']),median_ms=statistics.median(samples))
    def summarize(items):
        values=[v for case in items for v in case['samples_ms']]
        return {'n_cases':len(items),'n_samples':len(values),'p50_ms':statistics.median(values),
                'p95_ms':percentile(values,.95),'mean_ms':statistics.mean(values),
                'min_input_tokens':min(c['input_tokens'] for c in items),
                'max_input_tokens':max(c['input_tokens'] for c in items)} if values else None
    report={'backend':args.backend,'runtime':runtime,'threads':args.threads,'repeats':args.repeats,
            'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'temperature':temperature,
            'scope':'warm batch-one pretokenized input through final candidate probabilities; excludes tokenization/HTTP/network',
            'state_policy':'independent requests; full hybrid state reset; no prefix cache reuse',
            'sampling':'first two seed-42 quick-set rows per domain/type, plus longest quick-set input',
            'load_seconds':load_seconds,'rss_bytes':psutil.Process().memory_info().rss,
            'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'all':summarize(cases),'short_under_256_tokens':summarize([c for c in cases if c['input_tokens']<256]),
            'by_domain':{domain:summarize([c for c in cases if c['domain']==domain]) for domain in sorted({c['domain'] for c in cases})},
            'cases':cases,'hardware':subprocess.run(['lscpu'],capture_output=True,text=True).stdout,
            'gpu_processes_after':subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True).stdout,
            'pid':os.getpid(),'completed_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    write_json(args.out,report)
    log('latency_completed',backend=args.backend,summary=report['all'])
    engine.close()


if __name__=='__main__':
    main()
