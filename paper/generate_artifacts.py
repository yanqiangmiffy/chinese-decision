"""Rebuild paper tables and vector figures from the bundled evidence snapshot.

python generate_artifacts.py
python generate_artifacts.py --refresh-snapshot  # requires the original repository reports
"""
from pathlib import Path
import argparse
import hashlib
import json
import statistics

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
DOMAINS = ['general', 'medical', 'legal', 'finance']
LABELS = ['General', 'Medical', 'Legal', 'Finance']
SOURCE_RUNS = {
    '08b_2000': 'runs/cj-bench/step-2000-20261014-095033/full.json',
    '08b_3000': 'runs/cj-bench/latest-best-20261014-133012/full.json',
    '08b_5000': 'runs/cj-bench/latest-best-20261014-174339/full.json',
    '08b_10000': 'runs/cj-bench/latest-best-20261015-101241/full.json',
    '2b_1000': 'reports/cj-bench-2b-comparison-20261016/full.json',
    '2b_2000': 'reports/cj-bench-2b-step2000-20261016/full.json',
    'merged_T': 'reports/server-gguf-20261015/reference/full.json',
    'q8_T': 'reports/server-gguf-20261015/q8-b8192/full.json',
}


def read_source(name, manifest):
    path = REPO / name
    content = path.read_bytes()
    manifest.append({'path': name, 'sha256': hashlib.sha256(content).hexdigest()})
    return json.loads(content)


def refresh():
    manifest = []
    runs = {}
    for key, name in SOURCE_RUNS.items():
        d = read_source(name, manifest)
        assert d['status'] == 'completed', (key, d['status'])
        assert d['coverage']['evaluated'] == 307900
        assert d['coverage']['filtered'] == d['coverage']['truncated'] == 0
        assert d['metrics']['groups']['all']['hard_n'] == 304822
        runs[key] = {k: d.get(k) for k in ['selected_checkpoint', 'coverage', 'timing', 'metrics',
                                      'temperature', 'adapter_sha256', 'completed_at']}
        runs[key]['metrics'] = dict(runs[key]['metrics'])
        runs[key]['metrics']['groups'] = {
            k: v for k, v in runs[key]['metrics']['groups'].items()
            if k == 'all' or k.startswith(('domain:', 'type:', 'domain_type:'))
        }
    comparison = read_source('reports/cj-bench-2b-comparison-20261016/comparison.json', manifest)
    comparison2 = read_source('reports/cj-bench-2b-step2000-20261016/comparison.json', manifest)
    quant = read_source('reports/server-gguf-20261015/comparison.json', manifest)
    audit = read_source('reports/candidate-count-audit-20261016/analysis.json', manifest)
    calibration = read_source('models/chinese-decision-0.8b/validation_report.json', manifest)
    train = read_source('models/chinese-decision-0.8b/training_summary.json', manifest)
    cfg08 = read_source('runs/cj-ddp/config.json', manifest)
    cfg2 = read_source('reports/cj-bench-2b-step2000-20261016/training_config.json', manifest)
    lat08 = read_source('runs/cj-bench/latest-best-20261015-101241/latency.json', manifest)
    lat2 = read_source('reports/cj-bench-2b-step2000-20261016/latency.json', manifest)
    cpu = read_source('reports/cpu-benchmark.json', manifest)
    cpu_summary = {}
    for key, backend in cpu['backends'].items():
        short = [t for c in backend['cases'] if 64 <= c['input_tokens'] <= 89 for t in c['samples_ms']]
        assert len(short) == 36, (key, len(short))
        cpu_summary[key] = {'short_median_ms': statistics.median(short),
                            'peak_rss_gib': backend['peak_rss_bytes'] / (1024**3),
                            'short_measurements': len(short)}
    data = {'snapshot_date': '2026-10-16', 'runs': runs, 'paper_baselines': comparison['paper'],
            'trained_decisions_2b': [comparison['local_runs']['qwen35_2b_rlcd_step1000']['trained_samples'],
                                     comparison2['current']['trained_samples']],
            'candidate_audit_totals': audit['totals'], 'calibration_temperature': calibration['temperature'],
            'calibration_hard_n': calibration['calibration_hard_n'],
            'quantization_agreement': quant['groups'], 'training_08b': train,
            'training_config_08b': cfg08, 'training_config_2b': cfg2,
            'latency_08b': lat08, 'latency_2b': lat2, 'cpu_diagnostics': cpu_summary,
            'baseline_note': comparison['paper_note']}
    for name in ['decision/model.py', 'decision/schema.py', 'decision/data.py', 'decision/metrics.py',
                 'decision/rlcd.py', 'scripts/train.py', 'scripts/finalize.py', 'docs/CPU.md',
                 'runs/cj-ddp/world_size_migration.json']:
        content = (REPO / name).read_bytes()
        manifest.append({'path': name, 'sha256': hashlib.sha256(content).hexdigest()})
    for name, d in [('evidence.json', data), ('source_manifest.json', manifest)]:
        (ROOT / 'data' / name).write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding='utf-8')


def table(name, caption, label, spec, header, rows, wide=False, note='', size='small'):
    env = 'table*' if wide else 'table'
    text = '\\begin{' + env + '}[t]\n\\centering\n\\' + size + '\n'
    text += '\\setlength{\\tabcolsep}{4pt}\n'
    text += '\\caption{' + caption + '}\n\\label{' + label + '}\n'
    text += '\\begin{tabular}{' + spec + '}\n\\toprule\n' + header + ' \\\\\n\\midrule\n'
    text += '\n'.join(' & '.join(map(str, row)) + ' \\\\' for row in rows)
    text += '\n\\bottomrule\n\\end{tabular}\n'
    if note:
        text += '\\par\\vspace{3pt}\\begin{minipage}{' + ('\\textwidth' if wide else '\\columnwidth') + '}\\footnotesize ' + note + '\\end{minipage}\n'
    text += '\\end{' + env + '}\n'
    (ROOT / 'tables' / (name + '.tex')).write_text(text, encoding='utf-8')


def tables(d):
    runs = d['runs']
    def group(key, sub='all'):
        return runs[key]['metrics']['groups'][sub]
    pct = lambda x: f'{100*x:.2f}'
    f3 = lambda x: f'{x:.3f}'
    table('data', 'Training and full CJ-Bench sizes. Accuracy and ECE use only hard test targets.',
          'tab:data', 'lrrr', 'Domain & Train & Test & Hard test',
          [[l, f'{n:,}', f"{group('08b_10000','domain:'+k)['n']:,}",
            f"{group('08b_10000','domain:'+k)['hard_n']:,}"]
           for k,l,n in zip(DOMAINS,LABELS,[10000000,3000000,24000,24000])]
          + [['Total','13,048,000','307,900','304,822']], size='footnotesize')
    table('training', 'Recorded training configurations. The 2B run is ongoing at the evidence snapshot.',
          'tab:training', 'p{0.40\\columnwidth}rr', 'Setting & 0.8B & 2B', [
              ['Objective','CE','CE + RLCD'],['LoRA rank / scaling','32 / 64','32 / 64'],
              ['Dropout','0.05','0.05'],['Peak learning rate',r'$10^{-4}$',r'$10^{-4}$'],
              ['Nominal minimum batch','1,024','2,048'],['Global token budget','262,144','524,288'],
              ['Bucketing pool','4,096','4,096'],['Maximum input','16,384','16,384'],
              ['Planned epochs','1','1'],['Reported latest step','10,000','2,000'],
              ['Final optimizer step','10,132','Ongoing'],['RLCD perturbations','--','4'],
              [r'RLCD $\sigma$ schedule','--',r'$0.4\to0.1$'],['Random seed','42','42']
          ], size='footnotesize', note='Token budgets are padded-token limits shared across ranks. Accumulation can exceed the nominal minimum batch.')
    names = {'qwen35_2b':'Qwen3.5-2B (imported)', 'open_jev':'Open-Jev', 'semif':'SemIF',
             'laya':'Laya Multilingual', 'jev':'Jev',
             'chinese_jev_general_and_specialists':'Chinese-Jev General'}
    rows = [[names[k], f"{v['accuracy_pct'][0]:.2f}", f"{v['ece_pct'][0]:.2f}"]
            for k,v in d['paper_baselines'].items()]
    for key, name in [('08b_10000','Ours 0.8B (step 10000)'),('2b_1000','Ours 2B (step 1000)'),('2b_2000','Ours 2B (step 2000)')]:
        g = group(key,'domain:general'); rows.append([name,pct(g['accuracy']),pct(g['ece'])])
    table('general', 'General-domain comparison. All local models use $T=1$; 2B results are preliminary.',
          'tab:general', 'lrr', r'Model & ACC (\%) $\uparrow$ & ECE (\%) $\downarrow$', rows,
          note=r'Baseline rows are imported from Chinese-Jev~\citep{wang2026chinesejev}, not reproduced. The decision-fine-tuned 2B members differ from the imported Qwen baseline.',size='footnotesize')
    rows = []
    for key,v in d['paper_baselines'].items():
        name = 'Chinese-Jev-S (3 models)' if key=='chinese_jev_general_and_specialists' else names[key]
        nums = []
        for a,e in zip(v['accuracy_pct'][1:],v['ece_pct'][1:]): nums += [f'{a:.2f}',f'{e:.2f}']
        rows.append([name]+nums+[f"{v['professional_macro_accuracy_pct']:.2f}",f"{v['professional_macro_ece_pct']:.2f}"])
    for key,name in [('08b_10000','Ours 0.8B (single model)'),('2b_1000','Ours 2B (step 1000)'),('2b_2000','Ours 2B (step 2000)')]:
        gs=[group(key,'domain:'+k) for k in DOMAINS[1:]];nums=[]
        for g in gs: nums += [pct(g['accuracy']),pct(g['ece'])]
        rows.append([name]+nums+[pct(sum(g['accuracy'] for g in gs)/3),pct(sum(g['ece'] for g in gs)/3)])
    table('professional', r'Professional-domain comparison: accuracy / ECE (\%). Each local row uses one checkpoint for all domains; 2B snapshots are preliminary and uncalibrated.',
          'tab:professional', 'lrrrrrrrr',
          r'& \multicolumn{2}{c}{Medical} & \multicolumn{2}{c}{Legal} & \multicolumn{2}{c}{Finance} & \multicolumn{2}{c}{Professional mean} \\ Model & ACC $\uparrow$ & ECE $\downarrow$ & ACC $\uparrow$ & ECE $\downarrow$ & ACC $\uparrow$ & ECE $\downarrow$ & ACC $\uparrow$ & ECE $\downarrow$',
          rows,wide=True,note=r'All baseline values are imported from Chinese-Jev~\citep{wang2026chinesejev}. The last baseline combines three specialist models. Means are unweighted across the three domains.',size='small')
    rows=[]
    for key,name in [('08b_10000','0.8B / step 10000'),('2b_1000','2B / step 1000'),('2b_2000','2B / step 2000')]:
        g=group(key)
        rows.append([name,pct(g['accuracy']),pct(runs[key]['metrics']['macro_domain_accuracy']),
                     pct(g['ece']),f3(g['distribution_nll']),f3(g['brier']),f3(g['expected_score_mae'])])
    table('aggregate', 'Full-benchmark aggregate results at $T=1$. Macro ACC averages four domains; score MAE is restricted to ordinal questions.',
          'tab:aggregate','lrrrrrr',r'Model / checkpoint & Micro ACC (\%) & Macro ACC (\%) & ECE (\%) & NLL & Brier & Score MAE',rows,wide=True)
    rows=[]
    for key,name in [('08b_10000','0.8B / 10000'),('2b_2000','2B / 2000')]:
        for kind in ['noul','choice','score']:
            g=group(key,'type:'+kind)
            rows.append([name, '\\texttt{'+kind+'}', f"{g['n']:,} / {g['hard_n']:,}",pct(g['accuracy']),pct(g['ece']),f3(g['distribution_nll']),f3(g['expected_score_mae']) if g['score_n'] else '--'])
    table('types','Question-type results at $T=1$. The 2B row group is preliminary. The total/hard counts identify the metric denominators.',
          'tab:types','llrrrrr',r'Model / step & Type & Total / hard & ACC (\%) & ECE (\%) & NLL & Score MAE', rows,wide=True)
    rows=[]
    for step in [2000,3000,5000,10000]:
        key=f'08b_{step}';gs=[group(key,'domain:'+k) for k in DOMAINS];g=group(key)
        assert runs[key]['selected_checkpoint']['step']==step
        rows.append([f'{step:,}', f3(runs[key]['selected_checkpoint']['value'])]+[pct(x['accuracy']) for x in gs]+
                    [pct(sum(x['accuracy'] for x in gs[1:])/3),pct(sum(x['ece'] for x in gs[1:])/3),pct(g['ece'])])
    table('trajectory','Checkpoint trajectory for the 0.8B cross-entropy run. All test results are full-benchmark, uncalibrated measurements.',
          'tab:trajectory','rrrrrrrrr',r'Step & Val. NLL & General ACC & Medical ACC & Legal ACC & Finance ACC & Prof. ACC & Prof. ECE & Micro ECE',rows,wide=True,
          note='Accuracy and ECE are percentages. Checkpoints are selected using validation distribution NLL, not these test metrics.',size='footnotesize')
    a=group('merged_T');b=group('q8_T')
    metrics=[('Hard ACC (\\%)','accuracy',100,4),('Domain macro ACC (\\%)',None,100,4),
             ('ECE (\\%)','ece',100,4),('Distribution NLL','distribution_nll',1,6),
             ('Brier','brier',1,6),('Expected-score MAE','expected_score_mae',1,6)]
    rows=[]
    for label,key,mul,digits in metrics:
        vals=[runs[k]['metrics']['macro_domain_accuracy'] if key is None else group(k)[key] for k in ['merged_T','q8_T']]
        rows.append([label]+[f'{v*mul:.{digits}f}' for v in vals])
    table('deployment','Merged 0.8B deployment model and Q8\\_0, both using held-out $T=1.108133$. Full CJ-Bench coverage.',
          'tab:deployment','lrr','Metric & Merged BF16 & Q8\\_0',rows,size='footnotesize')
    table('efficiency','Scoped performance measurements. GPU peak allocation is in decimal GB; CPU process memory is in GiB. These workloads are not hardware-equivalent to imported baselines.',
          'tab:efficiency','llll', 'Configuration & Workload / scope & Warm p50 & Peak memory',[
              ['0.8B / A800 / BF16','60 decisions; local end-to-end',f"{d['latency_08b']['median_ms']:.2f} ms",f"{runs['08b_10000']['timing']['gpu_peak_gb']:.1f} GB (batched)"],
              ['2B / A800 / BF16','60 decisions; local end-to-end',f"{d['latency_2b']['median_ms']:.2f} ms",f"{runs['2b_2000']['timing']['gpu_peak_gb']:.2f} GB (batched)"],
              *[[label,'36 short-prompt forwards',f"{d['cpu_diagnostics'][key]['short_median_ms']:.0f} ms",f"{d['cpu_diagnostics'][key]['peak_rss_gib']:.2f} GiB"]
                for key,label in [('transformers_fp32','0.8B / CPU / FP32'),('gguf_bf16','0.8B / CPU / BF16 GGUF'),('gguf_q8_0','0.8B / CPU / Q8\\_0')]],
          ],wide=True,note='CPU: Intel Core i7-9700, eight threads, 2,048-token capacity; no tokenization or HTTP. GPU: A800 80GB PCIe; includes input preparation, excludes HTTP/network. GPU memory is measured in separate full-benchmark batched runs.',size='footnotesize')
    rows=[]
    for k,l in zip(DOMAINS,LABELS):
        g=group('08b_10000','domain:'+k); rows.append([l,pct(g['accuracy']),pct(g['ece']),f3(g['distribution_nll']),f3(g['brier'])])
    g=group('08b_10000'); rows.append(['Micro',pct(g['accuracy']),pct(g['ece']),f3(g['distribution_nll']),f3(g['brier'])])
    table('domain_detail','Completed 0.8B model by domain ($T=1$).','tab:domain-detail','lrrrr',r'Domain & ACC (\%) & ECE (\%) & NLL & Brier',rows,size='footnotesize')
    rows=[]
    for k,l in zip(DOMAINS,LABELS):
        rows.append([l]+[pct(group('08b_10000',f'domain_type:{k}:{t}')['accuracy']) for t in ['noul','choice','score']])
    table('domain_type',r'Domain-by-type hard-target accuracy (\%) for the completed 0.8B model ($T=1$).','tab:domain-type','lrrr',r'Domain & \noul{} & \choice{} & \score{}',rows)


def figures(d):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    colors=['#2563a6','#138478','#ba6934','#8c5696']
    fig,ax=plt.subplots(figsize=(10,2.3)); ax.set_xlim(0,10);ax.set_ylim(0,2.3);ax.axis('off')
    boxes=[(.05,1.05,1.65,.88,'Context + question\nCandidate descriptions'),
           (2.0,1.05,1.65,.88,'Shared Chinese\nprompt compiler'),
           (3.95,1.05,1.65,.88,'Qwen text backbone\n+ LoRA adaptation'),
           (5.9,1.05,1.65,.88,'Final-position logits\nSelect A ... Z'),
           (7.85,1.05,2.05,.88,'Valid-option softmax\nProbability distribution')]
    for i,(x,y,w,h,txt) in enumerate(boxes):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=0.06',
                                  facecolor='#edf4fb' if i<4 else '#e7f4ee',edgecolor='#536b82',linewidth=.8))
        ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=8.5)
        if i<4: ax.add_patch(FancyArrowPatch((x+w+.04,y+h/2),(boxes[i+1][0]-.04,y+h/2),arrowstyle='-|>',mutation_scale=10,color='#536b82'))
    ax.text(.15,.55,'Training: hard / soft targets',color=colors[0],fontweight='bold')
    ax.text(.15,.22,'0.8B: distribution CE     |     2B: CE + RLCD (preliminary)',fontsize=8)
    ax.text(6.0,.55,'Application readouts',color=colors[1],fontweight='bold')
    ax.text(6.0,.22,'Boolean probability   /   categorical key   /   expected score',fontsize=8)
    fig.savefig(ROOT/'figures/pipeline.pdf',bbox_inches='tight',pad_inches=.04);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.05))
    steps=[2000,3000,5000,10000]
    for dom,label,color in zip(DOMAINS,LABELS,colors):
        gs=[d['runs'][f'08b_{s}']['metrics']['groups']['domain:'+dom] for s in steps]
        axes[0].plot(steps,[g['accuracy']*100 for g in gs],marker='o',label=label,color=color,lw=1.6,ms=4)
        if dom!='general':axes[1].plot(steps,[g['ece']*100 for g in gs],marker='o',label=label,color=color,lw=1.6,ms=4)
    for ax in axes:
        ax.set_xlabel('Optimizer step (0.8B run)');ax.set_xticks(steps,['2k','3k','5k','10k'])
        ax.grid(axis='y',alpha=.18);ax.legend(frameon=False,fontsize=8)
    axes[0].set_ylabel('Hard-target accuracy (%)');axes[0].set_title('(a) Domain accuracy',fontsize=10)
    axes[1].set_ylabel('ECE, 15 bins (%)');axes[1].set_title('(b) Professional-domain calibration',fontsize=10)
    axes[0].set_ylim(50,93);axes[1].set_ylim(0,6.2)
    fig.tight_layout(w_pad=3)
    fig.savefig(ROOT/'figures/trajectory.pdf',bbox_inches='tight',pad_inches=.04);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-snapshot',action='store_true')
    args=parser.parse_args()
    for directory in ['data','tables','figures']:(ROOT/directory).mkdir(exist_ok=True)
    if args.refresh_snapshot:refresh()
    data=json.loads((ROOT/'data/evidence.json').read_text(encoding='utf-8'))
    tables(data);figures(data)
    print('Generated 11 tables and 2 vector figures from verified evaluation records.')
