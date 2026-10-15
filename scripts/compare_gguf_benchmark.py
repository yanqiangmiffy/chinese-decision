"""Paired comparison of two complete CJ-Bench prediction files."""
import argparse
import gzip
import heapq
import json
import statistics
from collections import defaultdict
from pathlib import Path

from decision.utils import write_json


def key(row):
    return row['domain'],row['line'],row['case_id']


def records(path):
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',required=True,help='reference report JSON')
    parser.add_argument('--candidate',required=True,help='candidate report JSON')
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    reference=json.loads(Path(args.reference).read_text())
    candidate=json.loads(Path(args.candidate).read_text())
    for field in ['manifest_sha256','temperature','prompt_version','selected_checkpoint']:
        if reference[field] != candidate[field]:
            raise ValueError(f'incompatible comparison field: {field}')
    base={}
    for row in records(reference['predictions']):
        identity=key(row)
        if identity in base:
            raise ValueError('duplicate reference case')
        base[identity]=row
    groups=defaultdict(lambda:{'n':0,'candidates':0,'top1_agree':0,'hard_n':0,
                               'correct_to_wrong':0,'wrong_to_correct':0,'probability_abs_sum':0.,
                               'max_probability_delta':0.,'per_row_max_deltas':[]})
    examples=[]
    for serial,row in enumerate(records(candidate['predictions'])):
        previous=base.pop(key(row))
        for field in ['target','hard','option_keys','domain','kind','parent_id']:
            if row[field] != previous[field]:
                raise ValueError(f'case definition mismatch: {key(row)}, {field}')
        old,new=previous['probabilities'],row['probabilities']
        deltas=[abs(a-b) for a,b in zip(old,new)]
        maximum=max(deltas)
        old_winner=max(range(len(old)),key=old.__getitem__)
        new_winner=max(range(len(new)),key=new.__getitem__)
        target=max(range(len(row['target'])),key=row['target'].__getitem__)
        for group in ['all',f"domain:{row['domain']}",f"type:{row['kind']}"]:
            stats=groups[group]
            stats['n']+=1
            stats['candidates']+=len(old)
            stats['top1_agree']+=int(old_winner==new_winner)
            stats['probability_abs_sum']+=sum(deltas)
            stats['max_probability_delta']=max(stats['max_probability_delta'],maximum)
            stats['per_row_max_deltas'].append(maximum)
            if row['hard']:
                stats['hard_n']+=1
                stats['correct_to_wrong']+=int(old_winner==target and new_winner!=target)
                stats['wrong_to_correct']+=int(old_winner!=target and new_winner==target)
        item=(maximum,serial,{k:row[k] for k in ['domain','kind','case_id','target','option_keys']}
              | {'reference':old,'candidate':new,'max_probability_delta':maximum})
        if len(examples)<10:
            heapq.heappush(examples,item)
        elif item[:2] > examples[0][:2]:
            heapq.heapreplace(examples,item)
    if base:
        raise ValueError(f'candidate missing {len(base)} reference rows')
    if groups['all']['n'] != reference['coverage']['evaluated'] or groups['all']['n'] != candidate['coverage']['evaluated']:
        raise ValueError('report and predictions disagree on coverage')
    if reference['timing']['input_tokens'] != candidate['timing']['input_tokens']:
        raise ValueError('reference and candidate consumed different input token totals')
    for group,stats in groups.items():
        ordered=sorted(stats.pop('per_row_max_deltas'))
        stats['p50_per_row_max_probability_delta']=statistics.median(ordered)
        stats['p95_per_row_max_probability_delta']=ordered[int((len(ordered)-1)*.95)]
        stats['mean_absolute_probability_delta']=stats.pop('probability_abs_sum')/stats['candidates']
        stats['top1_agreement']=stats['top1_agree']/stats['n']
        stats['accuracy_delta']=(stats['wrong_to_correct']-stats['correct_to_wrong'])/max(stats['hard_n'],1)
    metric_names=['accuracy','ece','distribution_nll','brier','expected_score_mae']
    metric_delta={}
    for group,stats in candidate['metrics']['groups'].items():
        if group in reference['metrics']['groups']:
            before=reference['metrics']['groups'][group]
            metric_delta[group]={name:stats[name]-before[name] for name in metric_names
                                 if stats.get(name) is not None and before.get(name) is not None}
    for group,stats in groups.items():
        if abs(stats['accuracy_delta']-metric_delta[group]['accuracy']) > 1e-12:
            raise ValueError(f'paired accuracy and aggregate report mismatch: {group}')
    write_json(args.out,{'reference':args.reference,'candidate':args.candidate,
                         'n':groups['all']['n'],'temperature':reference['temperature'],
                         'groups':dict(groups),'metric_delta':metric_delta,
                         'macro_domain_accuracy_delta':candidate['metrics']['macro_domain_accuracy']-reference['metrics']['macro_domain_accuracy'],
                         'largest_probability_changes':[item[2] for item in sorted(examples,reverse=True)],
                         'interpretation':'End-to-end comparison includes weight precision, inference backend and batch kernel numerics; no test fitting.'})
    print(json.dumps({'out':args.out,'all':groups['all'],'metrics':metric_delta['all']},ensure_ascii=False))


if __name__=='__main__':
    main()
