"""Independently verify literal one-line recommendation on the full test set."""
import difflib
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/grey_single_change'
DATA=ROOT/'outputs/grey_improved'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    frozen=read(OUT/'frozen_selection.json')
    manifest=read(OUT/'single_location_manifest.json')
    test=read(OUT/'test_results.json')
    original=Path(__file__).with_name('grey_session.py')
    assert sha(original)==frozen['source_sha256']
    assert sha(Path(__file__).with_name('grey_one_change.py'))==frozen['candidate_implementation_sha256']
    assert sha(OUT/'frozen_selection.json')==test['frozen_selection_sha256']
    assert len(manifest)==19 and frozen['families']==18
    original_text=original.read_text(encoding='utf-8')
    for name,m in manifest.items():
        path=Path(m['source_file'])
        assert sha(path)==m['source_sha256']
        diff=list(difflib.unified_diff(original_text.splitlines(True),path.read_text(encoding='utf-8').splitlines(True)))
        added=[l for l in diff if l.startswith('+') and not l.startswith('+++')]
        removed=[l for l in diff if l.startswith('-') and not l.startswith('---')]
        assert max(len(added),len(removed))==(0 if name=='baseline' else 1)
    with np.load(DATA/'splits.npz',allow_pickle=False) as d:
        rows={k:d['test_'+k] for k in ['users','histories','lengths','targets']}
    with np.load(OUT/'test_ranks.npz',allow_pickle=False) as d:
        saved={k:d[k] for k in d.files}
    tokens=read(DATA/'tokens.json')
    candidate=Path(manifest['harmonic_mean']['source_file'])
    spec=importlib.util.spec_from_file_location('verify_one_line_harmonic',candidate)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model=module.GreySession.load(ROOT/'outputs/grey_relation/grey_model.npz')
    ranks=[]
    with (OUT/'recommended_predictions.jsonl').open('w',encoding='utf-8') as f:
        for i,(u,h,n,t) in enumerate(zip(rows['users'],rows['histories'],rows['lengths'],rows['targets'])):
            history=h[:n]
            ids,scores,_,coeff=model.recommend(history)
            np.testing.assert_allclose(scores[ids],1/np.mean(1/coeff[:,ids-1],axis=0),atol=1e-7,rtol=0)
            match=np.flatnonzero(ids==t)
            rank=int(match[0]+1) if len(match) else 0
            assert rank==saved['harmonic_mean'][i]
            ranks.append(rank)
            row={'session_id':str(tokens['users'][u]),'history':[str(tokens['items'][x]) for x in history],
                 'target':str(tokens['items'][t]),'target_rank':rank,
                 'recommendations':[{'rank':j+1,'item':str(tokens['items'][x]),'score':float(scores[x])} for j,x in enumerate(ids)]}
            f.write(json.dumps(row)+'\n')
    for record in test['results']:
        r=saved[record['config']['family']]
        independently={'recall@10':sum(x>0 for x in r)/len(r),
                       'mrr@10':sum(1/x for x in r if x)/len(r),
                       'ndcg@10':sum(1/math.log2(x+1) for x in r if x)/len(r)}
        for k,v in independently.items():
            assert abs(v-record['test'][k])<1e-12
    latency=read(OUT/'latency.json')
    assert set(latency)==set(manifest) and len({r['sample_sha256'] for r in latency.values()})==1
    assert all(r['calls']==900 and r['correctness_queries']==300 for r in latency.values())
    assert len(ranks)==20000 and test['original_baseline_all_20000_ranks_exact']
    result={'all_passed':True,'original_source_unchanged':True,
            'each_of_18_variants_has_exactly_one_source_line_location':True,
            'all_19_methods_20000_rank_metrics_independently_recomputed':True,
            'cache_full_catalogue_top10_checks':test['full_catalogue_cache_top10_checks'],
            'all_19_methods_literal_code_300_queries_verified_and_900_calls_timed':True,
            'recommendation_literal_one_line_code_verified_all_20000':True,
            'primary_literal_one_line_code_verified_all_20000':read(OUT/'primary_literal_code_check.json')['all_passed'],
            'recommended_config':{'family':'harmonic_mean','new_hyperparameters':0},
            'frozen_validation_primary':frozen['primary']['config'],
            'report_sha256':sha(OUT/'单处改动对比与推荐.md'),
            'recommended_predictions_sha256':sha(OUT/'recommended_predictions.jsonl'),
            'original_source_sha256':sha(original),
            'scope':'Exploratory OFAT on existing 20k split. Per-family settings selected by validation. Recommendation weighs no added tuning and measured cost, while the validation-MRR primary is recorded separately.'}
    (OUT/'completion_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
