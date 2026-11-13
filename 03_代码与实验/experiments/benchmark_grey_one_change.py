"""Run actual one-line-patched code; original dense full-catalogue computation."""
import gc
import importlib.util
import json
from pathlib import Path
import time
import hashlib
import numpy as np
import torch

from materialize_grey_one_change import materialize
from grey_session import ranking_metrics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/grey_single_change'
DATA=ROOT/'outputs/grey_improved'


def main():
    torch.set_num_threads(4)
    manifest=materialize()
    frozen=json.loads((OUT/'frozen_selection.json').read_text(encoding='utf-8'))
    primary=frozen['primary']['config']['family']
    with np.load(DATA/'splits.npz',allow_pickle=False) as d:
        rows={k:d['test_'+k] for k in ['users','histories','lengths','targets']}
    with np.load(OUT/'test_ranks.npz',allow_pickle=False) as d:
        expected_ranks={k:d[k] for k in d.files}
    tokens=json.loads((DATA/'tokens.json').read_text(encoding='utf-8'))
    token2id={str(token):i for i,token in enumerate(tokens['items'])}
    selected=np.random.default_rng(20261111).choice(len(rows['users']),300,replace=False)
    users=rows['users'][selected]
    raw_queries=[[str(tokens['items'][x]) for x in h[:n]] for h,n in zip(rows['histories'][selected],rows['lengths'][selected])]
    sample_sha=hashlib.sha256(json.dumps(list(zip(users.tolist(),raw_queries)),sort_keys=True).encode()).hexdigest()
    output={}
    # Baseline first, then deterministic permutation to avoid ordering by performance.
    order=['baseline']+np.random.default_rng(19).permutation([k for k in manifest if k!='baseline']).tolist()
    for name in order:
        spec=importlib.util.spec_from_file_location('single_patch_'+name,manifest[name]['source_file'])
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        weights_file=OUT/'keep_self_model.npz' if name=='keep_self_cooccurrence' else ROOT/'outputs/grey_relation/grey_model.npz'
        model=module.GreySession.load(weights_file)
        def predict(raw):
            history=np.array([token2id[str(x)] for x in raw],dtype=np.int64)
            ids,scores,_,_=model.recommend(history)
            return [{'rank':i+1,'item':str(tokens['items'][x]),'score':float(scores[x])} for i,x in enumerate(ids)]
        for i,raw in zip(selected,raw_queries):
            recs=predict(raw)
            target=str(tokens['items'][rows['targets'][i]])
            ids=[r['item'] for r in recs]
            rank=ids.index(target)+1 if target in ids else 0
            assert rank==expected_ranks[name][i],(name,int(i),rank,int(expected_ranks[name][i]))
        # The PRIMARY's literal one-line implementation is checked on ALL 20k cases.
        if name==primary:
            ranks=[]
            with (OUT/'primary_predictions.jsonl').open('w',encoding='utf-8') as f:
                for i,(u,h,n,t) in enumerate(zip(rows['users'],rows['histories'],rows['lengths'],rows['targets'])):
                    history=h[:n]
                    ids,scores,_,_=model.recommend(history)
                    match=np.flatnonzero(ids==t)
                    rank=int(match[0]+1) if len(match) else 0
                    assert rank==expected_ranks[name][i],(name,i)
                    ranks.append(rank)
                    row={'session_id':str(tokens['users'][u]),'history':[str(tokens['items'][x]) for x in history],
                         'target':str(tokens['items'][t]),'target_rank':rank,
                         'recommendations':[{'rank':j+1,'item':str(tokens['items'][x]),'score':float(scores[x])}
                                            for j,x in enumerate(ids)]}
                    f.write(json.dumps(row)+'\n')
            (OUT/'primary_literal_code_check.json').write_text(json.dumps({'all_passed':True,'queries':20000,
                'config':manifest[name]['config'],'test':ranking_metrics(ranks),
                'one_line_implementation_matches_cached_ranks_all_queries':True},indent=2),encoding='utf-8')
        for raw in raw_queries[:30]:
            predict(raw)
        rounds=[]
        samples=[]
        gc.collect()
        gc.disable()
        try:
            for r in range(3):
                samples_r=[]
                for i in np.random.default_rng(20261111+r).permutation(len(raw_queries)):
                    start=time.perf_counter_ns()
                    pred=predict(raw_queries[i])
                    elapsed=(time.perf_counter_ns()-start)/1e6
                    samples_r.append(elapsed)
                samples.extend(samples_r)
                rounds.append({'mean_ms':float(np.mean(samples_r)),'median_ms':float(np.median(samples_r))})
        finally:
            gc.enable()
        numeric_bytes=sum(a.nbytes for a in [model.counts,model.log_popularity,model.weights])
        for matrix in [model.transitions,model.cooccurrence]:
            numeric_bytes+=matrix.data.nbytes+matrix.indices.nbytes+matrix.indptr.nbytes
        measurement={'config':manifest[name]['config'],'mean_ms':float(np.mean(samples)),
                     'median_ms':float(np.median(samples)),'p95_ms':float(np.percentile(samples,95)),
                     'numeric_state_bytes':numeric_bytes,'calls':900,'correctness_queries':300,
                     'sample_sha256':sample_sha,'per_round':rounds,'samples_ms':samples,
                     'scope':'CPU4-thread allowance, batch=1, same 300 raw histories, warm actual single-line code; includes ID mapping, dense features, scoring full catalogue, Top10 and raw-ID output; no sparse acceleration.'}
        output[name]=measurement
        print(f'{name}: mean={measurement["mean_ms"]:.4f}ms, bytes={numeric_bytes}',flush=True)
        (OUT/'latency.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
        del model
    print('All literal one-line implementations verified and benchmarked.',flush=True)


if __name__=='__main__':
    main()
