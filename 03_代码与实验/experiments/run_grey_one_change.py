"""Validation-only OFAT selection, then one frozen test comparison per family."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import argparse
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np

from grey_session import GreySession, ranking_metrics
from grey_one_change import configurations, scores_from_cache, OneChangePredictor, DEFAULT

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/grey_single_change'
DATA=ROOT/'outputs/grey_improved'


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(name, value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def load_split(split):
    with np.load(DATA/'splits.npz',allow_pickle=False) as d:
        return {k:d[split+'_'+k] for k in ['users','histories','lengths','targets']}


def prepare_models():
    original=GreySession.load(ROOT/'outputs/grey_relation/grey_model.npz')
    source=Path(__file__).with_name('grey_session.py')
    text=source.read_text(encoding='utf-8')
    needle='        cooc.setdiag(0)\n'
    assert text.count(needle)==1
    modified=text.replace(needle,'',1)
    candidate_dir=OUT/'candidate_code'
    candidate_dir.mkdir(exist_ok=True)
    candidate=candidate_dir/'grey_session_keep_self.py'
    candidate.write_text(modified,encoding='utf-8')
    patch=''.join(difflib.unified_diff(text.splitlines(True),modified.splitlines(True),
                                     fromfile='grey_session.py',tofile='grey_session.py'))
    (candidate_dir/'keep_self.patch').write_text(patch,encoding='utf-8')
    module_spec=importlib.util.spec_from_file_location('grey_session_keep_self_candidate',candidate)
    module=importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    valid=load_split('valid')
    sequences=[h[:n] for h,n in zip(valid['histories'],valid['lengths'])]
    start=time.perf_counter()
    self_model=module.GreySession(original.item_num).fit(sequences)
    self_model.save(OUT/'keep_self_model.npz')
    assert np.allclose(self_model.cooccurrence.diagonal()[original.counts>0],1,atol=2e-7)
    np.testing.assert_array_equal(original.counts,self_model.counts)
    delta=(self_model.transitions-original.transitions).tocsr()
    assert delta.nnz==0
    offdiag=self_model.cooccurrence.copy()
    offdiag.setdiag(0)
    offdiag.eliminate_zeros()
    delta=(offdiag-original.cooccurrence).tocsr()
    assert delta.nnz==0 or np.max(np.abs(delta.data))<1e-7
    save('preparation.json',{'original_source_sha256':digest(source),
                            'original_model_sha256':digest(ROOT/'outputs/grey_relation/grey_model.npz'),
                            'single_line_deleted':'cooc.setdiag(0)',
                            'self_model_refit_seconds':time.perf_counter()-start,
                            'all_other_statistics_unchanged':True})
    return original,self_model


def make_cache(split, model, self_model):
    rows=load_split(split)
    features, self_cooc, repeats, lastonly, ids_all = [],[],[],[],[]
    offsets, target_index=[0],[]
    pop=np.arange(1,model.item_num)
    pop=pop[np.lexsort((pop,-model.log_popularity[1:]))]
    start=time.perf_counter()
    for i,(history,n,target) in enumerate(zip(rows['histories'],rows['lengths'],rows['targets'])):
        history=history[:n]
        x=model.normalized_features(history)
        with_self=self_model.normalized_features(history)
        np.testing.assert_array_equal(x[[0,2,3]],with_self[[0,2,3]])
        support=np.unique(np.concatenate((np.flatnonzero(np.any(x[:3]>0,axis=0)),
                                          pop[:10],np.arange(1,11))))
        support=support[support>0]
        match=np.flatnonzero(support==target)
        target_index.append(offsets[-1]+int(match[0]) if len(match) else -1)
        features.append(x[:,support].T)
        self_cooc.append(with_self[1,support])
        count=np.bincount(history,minlength=model.item_num).astype(np.float32)
        count/=count.max()
        repeats.append(count[support])
        lastonly.append((support==history[-1]).astype(np.float32))
        ids_all.append(support)
        offsets.append(offsets[-1]+len(support))
        if (i+1)%4000==0:
            print(f'{split} feature cache {i+1}/20000, {time.perf_counter()-start:.1f}s',flush=True)
    cache={'x':np.concatenate(features),'self_cooccurrence':np.concatenate(self_cooc),
           'repeat_count':np.concatenate(repeats),'repeat_last_only':np.concatenate(lastonly),
           'ids':np.concatenate(ids_all),'offsets':np.asarray(offsets),
           'target_index':np.asarray(target_index),'targets':rows['targets'],'users':rows['users']}
    np.savez_compressed(OUT/f'{split}_features.npz',**cache)
    return cache


def load_cache(split, model, self_model):
    path=OUT/f'{split}_features.npz'
    if path.exists():
        with np.load(path,allow_pickle=False) as d:
            return {k:d[k] for k in d.files}
    return make_cache(split,model,self_model)


def cache_ranks(cache, config):
    scores=scores_from_cache(cache['x'],config,cache)
    target_index=cache['target_index']
    present=target_index>=0
    target_score=scores[np.maximum(target_index,0)]
    sizes=np.diff(cache['offsets'])
    repeated=np.repeat(target_score,sizes)
    targets=np.repeat(cache['targets'],sizes)
    above=(scores>repeated)|((scores==repeated)&(cache['ids']<targets))
    ranks=1+np.add.reduceat(above.astype(np.int32),cache['offsets'][:-1])
    ranks[(~present)|(ranks>10)]=0
    return ranks


def select():
    OUT.mkdir(parents=True,exist_ok=True)
    model,self_model=prepare_models()
    valid=load_cache('valid',model,self_model)
    trials=[]
    for config in configurations():
        metrics=ranking_metrics(cache_ranks(valid,config))
        trials.append({'config':config,'valid':metrics})
        print(f"VALID {config}: Recall={metrics['recall@10']:.5f} MRR={metrics['mrr@10']:.5f}",flush=True)
    families=sorted({c['config']['family'] for c in trials if c['config']['family']!='baseline'})
    selected=[]
    for family in families:
        candidates=[c for c in trials if c['config']['family']==family]
        best=max(candidates,key=lambda c:(c['valid']['mrr@10'],c['valid']['recall@10']))
        selected.append(best)
    primary=max(selected,key=lambda c:(c['valid']['mrr@10'],c['valid']['recall@10']))
    baseline=next(c for c in trials if c['config']['family']=='baseline')
    frozen={'criterion':'Within each family maximize full-validation MRR@10, tie-break Recall@10; primary family selected by the same rule before test.',
            'test_used_in_selection':False,'valid_queries':20000,'families':len(families),
            'configurations_including_baseline':len(trials),'baseline':baseline,
            'selected_per_family':selected,'primary':primary,
            'invariants':['four original features only','original training statistics except the sole self-diagonal variant',
                          'no session neighbours','no simultaneous interventions','rho=.5 unless rho is the sole changed parameter',
                          'equal weights unless one weight multiplier is the sole intervention'],
            'source_sha256':digest(Path(__file__).with_name('grey_session.py')),
            'candidate_implementation_sha256':digest(Path(__file__).with_name('grey_one_change.py'))}
    save('validation_search.json',trials)
    save('frozen_selection.json',frozen)
    print(json.dumps(frozen,ensure_ascii=True,indent=2),flush=True)


def evaluate_test():
    from scipy.stats import binomtest
    frozen_path=OUT/'frozen_selection.json'
    frozen=json.loads(frozen_path.read_text(encoding='utf-8'))
    sha_before=digest(frozen_path)
    assert frozen['source_sha256']==digest(Path(__file__).with_name('grey_session.py'))
    assert frozen['candidate_implementation_sha256']==digest(Path(__file__).with_name('grey_one_change.py'))
    model=GreySession.load(ROOT/'outputs/grey_relation/grey_model.npz')
    self_model=GreySession.load(OUT/'keep_self_model.npz')
    data=load_cache('test',model,self_model)
    rows=load_split('test')
    tokens=json.loads((DATA/'tokens.json').read_text(encoding='utf-8'))
    rank_arrays={'baseline':cache_ranks(data,DEFAULT)}
    baseline_metrics=ranking_metrics(rank_arrays['baseline'])
    original={json.loads(line)['session_id']:json.loads(line) for line in (ROOT/'outputs/grey_relation/GRA.jsonl').open(encoding='utf-8')}
    for i,(u,h,n,t) in enumerate(zip(rows['users'],rows['histories'],rows['lengths'],rows['targets'])):
        ref=original[str(tokens['users'][u])]
        assert ref['history']==[str(tokens['items'][x]) for x in h[:n]]
        assert ref['target']==str(tokens['items'][t])
        assert ref['target_rank']==int(rank_arrays['baseline'][i])
    dense_checks=0
    check_rows=np.random.default_rng(20261112).choice(len(rows['users']),40,replace=False)
    selected=[frozen['baseline']]+frozen['selected_per_family']
    results=[]
    for candidate in selected:
        config=candidate['config']
        ranks=cache_ranks(data,config)
        family=config['family']
        rank_arrays[family]=ranks
        predictor=OneChangePredictor(model,self_model,config)
        for i in check_rows:
            ids,scores=predictor.predict(rows['histories'][i,:rows['lengths'][i]])
            a,b=data['offsets'][i:i+2]
            alternatives={k:data[k][a:b] for k in ['self_cooccurrence','repeat_count','repeat_last_only']}
            cached=scores_from_cache(data['x'][a:b],config,alternatives)
            support=data['ids'][a:b]
            order=np.lexsort((support,-cached))[:10]
            np.testing.assert_array_equal(ids,support[order])
            np.testing.assert_allclose(scores,cached[order],atol=3e-7,rtol=0)
            target=int(rows['targets'][i])
            pos=np.flatnonzero(ids==target)
            rank=int(pos[0]+1) if len(pos) else 0
            assert rank==int(ranks[i])
            dense_checks+=1
        hit=ranks>0
        base_hit=rank_arrays['baseline']>0
        wins=int(np.sum(hit&~base_hit))
        losses=int(np.sum(~hit&base_hit))
        p=float(binomtest(min(wins,losses),wins+losses,.5).pvalue) if wins+losses else 1.
        repeated=np.array([int(t) in h[:n] for t,h,n in zip(rows['targets'],rows['histories'],rows['lengths'])])
        measured={'config':config,'valid':candidate['valid'],'test':ranking_metrics(ranks),
                  'recall_gain_percentage_points':100*(float(hit.mean())-baseline_metrics['recall@10']),
                  'additional_hits_net':wins-losses,'paired_wins':wins,'paired_losses':losses,
                  'mcnemar_exact_two_sided_p':p,
                  'repeat_target_recall':float(hit[repeated].mean()),
                  'novel_target_recall':float(hit[~repeated].mean())}
        results.append(measured)
        print(f"TEST {family}: Recall={measured['test']['recall@10']:.5f} MRR={measured['test']['mrr@10']:.5f}, net hits {wins-losses}",flush=True)
    # Holm across the pre-frozen family representatives, not arbitrary test-picked settings.
    tests=sorted([r for r in results if r['config']['family']!='baseline'],key=lambda r:r['mcnemar_exact_two_sided_p'])
    previous=0.
    for i,r in enumerate(tests):
        adjusted=min(1.,max(previous,(len(tests)-i)*r['mcnemar_exact_two_sided_p']))
        r['mcnemar_holm_p']=adjusted
        previous=adjusted
    np.savez_compressed(OUT/'test_ranks.npz',**rank_arrays)
    assert digest(frozen_path)==sha_before
    save('test_results.json',{'baseline':baseline_metrics,'results':results,
                             'primary_preselected_config':frozen['primary']['config'],
                             'frozen_selection_sha256':sha_before,'test_queries':20000,
                             'original_baseline_all_20000_ranks_exact':True,
                             'full_catalogue_cache_top10_checks':dense_checks,
                             'original_source_unchanged':digest(Path(__file__).with_name('grey_session.py'))==frozen['source_sha256']})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['select','test'])
    args=parser.parse_args()
    select() if args.stage=='select' else evaluate_test()
