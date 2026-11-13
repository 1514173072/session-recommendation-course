"""OFAT experiments INSIDE grey_coefficients; original input/weights/mean fixed.

Full-catalogue rank evaluation accounts for ALL 32,968 items, including
floating-point plateaus from nonlinear coefficients. Only 85 distinct static
popularity values exist; group background scores plus sparse corrections give
exact ranks without assuming strict monotonicity or dropping unseen support.
"""
import argparse
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np

from grey_session import GreySession, ranking_metrics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/grey_coefficient_change'
OLD=ROOT/'outputs/grey_single_change'
DATA=ROOT/'outputs/grey_improved'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def save(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def configs():
    values={'distance_power':[.25,.5,2,4,8,16,32,64,128,256],
            'rational_distance':[1,3,7,15,31,63,127,255],
            'exponential_distance':[.5,1,2,4,8,16,32],
            'adaptive_global_mean':[None], 'adaptive_column_mean':[None],
            'adaptive_column_cv':[None], 'adaptive_column_inverse_cv':[None],
            'adaptive_positive_deviation':[None]}
    return [{'family':'baseline','value':None}]+[{'family':f,'value':v} for f,vs in values.items() for v in vs]


def transform_delta(delta,config):
    family,value=config['family'],config['value']
    if family=='distance_power':
        return np.power(delta,value)
    if family=='rational_distance':
        return delta/(1.0+value*(1.0-delta))
    if family=='exponential_distance':
        return np.expm1(value*delta)/np.expm1(value)
    return delta


def rho_from_moments(meta,config):
    family=config['family']
    mu,std=meta['mu'],meta['std']
    if family=='adaptive_global_mean':
        return np.repeat(np.clip(1-mu.mean(axis=1,keepdims=True),.05,1),4,axis=1).astype(np.float32)
    if family=='adaptive_column_mean':
        return np.clip(1-mu,.05,1).astype(np.float32)
    if family=='adaptive_column_cv':
        return np.clip(mu/(std+1e-12),.05,1).astype(np.float32)
    if family=='adaptive_column_inverse_cv':
        return np.clip(std/(mu+1e-12),.05,1).astype(np.float32)
    if family=='adaptive_positive_deviation':
        return np.clip(meta['positive_mean_delta'],.05,1).astype(np.float32)
    return np.full_like(mu,.5,dtype=np.float32)


def coefficient_scores(x,rho,config):
    delta=transform_delta(np.abs(1.0-x),config)
    coefficients=rho/(delta+rho)
    return np.sum(np.float32(.25)*coefficients.T,axis=0)


def cache_metadata(split,model):
    with np.load(OLD/f'{split}_features.npz',allow_pickle=False) as d:
        cache={k:d[k] for k in ['x','ids','offsets','target_index','targets','users']}
    n=len(cache['targets'])
    m=model.item_num-1
    offsets=cache['offsets']
    qindex=np.repeat(np.arange(n),np.diff(offsets))
    sums=np.add.reduceat(cache['x'].astype(np.float64),offsets[:-1],axis=0)
    squares=np.add.reduceat(cache['x'].astype(np.float64)**2,offsets[:-1],axis=0)
    positives=np.add.reduceat((cache['x']>0).astype(np.int32),offsets[:-1],axis=0)
    pop=model.log_popularity[1:].copy()
    pop=(pop-pop.min())/(pop.max()-pop.min())
    sums[:,3]=np.sum(pop,dtype=np.float64)
    squares[:,3]=np.sum(pop.astype(np.float64)**2)
    positives[:,3]=np.count_nonzero(pop)
    mu=sums/m
    std=np.sqrt(np.maximum(squares/m-mu**2,0))
    positive_delta=np.ones_like(mu)
    np.divide(positives-sums,positives,out=positive_delta,where=positives>0)
    groups,inverse,counts=np.unique(pop,return_inverse=True,return_counts=True)
    ids_per_group=[np.flatnonzero(inverse==g)+1 for g in range(len(groups))]
    tie_less=np.column_stack([np.searchsorted(ids,cache['targets']) for ids in ids_per_group]).astype(np.int32)
    support_group=inverse[cache['ids']-1]
    np.testing.assert_array_equal(groups[support_group],cache['x'][:,3])
    target_x=np.zeros((n,4),dtype=np.float32)
    present=cache['target_index']>=0
    target_x[present]=cache['x'][cache['target_index'][present]]
    target_x[~present,3]=pop[cache['targets'][~present]-1]
    cache.update({'mu':mu,'std':std,'positive_mean_delta':positive_delta,
                  'groups':groups,'group_counts':counts,'tie_less':tie_less,
                  'support_group':support_group,'target_x':target_x,'qindex':qindex})
    return cache


def exact_ranks(cache,config):
    rho=rho_from_moments(cache,config)
    target_scores=coefficient_scores(cache['target_x'],rho,config)
    actual=coefficient_scores(cache['x'],rho[cache['qindex']],config)
    # Background: all local indicators zero; only popularity varies.
    baseline=rho/(1.0+rho)
    base=np.float32(.25)*baseline[:,0]
    base+=np.float32(.25)*baseline[:,1]
    base+=np.float32(.25)*baseline[:,2]
    pop_delta=transform_delta(1.0-cache['groups'],config)
    pop_coeff=rho[:,3,None]/(pop_delta[None,:]+rho[:,3,None])
    background=base[:,None]+np.float32(.25)*pop_coeff
    greater=background>target_scores[:,None]
    ties=background==target_scores[:,None]
    all_background_above=np.sum(greater*cache['group_counts'][None,:]+ties*cache['tie_less'],axis=1)
    repeated=target_scores[cache['qindex']]
    earlier=cache['ids']<cache['targets'][cache['qindex']]
    support_above=(actual>repeated)|((actual==repeated)&earlier)
    support_background=background[cache['qindex'],cache['support_group']]
    subtract=(support_background>repeated)|((support_background==repeated)&earlier)
    correction=np.add.reduceat(support_above.astype(np.int32)-subtract.astype(np.int32),cache['offsets'][:-1])
    rank=1+all_background_above+correction
    assert np.all(rank>=1),config
    rank[rank>10]=0
    return rank.astype(np.int32)


def candidate_source(original,config):
    family,value=config['family'],config['value']
    if family=='baseline':
        return original
    if family=='distance_power':
        return original.replace('    delta = np.abs(1.0 - normalized)\n',
                                f'    delta = np.abs(1.0 - normalized) ** {value!r}\n',1)
    if family=='rational_distance':
        line=f'    delta = delta / (1.0 + {value!r} * (1.0 - delta))\n'
    elif family=='exponential_distance':
        line=f'    delta = np.expm1({value!r} * delta) / np.expm1({value!r})\n'
    elif family=='adaptive_global_mean':
        line='    rho = np.float32(np.clip(delta.mean(dtype=np.float64), 0.05, 1.0))\n'
    elif family=='adaptive_column_mean':
        line='    rho = np.clip(delta.mean(axis=1, keepdims=True, dtype=np.float64), 0.05, 1.0).astype(np.float32)\n'
    elif family=='adaptive_column_cv':
        line=('    mu = normalized.mean(axis=1, keepdims=True, dtype=np.float64)\n'
              '    rho = np.clip(mu / (normalized.std(axis=1, keepdims=True, dtype=np.float64) + 1e-12), 0.05, 1.0).astype(np.float32)\n')
    elif family=='adaptive_column_inverse_cv':
        line=('    mu = normalized.mean(axis=1, keepdims=True, dtype=np.float64)\n'
              '    rho = np.clip(normalized.std(axis=1, keepdims=True, dtype=np.float64) / (mu + 1e-12), 0.05, 1.0).astype(np.float32)\n')
    elif family=='adaptive_positive_deviation':
        line=('    positive = normalized > 0\n'
              '    positive_count = positive.sum(axis=1, keepdims=True)\n'
              '    rho = np.clip(np.divide((delta * positive).sum(axis=1, keepdims=True, dtype=np.float64), positive_count, out=np.ones((normalized.shape[0], 1), dtype=np.float64), where=positive_count > 0), 0.05, 1.0).astype(np.float32)\n')
    else:
        raise ValueError(family)
    anchor='    lo, hi = float(delta.min()), float(delta.max())\n'
    return original.replace(anchor,line+anchor,1)


def literal_model(config):
    source=Path(__file__).with_name('grey_session.py').read_text(encoding='utf-8')
    changed=candidate_source(source,config)
    directory=OUT/'candidate_code'
    directory.mkdir(parents=True,exist_ok=True)
    family=config['family']
    path=directory/(family+'.py')
    path.write_text(changed,encoding='utf-8')
    patch=''.join(difflib.unified_diff(source.splitlines(True),changed.splitlines(True),fromfile='grey_session.py',tofile='grey_session.py'))
    (directory/(family+'.patch')).write_text(patch,encoding='utf-8')
    # All edits are contained strictly inside grey_coefficients.
    assert source.split('\ndef stable_topk',1)[1]==changed.split('\ndef stable_topk',1)[1]
    spec=importlib.util.spec_from_file_location('grey_core_candidate_'+family,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GreySession.load(ROOT/'outputs/grey_relation/grey_model.npz')


def verify_queries(split,configs_to_check,rank_arrays,amount=30):
    with np.load(DATA/'splits.npz',allow_pickle=False) as d:
        histories=d[split+'_histories']
        lengths=d[split+'_lengths']
        targets=d[split+'_targets']
    indices=np.random.default_rng(20261113).choice(len(targets),amount,replace=False)
    for config in configs_to_check:
        model=literal_model(config)
        for i in indices:
            ids,scores,_,_=model.recommend(histories[i,:lengths[i]])
            positions=np.flatnonzero(ids==targets[i])
            rank=int(positions[0]+1) if len(positions) else 0
            assert rank==rank_arrays[config['family']][i],(split,config,int(i),rank,int(rank_arrays[config['family']][i]))
    return len(indices)*len(configs_to_check)


def select():
    OUT.mkdir(parents=True,exist_ok=True)
    model=GreySession.load(ROOT/'outputs/grey_relation/grey_model.npz')
    data=cache_metadata('valid',model)
    trials=[]
    for config in configs():
        ranks=exact_ranks(data,config)
        metrics=ranking_metrics(ranks)
        trials.append({'config':config,'valid':metrics})
        print(f'VALID {config}: {metrics}',flush=True)
    families=sorted({c['config']['family'] for c in trials if c['config']['family']!='baseline'})
    selected=[max([t for t in trials if t['config']['family']==f],
                   key=lambda t:(t['valid']['mrr@10'],t['valid']['recall@10'])) for f in families]
    baseline=next(t for t in trials if t['config']['family']=='baseline')
    primary=max(selected,key=lambda t:(t['valid']['mrr@10'],t['valid']['recall@10']))
    checks_configs=[baseline['config']]+[t['config'] for t in selected]
    rank_arrays={c['family']:exact_ranks(data,c) for c in checks_configs}
    checked=verify_queries('valid',checks_configs,rank_arrays)
    save('validation_search.json',trials)
    save('frozen_selection.json',{'baseline':baseline,'selected_per_family':selected,'primary':primary,
                                'configurations':len(trials),'families':len(families),
                                'criterion':'Validation MRR@10, tie-break Recall@10; no test in selection',
                                'full_catalogue_validation_checks':checked,
                                'original_source_sha256':sha(Path(__file__).with_name('grey_session.py')),
                                'invariants':['four original inputs','same min-max normalization','equal weights .25','arithmetic aggregation','training statistics unchanged'],
                                'only_change':'inside grey_coefficients function'})
    print('FROZEN '+json.dumps(primary),flush=True)


def test():
    frozen=read(OUT/'frozen_selection.json')
    frozen_sha=sha(OUT/'frozen_selection.json')
    model=GreySession.load(ROOT/'outputs/grey_relation/grey_model.npz')
    cache=cache_metadata('test',model)
    records=[frozen['baseline']]+frozen['selected_per_family']
    rank_arrays={}
    results=[]
    for r in records:
        config=r['config']
        ranks=exact_ranks(cache,config)
        rank_arrays[config['family']]=ranks
        record={'config':config,'valid':r['valid'],'test':ranking_metrics(ranks)}
        results.append(record)
        print('TEST '+json.dumps(record),flush=True)
    # Original 20k test ranks must match previous output exactly.
    with np.load(OLD/'test_ranks.npz',allow_pickle=False) as d:
        np.testing.assert_array_equal(rank_arrays['baseline'],d['baseline'])
    checks=verify_queries('test',[r['config'] for r in records],rank_arrays,amount=80)
    np.savez_compressed(OUT/'test_ranks.npz',**rank_arrays)
    assert frozen_sha==sha(OUT/'frozen_selection.json')
    save('test_results.json',{'results':results,'full_catalogue_test_checks':checks,
                             'original_20000_ranks_exact':True,'frozen_selection_sha256':frozen_sha})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['select','test'])
    stage=parser.parse_args().stage
    select() if stage=='select' else test()
