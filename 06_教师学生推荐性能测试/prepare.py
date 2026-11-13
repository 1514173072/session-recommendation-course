"""Prepare real training evidence before the assistant writes teacher prototypes."""
import os
import sys
import time
import json
import hashlib
from collections import defaultdict
import numpy as np
import torch
from common import *

def main():
    begin=time.perf_counter()
    torch.set_num_threads(4)
    os.chdir(CODE)
    sys.path.insert(0,str(CODE/'RecBole-GNN'))
    sys.argv=sys.argv[:1]
    from recbole_gnn.config import Config
    from recbole_gnn.utils import create_dataset,data_preparation
    from recbole.utils import init_seed
    from run_grey import examples
    config=Config(model='SRGNN',dataset='diginetica-formal',
        config_file_list=[str(CODE/'configs/formal.yaml')],
        config_dict={'data_path':str(CODE/'data'),'seed':42,'use_gpu':False,
                     'checkpoint_dir':str(HERE/'cache/unused_checkpoints')})
    init_seed(42,True)
    dataset=create_dataset(config)
    loaders=data_preparation(config,dataset)
    rows={}
    hashes={}
    for name,loader in zip(['train','valid','test'],loaders):
        feat=loader.dataset.inter_feat
        identity=torch.stack([feat[config['USER_ID_FIELD']],feat[config['ITEM_ID_FIELD']]],1)
        hashes[name]=hashlib.sha256(identity.numpy().tobytes()).hexdigest()
        rows[name]=examples(loader,config)
    for name in ['SRGNN','GCSAN','GCEGNN']:
        ref=json.loads((CODE/f'outputs/formal/{name}/seed42/result.json').read_text(encoding='utf-8'))
        assert hashes==ref['split_session_target_sha256']
    tokens=json.loads((CODE/'outputs/grey_improved/tokens.json').read_text(encoding='utf-8'))
    assert tokens['items']==dataset.field2id_token[config['ITEM_ID_FIELD']].tolist()
    data=HERE/'cache'
    data.mkdir(parents=True,exist_ok=True)
    payload={}
    for split,rr in rows.items():
        hist=np.zeros((len(rr),20),dtype=np.int64)
        for i,(_,h,_) in enumerate(rr): hist[i,:len(h)]=h
        payload.update({split+'_histories':hist,split+'_users':np.array([r[0] for r in rr]),
                        split+'_targets':np.array([r[2] for r in rr]),
                        split+'_lengths':np.array([len(r[1]) for r in rr])})
    np.savez_compressed(data/'splits.npz',**payload)
    grey=load_grey()
    # Independently recover every longest training prefix; verify existing statistics.
    longest={}
    for u,h,t in rows['train']:
        seq=np.append(h,t)
        if u not in longest or len(seq)>len(longest[u]): longest[u]=seq
    for u,h,t in rows['valid']: np.testing.assert_array_equal(longest[u],h)
    for u,h,t in rows['test']: np.testing.assert_array_equal(longest[u],h[:-1])
    rebuilt=GreySession(grey.item_num).fit(longest.values())
    np.testing.assert_array_equal(rebuilt.counts,grey.counts)
    for a,b in [(rebuilt.transitions,grey.transitions),(rebuilt.cooccurrence,grey.cooccurrence)]:
        assert (a-b).nnz==0
    ctx=Context(grey)
    # Cross-fit behavioral statistics at SESSION level for supervised inputs.
    # Otherwise the target's own transition/session incidence leaks into its
    # training predictors, especially for items seen in only one session.
    fold_models=[]
    fold_context=[]
    for fold in range(5):
        m=GreySession(grey.item_num).fit(seq for u,seq in longest.items() if u%5!=fold)
        cc=Context(m)
        cc.pop_threshold=ctx.pop_threshold
        fold_models.append(m)
        fold_context.append(cc)
    rng=np.random.default_rng(20261112)
    n=len(rows['train'])
    x=np.zeros((n,32,4),dtype=np.float32)
    contexts=np.zeros((n,6),dtype=np.float32)
    sampled=np.zeros((n,32),dtype=np.int32)
    groups=[]
    aggregate=defaultdict(list)
    teacher_pool=defaultdict(list)
    for i,(u,h,target) in enumerate(rows['train']):
        training_model=fold_models[u%5]
        training_context=fold_context[u%5]
        z=training_model.normalized_features(h)
        c=training_context.get(h)
        g=training_context.group(h,c)
        contexts[i]=c
        groups.append(g)
        pool=np.unique(np.concatenate([stable_topk(np.r_[-np.inf,z[k,1:]],16) for k in range(4)]))
        pool=pool[pool!=target]
        hard=rng.choice(pool,min(16,len(pool)),replace=False).tolist()
        chosen=[target]+hard
        while len(chosen)<32:
            candidate=int(rng.integers(1,grey.item_num))
            if candidate not in chosen: chosen.append(candidate)
        sampled[i]=chosen
        x[i]=z[:,chosen].T
        aggregate[g].append(np.r_[c,z[:,target],z[:,chosen[1:]].mean(axis=1),float(target in h)])
        if len(teacher_pool[g])<8:
            teacher_pool[g].append({'train_index':i,'session':str(tokens['users'][u]),
                'history':[str(tokens['items'][v]) for v in h],
                'target':str(tokens['items'][target]),'context':c.tolist(),
                'target_features':z[:,target].tolist(),
                'candidate_examples':[{'item':str(tokens['items'][v]),'features':z[:,v].tolist(),
                                       'is_training_target':bool(v==target)} for v in chosen[:8]]})
        if (i+1)%5000==0: print(f'prepared {i+1}/{n}, elapsed={time.perf_counter()-begin:.1f}s',flush=True)
    np.savez_compressed(data/'training.npz',features=x,contexts=contexts,candidates=sampled,
                        groups=np.asarray(groups))
    summaries=[]
    for g,rr in sorted(aggregate.items()):
        a=np.stack(rr)
        summaries.append({'group':g,'count':len(a),'context_mean':a[:,:6].mean(0).tolist(),
            'positive_features_mean':a[:,6:10].mean(0).tolist(),
            'negative_features_mean':a[:,10:14].mean(0).tolist(),
            'repeat_target_fraction':float(a[:,14].mean()),'examples':teacher_pool[g]})
    write_json(HERE/'teacher/training_evidence.json',{'source':'train split only; real positive and sampled negative examples',
        'features':['transition','cooccurrence','history_support','popularity'],
        'context_names':CONTEXT_NAMES,'popularity_group_threshold':ctx.pop_threshold,'groups':summaries})
    picked=rng.choice(n,2000,replace=False)
    np.save(data/'teacher_indices.npy',picked)
    valid=rng.choice(len(rows['valid']),2000,replace=False)
    np.save(data/'selection_valid_indices.npy',valid)
    write_json(data/'manifest.json',{'split_hashes':hashes,'train_examples':n,
        'valid_examples':len(rows['valid']),'test_examples':len(rows['test']),
        'candidate_items':grey.item_num-1,'statistics_rebuilt_exact':True,
        'training_feature_shape':list(x.shape),'teacher_samples':len(picked),
        'teacher_input_split':'train only','seed':20261112,'prepare_seconds':time.perf_counter()-begin,
        'training_statistics':'5-fold session-level cross-fit; userID modulo5; exclude all prefixes from own session fold',
        'serving_statistics':'all training longest prefixes; existing GRA model unchanged',
        'source_sha256':digest(CODE/'data/diginetica-formal/diginetica-formal.inter'),
        'original_grey_sha256':digest(CODE/'experiments/grey_session.py'),
        'existing_split_sha256':digest(CODE/'outputs/grey_improved/splits.npz'),
        'sampling':'positive + up to16 hard items from per-feature top16 + uniform fill to32; only train positive inserted'})
    print('Training evidence ready for assistant review.',flush=True)

if __name__=='__main__': main()
