"""Independent artifact audit: all test rows, OOF feature samples, label provenance."""
import json
import math
from common import *

def main():
    torch.set_num_threads(4)
    manifest=json.loads((HERE/'cache/manifest.json').read_text(encoding='utf-8'))
    frozen=json.loads((HERE/'results/frozen_before_test.json').read_text(encoding='utf-8'))
    expected=json.loads((HERE/'results/test_summary.json').read_text(encoding='utf-8'))
    with np.load(HERE/'cache/splits.npz',allow_pickle=False) as z:
        split={k:z[k] for k in z.files}
    # Recompute actual cached identities instead of trusting the manifest alone.
    for part in ['train','valid','test']:
        # NumPy on Windows stores Python-int arrays as int32; RecBole hashes
        # torch.long/int64 identities. Normalize the byte representation.
        identity=np.stack([split[part+'_users'],split[part+'_targets']],axis=1).astype(np.int64)
        assert hashlib.sha256(identity.tobytes()).hexdigest()==manifest['split_hashes'][part]
    for name in ['SRGNN','GCSAN','GCEGNN']:
        prior=json.loads((CODE/f'outputs/formal/{name}/seed42/result.json').read_text(encoding='utf-8'))
        assert prior['split_session_target_sha256']==manifest['split_hashes']
    tokens=json.loads((CODE/'outputs/grey_improved/tokens.json').read_text(encoding='utf-8'))
    assert digest(CODE/'experiments/grey_session.py')==manifest['original_grey_sha256']
    assert digest(CODE/'data/diginetica-formal/diginetica-formal.inter')==manifest['source_sha256']
    assert digest(CODE/'outputs/grey_improved/splits.npz')==manifest['existing_split_sha256']
    assert digest(HERE/'teacher/prototypes.json')==frozen['teacher_prototypes_sha256']
    assert digest(HERE/'teacher/guidance.jsonl')==frozen['guidance_sha256']
    for name,h in frozen['models'].items(): assert digest(HERE/f'models/{name}.pt')==h
    with np.load(HERE/'cache/training.npz',allow_pickle=False) as z:
        train={k:z[k] for k in z.files}
    teacher=load_teacher()
    proto=json.loads((HERE/'teacher/prototypes.json').read_text(encoding='utf-8'))
    evidence=json.loads((HERE/'teacher/training_evidence.json').read_text(encoding='utf-8'))
    assert proto['evidence_sha256']==digest(HERE/'teacher/training_evidence.json')
    assert len(teacher)==12
    assert set(g['group'] for g in evidence['groups']).issubset(teacher)
    for coeff in teacher.values():
        assert np.isfinite(coeff).all() and np.all(coeff[:4]>=0)
        assert abs(float(coeff[:4].sum())-1)<1e-6 and np.max(np.abs(coeff[4:]))<=.5
    indices=np.load(HERE/'cache/teacher_indices.npy')
    guidance=[json.loads(line) for line in (HERE/'teacher/guidance.jsonl').open(encoding='utf-8')]
    assert len(guidance)==2000 and len(set(indices.tolist()))==2000
    for row,index in zip(guidance,indices):
        assert row['train_index']==index
        assert row['group']==train['groups'][index]
        np.testing.assert_array_equal(row['context'],train['contexts'][index])
        np.testing.assert_array_equal(row['weights']+row['interactions'],teacher[row['group']])
    # Independently reconstruct training prefixes from validation histories.
    longest={int(u):h[:n] for u,h,n in zip(split['valid_users'],split['valid_histories'],split['valid_lengths'])}
    for u,h,n,t in zip(split['train_users'],split['train_histories'],split['train_lengths'],split['train_targets']):
        np.testing.assert_array_equal(np.append(h[:n],t),longest[int(u)][:n+1])
    for u,h,n in zip(split['test_users'],split['test_histories'],split['test_lengths']):
        np.testing.assert_array_equal(h[:n-1],longest[int(u)])
    # Verify training cross-fitting, including target insertion only at training time.
    grey=load_grey()
    contexts=Context(grey)
    sampled_indices=np.random.default_rng(777).choice(len(train['features']),50,replace=False)
    for fold in range(5):
        model=GreySession(grey.item_num).fit(seq for u,seq in longest.items() if u%5!=fold)
        ctx=Context(model)
        ctx.pop_threshold=contexts.pop_threshold
        for i in sampled_indices:
            u=int(split['train_users'][i])
            if u%5!=fold: continue
            h=split['train_histories'][i,:split['train_lengths'][i]]
            candidate=train['candidates'][i]
            assert candidate[0]==split['train_targets'][i] and len(set(candidate.tolist()))==32
            np.testing.assert_allclose(train['features'][i],model.normalized_features(h)[:,candidate].T,rtol=0,atol=1e-7)
            np.testing.assert_allclose(train['contexts'][i],ctx.get(h),rtol=0,atol=1e-7)
            assert train['groups'][i]==ctx.group(h)
    a=json.loads((HERE/'results/student_plain_training.json').read_text(encoding='utf-8'))
    b=json.loads((HERE/'results/student_teacher_training.json').read_text(encoding='utf-8'))
    assert a['completed_epochs']==b['completed_epochs']==30
    assert a['initial_parameters_sha256']==b['initial_parameters_sha256']
    assert a['batch_order_sha256']==b['batch_order_sha256']
    assert a['teacher_anchor_count']==0 and b['teacher_anchor_count']==2000
    assert all(r['teacher_loss']==0 for r in a['history'])
    assert any(r['teacher_loss']>0 for r in b['history'])
    reload_checks={}
    for name in ['student_plain','student_teacher']:
        checkpoint=torch.load(HERE/f'models/{name}.pt',map_location='cpu')
        assert checkpoint['training_data_sha256']==digest(HERE/'cache/training.npz')
        assert checkpoint['selection_indices_sha256']==digest(HERE/'cache/selection_valid_indices.npy')
        net=load_student(name)
        assert sum(p.numel() for p in net.parameters())==282
        assert sum(p.numel()*p.element_size() for p in net.parameters())==1128
        with torch.inference_mode():
            xx=train['contexts'][:40]
            w=net(torch.from_numpy(xx)).numpy()
        params=net.state_dict()
        hidden=np.maximum(xx@params['net.0.weight'].numpy().T+params['net.0.bias'].numpy(),0)
        raw=hidden@params['net.2.weight'].numpy().T+params['net.2.bias'].numpy()
        a4=np.exp(raw[:,:4]-raw[:,:4].max(1,keepdims=True)); a4/=a4.sum(1,keepdims=True)
        independent=np.c_[a4,.5*np.tanh(raw[:,4:])]
        np.testing.assert_allclose(w,independent,rtol=1e-5,atol=1e-7)
        reload_checks[name]={'parameters':282,'fp32_parameter_bytes':1128,'numpy_forward_matches':True}
    references={str(tokens['users'][u]):([str(tokens['items'][x]) for x in h[:n]],str(tokens['items'][t]))
        for u,h,n,t in zip(split['test_users'],split['test_histories'],split['test_lengths'],split['test_targets'])}
    items=set(str(x) for x in tokens['items'][1:])
    verified={}
    latency_hashes=set()
    for name in expected:
        source=CODE/f'outputs/formal/predictions/{name}.jsonl' if name in ['SRGNN','GCSAN','GCEGNN'] else HERE/f'predictions/{name}.jsonl'
        assert digest(source)==expected[name]['predictions_sha256']
        seen=set(); hits=0; rr=dcg=0.
        for line in source.open(encoding='utf-8'):
            row=json.loads(line)
            sid=row['session_id']
            assert sid not in seen
            seen.add(sid)
            assert references[sid]==(row['history'],row['target'])
            rec=row['recommendations']
            ids=[r['item'] for r in rec]
            assert len(ids)==len(set(ids))==10 and set(ids)<=items
            assert all(math.isfinite(r['score']) for r in rec)
            assert all(rec[j]['score']>=rec[j+1]['score'] for j in range(9))
            rank=ids.index(row['target'])+1 if row['target'] in ids else 0
            assert rank==row['target_rank']
            if rank: hits+=1; rr+=1/rank; dcg+=1/math.log2(rank+1)
        assert len(seen)==20000
        metrics={'recall@10':hits/20000,'mrr@10':rr/20000,'ndcg@10':dcg/20000}
        for key,value in metrics.items(): assert abs(value-expected[name]['test'][key])<1e-12
        lat=json.loads((HERE/f'results/latency/{name}.json').read_text(encoding='utf-8'))
        assert lat['correctness_queries']==300 and lat['calls']==900 and len(lat['samples_ms'])==900
        assert abs(np.mean(lat['samples_ms'])-lat['mean_ms'])<1e-12
        assert abs(np.percentile(lat['samples_ms'],95)-lat['p95_ms'])<1e-12
        latency_hashes.add(lat['sample_sha256'])
        verified[name]={'full_prediction_rows':20000,'independent_metrics':metrics,
                        'latency_correctness_queries':300,'timed_calls':900}
    assert len(latency_hashes)==1
    for path in ['性能比较报告.md','results/comparison.csv','results/cases.json','README.md']:
        assert (HERE/path).is_file()
    audit={'all_passed':True,'scope':'independent performance pilot, not part of course paper',
        'source_and_existing_GRA_unchanged':True,'split_hashes':manifest['split_hashes'],
        'train_prefixes_exclude_heldout_positions':True,'session_oof_training_features_checked':50,
        'teacher_evidence_train_only':True,'teacher_prototypes':12,'actual_observed_groups':len(evidence['groups']),
        'teacher_guidance_records_checked':2000,'independent_llm_calls':0,
        'same_student_init_batches_budget':True,'reload_and_independent_forward':reload_checks,
        'models_frozen_match_test_freeze':True,'models':verified,'identical_benchmark_queries':True,
        'online_llm_latency_measured':False,'no_API_speedup_claim':True,
        'script_sha256':{p.name:digest(p) for p in HERE.glob('*.py')}}
    write_json(HERE/'results/completion_audit.json',audit)
    print(json.dumps(audit,indent=2),flush=True)

if __name__=='__main__': main()
