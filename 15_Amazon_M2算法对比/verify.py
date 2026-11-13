import json,math
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from common import ROOT,RUN,read,data,sha,Features,metrics,topk,state,save
from train import load_model,batch
def main():
    torch.set_num_threads(4);d=data();m=read(RUN/'manifest.json');f=Features(d)
    assert sha(RUN/'data.json')==m['data_sha256'] and sha(RUN/'attributes.npz')==m['attributes_sha256']
    for name,h in m['raw_sha256'].items():assert sha(ROOT.parent/'14_Amazon_M2推荐数据/data/raw'/name)==h
    for p,h in read(RUN/'results/test_configuration.json')['sha256'].items():assert sha(p)==h
    group_sets={}
    for split in ['train','validation','test']:
        group_sets[split]={tuple(r['history']+[r['target']]) for r in d[split]}
    # Exact raw grouping was checked before truncation; truncated duplicates are reported separately.
    overlaps={a+'_'+b:len(group_sets[a]&group_sets[b]) for a,b in [('train','validation'),('train','test'),('validation','test')]}
    reports=read(RUN/'results/comparison.json');saved={};total=0
    for r in reports:
        path=RUN/f'predictions/{r["key"]}.jsonl';assert sha(path)==r['prediction_sha256']
        rows=[json.loads(x) for x in path.open(encoding='utf-8')];assert len(rows)==len(d['test'])
        for i,(pred,source) in enumerate(zip(rows,d['test'])):
            assert pred['test_index']==i and pred['source_row']==source['source_row']
            assert pred['history']==source['history'] and pred['target']==source['target']
            ids=pred['top10'];assert len(ids)==len(set(ids))==10
            assert all(0<x<len(d['tokens']) for x in ids) and not(set(ids)&set(pred['history']))
            rank=ids.index(pred['target'])+1 if pred['target'] in ids else 0;assert rank==pred['rank'];total+=1
        for key,value in metrics([p['rank'] for p in rows]).items():assert math.isclose(value,r[key],abs_tol=1e-12)
        saved[r['key']]=rows
    gnn_checks={}
    for name in ['SRGNN','GCSAN','GCEGNN']:
        model=load_model(name,d)
        with torch.inference_mode():
            w=model.item_embedding.weight
            assert float((w-model.item_embedding.ids.weight*model.item_embedding.available)[1:].abs().max())>1e-6
            for i,source in enumerate(d['test'][:40]):
                score=model.full_sort_predict(batch([source],name))[0].numpy();assert np.isfinite(score).all()
                assert topk(score,source['history']).tolist()==saved[name][i]['top10']
        gnn_checks[name]={'reloaded_test_predictions_exact':40,'attribute_contribution_nonzero':True,
                        'training_validation_reload_exact':read(RUN/f'results/{name}_training.json')['independent_reload_validation_exact']}
    policy=read(RUN/'results/frozen_policy.json');w=np.asarray(policy['weights_by_state'],dtype=np.float32)
    for i,row in enumerate(d['test']):
        code=state(f.context(row['history']),policy['thresholds']);score=f.policy_score(row['history'],w[code],compiled=False)
        assert topk(score,row['history']).tolist()==saved['qwen_offline'][i]['top10']
    calls=[read(p) for p in sorted((RUN/'qwen_requests').glob('state_*.json'))]
    assert len(calls)==17 and all(c['valid'] and len(c['response']['tokens'])==1 for c in calls)
    evidence=read(RUN/'results/training_policy_evidence.json')
    proposal_by_state={g['state']:g['proposed'] for g in policy['qwen_proposals']}
    for call in calls:
        assert call['response']['model'].endswith('sha256-20cb277f0967ace47b0b5d5658e5e494a88937f7378b74b5a266496400938f4c')
        assert call['response']['content'].strip()==call['label']
        assert call['response']['timings']['predicted_n']==1 and call['response']['timings']['prompt_n']>0
        if call['state']!=99:
            option=ord(call['label'])-65
            assert evidence['groups'][call['state']]['options'][option]['candidate']==proposal_by_state[call['state']]
    assert policy['selected_alpha']==max(policy['alpha_choices_validation'],key=lambda r:r['mrr@10'])['alpha']
    assert policy['selected_beta']==int(max(policy['grey_beta_validation'],key=lambda b:policy['grey_beta_validation'][b]['mrr@10']))
    assert policy['online_llm_calls_per_request']==0 and not policy['test_targets_used_for_selection']
    result={'passed':True,'test_sessions':len(d['test']),'methods':len(reports),'aligned_prediction_rows':total,
            'raw_and_frozen_hashes_match':True,'all_metrics_recomputed':True,'history_filter_and_catalogue_aligned':True,
            'gnn_checks':gnn_checks,'qwen_full_dense_equivalence_cases':len(d['test']),
            'actual_local_llm_calls':len(calls),'native_model_identity_and_outputs_verified':True,
            'raw_qwen_choices_used_in_ablation':True,'online_llm_calls_per_request':0,
            'exact_full_sequence_overlap_before_truncation':m['cross_split_full_sequence_overlap'],
            'truncated_sequence_overlaps':overlaps,'verify_source_sha256':sha(__file__)}
    save(RUN/'results/verification.json',result);print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=4):main()
