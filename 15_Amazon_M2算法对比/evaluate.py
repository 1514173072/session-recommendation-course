"""Freeze the finished configurations, then test each method on every held-out session."""
import csv,json,time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from common import ROOT,RUN,Features,data,read,save,sha,metrics,topk,state,benchmark
from train import load_model,batch

def quality(key,rows,score):
    path=RUN/f'predictions/{key}.jsonl';rr=[]
    with path.open('w',encoding='utf-8') as stream:
        for i,row in enumerate(rows):
            ids=topk(score(row['history']),row['history']).tolist()
            rank=ids.index(row['target'])+1 if row['target'] in ids else 0;rr.append(rank)
            stream.write(json.dumps({'test_index':i,'source_row':row['source_row'],'history':row['history'],
                'target':row['target'],'top10':ids,'rank':rank},ensure_ascii=False)+'\n')
            if (i+1)%2000==0:print(f'Test {key} {i+1}/{len(rows)}',flush=True)
    return rr
@torch.inference_mode()
def gnn_quality(key,rows,model):
    path=RUN/f'predictions/{key}.jsonl';rr=[]
    with path.open('w',encoding='utf-8') as stream:
        for start in range(0,len(rows),128):
            chunk=rows[start:start+128];scores=model.full_sort_predict(batch(chunk,key)).numpy()
            for j,(row,scores_i) in enumerate(zip(chunk,scores)):
                ids=topk(scores_i,row['history']).tolist();rank=ids.index(row['target'])+1 if row['target'] in ids else 0;rr.append(rank)
                stream.write(json.dumps({'test_index':start+j,'source_row':row['source_row'],'history':row['history'],
                    'target':row['target'],'top10':ids,'rank':rank},ensure_ascii=False)+'\n')
    return rr
def main():
    torch.set_num_threads(4);d=data();policy=read(RUN/'results/frozen_policy.json');rows=d['test'];f=Features(d)
    files=[RUN/'data.json',RUN/'attributes.npz',RUN/'results/frozen_policy.json']+[RUN/f'models/{name}.pt' for name in ['SRGNN','GCSAN','GCEGNN']]
    files += [ROOT/name for name in ['common.py','train.py','policy.py','evaluate.py']]
    freeze=RUN/'results/test_configuration.json'
    if freeze.exists():
        for p,digest in read(freeze)['sha256'].items():assert sha(p)==digest
    else:save(freeze,{'sha256':{str(p):sha(p) for p in files},'protocol':'Single fixed held-out test; all decisions chosen from training/calibration/validation before first test'})
    state_weights=np.asarray(policy['weights_by_state'],dtype=np.float32);global_weights=np.asarray(policy['global_weights'],dtype=np.float32)
    candidate_weights=np.asarray([c['weights'] for c in policy['candidates']],dtype=np.float32)
    raw_weights=candidate_weights[[g['proposed'] for g in policy['qwen_proposals']]]
    def qwen_score(h):return f.policy_score(h,state_weights[state(f.context(h),policy['thresholds'])])
    def raw_qwen_score(h):return f.policy_score(h,raw_weights[state(f.context(h),policy['thresholds'])])
    methods=[('popularity','热门推荐',lambda h:f.pop,True),('grey','灰色关联',lambda h:f.grey_score(h),False),
       ('grey_improved',f'改进灰关联β={policy["selected_beta"]}',lambda h:f.grey_score(h,policy['selected_beta']),False),
       ('fixed_policy','固定策略（消融）',lambda h:f.policy_score(h,global_weights),True),
       ('qwen_raw','离线Qwen原始选策（消融）',raw_qwen_score,True),('qwen_offline','离线qwen3.5-2B（验证校准）',qwen_score,False)]
    cold=np.asarray([f.counts[r['target']]==0 for r in rows]);reports=[]
    for key,title,score,ablation in methods:
        start=time.perf_counter();rr=quality(key,rows,score);evaluation_seconds=time.perf_counter()-start
        latency=benchmark(lambda h:topk(score(h),h),rows);rr=np.asarray(rr)
        report={'key':key,'method':title,'ablation':ablation,'test_sessions':len(rows),'catalogue_items':len(d['tokens'])-1,
             **metrics(rr),**latency,'warm_metrics':metrics(rr[~cold]),'cold_metrics':metrics(rr[cold]),
             'cold_targets':int(cold.sum()),'quality_evaluation_seconds':evaluation_seconds,'online_llm_calls':0,
             'prediction_sha256':sha(RUN/f'predictions/{key}.jsonl')}
        save(RUN/f'results/{key}_test.json',report);reports.append(report);print(json.dumps(report,ensure_ascii=False),flush=True)
    for name in ['SRGNN','GCSAN','GCEGNN']:
        start=time.perf_counter();model=load_model(name,d);loading=time.perf_counter()-start
        start=time.perf_counter();rr=np.asarray(gnn_quality(name,rows,model));quality_time=time.perf_counter()-start
        @torch.inference_mode()
        def predict(h):return topk(model.full_sort_predict(batch([{'history':h}],name))[0].numpy(),h)
        latency=benchmark(predict,rows)
        report={'key':name,'method':{'SRGNN':'SR-GNN＋属性','GCSAN':'GC-SAN＋属性','GCEGNN':'GCE-GNN＋属性'}[name],
            'ablation':False,'test_sessions':len(rows),'catalogue_items':len(d['tokens'])-1,**metrics(rr),**latency,
            'warm_metrics':metrics(rr[~cold]),'cold_metrics':metrics(rr[cold]),'cold_targets':int(cold.sum()),
            'training_seconds':read(RUN/f'results/{name}_training.json')['training_seconds'],
            'loading_and_catalogue_compile_seconds':loading,'quality_evaluation_seconds':quality_time,
            'online_llm_calls':0,'prediction_sha256':sha(RUN/f'predictions/{name}.jsonl')}
        save(RUN/f'results/{name}_test.json',report);reports.append(report);print(json.dumps(report,ensure_ascii=False),flush=True)
        del model
    save(RUN/'results/comparison.json',reports)
    with (RUN/'results/算法对比.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,['method','ablation','test_sessions','catalogue_items','recall@10','mrr@10','ndcg@10','mean_ms','p95_ms'],extrasaction='ignore')
        writer.writeheader();writer.writerows(reports)
    print('Full test completed',flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=4):main()
