"""Full-catalogue test predictions and exact comparison to stored GNN outputs."""
import json
import time
from common import *

def main():
    torch.set_num_threads(4)
    frozen=json.loads((HERE/'results/frozen_before_test.json').read_text(encoding='utf-8'))
    assert frozen['teacher_prototypes_sha256']==digest(HERE/'teacher/prototypes.json')
    for name,h in frozen['models'].items(): assert h==digest(HERE/f'models/{name}.pt')
    split=np.load(HERE/'cache/splits.npz',allow_pickle=False)
    tokens=json.loads((CODE/'outputs/grey_improved/tokens.json').read_text(encoding='utf-8'))
    (HERE/'predictions').mkdir(exist_ok=True)
    summary={}
    for name in ['gra_original','gra_beta7','teacher_rules','student_plain','student_teacher']:
        model=Recommender(name)
        ranks=[]
        begin=time.perf_counter()
        with (HERE/f'predictions/{name}.jsonl').open('w',encoding='utf-8') as f:
            for i,(u,h,n,t) in enumerate(zip(split['test_users'],split['test_histories'],split['test_lengths'],split['test_targets'])):
                history=h[:n]
                ids,score,coeff=model.recommend(history)
                matched=np.flatnonzero(ids==t)
                rank=int(matched[0]+1) if len(matched) else 0
                assert len(ids)==10 and len(set(ids.tolist()))==10 and np.all(ids>0)
                assert np.all(np.isfinite(score[ids]))
                ranks.append(rank)
                row={'model':name,'test_index':i,'session_id':str(tokens['users'][u]),
                    'history':[str(tokens['items'][x]) for x in history],
                    'target':str(tokens['items'][t]),'target_rank':rank,
                    'coefficients':coeff.tolist() if coeff is not None else None,
                    'recommendations':[{'rank':j+1,'item':str(tokens['items'][x]),'score':float(score[x])} for j,x in enumerate(ids)]}
                f.write(json.dumps(row)+'\n')
                if (i+1)%5000==0: print(f'{name}: {i+1}/20000, elapsed={time.perf_counter()-begin:.1f}s',flush=True)
        summary[name]={'test':ranking_metrics(ranks),'queries':len(ranks),
            'evaluation_wall_seconds_including_output_io':time.perf_counter()-begin,
            'numeric_state_bytes':model.state_bytes,'predictions_sha256':digest(HERE/f'predictions/{name}.jsonl')}
        write_json(HERE/'results/test_summary.json',summary)
    reference={}
    for i,(u,h,n,t) in enumerate(zip(split['test_users'],split['test_histories'],split['test_lengths'],split['test_targets'])):
        reference[str(tokens['users'][u])]=([str(tokens['items'][x]) for x in h[:n]],str(tokens['items'][t]))
    for name in ['SRGNN','GCSAN','GCEGNN']:
        path=CODE/f'outputs/formal/predictions/{name}.jsonl'
        ranks=[]
        seen=set()
        for line in path.open(encoding='utf-8'):
            row=json.loads(line)
            sid=row['session_id']
            assert sid not in seen
            seen.add(sid)
            assert reference[sid]==(row['history'],row['target'])
            ids=[r['item'] for r in row['recommendations']]
            rank=ids.index(row['target'])+1 if row['target'] in ids else 0
            assert rank==row['target_rank']
            ranks.append(rank)
        assert len(seen)==20000
        summary[name]={'test':ranking_metrics(ranks),'queries':len(ranks),
                       'predictions_sha256':digest(path),'reused_existing_predictions':True,
                       'history_and_target_exact':True}
    # Original and beta7 must match their independently saved prior results.
    old=json.loads((CODE/'outputs/grey_coefficient_change/comparison.json').read_text(encoding='utf-8'))
    for name,family in [('gra_original','baseline'),('gra_beta7','rational_distance')]:
        ref=next(r for r in old if r['family']==family)
        for k in ['recall@10','mrr@10','ndcg@10']:
            assert abs(summary[name]['test'][k]-ref[k])<1e-12,(name,k)
    write_json(HERE/'results/test_summary.json',summary)
    print(json.dumps({k:v['test'] for k,v in summary.items()},indent=2),flush=True)

if __name__=='__main__': main()
