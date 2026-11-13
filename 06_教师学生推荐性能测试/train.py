"""Matched supervised-only and teacher-guided 282-parameter students."""
import time
import json
import numpy as np
import torch
from torch.nn import functional as F
from common import *

@torch.inference_mode()
def validate(model,grey,context,split,indices):
    ranks=[]
    for i in indices:
        h=split['valid_histories'][i,:split['valid_lengths'][i]]
        c=context.get(h)
        coeff=model(torch.from_numpy(c)[None])[0].numpy()
        score=numpy_score(grey.normalized_features(h),coeff)
        score[0]=-np.inf
        ids=stable_topk(score)
        matched=np.flatnonzero(ids==split['valid_targets'][i])
        ranks.append(int(matched[0]+1) if len(matched) else 0)
    return ranking_metrics(ranks)

def main():
    torch.set_num_threads(4)
    (HERE/'models').mkdir(exist_ok=True)
    (HERE/'results').mkdir(exist_ok=True)
    teacher=load_teacher()
    with np.load(HERE/'cache/training.npz',allow_pickle=False) as d:
        x=torch.from_numpy(d['features'])
        context=torch.from_numpy(d['contexts'])
        groups=d['groups'].copy()
    teacher_indices=np.load(HERE/'cache/teacher_indices.npy')
    teacher_labels=np.stack([teacher[str(groups[i])] for i in teacher_indices])
    teacher_context=context[teacher_indices]
    teacher_targets=torch.from_numpy(teacher_labels)
    with (HERE/'teacher/guidance.jsonl').open('w',encoding='utf-8') as f:
        for index,c,t in zip(teacher_indices,teacher_context,teacher_labels):
            f.write(json.dumps({'train_index':int(index),'group':str(groups[index]),
                'context':c.tolist(),'weights':t[:4].tolist(),'interactions':t[4:].tolist(),
                'origin':'assistant-authored train-evidence prototypes, deterministically assigned; not an independent LLM call'})+'\n')
    split=np.load(HERE/'cache/splits.npz',allow_pickle=False)
    valid_indices=np.load(HERE/'cache/selection_valid_indices.npy')
    grey=load_grey()
    ctx=Context(grey)
    results={}
    for name in ['student_plain','student_teacher']:
        torch.manual_seed(42)
        model=Student()
        assert sum(p.numel() for p in model.parameters())==282
        initial_sha=hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
        optimizer=torch.optim.Adam(model.parameters(),lr=0.003)
        order_rng=np.random.default_rng(42)
        anchor_rng=np.random.default_rng(20261113)
        best=-1.
        history=[]
        start=time.perf_counter()
        optimize_seconds=0.
        permutation_hashes=[]
        for epoch in range(1,31):
            model.train()
            order=order_rng.permutation(len(x))
            permutation_hashes.append(hashlib.sha256(order.tobytes()).hexdigest())
            begin=time.perf_counter()
            ce_total=kd_total=0.
            for offset in range(0,len(x),512):
                batch=order[offset:offset+512]
                coeff=model(context[batch])
                scores=torch_score(x[batch],coeff)/0.1
                ce=F.cross_entropy(scores,torch.zeros(len(batch),dtype=torch.long))
                kd=torch.zeros(())
                if name=='student_teacher':
                    anchor=anchor_rng.integers(0,len(teacher_context),size=min(128,len(batch)))
                    kd=F.mse_loss(model(teacher_context[anchor]),teacher_targets[anchor])
                loss=ce+kd
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ce_total+=float(ce.detach())*len(batch)
                kd_total+=float(kd.detach())*len(batch)
            optimize_seconds+=time.perf_counter()-begin
            record={'epoch':epoch,'click_loss':ce_total/len(x),'teacher_loss':kd_total/len(x)}
            if epoch%5==0:
                model.eval()
                valid=validate(model,grey,ctx,split,valid_indices)
                record['selection_valid']=valid
                if valid['mrr@10']>best:
                    best=valid['mrr@10']
                    checkpoint={'state_dict':{k:v.detach().clone() for k,v in model.state_dict().items()},
                        'epoch':epoch,'selection_valid':valid,'seed':42,'parameter_count':282,
                        'context_names':CONTEXT_NAMES,'teacher_supervision':name=='student_teacher',
                        'teacher_prototypes_sha256':digest(HERE/'teacher/prototypes.json'),
                        'training_data_sha256':digest(HERE/'cache/training.npz'),
                        'selection_indices_sha256':digest(HERE/'cache/selection_valid_indices.npy')}
                    torch.save(checkpoint,HERE/f'models/{name}.pt')
                print(f'{name} epoch {epoch}: ce={record["click_loss"]:.4f}, kd={record["teacher_loss"]:.5f}, valid={valid}, elapsed={time.perf_counter()-start:.1f}s',flush=True)
            history.append(record)
            write_json(HERE/f'results/{name}_training.json',{'model':name,'history':history,
                'initial_parameters_sha256':initial_sha,'batch_order_sha256':permutation_hashes,
                'optimization_seconds':optimize_seconds,'elapsed_seconds':time.perf_counter()-start,
                'planned_epochs':30,'completed_epochs':epoch,'training_examples':len(x),
                'selection_valid_queries':len(valid_indices),'teacher_anchor_count':len(teacher_indices) if name=='student_teacher' else 0})
        chosen=torch.load(HERE/f'models/{name}.pt',map_location='cpu')
        reloaded=load_student(name)
        reference=Student()
        reference.load_state_dict(chosen['state_dict'])
        with torch.inference_mode():
            np.testing.assert_array_equal(reference(context[:50]).numpy(),reloaded(context[:50]).numpy())
        results[name]={'selected_epoch':chosen['epoch'],'selection_valid':chosen['selection_valid'],
            'initial_parameters_sha256':initial_sha,'batch_order_sha256':permutation_hashes,
            'optimization_seconds':optimize_seconds,'training_and_selection_seconds':time.perf_counter()-start,
            'checkpoint_sha256':digest(HERE/f'models/{name}.pt'),'reload_exact':True,
            'parameter_count':282,'parameter_bytes':1128}
    assert results['student_plain']['initial_parameters_sha256']==results['student_teacher']['initial_parameters_sha256']
    assert results['student_plain']['batch_order_sha256']==results['student_teacher']['batch_order_sha256']
    write_json(HERE/'results/training_summary.json',results)
    freeze={'teacher_prototypes_sha256':digest(HERE/'teacher/prototypes.json'),
            'guidance_sha256':digest(HERE/'teacher/guidance.jsonl'),
            'models':{k:v['checkpoint_sha256'] for k,v in results.items()},
            'selection':'best MRR@10 among epoch5,10,15,20,25,30 on fixed2000 validation samples',
            'test_targets_used_for_training_or_selection':False,
            'teacher_kind':'current-conversation assistant authored 12 evidence-based prototype policies; deterministic assignment to2000 anchors',
            'teacher_online_llm_latency_measured':False}
    write_json(HERE/'results/frozen_before_test.json',freeze)
    print('Both students frozen before test evaluation.',flush=True)

if __name__=='__main__': main()
