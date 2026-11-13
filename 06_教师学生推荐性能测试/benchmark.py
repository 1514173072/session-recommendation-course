"""Fresh isolated-process warm batch1 end-to-end local benchmarks."""
import argparse
import gc
import json
import os
import platform
import sys
import time
import psutil
from common import *

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True)
    args=parser.parse_args()
    name=args.model
    sys.argv=sys.argv[:1]
    torch.set_num_threads(4)
    tokens=json.loads((CODE/'outputs/grey_improved/tokens.json').read_text(encoding='utf-8'))
    token2id={str(t):i for i,t in enumerate(tokens['items'])}
    split=np.load(HERE/'cache/splits.npz',allow_pickle=False)
    selected=np.random.default_rng(20261111).choice(len(split['test_users']),300,replace=False)
    users=split['test_users'][selected]
    queries=[[str(tokens['items'][x]) for x in split['test_histories'][i,:split['test_lengths'][i]]] for i in selected]
    sample_sha=hashlib.sha256(json.dumps(list(zip(users.tolist(),queries)),sort_keys=True).encode()).hexdigest()
    process=psutil.Process(os.getpid())
    before=process.memory_info().rss
    start=time.perf_counter()
    if name in ['SRGNN','GCSAN','GCEGNN']:
        sys.path.insert(0,str(HERE.parent/'app'))
        from predictor import SessionPredictor
        adapter=SessionPredictor(name)
        tensors=list(adapter.model.parameters())+list(adapter.model.buffers())
        for key in getattr(adapter.model,'other_parameter_name',[]): tensors.append(getattr(adapter.model,key))
        unique={t.data_ptr():t for t in tensors if torch.is_tensor(t)}
        state_bytes=sum(t.numel()*t.element_size() for t in unique.values())
        params=sum(t.numel() for t in adapter.model.parameters())
        def predict(raw): return adapter.predict(raw)
        source=CODE/f'outputs/formal/predictions/{name}.jsonl'
    else:
        model=Recommender(name)
        state_bytes=model.state_bytes
        params=282 if model.student is not None else 0
        def predict(raw):
            history=np.asarray([token2id[str(x)] for x in raw],dtype=np.int64)
            ids,scores,_=model.recommend(history)
            return [{'rank':j+1,'item':str(tokens['items'][x]),'score':float(scores[x])} for j,x in enumerate(ids)]
        source=HERE/f'predictions/{name}.jsonl'
    startup=time.perf_counter()-start
    after_load=process.memory_info().rss
    wanted={str(tokens['users'][u]) for u in users}
    prior={}
    for line in source.open(encoding='utf-8'):
        row=json.loads(line)
        if row['session_id'] in wanted: prior[row['session_id']]=row
    assert len(prior)==300
    max_diff=0.
    for u,q in zip(users,queries):
        pred=predict(q)
        expected=prior[str(tokens['users'][u])]['recommendations']
        assert [r['item'] for r in pred]==[r['item'] for r in expected],(name,int(u))
        diff=max(abs(a['score']-b['score']) for a,b in zip(pred,expected))
        max_diff=max(diff,max_diff)
        assert diff<1e-4
    for q in queries[:30]: predict(q)
    samples=[]
    rounds=[]
    gc.collect()
    gc.disable()
    try:
        for r in range(3):
            current=[]
            for i in np.random.default_rng(20261111+r).permutation(len(queries)):
                start=time.perf_counter_ns()
                pred=predict(queries[i])
                current.append((time.perf_counter_ns()-start)/1e6)
                assert len(pred)==10
            samples.extend(current)
            rounds.append({'mean_ms':float(np.mean(current)),'p95_ms':float(np.percentile(current,95))})
    finally: gc.enable()
    result={'model':name,'sample_sha256':sample_sha,'distinct_queries':300,'calls':900,
        'mean_ms':float(np.mean(samples)),'median_ms':float(np.median(samples)),
        'p95_ms':float(np.percentile(samples,95)),'numeric_state_bytes':state_bytes,
        'trainable_parameters':params,'model_load_seconds_excluded':startup,
        'process_rss_before_model_load_bytes':before,'process_rss_after_load_bytes':after_load,
        'process_rss_after_requests_bytes':process.memory_info().rss,
        'process_peak_working_set_bytes':getattr(process.memory_info(),'peak_wset',None),
        'correctness_queries':300,'max_score_diff':max_diff,'samples_ms':samples,'per_round':rounds,
        'scope':'warm CPU4threads, raw item IDs -> feature/graph construction -> full catalogue score -> top10 raw IDs; excludes startup, disk IO, HTTP and queueing',
        'memory_scope':'RSS includes Python, torch and loader/dataset objects; numeric state includes retained statistics/context arrays and network; these are different quantities',
        'teacher_scope':'teacher_rules is a compiled assistant-authored policy, NOT an online LLM/API call' if name=='teacher_rules' else None,
        'python':platform.python_version(),'torch':torch.__version__,'platform':platform.platform()}
    write_json(HERE/f'results/latency/{name}.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['samples_ms']},indent=2),flush=True)

if __name__=='__main__': main()
