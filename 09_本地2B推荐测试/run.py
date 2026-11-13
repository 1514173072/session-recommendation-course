"""Same frozen 30 sessions, two local Qwen conditions, no cloud requests."""
from pathlib import Path
import csv, hashlib, json, math, sys, time, urllib.request, subprocess
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
SOURCE=ROOT/'07_云端模型八组测试'
sys.path.insert(0,str(ROOT/'06_教师学生推荐性能测试'))
from common import load_grey,numpy_score,stable_topk,ranking_metrics,Recommender
MODEL='qwen3.5:2b-q4_K_M'
BASE='http://127.0.0.1:11434'
OPTIONS={'num_ctx':2048,'num_predict':2048,'temperature':0,'seed':42,'num_thread':4}

def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def api(path,body=None,timeout=240):
    data=None if body is None else json.dumps(body,ensure_ascii=False).encode('utf-8')
    req=urllib.request.Request(BASE+path,data=data,headers={'Content-Type':'application/json'})
    # Loopback API must never inherit a system HTTP proxy.
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=timeout) as r:
        return json.load(r)

def gpu():
    p=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True)
    return p.stdout.strip()

def parse(response):
    text=response.get('message',{}).get('content','').strip()
    out={'success':False,'text':text,'thinking_characters':len(response.get('message',{}).get('thinking',''))}
    try:
        if not response.get('done') or response.get('done_reason')!='stop':raise ValueError('generation did not stop normally')
        obj=json.loads(text)
        w=np.asarray(obj['weights'],dtype=np.float32); b=np.asarray(obj['interactions'],dtype=np.float32)
        if w.shape!=(4,) or b.shape!=(6,):raise ValueError('coefficient shape')
        if not np.isfinite(w).all() or not np.isfinite(b).all():raise ValueError('nonfinite')
        if np.any(w<0) or np.any(np.abs(b)>.500001) or abs(float(w.sum())-1)>.02:raise ValueError('coefficient range')
        out.update(success=True,coefficients=np.r_[w/w.sum(),b].tolist(),raw_weight_sum=float(w.sum()))
    except (ValueError,KeyError,TypeError) as e:out['error']=str(e)
    return out

def call(condition,index,attempt):
    prompt=(SOURCE/f'prompts/{condition}_{index:02d}.txt').read_text(encoding='utf-8')
    body={'model':MODEL,'messages':[{'role':'user','content':prompt}],
          'stream':False,'think':False,'keep_alive':'15m','options':OPTIONS}
    stem=f'{condition}_{index:02d}_a{attempt}'
    path=HERE/f'requests/{stem}.json'
    if path.exists():return json.loads(path.read_text(encoding='utf-8'))
    start=time.perf_counter()
    try:
        response=api('/api/chat',body)
        result={'response':response,**parse(response)}
    except Exception as e:result={'success':False,'error':repr(e)}
    result.update(condition=condition,case_index=index,attempt=attempt,
                  wall_seconds=time.perf_counter()-start,
                  prompt_sha256=hashlib.sha256(prompt.encode('utf-8')).hexdigest())
    save(path,result)
    return result

def rank(ids,target):
    found=np.flatnonzero(ids==target)
    return int(found[0]+1) if len(found) else 0

def evaluate():
    torch.set_num_threads(4)
    cases=json.loads((SOURCE/'cases.json').read_text(encoding='utf-8'))
    frozen={'model':MODEL,'options':OPTIONS,'think':False,'format_constraint':None,
       'cases_sha256':sha(SOURCE/'cases.json'),
       'teacher_sha256':sha(ROOT/'06_教师学生推荐性能测试/teacher/prototypes.json'),
       'prompts':{f'{c}_{i:02d}.txt':sha(SOURCE/f'prompts/{c}_{i:02d}.txt') for c in ['plain','teacher'] for i in range(30)},
       'run_script_sha256':sha(Path(__file__)),
       'timing':'Warm resident model, sequential loopback API call including tokenization, prompt evaluation and output generation. Local ranking separately recorded. Prior cloud calls include Harness startup and up to 4-way concurrency.',
       'max_attempts':2,'sample_count':30,'teacher_type':'Existing training-only prototype in prompt; model weights unchanged'}
    freeze_path=HERE/'frozen_before_requests.json'
    if freeze_path.exists():assert json.loads(freeze_path.read_text(encoding='utf-8'))==frozen
    else:save(freeze_path,frozen)
    save(HERE/'deployment/model_show.json',api('/api/show',{'model':MODEL}))
    save(HERE/'deployment/version.json',api('/api/version'))
    # One generic warmup, no held-out example and no coefficient quality tuning.
    if not (HERE/'deployment/warmup.json').exists():
        start=time.perf_counter()
        warm=api('/api/chat',{'model':MODEL,'messages':[{'role':'user','content':'Return only the JSON object {"ready":true}.'}],
                            'think':False,'stream':False,'keep_alive':'15m','options':{**OPTIONS,'num_predict':32}})
        save(HERE/'deployment/warmup.json',{'wall_seconds':time.perf_counter()-start,'response':warm,'gpu_after':gpu(),'ps':api('/api/ps')})
        print('Warmup completed',flush=True)
    grey=load_grey()
    features=[grey.normalized_features(np.asarray(c['internal_history'],dtype=np.int64)) for c in cases]
    tokens=json.loads((ROOT/'03_代码与实验/outputs/grey_improved/tokens.json').read_text(encoding='utf-8'))['items']
    results=[]
    start=time.perf_counter()
    for i,case in enumerate(cases):
        # Alternate condition order, one request at a time on the 4GB GPU.
        conditions=['plain','teacher'] if i%2==0 else ['teacher','plain']
        for condition in conditions:
            attempts=[]
            for a in range(2):
                attempts.append(call(condition,i,a))
                if attempts[-1]['success']:break
            success=attempts[-1]['success']
            row={'model':MODEL,'condition':condition,'case_index':i,'test_index':case['test_index'],
                 'success':success,'attempts':attempts,'api_wall_seconds':sum(a['wall_seconds'] for a in attempts),
                 'target_rank':0,'recommendations':[],'ranking_seconds':0.0}
            if success:
                coeff=np.asarray(attempts[-1]['coefficients'],dtype=np.float32)
                t=time.perf_counter();scores=numpy_score(features[i],coeff);scores[0]=-np.inf;ids=stable_topk(scores)
                row['ranking_seconds']=time.perf_counter()-t
                row['target_rank']=rank(ids,case['internal_target'])
                row['coefficients']=coeff.tolist()
                row['recommendations']=[str(tokens[x]) for x in ids]
            row['recommendation_seconds']=row['api_wall_seconds']+row['ranking_seconds']
            results.append(row)
            save(HERE/'results/progress.json',{'completed':len(results),'planned':60,'valid':sum(r['success'] for r in results)})
            print(f'{len(results):02d}/60 {condition} case={i:02d} valid={success} API={row["api_wall_seconds"]:.3f}s elapsed={time.perf_counter()-start:.1f}s',flush=True)
    results.sort(key=lambda r:(r['condition'],r['case_index']))
    summary=[]
    for condition in ['plain','teacher']:
        rows=[r for r in results if r['condition']==condition]
        valid=[r for r in rows if r['success']]
        responses=[r['attempts'][-1]['response'] for r in valid]
        summary.append({'model':MODEL,'condition':condition,'queries':len(rows),'valid_outputs':len(valid),
             'actual_attempts':sum(len(r['attempts']) for r in rows),**ranking_metrics([r['target_rank'] for r in rows]),
             'mean_api_wall_seconds':float(np.mean([r['api_wall_seconds'] for r in rows])),
             'mean_recommendation_seconds':float(np.mean([r['recommendation_seconds'] for r in rows])),
             'p95_api_wall_seconds':float(np.percentile([r['api_wall_seconds'] for r in rows],95)),
             'mean_prompt_eval_seconds':float(np.mean([r.get('prompt_eval_duration',0)/1e9 for r in responses])) if responses else None,
             'mean_decode_seconds':float(np.mean([r.get('eval_duration',0)/1e9 for r in responses])) if responses else None,
             'output_tokens':sum(r.get('eval_count',0) for r in responses),
             'decode_tokens_per_second':sum(r.get('eval_count',0) for r in responses)/(sum(r.get('eval_duration',0) for r in responses)/1e9) if responses else None,
             'thinking_outputs':sum(r['attempts'][-1]['thinking_characters']>0 for r in valid)})
    save(HERE/'results/predictions.json',results)
    save(HERE/'results/summary.json',summary)
    save(HERE/'deployment/loaded_after_test.json',{'gpu':gpu(),'ps':api('/api/ps')})
    # Same-30-case deterministic local baselines; their timing includes feature extraction and Top10.
    baseline=[]
    for name in ['gra_original','student_plain','student_teacher']:
        recommender=Recommender(name);recommender.recommend(np.asarray(cases[0]['internal_history'],dtype=np.int64))
        rr=[];times=[]
        for case in cases:
            t=time.perf_counter();ids,_,_=recommender.recommend(np.asarray(case['internal_history'],dtype=np.int64));times.append(time.perf_counter()-t)
            rr.append(rank(ids,case['internal_target']))
        baseline.append({'model':name,'queries':30,**ranking_metrics(rr),'mean_seconds':float(np.mean(times)),'ranks':rr,'seconds':times})
    save(HERE/'results/same30_baselines.json',baseline)
    export(summary,baseline)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

def export(summary,baseline):
    dest=ROOT/'08扩展对比/本地2B对比';dest.mkdir(parents=True,exist_ok=True)
    cloud=json.loads((SOURCE/'results/summary.json').read_text(encoding='utf-8'))
    rows=[]
    for r in cloud:
        rows.append({'模型':r['model'],'条件':'有教师' if r['condition']=='teacher' else '无教师','运行位置':'云端Harness',
          '样本数':r['queries'],'有效输出':r['valid_outputs'],'Recall@10':r['recall@10'],'MRR@10':r['mrr@10'],'NDCG@10':r['ndcg@10'],
          '平均调用秒':r['mean_recommendation_wall_seconds'],'平均推荐秒':'','实际请求数':r['actual_attempts']})
    for r in summary:
        rows.append({'模型':MODEL,'条件':'有教师' if r['condition']=='teacher' else '无教师','运行位置':'本地Ollama',
          '样本数':r['queries'],'有效输出':r['valid_outputs'],'Recall@10':r['recall@10'],'MRR@10':r['mrr@10'],'NDCG@10':r['ndcg@10'],
          '平均调用秒':r['mean_api_wall_seconds'],'平均推荐秒':r['mean_recommendation_seconds'],'实际请求数':r['actual_attempts']})
    for r in baseline:
        rows.append({'模型':r['model'],'条件':'本地确定性模型','运行位置':'本地CPU','样本数':30,'有效输出':30,
          'Recall@10':r['recall@10'],'MRR@10':r['mrr@10'],'NDCG@10':r['ndcg@10'],'平均调用秒':'','平均推荐秒':r['mean_seconds'],'实际请求数':0})
    with (dest/'本地2B与已有模型对比.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# 本地2B推荐测试','', '| 项目 | 设置 |','|---|---|',
        '| 模型 | Qwen3.5-2B，Q4_K_M；权重未微调 |',
        '| 数据 | 沿用云端八组的同30条真实Diginetica会话；每组全32,968商品Top10 |',
        '| 教师 | 已有训练数据教师原型加入提示词，未新增云端教师调用 |',
        '| 推理 | think=false；上下文2048；输出上限2048；温度0；逐条串行 |',
        '| 输出 | 4个权重和6个交互系数；无额外JSON语法约束；失败最多重试1次并保留代价 |',
        '| 本地计时 | 预热常驻后的回环API请求；另记全商品评分与Top10时间；冷启动单独存档 |',
        '| 云端计时 | 之前实测，包含Harness启动和网络；最多4路并发，与本地计时路径不同 |',
        '| 解释范围 | 30条会话的小样本功能与性能试测；不同模型体量和运行路径一起变化，不能归因为部署位置单一因素 |',
        '| 归档 | 原始输出、提示词指纹、模型信息、逐条预测及耗时见09_本地2B推荐测试 |','',
        '| 条件 | 有效输出 | Recall@10 | MRR@10 | NDCG@10 | 平均调用(s) |', '|---|---:|---:|---:|---:|---:|']
    for r in summary:lines.append(f'| {"有教师" if r["condition"]=="teacher" else "无教师"} | {r["valid_outputs"]}/30 | {r["recall@10"]:.2%} | {r["mrr@10"]:.5f} | {r["ndcg@10"]:.5f} | {r["mean_api_wall_seconds"]:.3f} |')
    (dest/'测试说明.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':evaluate()
