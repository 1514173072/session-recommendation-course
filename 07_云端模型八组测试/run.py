"""Resume-safe eight-group live Harness comparison, max four in-flight requests."""
from pathlib import Path
import concurrent.futures
import hashlib,json,re,sys,time
import numpy as np
import torch
from one_call import call,HERE

ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'06_教师学生推荐性能测试'))
from common import load_grey,numpy_score,stable_topk,ranking_metrics

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def parse(meta):
    path=HERE/'requests'/(meta['stem']+'.ndjson')
    events=[]
    for line in path.read_text(encoding='utf-8',errors='replace').splitlines():
        try: events.append(json.loads(line))
        except ValueError:pass
    final=[x.get('text','') for x in events if x.get('type')=='final']
    ends=[x.get('reason',{}).get('kind') for x in events if x.get('type')=='status' and x.get('phase')=='turn_end']
    text=final[-1].strip() if final else ''
    usage=[x.get('usage',{}) for x in events if x.get('type')=='status' and x.get('phase')=='step_end']
    result={'success':False,'error':None,'final_text':text,'usage':usage,
            'thinking_characters':sum(len(x.get('text','')) for x in events if x.get('type')=='thinking'),
            'tool_calls':sum(x.get('type') in ['tool_call','tool-call','tool_start'] for x in events),
            'tool_names':[x.get('tool') for x in events if x.get('type')=='tool_call'],
            'terminal_reasons':ends}
    if meta['exit_code'] or meta['timed_out'] or not ends or ends[-1]!='completed':
        err=(HERE/'requests'/(meta['stem']+'.stderr.txt')).read_text(encoding='utf-8',errors='replace')
        result['error']=err[-1800:] or 'request did not complete'
        return result
    strict=True
    try: obj=json.loads(text)
    except ValueError:
        strict=False
        cleaned=re.sub(r'^```(?:json)?\s*|\s*```$','',text).strip()
        try:obj=json.loads(cleaned)
        except ValueError:
            result['error']='invalid JSON';return result
    try:
        w=np.asarray(obj['weights'],dtype=np.float32)
        b=np.asarray(obj['interactions'],dtype=np.float32)
        assert w.shape==(4,) and b.shape==(6,)
        assert np.isfinite(w).all() and np.isfinite(b).all()
        assert np.all(w>=0) and np.all(np.abs(b)<=.500001)
        assert abs(float(w.sum())-1)<=.02
        raw_sum=float(w.sum())
        w=w/w.sum()
    except (AssertionError,KeyError,TypeError,ValueError):
        result['error']='invalid coefficient schema/range or tool use';return result
    result.update(success=True,strict_json=strict,strict_protocol=strict and not result['tool_calls'],
                  coefficients=np.r_[w,b].tolist(),raw_weight_sum=raw_sum)
    return result

def task(model,condition,index,freeze):
    stem=f'{model}_{condition}_{index:02d}'
    prompt_path=HERE/f'prompts/{condition}_{index:02d}.txt'
    request_sha=hashlib.sha256(prompt_path.read_text(encoding='utf-8').encode('utf-8')).hexdigest()
    result_path=HERE/'results'/(stem+'.json')
    if result_path.is_file():
        old=json.loads(result_path.read_text(encoding='utf-8'))
        assert old['patch_sha256']==freeze['patches'][model]
        assert old['prompt_sha256']==request_sha
        return old
    attempts=[]
    for attempt in range(2):
        meta_path=HERE/'requests'/(stem+f'_a{attempt}.meta.json')
        if meta_path.is_file():meta=json.loads(meta_path.read_text(encoding='utf-8'))
        else:meta=call(model,condition,index,attempt=attempt)
        assert meta['patch_sha256']==freeze['patches'][model]
        assert meta['prompt_sha256']==request_sha
        outcome=parse(meta)
        attempts.append({**meta,**outcome})
        if outcome['success']:break
        # A single retry covers transient service or output failures; both remain recorded.
        if attempt==0:time.sleep(1)
    selected=next((i for i,a in enumerate(attempts) if a['success']),None)
    result={'model':model,'condition':condition,'case_index':index,
        'patch_sha256':freeze['patches'][model],
        'prompt_sha256':request_sha,
        'prompt_file_sha256':freeze['prompts'][f'{condition}_{index:02d}.txt'],
        'attempts':attempts,'success':selected is not None,'selected_attempt_index':selected,
        'coefficients':attempts[selected].get('coefficients') if selected is not None else None,
        'total_attempt_wall_seconds':sum(a['wall_seconds_including_harness_startup'] for a in attempts),
        'recommendation_wall_seconds':sum(a['wall_seconds_including_harness_startup'] for a in attempts[:selected+1]) if selected is not None else sum(a['wall_seconds_including_harness_startup'] for a in attempts),
        'successful_model_step_seconds':attempts[selected]['model_step_seconds'] if selected is not None else None}
    result_path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result

def summarize(results,cases):
    grey=load_grey()
    tokens=json.loads((ROOT/'03_代码与实验/outputs/grey_improved/tokens.json').read_text(encoding='utf-8'))
    features={i:grey.normalized_features(np.asarray(c['internal_history'],dtype=np.int64)) for i,c in enumerate(cases)}
    grouped={}
    for row in results:
        i=row['case_index'];case=cases[i]
        row['target_rank']=0;row['recommendations']=[]
        if row['success']:
            coeff=np.asarray(row['coefficients'],dtype=np.float32)
            scores=numpy_score(features[i],coeff);scores[0]=-np.inf
            ids=stable_topk(scores)
            matched=np.flatnonzero(ids==case['internal_target'])
            row['target_rank']=int(matched[0]+1) if len(matched) else 0
            row['recommendations']=[{'rank':j+1,'item':str(tokens['items'][x]),'score':float(scores[x])} for j,x in enumerate(ids)]
        row['session_id']=case['session_id'];row['history']=case['history'];row['target']=case['target']
        grouped.setdefault((row['model'],row['condition']),[]).append(row)
    summary=[]
    for (model,condition),rr in grouped.items():
        valid=[r for r in rr if r['success']]
        steps=[r['successful_model_step_seconds'] for r in valid]
        summary.append({'model':model,'condition':condition,'queries':len(rr),'valid_outputs':len(valid),
            'valid_output_rate':len(valid)/len(rr),'actual_attempts':sum(len(r['attempts']) for r in rr),
            **ranking_metrics([r['target_rank'] for r in rr]),
            'mean_model_step_seconds':float(np.mean(steps)) if steps else None,
            'median_model_step_seconds':float(np.median(steps)) if steps else None,
            'p95_model_step_seconds':float(np.percentile(steps,95)) if steps else None,
            'mean_all_attempt_wall_seconds':float(np.mean([r['total_attempt_wall_seconds'] for r in rr])),
            'mean_recommendation_wall_seconds':float(np.mean([r['recommendation_wall_seconds'] for r in rr])),
            'thinking_response_count':sum(r['attempts'][r['selected_attempt_index']]['thinking_characters']>0 for r in valid),
            'requests_with_tool_calls':sum(r['attempts'][r['selected_attempt_index']]['tool_calls']>0 for r in valid),
            'strict_protocol_count':sum(r['attempts'][r['selected_attempt_index']]['strict_protocol'] for r in valid),
            'total_input_tokens':sum(u.get('inputTokens',0) for r in rr for a in r['attempts'] for u in a['usage']),
            'total_output_tokens':sum(u.get('outputTokens',0) for r in rr for a in r['attempts'] for u in a['usage'])})
    (HERE/'results/summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (HERE/'results/predictions.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary

def main():
    torch.set_num_threads(4)
    manifest=json.loads((HERE/'manifest.json').read_text(encoding='utf-8'))
    cases=json.loads((HERE/'cases.json').read_text(encoding='utf-8'))
    freeze_path=HERE/'frozen_before_requests.json'
    current={'manifest_sha256':sha(HERE/'manifest.json'),'cases_sha256':sha(HERE/'cases.json'),
        'teacher_sha256':sha(ROOT/'06_教师学生推荐性能测试/teacher/prototypes.json'),
        'patches':{m:sha(HERE/f'patches/{m}.yml') for m in manifest['models']},
        'prompts':manifest['prompt_sha256'],
        'desktop_config_sha256':sha(Path.home()/'.dsh/profiles/desktop/cordis.patch.yml'),
        'concurrency':4,'queries_per_group':30,'primary_metric':'recall@10',
        'timing_scope':'Harness step_start to step_end; network, provider routing and generation included; startup separately recorded; up to4 requests in flight',
        'failed_output_metric_policy':'count as miss, no fabricated fallback; one retry maximum',
        'created_unix_time':time.time()}
    if freeze_path.is_file():
        freeze=json.loads(freeze_path.read_text(encoding='utf-8'))
        for key in ['manifest_sha256','cases_sha256','teacher_sha256','patches','prompts','desktop_config_sha256']:
            assert current[key]==freeze[key]
    else:
        freeze=current
        freeze_path.write_text(json.dumps(freeze,ensure_ascii=False,indent=2),encoding='utf-8')
    jobs=[]
    for index in range(30):
        models=manifest['models'][index%4:]+manifest['models'][:index%4]
        conditions=['plain','teacher'] if index%2==0 else ['teacher','plain']
        for model in models:
            for condition in conditions:jobs.append((model,condition,index))
    begin=time.perf_counter();results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        future_to_job={pool.submit(task,*job,freeze):job for job in jobs}
        for future in concurrent.futures.as_completed(future_to_job):
            result=future.result();results.append(result)
            print(f'{len(results):03d}/240 {result["model"]} {result["condition"]} case={result["case_index"]:02d} valid={result["success"]} step={result["successful_model_step_seconds"]} elapsed={time.perf_counter()-begin:.1f}s',flush=True)
            progress={'completed':len(results),'planned':240,'valid':sum(r['success'] for r in results),
                      'wall_elapsed_seconds':time.perf_counter()-begin}
            (HERE/'progress.json').write_text(json.dumps(progress,indent=2),encoding='utf-8')
    results.sort(key=lambda r:(manifest['models'].index(r['model']),r['condition'],r['case_index']))
    summary=summarize(results,cases)
    assert len(summary)==8 and all(r['queries']==30 for r in summary)
    assert sha(Path.home()/'.dsh/profiles/desktop/cordis.patch.yml')==freeze['desktop_config_sha256']
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    print('All eight groups attempted on the same30 cases.',flush=True)

if __name__=='__main__':main()
