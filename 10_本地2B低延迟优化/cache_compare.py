"""Separate offline LLM policy cache from synchronous per-query 2B latency."""
import csv,json,math,sys,time
from pathlib import Path
import numpy as np
import refine
opt=refine.opt

def main():
 opt.torch.set_num_threads(4)
 selected=json.loads((opt.HERE/'results/refined_selection_before_test.json').read_text(encoding='utf-8'))['selected']
 freeze=json.loads((opt.HERE/'frozen_before_selection.json').read_text(encoding='utf-8'))
 with np.load(opt.ROOT/'06_教师学生推荐性能测试/cache/splits.npz') as archive:
  data={k:archive[k] for k in archive.files}
 grey=opt.load_grey();ctx=opt.Context(grey)
 cache={};starts=time.perf_counter()
 # Exhaust the exact input domain of examples12/raw12: 3 length bins x 2 repeat x 2 popularity.
 # Other variants are intentionally unsupported: cache must not drop model input fields.
 assert selected in ['examples12','raw12']
 cached_path=opt.HERE/'results/offline_llm_policy_cache.json'
 previous=json.loads(cached_path.read_text(encoding='utf-8')) if cached_path.exists() else None
 if previous:
  assert previous['selected']==selected and previous['teacher_sha256']==opt.sha(opt.ROOT/'06_教师学生推荐性能测试/teacher/prototypes.json')
 for length_bin in range(3):
  for repeat in range(2):
   for popular in range(2):
    length=[2,5,8][length_bin]
    h=np.arange(1,length+1,dtype=np.int64)
    # Synthetic contexts enumerate valid codes, not synthetic performance data.
    c=np.asarray([0,1,repeat,0,popular*.5,0],dtype=np.float32)
    key=f'{length_bin},{repeat},{popular}'
    r=previous['entries'][key] if previous else opt.infer(selected,h,c,'cache_build_'+key)
    assert r['valid'];cache[key]=r
 build_seconds=previous['build_seconds'] if previous else time.perf_counter()-starts
 opt.save(opt.HERE/'results/offline_llm_policy_cache.json',{'selected':selected,'build_seconds':build_seconds,
   'entries':cache,'teacher_sha256':opt.sha(opt.ROOT/'06_教师学生推荐性能测试/teacher/prototypes.json'),
   'scope':'12 actual local full-2B calls, then exact memoization of the model input; online hits do not execute the 2B model'})
 tokens=json.loads((opt.ROOT/'03_代码与实验/outputs/grey_improved/tokens.json').read_text(encoding='utf-8'))
 token2id={str(x):i for i,x in enumerate(tokens['items'])}
 def cached(raw):
  h=np.asarray([token2id[str(x)] for x in raw],dtype=np.int64)
  c=ctx.get(h);key=f'{0 if len(h)<=3 else (1 if len(h)<=7 else 2)},{int(c[2])},{int(c[4]>=0.23069553)}'
  coeff=np.asarray(cache[key]['coefficients'],dtype=np.float32)
  z=grey.normalized_features(h);scores=opt.numpy_score(z,coeff);scores[0]=-np.inf;ids=opt.stable_topk(scores)
  return ids,[str(tokens['items'][i]) for i in ids],key
 result=[]
 for name,indices in [('test1000',freeze['test_indices']),('original30',freeze['original_test_indices']),('test20000',list(range(len(data['test_targets']))))]:
  ranks=[];durations=[];keys=[]
  for i in indices:
   h=opt.history(data,'test',i);raw=[str(tokens['items'][x]) for x in h]
   t=time.perf_counter();ids,_,key=cached(raw);durations.append((time.perf_counter()-t)*1000);keys.append(key)
   matches=np.flatnonzero(ids==data['test_targets'][i]);ranks.append(int(matches[0]+1) if len(matches) else 0)
  out={'model':'offline_2b_policy_cache','split':name,'queries':len(indices),**opt.ranking_metrics(ranks),
       'mean_end_to_end_ms':float(np.mean(durations)),'p95_end_to_end_ms':float(np.percentile(durations,95)),
       'online_model_calls':0,'cache_hit_rate':1.0,'ranks':ranks,'per_case_ms':durations}
  result.append(out);print(json.dumps({k:v for k,v in out.items() if k not in ['ranks','per_case_ms']},ensure_ascii=False),flush=True)
 opt.save(opt.HERE/'results/cache_comparison.json',result)

if __name__=='__main__':main()
