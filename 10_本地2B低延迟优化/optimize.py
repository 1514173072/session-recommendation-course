"""Validation-selected, real per-request 2B policy inference; no response cache."""
from pathlib import Path
import csv,hashlib,json,math,sys,time
import numpy as np
import requests,torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'06_教师学生推荐性能测试'))
from common import load_grey,Context,numpy_score,stable_topk,ranking_metrics,Recommender
LABELS='ABCDEFGHIJKL'
PROTOTYPES=json.loads((ROOT/'06_教师学生推荐性能测试/teacher/prototypes.json').read_text(encoding='utf-8'))['prototypes']
COEFF=np.asarray([r['weights']+r['interactions'] for r in PROTOTYPES],dtype=np.float32)
GROUPS=[r['group'] for r in PROTOTYPES]
VARIANTS=['compact12','examples12','semantic4']
SESSION=requests.Session();SESSION.trust_env=False
BASE='http://127.0.0.1:11435'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,o):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding='utf-8')
def post(route,body):
 r=SESSION.post(BASE+route,json=body,timeout=120);r.raise_for_status();return r.json()

def prompt(variant,history,c):
 length=len(history);repeat=int(c[2]);pop=float(c[4]);unique=float(c[1]);entropy=float(c[5])
 if variant=='compact12':
  menu=' '.join(f'{LABELS[i]}={g.replace("_repeat0","/new").replace("_repeat1","/repeat").replace("_popular0","/rare").replace("_popular1","/popular")}' for i,g in enumerate(GROUPS))
  system=('Classify the session using this menu. short: length<=3; medium:4-7; long:>=8. repeat: last item appeared earlier. popular: popularity>=0.23069553, else rare. Output one letter only. '+menu)
  user=f'length={length};last_repeated={repeat};popularity={pop:.3f}'
 elif variant=='examples12':
  # Input compression uses history-side bins only, never the target.
  system='Map session codes to policy letters. L=0 short,1 medium,2 long; R=last repeated; P=popular. '+ ' '.join(f'{i//4},{(i//2)%2},{i%2}:{LABELS[i]}' for i in range(12))+'. Output one letter.'
  user=f'L={0 if length<=3 else (1 if length<=7 else 2)},R={repeat},P={int(pop>=0.23069553)}'
 else:
  system=('Choose a recommendation policy using session behavior. A=short non-repeating exploration; B=long non-repeating cooccurrence; C=short repeating interest; D=long repeating interest. Output one letter only.')
  user=f'length={length},last_repeat={repeat},unique={unique:.2f},transition_entropy={entropy:.2f},last_popularity={pop:.2f}'
 return '<|im_start|>system\n'+system+'<|im_end|>\n<|im_start|>user\n'+user+'<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

def codebook(variant):
 if variant!='semantic4':return COEFF
 # Four anchors come from the existing training-derived teacher, not test tuning.
 return COEFF[[0,8,2,10]]

def infer(variant,history,c,tag):
 p=prompt(variant,history,c)
 body={'prompt':p,'n_predict':1,'temperature':0,'cache_prompt':True,'grammar':'root ::= [A-D]' if variant=='semantic4' else 'root ::= [A-L]',
       'n_probs':0,'return_tokens':True,'seed':42,'repeat_penalty':1,'samplers':['temperature']}
 t=time.perf_counter();o=post('/completion',body);wall=(time.perf_counter()-t)*1000
 text=o.get('content','').strip();valid=len(text)==1 and text in LABELS[:len(codebook(variant))] and len(o.get('tokens',[]))==1 and not o.get('truncated')
 row={'variant':variant,'tag':tag,'prompt':p,'request':body,'response':o,'wall_ms':wall,'valid':valid}
 if valid:row['policy_index']=LABELS.index(text);row['coefficients']=codebook(variant)[row['policy_index']].tolist()
 return row

def history(data,split,index):return data[split+'_histories'][index,:int(data[split+'_lengths'][index])].astype(np.int64)

def evaluate_batch(variant,split,indices,data,grey,ctx,record_file):
 rows=[];target_values=data[split+'_targets'];begin=time.perf_counter()
 with record_file.open('w',encoding='utf-8') as f:
  for n,i in enumerate(indices):
   h=history(data,split,i);c=ctx.get(h)
   # Include raw internal ID -> context -> full catalogue score -> Top10 in end-to-end.
   t=time.perf_counter();c=ctx.get(h);r=infer(variant,h,c,f'{split}_{int(i)}');rank=0
   if r['valid']:
    z=grey.normalized_features(h);scores=numpy_score(z,np.asarray(r['coefficients'],dtype=np.float32));scores[0]=-np.inf;ids=stable_topk(scores)
    matches=np.flatnonzero(ids==target_values[i]);rank=int(matches[0]+1) if len(matches) else 0
    r['top10_internal']=ids.tolist()
   r.update(split=split,index=int(i),target_rank=rank,history_internal=h.tolist(),target_internal=int(target_values[i]),end_to_end_ms=(time.perf_counter()-t)*1000)
   f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush();rows.append(r)
   if (n+1)%50==0:print(f'{variant} {split} {n+1}/{len(indices)} elapsed={time.perf_counter()-begin:.1f}s',flush=True)
 metrics=ranking_metrics([r['target_rank'] for r in rows])
 summary={'variant':variant,'split':split,'queries':len(rows),'valid_outputs':sum(r['valid'] for r in rows),**metrics,
    'mean_api_ms':float(np.mean([r['wall_ms'] for r in rows])),'median_api_ms':float(np.median([r['wall_ms'] for r in rows])),
    'mean_end_to_end_ms':float(np.mean([r['end_to_end_ms'] for r in rows])),
    'p95_end_to_end_ms':float(np.percentile([r['end_to_end_ms'] for r in rows],95)),
    'mean_evaluated_tokens':float(np.mean([r['response']['timings']['prompt_n'] for r in rows])),
    'mean_cached_tokens':float(np.mean([r['response']['timings']['cache_n'] for r in rows])),
    'mean_model_ms':float(np.mean([r['response']['timings']['prompt_ms']+r['response']['timings']['predicted_ms'] for r in rows])),
    'different_policies':len(set(r.get('policy_index') for r in rows if r['valid']))}
 return summary,rows

def main():
 torch.set_num_threads(4)
 for d in ['requests','results']: (HERE/d).mkdir(exist_ok=True)
 data=np.load(ROOT/'06_教师学生推荐性能测试/cache/splits.npz')
 grey=load_grey();ctx=Context(grey)
 valid_indices=np.random.default_rng(20261113).choice(len(data['valid_targets']),128,replace=False)
 old=json.loads((ROOT/'07_云端模型八组测试/cases.json').read_text(encoding='utf-8'))
 old_indices=[r['test_index'] for r in old]
 available=np.setdiff1d(np.arange(len(data['test_targets'])),old_indices)
 test_indices=np.random.default_rng(20261114).choice(available,1000,replace=False)
 freeze={'script_sha256':sha(__file__),'splits_sha256':sha(ROOT/'06_教师学生推荐性能测试/cache/splits.npz'),
         'teacher_sha256':sha(ROOT/'06_教师学生推荐性能测试/teacher/prototypes.json'),
         'valid_indices':valid_indices.tolist(),'test_indices':test_indices.tolist(),'original_test_indices':old_indices,
         'selection':'highest valid MRR@10; no test-driven prompt/model selection','variants':VARIANTS,
         'runtime':'same full Qwen3.5-2B Q4, CUDA, single-slot, one output token, no response-result cache'}
 save(HERE/'frozen_before_selection.json',freeze)
 validation=[]
 for variant in VARIANTS:
  # Warmup is a VALIDATION history, saved separately and excluded from measured requests.
  h=history(data,'valid',int(valid_indices[-1]));warm=infer(variant,h,ctx.get(h),'warmup');save(HERE/f'requests/{variant}_warmup.json',warm)
  result,_=evaluate_batch(variant,'valid',valid_indices,data,grey,ctx,HERE/f'requests/{variant}_validation.jsonl')
  validation.append(result);print(json.dumps(result,ensure_ascii=False),flush=True)
 selected=max(validation,key=lambda r:(r['mrr@10'],-r['mean_end_to_end_ms']))['variant']
 save(HERE/'results/selection_before_test.json',{'selected':selected,'validation':validation,'freeze_sha256':sha(HERE/'frozen_before_selection.json')})
 print('Selected before test:',selected,flush=True)
 # Warm before each distinct evaluation segment, excluded from timing.
 h=history(data,'valid',int(valid_indices[-1]));infer(selected,h,ctx.get(h),'test_warmup')
 main_result,main_rows=evaluate_batch(selected,'test',test_indices,data,grey,ctx,HERE/'requests/selected_test1000.jsonl')
 h=history(data,'valid',int(valid_indices[-1]));infer(selected,h,ctx.get(h),'old30_warmup')
 old_result,old_rows=evaluate_batch(selected,'test',old_indices,data,grey,ctx,HERE/'requests/selected_original30.jsonl')
 save(HERE/'results/summary.json',{'selected':selected,'validation':validation,'test1000':main_result,'original30':old_result})
 save(HERE/'results/runtime_props.json',SESSION.get(BASE+'/props',timeout=10).json())
 print(json.dumps({'test1000':main_result,'original30':old_result},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
