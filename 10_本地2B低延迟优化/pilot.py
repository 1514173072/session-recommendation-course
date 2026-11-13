from pathlib import Path
import json,time,sys
import requests
ROOT=Path(__file__).resolve().parent.parent
HERE=Path(__file__).resolve().parent
s=requests.Session();s.trust_env=False
url='http://127.0.0.1:11435'
p=json.loads((ROOT/'06_教师学生推荐性能测试/teacher/prototypes.json').read_text(encoding='utf-8'))['prototypes']
labels='ABCDEFGHIJKL'
menu='\n'.join(f'{labels[i]}: {r["group"]}; weights={r["weights"]}; interactions={r["interactions"]}' for i,r in enumerate(p))
prefix=('<|im_start|>system\nYou choose a session-recommendation scoring policy from a training-derived menu. '
        'Use length, repeated interest, and training popularity to choose an appropriate policy. '
        'short means length<=3; medium means4..7; long means>=8. popular1 means last-click normalized popularity>=0.23069553; repeat1 means last item appeared before. '
        'Return only the matching policy letter.\n'+menu+'<|im_end|>\n<|im_start|>user\n')
cases=json.loads((ROOT/'07_云端模型八组测试/cases.json').read_text(encoding='utf-8'))
out=[]
for i in range(8):
 c=cases[i];ctx=c['context']; text=f'length={len(c["internal_history"])} repeat={int(ctx[2])} popularity={ctx[4]:.3f}'
 prompt=prefix+text+'<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
 body={'prompt':prompt,'n_predict':1,'temperature':0,'cache_prompt':True,'grammar':'root ::= [A-L]',
       'n_probs':12,'post_sampling_probs':True,'return_tokens':True,'seed':42}
 t=time.perf_counter();r=s.post(url+'/completion',json=body,timeout=120);r.raise_for_status();o=r.json();elapsed=time.perf_counter()-t
 out.append({'index':i,'expected':labels[next(j for j,r in enumerate(p) if r['group']==c['group'])],
             'wall_ms':elapsed*1000,'response':o})
 print(i,o.get('content'),round(elapsed*1000,2),o.get('timings'),flush=True)
 if i==0:print(json.dumps({k:v for k,v in o.items() if k in ['probs','completion_probabilities','generation_settings','tokens']},ensure_ascii=False)[:1200],flush=True)
(HERE/'pilot.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
