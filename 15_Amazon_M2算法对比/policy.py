"""Training-only evidence, real local 2B selections and validation-only calibration."""
import itertools,json,time
import numpy as np
import requests
from threadpoolctl import threadpool_limits
from common import RUN,FEATURES,Features,data,save,metrics,state,ranks,sha

def candidate_policies():
    manual=[('balanced',[1]*8),('behavior',[.5,.3,.1,.05,0,0,0,.05]),
        ('text',[.1,.1,.6,.1,0,0,.1,0]),('brand',[.1,.1,.1,.5,0,0,.1,.1]),
        ('price',[0,.1,.2,.1,.05,.05,.4,.1]),('color',[0,.1,.3,.1,.4,0,.1,0]),
        ('material',[0,.1,.3,.1,0,.4,.1,0]),('popular',[0,.2,.1,0,0,0,.1,.6])]
    entries=[{'name':n,'weights':(np.asarray(w)/sum(w)).tolist()} for n,w in manual]
    for i in range(8):entries.append({'name':FEATURES[i],'weights':np.eye(8)[i].tolist()})
    for i,j in itertools.combinations(range(8),2):
        w=np.zeros(8);w[[i,j]]=.5;entries.append({'name':FEATURES[i]+'+'+FEATURES[j],'weights':w.tolist()})
    rng=np.random.default_rng(44)
    for i in range(16):
        active=rng.choice(8,3,replace=False);w=np.zeros(8);w[active]=rng.dirichlet(np.ones(3));entries.append({'name':f'sparse_{i}','weights':w.tolist()})
    return entries
def evidence(f,rows,weights):
    out=np.zeros((len(weights),len(rows)),dtype=np.int16);ids=np.arange(f.n)[None,:]
    for q,row in enumerate(rows):
        scores=weights@f.matrix(row['history']);scores[:,0]=-np.inf;scores[:,row['history']]=-np.inf
        truth=scores[:,row['target'],None]
        r=1+np.sum(scores>truth,1)+np.sum((scores==truth)&(ids<row['target']),1);out[:,q]=np.where(r<=10,r,0)
        if (q+1)%256==0:print(f'Policy evidence {q+1}/{len(rows)}',flush=True)
    return out
def reciprocal(r):return np.where(r>0,1/np.maximum(r,1),0)
def llm_call(session,code,prompt):
    path=RUN/f'qwen_requests/state_{code:02d}.json'
    if path.exists():
        saved=json.loads(path.read_text(encoding='utf-8'));assert saved['prompt']==prompt;return saved
    text='<|im_start|>system\nChoose the best recommendation policy using the supplied evidence. Output one letter A-D only.<|im_end|>\n<|im_start|>user\n'+prompt+'<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
    body={'prompt':text,'n_predict':1,'temperature':0,'seed':42,'cache_prompt':True,'return_tokens':True,'grammar':'root ::= [A-D]','samplers':['temperature']}
    t=time.perf_counter();response=session.post('http://127.0.0.1:11437/completion',json=body,timeout=180);response.raise_for_status();raw=response.json()
    label=raw.get('content','').strip();valid=label in 'ABCD' and len(label)==1 and len(raw.get('tokens',[]))==1
    result={'state':code,'prompt':prompt,'request':body,'response':raw,'label':label,'valid':valid,'wall_seconds':time.perf_counter()-t}
    save(path,result);print(f'Actual local Qwen state={code} label={label} valid={valid}',flush=True);return result
def main():
    preparation_start=time.perf_counter()
    d=data();base=Features(d,base=True);f=Features(d);entries=candidate_policies();weights=np.asarray([e['weights'] for e in entries],dtype=np.float32)
    contexts=np.asarray([base.context(r['history']) for r in d['calibration']]);thresholds=np.median(contexts,0).tolist()
    codes=np.asarray([state(c,thresholds) for c in contexts]);counts=np.bincount(codes,minlength=16)
    train=evidence(base,d['calibration'],weights);rr=reciprocal(train);global_mrr=rr.mean(1);nnz=np.count_nonzero(weights,axis=1)
    best=int(np.lexsort((nnz,-global_mrr))[0]);menus=[];groups=[]
    for code in range(16):
        mask=codes==code;n=int(mask.sum());group=rr[:,mask].mean(1) if n else global_mrr;shrunk=(n*group+80*global_mrr)/(n+80)
        menu=np.lexsort((nnz,-shrunk))[:4].tolist();menus.append(menu)
        examples=[]
        for q in np.flatnonzero(mask)[:2]:
            items=[]
            for x in d['calibration'][int(q)]['history'][-2:]:
                items.append({'title':d['titles'][x][:80],'brand':d['brand'][x][:30],
                    'color':d['color'][x][:25],'material':d['material'][x][:25],
                    'price_EUR':None if f.a['missing'][x] else round(float(np.expm1(f.a['log_price'][x])),2)})
            examples.append(items)
        groups.append({'state':code,'support':n,'history_examples':examples,
            'options':[{'candidate':i,'name':entries[i]['name'],'corrected_mrr':float(shrunk[i]),'global_mrr':float(global_mrr[i]),'active_features':int(nnz[i])} for i in menu]})
    save(RUN/'results/training_policy_evidence.json',{'features':FEATURES,'global_best':best,'groups':groups,'thresholds':thresholds,'candidate_count':len(entries)})
    np.savez_compressed(RUN/'results/policy_calibration.npz',ranks=train)
    session=requests.Session();session.trust_env=False
    for attempt in range(90):
        try:
            if session.get('http://127.0.0.1:11437/health',timeout=2).ok:break
        except requests.RequestException:pass
        if attempt==89:raise RuntimeError('Local Qwen service did not become healthy')
        time.sleep(1)
    llm_call(session,99,'Warmup. A is the correct option. Select A.')
    selected=[];calls=[];guards=[]
    for code,g in enumerate(groups):
        meaning=('history '+('longer' if code&1 else 'shorter')+', last product '+('repeated' if code&2 else 'not repeated')+
                 ', brand '+('concentrated' if code&4 else 'diverse')+', price '+('consistent' if code&8 else 'variable'))
        prompt='Features: '+', '.join(FEATURES)+'.\n'+f'State {code}: {meaning}; calibration support {g["support"]}. CorrectedMRR already shrinks sparse groups toward global evidence. Prioritize highest measured correctedMRR; break close ties by fewer features.\n'
        prompt+='History product titles: '+json.dumps(g['history_examples'],ensure_ascii=False)+'\n'
        prompt+='\n'.join(f'{chr(65+j)} {o["name"]}: correctedMRR={o["corrected_mrr"]:.5f}, globalMRR={o["global_mrr"]:.5f}, feature_count={o["active_features"]}' for j,o in enumerate(g['options']))
        call=llm_call(session,code,prompt);calls.append(call);position=ord(call['label'])-65 if call['valid'] else 0
        proposed=g['options'][position];top=g['options'][0]
        choice=proposed['candidate']
        if proposed['corrected_mrr']+max(.0002,.03*top['corrected_mrr'])<top['corrected_mrr']:choice=top['candidate']
        if g['support']<20:choice=best
        selected.append(choice);guards.append({'state':code,'proposed':proposed['candidate'],'guarded':choice,'changed':choice!=proposed['candidate']})
    valid=evidence(f,d['policy_validation'],weights);vrr=reciprocal(valid)
    vcodes=np.asarray([state(f.context(r['history']),thresholds) for r in d['policy_validation']]);table=[]
    for code in range(16):
        mask=vcodes==code;n=int(mask.sum());options=list(dict.fromkeys([selected[code],menus[code][0],best]))
        def value(i):return float((n*(vrr[i,mask].mean() if n else 0)+80*vrr[i].mean())/(n+80))
        table.append(max(options,key=lambda i:(value(i),-nnz[i])) if n else best)
    mixture_choices=[];mixtures=[]
    for alpha in [0,.5,1]:
        mixed=alpha*weights[table]+(1-alpha)*weights[best];result=[]
        for row,code in zip(d['policy_validation'],vcodes):
            score=f.policy_score(row['history'],mixed[code]);result.extend(ranks(score[None,:],[row['target']],[row['history']]).tolist())
        mixture_choices.append({'alpha':alpha,**metrics(result)});mixtures.append(mixed)
    chosen=max(range(3),key=lambda i:mixture_choices[i]['mrr@10'])
    beta_results={}
    for beta in [0,1,3,7]:
        rr_beta=[]
        for row in d['policy_validation']:
            rr_beta.extend(ranks(f.grey_score(row['history'],beta)[None,:],[row['target']],[row['history']]).tolist())
        beta_results[str(beta)]=metrics(rr_beta)
    beta=max(beta_results,key=lambda b:beta_results[b]['mrr@10'])
    result={'candidates':entries,'global_policy':best,'global_weights':weights[best].tolist(),'thresholds':thresholds,
        'qwen_proposals':guards,'guarded_table':table,'weights_by_state':mixtures[chosen].tolist(),
        'alpha_choices_validation':mixture_choices,'selected_alpha':mixture_choices[chosen]['alpha'],
        'grey_beta_validation':beta_results,'selected_beta':int(beta),
        'actual_policy_llm_calls':len(calls),'warmup_calls':1,'valid_outputs':sum(c['valid'] for c in calls),
        'policy_llm_wall_seconds':sum(c['wall_seconds'] for c in calls),'online_llm_calls_per_request':0,
        'full_policy_preparation_seconds':time.perf_counter()-preparation_start,
        'training_evidence_only':True,'selection_validation_only':True,'test_targets_used_for_selection':False,
        'data_sha256':sha(RUN/'data.json'),'attributes_sha256':sha(RUN/'attributes.npz')}
    save(RUN/'results/frozen_policy.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['candidates','weights_by_state','grey_beta_validation']},ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=4):main()
