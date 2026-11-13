from pathlib import Path
import sys,json,hashlib
import numpy as np
import yaml

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'06_教师学生推荐性能测试'))
from common import Recommender,Context,load_grey,PAIRS

MODELS=['mimo-v2.6-flash','qwen3.8-flash','glm-5.3-flash','deepseek-v4.1-flash']

def main():
    for name in ['patches','prompts','requests','results']:(HERE/name).mkdir(exist_ok=True)
    with np.load(ROOT/'06_教师学生推荐性能测试/cache/splits.npz',allow_pickle=False) as z:
        split={k:z[k] for k in z.files}
    tokens=json.loads((ROOT/'03_代码与实验/outputs/grey_improved/tokens.json').read_text(encoding='utf-8'))
    prototypes=json.loads((ROOT/'06_教师学生推荐性能测试/teacher/prototypes.json').read_text(encoding='utf-8'))
    teacher={x['group']:x for x in prototypes['prototypes']}
    grey=load_grey(); ctx=Context(grey)
    indices=np.random.default_rng(20261112).choice(len(split['test_users']),30,replace=False)
    cases=[]
    for i in indices:
        h=split['test_histories'][i,:split['test_lengths'][i]]
        c=ctx.get(h)
        g=ctx.group(h,c)
        cases.append({'test_index':int(i),'session_id':str(tokens['users'][split['test_users'][i]]),
            'history':[str(tokens['items'][x]) for x in h],'internal_history':h.tolist(),
            'target':str(tokens['items'][split['test_targets'][i]]),
            'internal_target':int(split['test_targets'][i]),'context':c.tolist(),'group':g})
    (HERE/'cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2),encoding='utf-8')
    base='''You are implementing a small session-recommendation scoring component. This is a constrained code-integration test: return only one compact JSON object; do not use tools, read files, browse, or write code.\nThe existing local implementation scores ALL candidate products by\nscore=sum(weights[k]*z[k],k=0..3)+sum(interactions[j]*z[a]*z[b],j=0..5).\nThe four normalized candidate features z are, in order:\n0 transition: training-only probability of next product after the last click;\n1 cooccurrence: recency-weighted cosine-normalized cooccurrence with the session history;\n2 history_support: recency-weighted support of a product within this session;\n3 popularity: log training clicks. All normalized to [0,1] over the full catalogue.\nInteractions are ordered [0*1,0*2,0*3,1*2,1*3,2*3].\nSelect session-dependent scoring coefficients. Output exactly {"weights":[a,b,c,d],"interactions":[e,f,g,h,i,j]}. Weights must be nonnegative and sum to1. Each interaction must be between -0.5 and0.5. Do not output explanations or Markdown. Product IDs are anonymous; use only supplied behavioral evidence. The true next click is not supplied.\n'''
    for condition in ['plain','teacher']:
        for j,case in enumerate(cases):
            c=case['context']
            evidence={'history':case['history'],'length':len(case['history']),
                'unique_ratio':round(c[1],5),'last_product_repeated':bool(c[2]),
                'maximum_product_frequency_ratio':round(c[3],5),
                'last_product_training_popularity_normalized':round(c[4],5),
                'last_product_transition_entropy':round(c[5],5)}
            prompt=base+'Session evidence:\n'+json.dumps(evidence,separators=(',',':'))+'\n'
            if condition=='teacher':
                p=teacher[case['group']]
                guidance={'weights':p['weights'],'interactions':p['interactions'],'rationale':p['rationale']}
                prompt+='Teacher guidance derived exclusively from training data. Use this as a prior and adapt only if the session evidence warrants it:\n'+json.dumps(guidance,ensure_ascii=False,separators=(',',':'))+'\n'
            (HERE/f'prompts/{condition}_{j:02d}.txt').write_text(prompt,encoding='utf-8')
    disabled=['agent-instructions','session-log-deepseek','tool-plugin-manager','tool-working-directory',
        'tool-bash','tool-pwsh','tool-jobs','tool-fs','tool-fs-search','tool-skill',
        'tool-subagent-control','tool-subagent-list-agents','tool-subagent','tool-subagent-fork',
        'tool-workflow','tool-todo','tool-goal','tool-web']
    for model in MODELS:
        model_options={'maxTokens':2048,'reasoningEfforts':{'off':None,'low':'low','high':'high'}}
        provider_options={'apiKeyEnv':None,'compat':{'thinkingFormat':'deepseek'},
                          'modelOverrides':{model:model_options}}
        patch=[{'id':'agent-default-model','name':'@deepseek-ai/dsh-agent-default-model',
            'config':{'provider':'opencode-go','model':model,
                      'reasoningEffort':'low' if model=='glm-5.3-flash' else 'off'}},
            {'id':'llm-pi-ai','name':'@deepseek-ai/dsh-llm-pi-ai',
             'config':{'providers':{'opencode-go':provider_options}}}]
        # Carry the existing credential REFERENCE only, never read credential contents.
        auth=yaml.load(Path(r'D:\dsh-model-check\headless-route-auth.patch.yml').read_text(encoding='utf-8'),Loader=yaml.SafeLoader)
        original=next(r for r in auth if r['id']=='llm-pi-ai')['config']['providers']['opencode-go']
        patch[1]['config']['providers']['opencode-go']['apiKeyEnv']=original['apiKeyEnv']
        patch.extend({'id':k,'disabled':True} for k in disabled)
        (HERE/f'patches/{model}.yml').write_text(yaml.safe_dump(patch,allow_unicode=True,sort_keys=False),encoding='utf-8')
    manifest={'models':MODELS,'conditions':['plain','teacher'],'groups':8,'queries_per_group':30,
        'total_planned_requests':240,'sample_seed':20261112,'sample_indices':indices.tolist(),
        'candidate_items':grey.item_num-1,'teacher_source_sha256':hashlib.sha256((ROOT/'06_教师学生推荐性能测试/teacher/prototypes.json').read_bytes()).hexdigest(),
        'cases_sha256':hashlib.sha256((HERE/'cases.json').read_bytes()).hexdigest(),
        'prompt_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/'prompts').glob('*.txt')},
        'cloud_teacher_semantics':'prompt-guided inference, not cloud weight fine-tuning',
        'max_output_tokens':2048,
        'requested_reasoning_effort':{m:('low' if m=='glm-5.3-flash' else 'off') for m in MODELS},
        'reasoning_note':'GLM endpoint rejected off as thinking-only. Actual thinking events/usage are recorded; off is a request and is not assumed honored by each gateway.',
        'scope':'30-case fixed random pilot; full-catalogue deterministic scoring from actual generated coefficients'}
    (HERE/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Prepared eight groups, 30 matching cases each; prompts contain no test targets.')

if __name__=='__main__':main()
