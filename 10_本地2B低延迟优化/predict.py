"""Usable optimized recommender: cached local-LLM policy or actual online 2B."""
import argparse,json,time
import numpy as np
import refine
opt=refine.opt

class OptimizedRecommender:
 def __init__(self,mode='cache'):
  if mode not in ['cache','sync']:raise ValueError('mode must be cache or sync')
  self.mode=mode;self.grey=opt.load_grey();self.context=opt.Context(self.grey)
  self.tokens=json.loads((opt.ROOT/'03_代码与实验/outputs/grey_improved/tokens.json').read_text(encoding='utf-8'))['items']
  self.token2id={str(x):i for i,x in enumerate(self.tokens)}
  self.cache=json.loads((opt.HERE/'results/offline_llm_policy_cache.json').read_text(encoding='utf-8'))
  if self.cache['teacher_sha256']!=opt.sha(opt.ROOT/'06_教师学生推荐性能测试/teacher/prototypes.json'):
   raise ValueError('Teacher source changed; rebuild the policy cache before serving')
 def recommend(self,raw):
  t=time.perf_counter()
  if not raw or len(raw)>20:raise ValueError('Provide 1..20 known product IDs')
  if any(str(x) not in self.token2id or self.token2id[str(x)]==0 for x in raw):raise ValueError('Unknown or padding product ID')
  h=np.asarray([self.token2id[str(x)] for x in raw],dtype=np.int64);c=self.context.get(h)
  key=f'{0 if len(h)<=3 else (1 if len(h)<=7 else 2)},{int(c[2])},{int(c[4]>=0.23069553)}'
  if self.mode=='cache':r=self.cache['entries'][key]
  else:
   r=opt.infer(self.cache['selected'],h,c,'interactive_prediction')
   if not r['valid']:raise RuntimeError('Local model did not return a valid policy label')
  coeff=np.asarray(r['coefficients'],dtype=np.float32)
  z=self.grey.normalized_features(h);scores=opt.numpy_score(z,coeff);scores[0]=-np.inf;ids=opt.stable_topk(scores)
  recommendations=[{'rank':j+1,'item':str(self.tokens[i]),'score':float(scores[i])} for j,i in enumerate(ids)]
  return {'mode':self.mode,'online_2b_calls':int(self.mode=='sync'),'policy_state':key,
      'policy_label':r['response']['content'],'coefficients':coeff.tolist(),'recommendations':recommendations,
      'warm_recommendation_ms':(time.perf_counter()-t)*1000}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['cache','sync'],default='cache');p.add_argument('--items',nargs='+',required=True);a=p.parse_args()
 opt.torch.set_num_threads(4)
 recommender=OptimizedRecommender(a.mode)
 print(json.dumps(recommender.recommend(a.items),ensure_ascii=False,indent=2))
