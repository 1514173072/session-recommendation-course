"""Same samples, original model weights, warm raw-ID to full-catalogue Top10."""
import argparse,json,sys,time,gc
import numpy as np
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'06_教师学生推荐性能测试'))
from common import Recommender,ranking_metrics,torch

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--model',required=True);a=parser.parse_args();name=a.model
 torch.set_num_threads(4)
 freeze=json.loads((HERE/'frozen_before_selection.json').read_text(encoding='utf-8'))
 tokens=json.loads((ROOT/'03_代码与实验/outputs/grey_improved/tokens.json').read_text(encoding='utf-8'))
 with np.load(ROOT/'06_教师学生推荐性能测试/cache/splits.npz') as archive:
  data={k:archive[k] for k in archive.files}
 token2id={str(x):i for i,x in enumerate(tokens['items'])}
 def hist(i):return data['test_histories'][i,:int(data['test_lengths'][i])].astype(np.int64)
 if name in ['SRGNN','GCSAN','GCEGNN']:
  sys.path.insert(0,str(ROOT/'app'));from predictor import SessionPredictor
  model=SessionPredictor(name)
  def predict(raw):return [str(r['item']) for r in model.predict(raw)]
  source=ROOT/f'03_代码与实验/outputs/formal/predictions/{name}.jsonl'
 else:
  model=Recommender(name)
  def predict(raw):
   ids,_,_=model.recommend(np.asarray([token2id[x] for x in raw],dtype=np.int64))
   return [str(tokens['items'][i]) for i in ids]
  source=ROOT/f'06_教师学生推荐性能测试/predictions/{name}.jsonl'
 expected={r['session_id']:r for r in map(json.loads,source.open(encoding='utf-8'))}
 result=[]
 for split,indices in [('test1000',freeze['test_indices']),('original30',freeze['original_test_indices'])]:
  queries=[[str(tokens['items'][x]) for x in hist(i)] for i in indices]
  for raw in queries[:20]:predict(raw)
  ranks=[];durations=[]
  for i,raw in zip(indices,queries):
   t=time.perf_counter();ids=predict(raw);durations.append((time.perf_counter()-t)*1000)
   prior=expected[str(tokens['users'][data['test_users'][i]])]
   assert prior['history']==raw
   assert ids==[r['item'] for r in prior['recommendations']],(name,i)
   target=str(tokens['items'][data['test_targets'][i]])
   ranks.append(ids.index(target)+1 if target in ids else 0)
  result.append({'model':name,'split':split,'queries':len(indices),**ranking_metrics(ranks),
      'mean_end_to_end_ms':float(np.mean(durations)),'p95_end_to_end_ms':float(np.percentile(durations,95)),
      'ranks':ranks,'per_case_ms':durations,'timing_scope':'warm raw IDs -> feature/graph -> full catalogue -> Top10 raw IDs; CPU4 threads'})
 out=HERE/f'results/baseline_{name}.json';out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps([{k:v for k,v in r.items() if k not in ['ranks','per_case_ms']} for r in result],ensure_ascii=False),flush=True)

if __name__=='__main__':main()
