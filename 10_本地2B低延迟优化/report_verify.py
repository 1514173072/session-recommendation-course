"""Independent recomputation, then concise CSV/Markdown delivery."""
from pathlib import Path
import csv,json,math,hashlib,time
import numpy as np
import refine
opt=refine.opt
HERE=opt.HERE;ROOT=opt.ROOT

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def metrics(ranks):
 n=len(ranks)
 return {'recall@10':sum(x>0 for x in ranks)/n,
         'mrr@10':sum(1/x if x else 0 for x in ranks)/n,
         'ndcg@10':sum(1/math.log2(x+1) if x else 0 for x in ranks)/n}
def check_metrics(row,ranks):
 for key,value in metrics(ranks).items():assert math.isclose(row[key],value,abs_tol=1e-12),(key,row[key],value)
def independent_top(z,c):
 scores=(z*c[:4,None]).sum(axis=0)
 for k,(a,b) in enumerate([(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]):scores+=c[4+k]*z[a]*z[b]
 scores[0]=-np.inf
 return np.lexsort((np.arange(len(scores)),-scores))[:10]
def rank(ids,target):
 m=np.flatnonzero(ids==target);return int(m[0]+1) if len(m) else 0

def main():
 opt.torch.set_num_threads(4)
 freeze=read(HERE/'frozen_before_selection.json');rfreeze=read(HERE/'refine_before_selection.json')
 assert freeze['script_sha256']==opt.sha(HERE/'optimize.py')
 assert rfreeze['script_sha256']==opt.sha(HERE/'refine.py')
 assert freeze['teacher_sha256']==opt.sha(ROOT/'06_教师学生推荐性能测试/teacher/prototypes.json')
 assert freeze['splits_sha256']==opt.sha(ROOT/'06_教师学生推荐性能测试/cache/splits.npz')
 selection=read(HERE/'results/refined_selection_before_test.json');summary=read(HERE/'results/refined_summary.json')
 assert selection['config_sha256']==opt.sha(HERE/'refine_before_selection.json')
 assert max(selection['validation'],key=lambda r:(r['mrr@10'],-r['mean_end_to_end_ms']))['variant']==selection['selected']==summary['selected']
 with np.load(ROOT/'06_教师学生推荐性能测试/cache/splits.npz') as archive:data={k:archive[k] for k in archive.files}
 grey=opt.load_grey();ctx=opt.Context(grey)
 source_cases=read(ROOT/'07_云端模型八组测试/cases.json')
 assert [c['test_index'] for c in source_cases]==freeze['original_test_indices']
 assert len(set(freeze['test_indices']))==1000 and not(set(freeze['test_indices'])&set(freeze['original_test_indices']))
 audited=0
 for filename,indices,key in [('refined_test1000.jsonl',freeze['test_indices'],'test1000'),('refined_original30.jsonl',freeze['original_test_indices'],'original30')]:
  rr=[json.loads(s) for s in (HERE/'requests'/filename).read_text(encoding='utf-8').splitlines()]
  assert len(rr)==len(indices)
  for r,i in zip(rr,indices):
   h=opt.history(data,'test',i);c=ctx.get(h)
   assert r['history_internal']==h.tolist() and r['index']==i
   assert r['prompt']==opt.prompt(summary['selected'],h,c)
   assert r['target_internal']==int(data['test_targets'][i])
   o=r['response'];assert r['valid'] and len(o['tokens'])==1 and r['request']['n_predict']==1 and not o['truncated']
   label=o['content'].strip();assert label in opt.LABELS
   coeff=opt.codebook(summary['selected'])[opt.LABELS.index(label)]
   np.testing.assert_allclose(coeff,r['coefficients'],rtol=0,atol=0)
   ids=independent_top(grey.normalized_features(h),coeff)
   assert ids.tolist()==r['top10_internal'] and rank(ids,int(data['test_targets'][i]))==r['target_rank']
   audited+=1
  check_metrics(summary[key],[r['target_rank'] for r in rr])
  assert math.isclose(summary[key]['mean_end_to_end_ms'],np.mean([r['end_to_end_ms'] for r in rr]))
 cache=read(HERE/'results/offline_llm_policy_cache.json');cached=read(HERE/'results/cache_comparison.json')
 assert len(cache['entries'])==12 and cache['teacher_sha256']==freeze['teacher_sha256']
 for k,r in cache['entries'].items():
  l,re,p=map(int,k.split(','));h=np.arange(1,[2,5,8][l]+1);c=np.asarray([0,1,re,0,p*.5,0],dtype=np.float32)
  assert r['prompt']==opt.prompt(cache['selected'],h,c)
  assert r['valid'] and len(r['response']['tokens'])==1
  assert r['coefficients']==opt.codebook(cache['selected'])[opt.LABELS.index(r['response']['content'])].tolist()
 all_ranks=[]
 for i in range(len(data['test_targets'])):
  h=opt.history(data,'test',i);c=ctx.get(h)
  key=f'{0 if len(h)<=3 else (1 if len(h)<=7 else 2)},{int(c[2])},{int(c[4]>=0.23069553)}'
  coeff=np.asarray(cache['entries'][key]['coefficients'],dtype=np.float32)
  ids=independent_top(grey.normalized_features(h),coeff)
  all_ranks.append(rank(ids,int(data['test_targets'][i])))
 for r in cached:
  indices=range(20000) if r['split']=='test20000' else freeze['test_indices' if r['split']=='test1000' else 'original_test_indices']
  ranks=[all_ranks[i] for i in indices]
  assert ranks==r['ranks'];check_metrics(r,ranks)
  assert r['online_model_calls']==0 and r['cache_hit_rate']==1
  assert math.isclose(r['mean_end_to_end_ms'],np.mean(r['per_case_ms']))
 baselines={}
 for name in ['gra_original','teacher_rules','SRGNN','GCSAN','GCEGNN']:
  rows=read(HERE/f'results/baseline_{name}.json')
  for r in rows:check_metrics(r,r['ranks'])
  baselines[name]=rows
 # Exactly matched policy choices demonstrate that the cache reproduces this synchronous run.
 for filename,split in [('refined_test1000.jsonl','test1000'),('refined_original30.jsonl','original30')]:
  rr=[json.loads(s) for s in (HERE/'requests'/filename).read_text(encoding='utf-8').splitlines()]
  cr=next(r for r in cached if r['split']==split)
  assert [r['target_rank'] for r in rr]==cr['ranks']
 export(summary,cached,baselines,cache)
 audit={'all_passed':True,'synchronous_response_rank_audits':audited,'cached_full_test_rank_audits':20000,
        'offline_actual_local_2b_calls':12,'cached_online_model_calls':0,
        'selection_used_validation_only':True,'cache_and_synchronous_ranks_match_on_1030_cases':True,
        'model_sha256':opt.sha(Path(r'D:\Models\Ollama\blobs\sha256-20cb277f0967ace47b0b5d5658e5e494a88937f7378b74b5a266496400938f4c')),
        'scripts':{p.name:opt.sha(p) for p in [HERE/'optimize.py',HERE/'refine.py',HERE/'cache_compare.py',HERE/'baselines.py',Path(__file__)]}}
 opt.save(HERE/'results/completion_audit.json',audit)
 print(json.dumps(audit,ensure_ascii=False,indent=2))

def export(summary,cached,baselines,cache):
 dest=ROOT/'08扩展对比/本地2B低延迟优化';dest.mkdir(parents=True,exist_ok=True)
 names={'gra_original':'原始灰关联','teacher_rules':'固定教师分组策略','SRGNN':'SR-GNN','GCSAN':'GC-SAN','GCEGNN':'GCE-GNN'}
 rows=[]
 for split,key in [('test1000','test1000'),('original30','original30')]:
  for name,bb in baselines.items():
   r=next(x for x in bb if x['split']==split)
   rows.append({'样本范围':split,'样本数':r['queries'],'方法':names[name],
     'Recall@10':r['recall@10'],'MRR@10':r['mrr@10'],'NDCG@10':r['ndcg@10'],
     '平均完整耗时毫秒':r['mean_end_to_end_ms'],'P95毫秒':r['p95_end_to_end_ms'],'在线每请求执行2B':False,'计时轮次':'本次同样本实测'})
  r=summary[key]
  rows.append({'样本范围':split,'样本数':r['queries'],'方法':'2B单Token同步策略判别',
     'Recall@10':r['recall@10'],'MRR@10':r['mrr@10'],'NDCG@10':r['ndcg@10'],
     '平均完整耗时毫秒':r['mean_end_to_end_ms'],'P95毫秒':r['p95_end_to_end_ms'],'在线每请求执行2B':True,'计时轮次':'本次同样本实测'})
  r=next(x for x in cached if x['split']==split)
  rows.append({'样本范围':split,'样本数':r['queries'],'方法':'本地2B预计算策略表',
     'Recall@10':r['recall@10'],'MRR@10':r['mrr@10'],'NDCG@10':r['ndcg@10'],
     '平均完整耗时毫秒':r['mean_end_to_end_ms'],'P95毫秒':r['p95_end_to_end_ms'],'在线每请求执行2B':False,'计时轮次':'本次同样本实测'})
 with (ROOT/'06_教师学生推荐性能测试/results/comparison.csv').open(encoding='utf-8-sig',newline='') as f:old=list(csv.DictReader(f))
 for r in old:
  if r['model'] in names:
   rows.append({'样本范围':'test20000','样本数':20000,'方法':names[r['model']],
      'Recall@10':float(r['recall@10']),'MRR@10':float(r['mrr@10']),'NDCG@10':float(r['ndcg@10']),
      '平均完整耗时毫秒':float(r['mean_ms']),'P95毫秒':float(r['p95_ms']),'在线每请求执行2B':False,'计时轮次':'已有900次预热CPU计时；质量为完整20000条'})
 r=next(x for x in cached if x['split']=='test20000')
 rows.append({'样本范围':'test20000','样本数':20000,'方法':'本地2B预计算策略表',
    'Recall@10':r['recall@10'],'MRR@10':r['mrr@10'],'NDCG@10':r['ndcg@10'],
    '平均完整耗时毫秒':r['mean_end_to_end_ms'],'P95毫秒':r['p95_end_to_end_ms'],'在线每请求执行2B':False,'计时轮次':'本次全部20000条预热CPU计时'})
 csvpath=dest/'低延迟方案与灰关联及GNN对比.csv'
 with csvpath.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 lines=['# 本地2B低延迟优化','',
 '实现了两条路径：每次真实执行完整2B的同步策略判别，以及本地2B提前选择策略、在线查表。后者达到与灰关联和GNN相近的毫秒级耗时。','',
 '| 路径 | 本地2B的作用 | 在线是否执行2B |','|---|---|---|',
 '| 单Token同步判别 | 每次将会话状态映射为一个策略标签 | 是 |',
 '| 预计算策略表 | 对全部12类状态分别实际调用2B一次，保存选择结果 | 否，查表后重新计算全部商品排名 |','',
 '输入为长度档位（≤3、4—7、≥8）、最后商品是否重复、训练热度是否达到0.23069553，共12种状态。12套候选权重及交互系数来自之前训练侧教师原型。本地2B只选择策略，权重没有微调。评分沿用四个行为特征与六个二阶交互项，不等同于原始灰色关联系数。','',
 f'预计算共12次真实本地2B调用，用时{cache["build_seconds"]:.3f}秒。缓存保存的是“状态→策略”，商品得分和推荐名单仍根据每条实际会话重新计算。','',
 '先以128条验证会话比较提示词，再在同一验证集选择缓存与短输入形式；保留所有候选结果。最终选择examples12。测试使用固定的1000条会话，以及原30条；策略表额外检查全部20000条。该测试集此前已有其他实验，不表述为全新盲测。','',
 '| 同1000条测试 | Recall@10 | MRR@10 | NDCG@10 | 平均完整耗时(ms) | P95(ms) |',
 '|---|---:|---:|---:|---:|---:|']
 for r in rows:
  if r['样本范围']=='test1000':lines.append(f'| {r["方法"]} | {r["Recall@10"]:.2%} | {r["MRR@10"]:.5f} | {r["NDCG@10"]:.5f} | {r["平均完整耗时毫秒"]:.3f} | {r["P95毫秒"]:.3f} |')
 lines+=['','| 完整20000条质量比较 | Recall@10 | MRR@10 | NDCG@10 |','|---|---:|---:|---:|']
 for r in rows:
  if r['样本范围']=='test20000':lines.append(f'| {r["方法"]} | {r["Recall@10"]:.2%} | {r["MRR@10"]:.5f} | {r["NDCG@10"]:.5f} |')
 lines+=['',
 '计时：模型加载、文件读取、离线策略预计算均排除；同步方案包含会话状态提取、本地HTTP、完整2B前向和全商品Top10；缓存和CPU基线包含原始ID映射、特征/图构造、全商品评分、Top10及原始ID输出。CPU为4线程。同步2B使用GPU，其余为CPU。本次1000条基线顺序独立运行；CSV另标明完整20000条基线的旧900次计时口径。','',
 '主要变化：原约56Token的系数JSON改为1Token策略标签；省略匿名商品ID等不参与状态判别的字段；直接调用llama-server；CUDA全层加载、Flash Attention、32Token物理批次和16个上下文检查点用于减少固定前缀重复计算。这里的32Token是单条提示词内部批次，不是将32个请求的吞吐量当作单请求延迟。','',
 '单Token同步2B仍为几十毫秒，没有实现每次完整执行2B只需1—2毫秒。约0.6毫秒来自策略表路径，在线不执行2B。状态只有12类，无法表达更细的连续行为差异；目前的结果支持该策略分配表的推荐表现，尚未隔离预训练语言能力带来的独立收益。','',
 '2B给出的状态分配与人工按同名分组直接取策略不完全相同。固定教师分组策略已作为对照保留；状态→实际标签的完整表与全部真实响应见10_本地2B低延迟优化/results/offline_llm_policy_cache.json。','',
 '实现、逐条输出、验证选择记录与独立重算结果位于10_本地2B低延迟优化。原09实验及课程论文未修改。']
 (dest/'优化说明与结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':main()
