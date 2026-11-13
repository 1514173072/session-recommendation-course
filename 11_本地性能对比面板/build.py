"""Read existing result snapshots; embed them in a standalone local HTML."""
from pathlib import Path
import csv,hashlib,json,math
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
REG=[];RECORDS=[];SOURCES={}

def read(path):
 p=ROOT/path;SOURCES[path]=hashlib.sha256(p.read_bytes()).hexdigest()
 return json.loads(p.read_text(encoding='utf-8'))

def model(id,name,group,color,formal,mechanism,execution):
 REG.append(dict(id=id,name=name,group=group,color=color,formal=formal,mechanism=mechanism,execution=execution))

def record(id,scope,row,mean,p95,source,timing,device,valid=None,p95_note=''):
 RECORDS.append({'id':id,'scope':scope,'n':{'30':30,'1000':1000,'20000':20000}[scope],
  'recall':row['recall@10'],'mrr':row['mrr@10'],'ndcg':row['ndcg@10'],
  'mean_ms':mean,'p95_ms':p95,'source':source,'timing':timing,'device':device,
  'valid':valid,'p95_note':p95_note})

def main():
 colors=['#536b80','#a18c67','#397d9a','#526bc1','#8666aa','#6a8a59','#749e88']
 ids={'gra_original':'gra','gra_beta7':'gra_improved','SRGNN':'srgnn','GCSAN':'gcsan','GCEGNN':'gcegnn'}
 labels={'gra_original':'原始灰关联','gra_beta7':'改进灰关联（β=7）','SRGNN':'SR-GNN','GCSAN':'GC-SAN','GCEGNN':'GCE-GNN'}
 mechanisms={'gra_original':'四项训练侧行为证据归一化后，计算灰色关联系数并等权汇总。',
 'gra_beta7':'保留原始四项特征和等权汇总，仅将差异d替换为d/[1+7(1-d)]。',
 'teacher_rules':'按会话状态直接取既有训练侧教师原型的权重和交互系数；在线不调用LLM。',
 'SRGNN':'将当前会话构造成图，用图神经网络学习商品转移并对全部商品评分。',
 'GCSAN':'结合会话图表示与自注意力进行下一商品推荐。',
 'GCEGNN':'结合会话与全局上下文图信息进行下一商品推荐。'}
 for k,c in zip(ids,colors):model(ids[k],labels[k],'基础方法' if k in ['gra_original','gra_beta7','teacher_rules'] else '深度学习',c,labels[k],mechanisms[k],'本地CPU；每次重新计算推荐')
 model('qwen_plain','qwen3.5-2B · 无强模型关联分析','本地LLM','#b67743','在线 LLM 动态策略生成 · Online LLM Policy Generation',
  '每次将当前会话及行为证据送入本地Qwen3.5-2B Q4，生成4个权重和6个交互系数，再对全部商品评分。权重未微调。','每次实际运行本地2B')
 model('qwen_teacher','qwen3.5-2B · 有强模型关联分析','本地LLM','#c66b45','在线 LLM 动态策略生成 · Online LLM Policy Generation',
  '沿用原始在线流程，在提示词中加入训练侧教师原型；每次生成完整系数JSON。没有更新2B权重。','每次实际运行本地2B')
 model('qwen_offline','离线qwen3.5-2B','本地LLM','#168a80','离线 LLM 策略预计算 · Offline LLM Policy Precomputation',
  '本地2B提前为12类会话状态选择候选策略。在线查状态→策略表，重新计算当前会话下的商品得分和Top10。没有知识蒸馏或微调。','在线查表，不执行2B；不是缓存商品名单')
 # User now explicitly includes the single-d-change GRA improvement.
 path='06_教师学生推荐性能测试/results/comparison.csv';p=ROOT/path;SOURCES[path]=hashlib.sha256(p.read_bytes()).hexdigest()
 with p.open(encoding='utf-8-sig',newline='') as f:full=list(csv.DictReader(f))
 for r in full:
  if r['model'] not in ids:continue
  rr={k:float(r[k]) for k in ['recall@10','mrr@10','ndcg@10']}
  record(ids[r['model']],'20000',rr,float(r['mean_ms']),float(r['p95_ms']),path,
   '推荐质量：完整20,000条；耗时：历史300条请求×3轮，共900次预热CPU调用。含原始ID→特征/图→全商品评分→Top10；不含加载。','本地CPU · 4线程')
 # Fresh same-sample baselines from the optimization round.
 for k in ['gra_original','SRGNN','GCSAN','GCEGNN']:
  path=f'10_本地2B低延迟优化/results/baseline_{k}.json'
  for r in read(path):record(ids[k],'1000' if r['split']=='test1000' else '30',r,r['mean_end_to_end_ms'],r['p95_end_to_end_ms'],path,
    '本次对应会话的预热CPU实测：原始ID→特征/图构造→全商品评分→Top10原始ID。加载排除。','本地CPU · 4线程')
 path='06_教师学生推荐性能测试/predictions/gra_beta7.jsonl';p=ROOT/path;SOURCES[path]=hashlib.sha256(p.read_bytes()).hexdigest()
 improved=[json.loads(line) for line in p.open(encoding='utf-8')]
 assert len(improved)==20000 and all(r['test_index']==i for i,r in enumerate(improved))
 freeze=read('10_本地2B低延迟优化/frozen_before_selection.json')
 old_cases=read('07_云端模型八组测试/cases.json')
 for c in old_cases:
  r=improved[c['test_index']];assert r['session_id']==c['session_id'] and r['history']==c['history'] and r['target']==c['target']
 timing=next(r for r in full if r['model']=='gra_beta7')
 for scope,indices in [('30',freeze['original_test_indices']),('1000',freeze['test_indices'])]:
  ranks=[]
  for i in indices:
   r=improved[i];items=[x['item'] for x in r['recommendations']];actual=items.index(r['target'])+1 if r['target'] in items else 0
   assert actual==r['target_rank'];ranks.append(actual)
  n=len(ranks);quality={'recall@10':sum(x>0 for x in ranks)/n,'mrr@10':sum(1/x if x else 0 for x in ranks)/n,'ndcg@10':sum(1/math.log2(x+1) if x else 0 for x in ranks)/n}
  record('gra_improved',scope,quality,float(timing['mean_ms']),float(timing['p95_ms']),path,
   '质量由同一批会话的已保存预测重新汇总；耗时沿用历史900次预热CPU调用，未将其表述为本子集重新计时。','本地CPU · 4线程')
 path='09_本地2B推荐测试/results/summary.json';native=read(path)
 for r in native:
  record('qwen_teacher' if r['condition']=='teacher' else 'qwen_plain','30',r,r['mean_recommendation_seconds']*1000,None,path,
   '常驻模型的本地API调用＋全商品评分/Top10；商品特征预先准备。不含约71秒的首次加载与预热。','本地RTX 3050 · Q4',r['valid_outputs'],
   '源汇总的P95只覆盖API调用，未覆盖评分；为保持同列口径，此处显示未记录。')
 path='10_本地2B低延迟优化/results/cache_comparison.json'
 for r in read(path):
  scope={'test1000':'1000','original30':'30','test20000':'20000'}[r['split']]
  record('qwen_offline',scope,r,r['mean_end_to_end_ms'],r['p95_end_to_end_ms'],path,
   '对应全部会话的CPU实测：原始ID→状态查表→当前会话特征→全商品评分→Top10。不含12次离线LLM调用（共约0.875秒）；在线不运行2B。','本地CPU · 4线程')
 assert len({(r['id'],r['scope']) for r in RECORDS})==len(RECORDS)
 assert not any(m['id'].startswith(('cloud_','mlp')) for m in REG)
 for r in RECORDS:
  assert r['id'] in {m['id'] for m in REG}
  assert all(math.isfinite(r[k]) and 0<=r[k]<=1 for k in ['recall','mrr','ndcg'])
  assert r['mean_ms']>0 and (r['p95_ms'] is None or r['p95_ms']>0)
 assert not any(r['scope']!='30' and r['id'] in ['qwen_plain','qwen_teacher'] for r in RECORDS)
 payload={'models':REG,'records':RECORDS,'snapshot':'2026-11-12','scopes':[
  {'id':'30','name':'原30条会话','n':30,'description':'同一组真实会话，包含原版qwen3.5-2B。小样本试测。'},
  {'id':'1000','name':'固定1,000条','n':1000,'description':'低延迟优化实验的固定测试子集；不包含原30条会话。'},
  {'id':'20000','name':'完整20,000条','n':20000,'description':'完整测试集。部分基线耗时来自既有900次预热测量。'}]}
 data=json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('</',r'<\/')
 template=(HERE/'template.html').read_text(encoding='utf-8')
 assert template.count('__EMBEDDED_DATA__')==1
 html=template.replace('__EMBEDDED_DATA__',data)
 output=ROOT/'08扩展对比/推荐性能与耗时对比.html'
 output.write_text(html,encoding='utf-8');(HERE/'dist/index.html').write_text(html,encoding='utf-8')
 (HERE/'data_manifest.json').write_text(json.dumps({'sources':SOURCES,'models':len(REG),'records':len(RECORDS),'counts':{s:sum(r['scope']==s for r in RECORDS) for s in ['30','1000','20000']},'output':str(output),'sha256':hashlib.sha256(output.read_bytes()).hexdigest()},ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps({'output':str(output),'models':len(REG),'records':len(RECORDS),'bytes':output.stat().st_size},ensure_ascii=False))

if __name__=='__main__':main()
