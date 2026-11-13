"""Independent output audit and combined report for the four-axis experiment."""
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/grey_four_axes'
KEYS = ['Base','A','B','C','D','SRGNN','GCSAN','GCEGNN']
GRID = {'A':[.1,.25,.5,.75,1.], 'B':[0,1,2,7,15], 'C':[-2,-1,0,1,2], 'D':[0,.25,.5,.75,1.]}
BASE = {'A':.5,'B':0,'C':0,'D':0}


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for b in iter(lambda:stream.read(4*1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def quality(rr):
    n=len(rr)
    return {'recall@10':sum(r>0 for r in rr)/n,
            'mrr@10':math.fsum(1/r for r in rr if r)/n,
            'ndcg@10':math.fsum(1/math.log2(r+1) for r in rr if r)/n}


def audit(dataset):
    out = OUT/dataset
    manifest, frozen = read(out/'manifest.json'), read(out/'frozen_selection.json')
    assert manifest['grids']==GRID and frozen['grids']==GRID
    assert digest(out/'manifest.json')==frozen['manifest_sha256']
    for p,h in manifest['sha256'].items():
        assert digest(Path(p))==h, ('changed frozen input',p)
    assert not frozen['test_used_in_this_round_selection'] and frozen['prior_test_exposure']
    trials=list(csv.DictReader((out/'validation_search.csv').open(encoding='utf-8-sig')))
    with np.load(out/'validation_progress.npz') as d:
        ranks=d['ranks'].copy()
        assert int(d['next_index'])==manifest['validation_queries']
    assert ranks.shape==(manifest['validation_queries'],17) and (ranks>=0).all() and (ranks<=10).all()
    assert len(trials)==17
    for j,t in enumerate(trials):
        for k,v in quality(ranks[:,j].tolist()).items():
            assert abs(float(t[k])-v)<1e-13
    for g, values in GRID.items():
        entries=[]
        for order,v in enumerate(values):
            key='Base' if v==BASE[g] else f'{g}_{v:g}'
            t=next(t for t in trials if t['key']==key)
            entries.append((v,order,t))
        winner=max(entries,key=lambda e:(float(e[2]['mrr@10']),float(e[2]['recall@10']),
                   float(e[2]['ndcg@10']),e[0]==BASE[g],-e[1]))
        assert frozen['selected'][g]['value']==winner[0]
    comparison=list(csv.DictReader((out/'comparison.csv').open(encoding='utf-8-sig')))
    assert [r['method_key'] for r in comparison]==KEYS
    records={}
    for name in KEYS:
        rs=[json.loads(line) for line in (out/'predictions'/f'{name}.jsonl').open(encoding='utf-8')]
        assert len(rs)==manifest['test_queries']
        for i,r in enumerate(rs):
            assert r['query_index']==i
            ids=r['top10']
            assert len(ids)==len(set(ids))==10 and min(ids)>0 and max(ids)<=manifest['catalogue_items']
            assert r['rank']==(ids.index(r['target'])+1 if r['target'] in ids else 0)
            if manifest['filter_history']:
                assert not set(ids)&set(r['history'])
            sc=r['top10_scores']
            assert all(np.isfinite(sc)) and all(a>=b for a,b in zip(sc,sc[1:]))
            assert all(ids[j]<ids[j+1] for j in range(9) if sc[j]==sc[j+1])
        if name=='Base':
            reference=rs
        else:
            assert all((a['query_id'],a['history'],a['target'])==(b['query_id'],b['history'],b['target'])
                       for a,b in zip(reference,rs))
        m=quality([r['rank'] for r in rs])
        target=next(r for r in comparison if r['method_key']==name)
        for k,v in m.items():
            assert abs(float(target[k])-v)<1e-13
        records[name]=rs
        if dataset=='amazon_m2' and name in ['SRGNN','GCSAN','GCEGNN']:
            assert read(out/(name+'_results.json'))['catalogue_embedding_precomputed'], name
    latency=list(csv.DictReader((out/'latency.csv').open(encoding='utf-8-sig')))
    assert len(latency)==7200
    timing=read(out/'latency_summary.json')
    expected=set(timing['sample_indices'])
    assert len(expected)==300
    for name in KEYS:
        ts=[r for r in latency if r['method']==name]
        assert len(ts)==900
        for repeat in [1,2,3]:
            rows=[r for r in ts if int(r['round'])==repeat]
            assert len(rows)==300 and {int(r['query_index']) for r in rows}==expected
        values=np.array([float(r['ms']) for r in ts])
        assert np.isfinite(values).all() and (values>0).all()
        table=next(r for r in comparison if r['method_key']==name)
        assert abs(values.mean()-float(table['mean_ms']))<1e-12
        assert abs(np.percentile(values,95)-float(table['p95_ms']))<1e-12
        if dataset=='amazon_m2' and name in ['SRGNN','GCSAN','GCEGNN']:
            assert timing['methods'][name]['catalogue_embedding_precomputed'], name
    # Rebuild current features and independently express each selected formula.
    import grey_four_axes as run
    torch=run.torch
    torch.set_num_threads(4)
    d=run.Dataset(dataset)
    original=d.reference(d.reference_gra)
    assert all(a['top10']==b['top10'] and a['rank']==b['rank'] for a,b in zip(reference,original))
    prior_model_checks={}
    for name in ['SRGNN','GCSAN','GCEGNN']:
        old=d.reference(d.prior_predictions(name))
        exact=sum(a['top10']==b['top10'] and a['rank']==b['rank'] for a,b in zip(records[name],old))
        assert exact==len(reference), (dataset,name,'changed checkpoint predictions')
        prior_model_checks[name]=exact
    rng=np.random.default_rng(17419)
    sample=rng.choice(len(reference),16,replace=False)
    formula_checks=0
    for i in sample:
        row=d.rows['test'][i]
        z=d.matrix(row['history'])
        delta=np.abs(1.-z)
        for name in KEYS[:5]:
            val=None if name=='Base' else frozen['selected'][name]['value']
            dd=delta.copy()
            rho=val if name=='A' else .5
            if name=='B' and val:
                dd=dd/(1.+val*z)
            lo,hi=float(dd.min()),float(dd.max())
            c=(lo+rho*hi)/(dd+rho*hi) if hi else np.ones_like(dd)
            if name=='C' and val:
                w=np.array([math.exp(val)]+[1.]*(d.f-1),dtype=np.float64)
                w=(w/w.sum()).astype(np.float32)
                score=np.sum(c*w[:,None],axis=0)
            else:
                score=(np.sum(np.float32(.25)*c,axis=0) if d.f==4 else c.mean(axis=0))
                if name=='D' and val:
                    score=score-np.float32(val/d.f)*np.abs(c[0]-c[1])
            # B denominator uses z mathematically; 1-d can differ from z at float32 precision.
            actual=run.candidate_score(z,name,val,dataset)
            np.testing.assert_allclose(actual,score,rtol=1e-6,atol=1e-7)
            ids,_=run.top10(actual,row['history'],d.filter_history)
            assert ids.tolist()==records[name][int(i)]['top10']
            formula_checks+=1
    assert read(out/'verification.json')['passed']
    return comparison, {'passed':True,'dataset':dataset,'fully_audited_prediction_rows':len(reference)*8,
          'original_gra_top10_compared_directly':len(original),'prior_gnn_top10_compared_directly':prior_model_checks,
          'independent_validation_metric_columns':17,'selected_parameters_recomputed':4,
          'matched_latency_records':len(latency),'real_query_formula_and_prediction_checks':formula_checks,
          'manifest_sha256':digest(out/'manifest.json'),'verification_sha256':digest(out/'verification.json')}


def main():
    source_snapshot=read(OUT/'gnn_source_snapshot.json')
    for path, expected in source_snapshot['sha256'].items():
        assert digest(Path(path))==expected, ('changed imported GNN source',path)
    assert source_snapshot['amazon_training_source_hashes_match']
    all_rows=[]
    checks={}
    for dataset in ['diginetica','amazon_m2']:
        rows,check=audit(dataset)
        all_rows.extend(rows)
        checks[dataset]=check
        print(dataset,json.dumps(check),flush=True)
    with (OUT/'两数据集完整对比.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    costs=[]
    for dataset in ['diginetica','amazon_m2']:
        folder=OUT/dataset
        manifest=read(folder/'manifest.json')
        for name in KEYS:
            table=next(r for r in all_rows if r['dataset']==dataset and r['method_key']==name)
            neural=name in ['SRGNN','GCSAN','GCEGNN']
            row={'dataset':dataset,'method_key':name,'numeric_state_mb':table['numeric_state_mb'],
                 'shared_gra_statistics_seconds':'','statistics_measurement_scope':'',
                 'checkpoint_load_and_catalogue_precompute_seconds':'','historical_training_validation_seconds':'',
                 'historical_training_scope':'','learned_parameters':'','checkpoint_file_bytes':''}
            if neural:
                current=read(folder/(name+'_results.json'))
                prior_path=(ROOT/f'outputs/formal/{name}/seed42/result.json' if dataset=='diginetica'
                    else ROOT.parent/f'15_Amazon_M2算法对比/run/results/{name}_training.json')
                prior=read(prior_path)
                row.update({'checkpoint_load_and_catalogue_precompute_seconds':current['loading_and_catalogue_compile_seconds'],
                    'historical_training_validation_seconds':prior.get('training_and_validation_seconds',prior.get('training_seconds')),
                    'historical_training_scope':('stored previous training+validation wall time; no new training' if dataset=='diginetica'
                         else 'stored previous concurrent-training wall time; not an isolated training efficiency comparison'),
                    'learned_parameters':current['learned_parameters'],'checkpoint_file_bytes':current['checkpoint_file_bytes']})
            else:
                row.update({'shared_gra_statistics_seconds':manifest['statistics_fit_seconds'],
                    'statistics_measurement_scope':('previous fit measurement, shared by five GRA methods' if dataset=='diginetica'
                        else 'current adapter preparation, shared by five GRA methods; includes loading static attributes')})
            costs.append(row)
    with (OUT/'准备与模型成本.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,list(costs[0]))
        writer.writeheader()
        writer.writerows(costs)
    lines=['# 四组灰关联单因素试验：两数据集实测汇总','','状态：完整测试、现有GNN检查点重新推理与统一响应计时已完成。','','## 实验设置','',
      '- 四组分别改变rho、差异压缩beta、转移权重lambda、转移—共现交互eta，均从原版独立出发。每组5档，含原版恢复值，每数据集17个唯一验证配置。',
      '- 各数据集完整验证集按MRR优先选参，Recall、NDCG辅助处理并列，冻结后再执行完整测试；本轮未使用测试分数选参。',
      '- Dig为已有20k会话子集、32,968商品，20,000验证和20,000测试；保留历史商品。Amazon-M2 ES为42,503商品，8,886验证和8,970测试，过滤历史商品，使用8指标与属性增强GNN。',
      '- 三个GNN使用原有权重重新推理，本轮不重训、不加入Qwen/MLP。','','## 完整质量与响应结果','']
    for dataset in ['diginetica','amazon_m2']:
        lines += [f'### {dataset}','','| 方法 | 参数 | Recall@10 | MRR@10 | NDCG@10 | 平均/ms | P95/ms | 数值状态/MB |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        rows=[r for r in all_rows if r['dataset']==dataset]
        for r in rows:
            lines.append(f"| {r['method']} | {r['selected_parameter']} | {float(r['recall@10'])*100:.3f}% | {float(r['mrr@10']):.5f} | {float(r['ndcg@10']):.5f} | {float(r['mean_ms']):.3f} | {float(r['p95_ms']):.3f} | {float(r['numeric_state_mb']):.3f} |")
        baseline=rows[0]
        best_recall=max(rows,key=lambda r:float(r['recall@10']))
        best_mrr=max(rows,key=lambda r:float(r['mrr@10']))
        lines += ['',f"本配置下Recall最高：{best_recall['method']}（{float(best_recall['recall@10'])*100:.3f}%）；MRR最高：{best_mrr['method']}（{float(best_mrr['mrr@10']):.5f}）。",'',
                  '| 变体相对原版 | 新增命中 | 失去命中 | Recall变化/百分点 |','|---|---:|---:|---:|']
        for r in rows[1:5]:
            lines.append(f"| {r['method']} | {r['new_hits_vs_base']} | {r['lost_hits_vs_base']} | {float(r['recall_gain_pp_vs_base']):+.3f} |")
        lines += ['']
    lines += ['## 计算成本与解释范围','',
      '每种方法同机CPU4线程、batch=1，使用固定种子选出的同一批300个请求，预热30次，3轮共900次。算法顺序每轮随机，计时包括指标/图构建、全目录评分、过滤和Top10；模型加载、静态目录表示预计算及写盘不包含在单请求时间内。', '',
      'Amazon GNN在最后一次eval后预计算属性融合目录向量，单请求复用缓存；该目录缓存计入数值状态。早期未正确保留缓存的开发计时已弃用。D组两数据集均选择eta=0，与原版同一评分；两行微小耗时差异按重复测量波动解释。', '',
      '五种GRA均使用相同的各数据集特征路径与完整目录评分，评分部分为O(FM)，另加原有特征构建成本；交互组额外计算一对差值，增加O(M)。GRA统计矩阵与GNN参数/缓存都计入各自数值状态。数值状态MB不是进程运行内存或下载体积。', '',
      '两个数据集的特征、历史过滤和划分不同，分别在各数据集内部比较。旧测试集已有查看记录、GNN为单个既有训练种子，本轮是扩展性探索；一次表现不证明跨划分稳定性。四组若退化或恢复原版参数，完整保留结果。', '',
      '准备与模型成本.csv另列统计构建、当前模型加载及目录预计算、历史训练墙钟时间、参数量与检查点文件大小。五种GRA共享统计，准备时间不能相加；Amazon历史并行训练时间不用于隔离训练速度排名。', '',
      '文献支撑与公式边界见05目录的四组方案及文献核对记录。既有有理式差异压缩与单参数转移赋权不是对应文献原式的复现。', '',
      '各数据集目录保存manifest.json、validation_search.csv、frozen_selection.json、全部8方法逐会话预测、latency.csv、subgroup_results.json和verification.json。independent_audit.json保存另外的全量文件核对、验证选参复算与真实样本公式检查。','']
    (OUT/'两数据集实测汇总.md').write_text('\n'.join(lines),encoding='utf-8')
    (OUT/'independent_audit.json').write_text(json.dumps({'passed':True,'datasets':checks,
        'gnn_imported_source_snapshot_sha256':digest(OUT/'gnn_source_snapshot.json'),
        'gnn_imported_source_hashes_unchanged':True,
        'complete_method_rows':len(all_rows),'summary_csv_sha256':digest(OUT/'两数据集完整对比.csv')},ensure_ascii=False,indent=2),encoding='utf-8')
    print('BOTH DATASETS PASSED independent audit; 16 method rows.',flush=True)


if __name__=='__main__':
    sys.argv=sys.argv[:1]
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=4):
        main()
