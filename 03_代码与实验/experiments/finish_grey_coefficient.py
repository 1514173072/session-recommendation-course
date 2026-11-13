"""Verify actual coefficient-only source edits, benchmark, and write the report."""
import ast
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from scipy.stats import binomtest
from explore_grey_coefficient import ROOT,OUT,DATA,OLD,read,save,sha,literal_model,cache_metadata,exact_ranks,verify_queries

LABELS={'baseline':'原灰色关联系数','distance_power':'差异幂变换',
        'rational_distance':'有理差异压缩','exponential_distance':'指数差异变换',
        'adaptive_global_mean':'全局平均差异自适应ρ','adaptive_column_mean':'逐指标平均差异自适应ρ',
        'adaptive_column_cv':'均值/标准差自适应ρ','adaptive_column_inverse_cv':'标准差/均值自适应ρ',
        'adaptive_positive_deviation':'非零证据平均差异自适应ρ'}


def outside_core(text):
    tree=ast.parse(text)
    return [ast.dump(node,include_attributes=False) for node in tree.body
            if not isinstance(node,ast.FunctionDef) or node.name!='grey_coefficients']


def main():
    torch.set_num_threads(4)
    frozen=read(OUT/'frozen_selection.json')
    result=read(OUT/'test_results.json')
    source=Path(__file__).with_name('grey_session.py')
    assert sha(source)==frozen['original_source_sha256']
    assert sha(OUT/'frozen_selection.json')==result['frozen_selection_sha256']
    original_text=source.read_text(encoding='utf-8')
    with np.load(DATA/'splits.npz',allow_pickle=False) as d:
        rows={k:d['test_'+k] for k in ['users','histories','lengths','targets']}
    with np.load(OUT/'test_ranks.npz',allow_pickle=False) as d:
        rank_arrays={k:d[k] for k in d.files}
    tokens=read(DATA/'tokens.json')
    token2id={str(t):i for i,t in enumerate(tokens['items'])}
    chosen=np.random.default_rng(20261111).choice(20000,300,replace=False)
    queries=[[str(tokens['items'][x]) for x in h[:n]] for h,n in zip(rows['histories'][chosen],rows['lengths'][chosen])]
    sample_sha=hashlib.sha256(json.dumps(list(zip(rows['users'][chosen].tolist(),queries)),sort_keys=True).encode()).hexdigest()
    primary=frozen['primary']['config']['family']
    primary_record=None
    latencies={}
    primary_actual_metrics=None
    code_audit={}
    for record in result['results']:
        config=record['config']
        family=config['family']
        model=literal_model(config)
        candidate=OUT/'candidate_code'/f'{family}.py'
        assert outside_core(candidate.read_text(encoding='utf-8'))==outside_core(original_text),family
        code_audit[family]={'only_grey_coefficients_changed':True,'source_sha256':sha(candidate)}
        def predict(raw):
            history=np.asarray([token2id[str(x)] for x in raw],dtype=np.int64)
            ids,scores,_,_=model.recommend(history)
            return [{'rank':i+1,'item':str(tokens['items'][x]),'score':float(scores[x])} for i,x in enumerate(ids)]
        for i,raw in zip(chosen,queries):
            rec=predict(raw)
            items=[r['item'] for r in rec]
            target=str(tokens['items'][rows['targets'][i]])
            rank=items.index(target)+1 if target in items else 0
            assert rank==rank_arrays[family][i],(family,int(i))
        if family==primary:
            primary_record=record
            ranks=[]
            with (OUT/'recommended_predictions.jsonl').open('w',encoding='utf-8') as f:
                for i,(u,h,n,t) in enumerate(zip(rows['users'],rows['histories'],rows['lengths'],rows['targets'])):
                    ids,scores,_,_=model.recommend(h[:n])
                    pos=np.flatnonzero(ids==t)
                    rank=int(pos[0]+1) if len(pos) else 0
                    assert rank==rank_arrays[family][i],(family,i)
                    ranks.append(rank)
                    row={'session_id':str(tokens['users'][u]),'history':[str(tokens['items'][x]) for x in h[:n]],
                         'target':str(tokens['items'][t]),'target_rank':rank,
                         'recommendations':[{'rank':j+1,'item':str(tokens['items'][x]),'score':float(scores[x])} for j,x in enumerate(ids)]}
                    f.write(json.dumps(row)+'\n')
            primary_actual_metrics={'recall@10':sum(r>0 for r in ranks)/len(ranks),
                                    'mrr@10':sum(1/r for r in ranks if r)/len(ranks),
                                    'ndcg@10':sum(1/math.log2(r+1) for r in ranks if r)/len(ranks)}
            for k,v in primary_actual_metrics.items():
                assert abs(v-record['test'][k])<1e-12
        for raw in queries[:30]:
            predict(raw)
        gc.collect()
        gc.disable()
        samples=[]
        rounds=[]
        try:
            for round_id in range(3):
                times=[]
                for i in np.random.default_rng(20261111+round_id).permutation(len(queries)):
                    begin=time.perf_counter_ns()
                    pred=predict(queries[i])
                    times.append((time.perf_counter_ns()-begin)/1e6)
                samples+=times
                rounds.append({'mean_ms':float(np.mean(times)),'median_ms':float(np.median(times))})
        finally:
            gc.enable()
        state_bytes=sum(a.nbytes for a in [model.counts,model.log_popularity,model.weights])
        for matrix in [model.transitions,model.cooccurrence]:
            state_bytes+=matrix.data.nbytes+matrix.indices.nbytes+matrix.indptr.nbytes
        latencies[family]={'mean_ms':float(np.mean(samples)),'median_ms':float(np.median(samples)),
                           'p95_ms':float(np.percentile(samples,95)),'numeric_state_bytes':state_bytes,
                           'timed_calls':900,'verified_queries':300,'sample_sha256':sample_sha,
                           'rounds':rounds,'samples_ms':samples}
        print(f'{family}: {latencies[family]["mean_ms"]:.3f}ms',flush=True)
    save('latency.json',latencies)
    base=next(r for r in result['results'] if r['config']['family']=='baseline')
    summaries=[]
    for record in result['results']:
        family=record['config']['family']
        ranks=rank_arrays[family]
        baseline=rank_arrays['baseline']
        win=int(np.sum((ranks>0)&(baseline==0)))
        loss=int(np.sum((ranks==0)&(baseline>0)))
        p=float(binomtest(min(win,loss),win+loss,.5).pvalue) if win+loss else 1.
        summaries.append({'family':family,'method':LABELS[family],'value':record['config']['value'],
                          'valid_mrr@10':record['valid']['mrr@10'],**record['test'],
                          'recall_gain_points':100*(record['test']['recall@10']-base['test']['recall@10']),
                          'extra_hits_net':win-loss,'paired_wins':win,'paired_losses':loss,
                          'mcnemar_p':p,'mean_response_ms':latencies[family]['mean_ms'],
                          'numeric_state_mb':latencies[family]['numeric_state_bytes']/1e6})
    ordered=sorted([r for r in summaries if r['family']!='baseline'],key=lambda r:r['mcnemar_p'])
    previous=0
    for i,r in enumerate(ordered):
        adjusted=min(1.,max(previous,(len(ordered)-i)*r['mcnemar_p']))
        r['mcnemar_holm_p']=adjusted
        previous=adjusted
    for r in summaries:
        r.setdefault('mcnemar_holm_p',1.)
    save('comparison.json',summaries)
    with (OUT/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    best=next(r for r in summaries if r['family']==primary)
    b=next(r for r in summaries if r['family']=='baseline')
    beta=primary_record['config']['value']
    assert primary=='rational_distance'
    # Bounds, monotonicity, endpoints and recovery of the original beta=0 form.
    d=np.linspace(0,1,10001)
    compressed=d/(1+beta*(1-d))
    assert np.all(np.diff(compressed)>0) and compressed[0]==0 and compressed[-1]==1
    assert np.all((compressed>=0)&(compressed<=1))
    np.testing.assert_array_equal(d/(1+0*(1-d)),d)
    lines=['# 灰色关联系数内部的一处改动','',
           '## 推荐：有理差异压缩','',
           '这次固定原四项指标、原min-max归一化、四项等权、ρ=0.5和最后的算术平均，只改灰色关联系数函数内部的差异映射。','',
           '原灰色关联系数：','',
           'ξ=(Δmin+ρΔmax)/(Δ+ρΔmax)，Δ=|1−z|。','',
           '原公式可对照[公开研究的方法定义](https://link.springer.com/article/10.1007/s12540-025-02003-9)。以下压缩与自适应规则是本轮候选设计，未做文献新颖性核验。','',
           '在当前0—1归一化、参考向量全1的输入下，将差异改为：','',
           'Δ̃ = Δ / [1 + β(1−Δ)]，β≥0。','',
           '再按原灰色关联系数公式计算ξ。验证集从β=1、3、7、15、31、63、127、255中选出β=7；最后的等权算术平均保持不变。','',
           '实际代码仅在`grey_coefficients`中增加一行：','',
           '```python','delta = delta / (1.0 + 7 * (1.0 - delta))','```','',
           '添加位置是计算`delta`之后、计算`lo, hi`之前。它没有改输入特征，也没有改最后的平均方式。','',
           '## 为什么仍是对灰色关联机制的小改动','',
           '- 改动位于“差异 → 关联系数”的内部，改变的是对不同差异大小的响应曲线。',
           '- β=0时严格退化为原方法。0≤Δ≤1、β≥0时，Δ̃仍在[0,1]且随Δ严格递增，两端0和1不变。',
           '- 当前数据归一化后Δmin=0、Δmax=1，因此关联系数范围仍为[1/3,1]；完全匹配时仍为1。',
           '- 中间差异被非线性压缩，改变多项关联系数汇总后的商品排序。增加绝对分数本身不代表质量变好，质量提升由实际下一点击评估给出。','',
           '在当前ρ=0.5、β=7条件下，可写成：','',
           '原：ξ(z)=1/(3−2z)；改：ξ̃(z)=(1+7z)/(3+5z)。','',
           '| 归一化指标z | 原关联系数 | 改进关联系数 |', '|---:|---:|---:|']
    for z in [0,.01,.1,.5,1]:
        lines.append(f'| {z:g} | {1/(3-2*z):.5f} | {(1+7*z)/(3+5*z):.5f} |')
    lines+=['','该映射提高弱至中等非零证据的响应，压缩它与极强证据之间的差距；零证据和完全匹配的端点保持不变。','',
            '## 完整测试结果','',
            f"沿用相同20,000条测试会话和全部32,968件候选商品。Recall从{b['recall@10']:.3%}到{best['recall@10']:.3%}，提高{best['recall_gain_points']:.3f}个百分点，相对提升{best['recall@10']/b['recall@10']-1:.2%}，净多命中{best['extra_hits_net']:,}条。MRR={best['mrr@10']:.5f}，NDCG={best['ndcg@10']:.5f}。",'',
            '| 系数内部改动 | 验证选择的参数 | Recall@10 | MRR@10 | NDCG@10 | 平均响应/ms |',
            '|---|---:|---:|---:|---:|---:|']
    for r in [b]+sorted([r for r in summaries if r['family']!='baseline'],key=lambda r:r['valid_mrr@10'],reverse=True):
        value='—' if r['value'] is None else str(r['value'])
        lines.append(f"| {r['method']} | {value} | {r['recall@10']:.3%} | {r['mrr@10']:.5f} | {r['ndcg@10']:.5f} | {r['mean_response_ms']:.3f} |")
    lines+=['',f"本轮共8类、31个验证配置，按完整20,000条验证MRR选择每类代表并在测试前冻结。有理压缩也是验证MRR最高的主配置。五种自适应ρ分别使用全局差异均值、逐指标差异均值、指标均值/标准差、标准差/均值、非零证据差异均值；ρ裁剪到[0.05,1]，该边界是固定设计，未调参。部分自适应形式失败，均保留。",'',
            '## 轻量程度和局限','',
            f"实际源码版本在同机CPU、单请求、相同300条会话上各计时900次，包含ID转换、全目录特征构建、打分和Top10输出。原版平均{b['mean_response_ms']:.3f}ms，改版{best['mean_response_ms']:.3f}ms。模型数值状态均为{best['numeric_state_mb']:.2f}MB；新增一个标量β，无新训练统计、嵌入、邻域索引或神经网络。计算复杂度保持O(FM)级别（另加原有共现查表），多一次逐元素差异变换。",'',
            '当前GCE-GNN同划分的历史指标为Recall33.135%、MRR0.15904、NDCG0.19979；本次未重新训练或调参GNN。',
            f"相同会话的命中变化：新增{best['paired_wins']}、失去{best['paired_losses']}。精确McNemar检验对8类代表作Holm校正后p={best['mcnemar_holm_p']:.3e}，该检验针对Recall命中率。",'',
            '该改法适用于本项目的0—1指标与全1理想参考设定；对任意未归一化序列不能直接套用当前差异压缩式。这里保持了值域、端点、差异单调性及β=0还原性质，未据此声称已证明所有灰色关联理论公理。',
            '测试集沿用此前已查看的数据，本轮β只用验证集选择；结果是探索性比较。数学上这也可视为差异重参数化，是否已有同类方法需要后续文献查新。','',
            '## 核验与文件','',
            '- 原灰色关联源文件未修改；candidate_code中的候选仅改grey_coefficients函数，其他函数和类的AST逐项相同。',
            '- 本轮分组计数覆盖完整商品目录，而非截取少量候选。商品热度仅85种取值，背景商品按热度组完整计数，再校正有会话证据的商品；处理非线性带来的浮点并列，避免候选采样或遗漏。',
            f"- 验证集{frozen['full_catalogue_validation_checks']}次、测试集{result['full_catalogue_test_checks']}次全目录代码对照；每候选再核验300条真实预测。推荐改法额外用实际一行源码跑完全部20,000条测试，并与分组计数结果逐条一致。",'',
            '- `candidate_code/rational_distance.py`及`.patch`：可直接运行的源码与一行差异补丁。',
            '- `recommended_predictions.jsonl`：推荐改法全部20,000条真实预测。',
            '- `validation_search.json`、`frozen_selection.json`：31组验证结果与冻结选择。',
            '- `comparison.csv/json`：本轮各系数改法指标、响应、配对统计。',
            '- `latency.json`：900次实际计时记录。',
            '- `completion_audit.json`：源码范围、预测、性质检查与文件指纹。','',
            '复跑（在03_代码与实验目录）：','', '```powershell',
            "$greyCoefficientPython = 'D:\\Envs\\recbole-gnn-py39\\Scripts\\python.exe'",
            '& $greyCoefficientPython experiments/explore_grey_coefficient.py select',
            '& $greyCoefficientPython experiments/explore_grey_coefficient.py test',
            '& $greyCoefficientPython experiments/finish_grey_coefficient.py','```','']
    report=OUT/'灰色关联系数内部改动与实测.md'
    report.write_text('\n'.join(lines),encoding='utf-8')
    save('completion_audit.json',{'all_passed':True,'original_source_unchanged':sha(source)==frozen['original_source_sha256'],
                                 'all_candidate_edits_inside_coefficient_function':code_audit,
                                 'same_four_features_equal_weights_arithmetic_mean':True,
                                 'full_catalogue_evaluation_with_popularity_groups':True,
                                 'all_20000_primary_literal_predictions_match':True,
                                 'independent_primary_metrics':primary_actual_metrics,
                                 'each_method_300_correctness_queries_and_900_timed_calls':True,
                                 'same_sample_sha256':sample_sha,
                                 'distance_bounds_monotonicity_endpoints_and_beta_zero_recovery':True,
                                 'frozen_config_sha256':sha(OUT/'frozen_selection.json'),
                                 'report_sha256':sha(report),'predictions_sha256':sha(OUT/'recommended_predictions.jsonl')})
    print(json.dumps({'primary':best,'baseline':b,'audit_passed':True},ensure_ascii=True,indent=2),flush=True)


if __name__=='__main__':
    main()
