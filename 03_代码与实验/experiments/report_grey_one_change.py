"""Report all single-location interventions, including unsuccessful experiments."""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/grey_single_change'
LABELS={
 'baseline':'原版四指标GRA','all_power':'四项指标整体幂变换',
 'repeat_weight':'只调整再访权重','popularity_weight':'只调整热度权重',
 'cooccurrence_weight':'只调整共现权重','transition_weight':'只调整最近转移权重',
 'rho':'只调整分辨系数ρ','repeat_power':'只变换再访指标','popularity_power':'只变换热度指标',
 'cooccurrence_power':'只变换共现指标','transition_power':'只变换最近转移指标',
 'repeat_binary':'再访改为是否浏览','repeat_count':'再访改为点击次数',
 'repeat_last_only':'再访只保留最后商品','keep_self_cooccurrence':'保留共现对角线',
 'geometric_mean':'算术均值改为几何均值','harmonic_mean':'算术均值改为调和均值',
 'max_aggregation':'算术均值改为最大值','all_binary':'四项指标全部二值化'}


def main():
    results=json.loads((OUT/'test_results.json').read_text(encoding='utf-8'))
    selected=json.loads((OUT/'frozen_selection.json').read_text(encoding='utf-8'))
    latency=json.loads((OUT/'latency.json').read_text(encoding='utf-8'))
    records=[]
    for r in results['results']:
        family=r['config']['family']
        speed=latency[family]
        record={'family':family,'change':LABELS[family],'value':r['config']['value'],
                **r['test'],'valid_mrr@10':r['valid']['mrr@10'],
                'recall_gain_percentage_points':r['recall_gain_percentage_points'],
                'additional_hits_net':r['additional_hits_net'],
                'response_mean_ms':speed['mean_ms'],'response_p95_ms':speed['p95_ms'],
                'numeric_state_mb_decimal':speed['numeric_state_bytes']/1e6,
                'mcnemar_holm_p':r.get('mcnemar_holm_p',1.),
                'repeat_target_recall':r['repeat_target_recall'],
                'novel_target_recall':r['novel_target_recall']}
        records.append(record)
    with (OUT/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (OUT/'comparison.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    by_name={r['family']:r for r in records}
    base,best=by_name['baseline'],by_name['harmonic_mean']
    power=by_name['all_power']
    lines=['# 原版灰色关联推荐：单处改动逐一比较','',
           '## 建议采用：只将最终算术聚合改成调和聚合','',
           f"保持原四项指标、训练统计、等权重、ρ=0.5、归一化、全目录Top-10及数据划分不变，只替换最终评分的一行。Recall@10 从 {base['recall@10']:.3%} 提升到 {best['recall@10']:.3%}，高 {best['recall_gain_percentage_points']:.3f} 个百分点，相对提升 {(best['recall@10']/base['recall@10']-1):.2%}；净多命中 {best['additional_hits_net']:,} 条测试会话。",
           f"MRR@10 从 {base['mrr@10']:.5f} 提升到 {best['mrr@10']:.5f}，NDCG@10 从 {base['ndcg@10']:.5f} 提升到 {best['ndcg@10']:.5f}。平均响应 {base['response_mean_ms']:.3f} → {best['response_mean_ms']:.3f} ms，基本不变；模型数值状态保持 {best['numeric_state_mb_decimal']:.2f} MB。",'',
           '原代码：','', '```python',
           'scores[1:] = np.sum(self.weights[:, None] * coefficients, axis=0)','```','',
           '替换为：','', '```python',
           'scores[1:] = 1.0 / np.mean(1.0 / coefficients, axis=0)','```','',
           '这条式子针对原版四项等权设置。关联系数仍采用原公式，改变的是四个关联系数的聚合方式；没有引入会话邻域、第五个指标或新的可调参数。', '',
           '## 为什么这是一处有意义的改动','',
           '原算术聚合为 Γ_A=(ξ₁+ξ₂+ξ₃+ξ₄)/4；调和聚合为 Γ_H=4/(1/ξ₁+1/ξ₂+1/ξ₃+1/ξ₄)。原ρ=0.5时各关联系数均为正，倒数有定义。', '',
           '调和均值更看重较小的分量。一个商品若只在某一项很高、其他项弱，其综合分数会受到更强约束；多项证据比较均衡的商品可以获得相对优势。','',
           '纯示意：关联系数 (1,1/3,1/3,1/3) 的算术均值为0.5、调和均值为0.4；(0.5,0.5,0.5,0.5) 的两种均值都为0.5。该变化能压低“单项很强、其他项弱”的候选，例如主要依靠热度的候选。具体哪类商品因此改变排序，仍以真实输出为准。','',
           '## 实验控制与选择','',
           f"共探索 {selected['families']} 类单处改动、{selected['configurations_including_baseline']} 个配置（含原版）。每类只动一个标量、一个指标定义或一个聚合公式；其余保持原样。参数类先在完整20,000条验证样本上选MRR最高者，Recall作并列规则。搜索范围扩展由验证最优值落在边界触发，不依据测试指标扩展。各类最终只把一个冻结配置放入本轮统一测试表。",'',
           '数据为已有Diginetica正式子集：20,000会话、32,968候选商品，原97,273训练前缀样本、20,000验证和20,000测试样本；留末划分、训练统计、候选目录和评价指标与原GRA一致。', '',
           '验证MRR预先选中的主配置是整体幂变换γ=1/16。下方工程建议进一步结合用户要求的简单程度、是否新增调参和实际计算开销；调和均值是所有固定无超参改动中验证MRR最高者。两者的选择依据分别列出，未把测试排名改写成验证选择。', '',
           '## 全部单处改动的测试结果','',
           '权重倍率表示只将对应的一项原权重乘以该数，其他三项不变；与最后把整个权重向量归一化得到相同排序。幂指数只作用于指定指标，整体幂变换则对整张四指标矩阵执行一次统一变换。', '',
           '| 单处改动 | 验证选择的值 | Recall@10 | MRR@10 | NDCG@10 | Recall变化/百分点 | 平均响应/ms |',
           '|---|---:|---:|---:|---:|---:|---:|']
    display=[base]+sorted([r for r in records if r['family']!='baseline'],key=lambda r:r['valid_mrr@10'],reverse=True)
    for r in display:
        value='—' if r['value'] is None or isinstance(r['value'],bool) else str(r['value'])
        lines.append(f"| {r['change']} | {value} | {r['recall@10']:.3%} | {r['mrr@10']:.5f} | {r['ndcg@10']:.5f} | {r['recall_gain_percentage_points']:+.3f} | {r['response_mean_ms']:.3f} |")
    lines+=['','## 如何取舍','',
            f"1. **优先简单、无新增可调参数、维持响应开销：调和聚合。** Recall={best['recall@10']:.3%}，MRR={best['mrr@10']:.5f}；这是本轮建议采用的一行改法。",
            f"2. **验证MRR最高：整体幂变换γ=1/16。** 测试Recall={power['recall@10']:.3%}、MRR={power['mrr@10']:.5f}，但通用NumPy幂运算让平均响应增至{power['response_mean_ms']:.3f}ms，约为原版{power['response_mean_ms']/base['response_mean_ms']:.2f}倍。不能将它描述成不增加计算成本。",
            f"3. **更重视前十覆盖：仅把热度权重乘0.05。** 测试Recall={by_name['popularity_weight']['recall@10']:.3%}，MRR={by_name['popularity_weight']['mrr@10']:.5f}，平均{by_name['popularity_weight']['response_mean_ms']:.3f}ms。注意0.05是原热度权重的倍率，并非最终权重5%。",
            f"4. **更重视命中名次：仅把再访权重乘32。** 测试MRR={by_name['repeat_weight']['mrr@10']:.5f}，Recall={by_name['repeat_weight']['recall@10']:.3%}；再访目标命中率高，但未浏览目标Recall={by_name['repeat_weight']['novel_target_recall']:.3%}，低于原版{base['novel_target_recall']:.3%}。这是明显取舍。",'',
            '保留共现对角线虽然提高MRR，但降低Recall；仅保留最后一个浏览商品、最大值聚合也明显退化。失败方案全部保留在表中。','',
            '## 与原三个GNN的参照','',
            '| 方法 | Recall@10 | MRR@10 | NDCG@10 |', '|---|---:|---:|---:|',
            '| SR-GNN | 28.175% | 0.13396 | 0.16877 |',
            '| GC-SAN | 29.525% | 0.13874 | 0.17557 |',
            '| GCE-GNN | 33.135% | 0.15904 | 0.19979 |',
            f"| 一行调和聚合GRA | {best['recall@10']:.3%} | {best['mrr@10']:.5f} | {best['ndcg@10']:.5f} |",'',
            'GNN指标引用上一轮同划分的完整逐会话重算结果，未新增训练或调参；本轮响应表只测原GRA及各单处改动。上一轮同机GCE平均1.954ms仅作历史计算速度参照，不是本轮交错计时结果。','',
            '## 统计比较与分组','',
            f"调和聚合相对原版，在相同20,000条会话中新增命中1,315条、失去命中114条，净增1,201条。对命中/未命中作配对精确McNemar检验，并对18个冻结类代表作Holm校正，调和聚合校正后p={best['mcnemar_holm_p']:.3e}。这里检验的是Recall对应的命中概率，不是MRR或响应时间。",'',
            '| 方法 | 目标已在历史出现：Recall | 目标未在历史出现：Recall |',
            '|---|---:|---:|']
    for name in ['baseline','harmonic_mean','all_power','popularity_weight','repeat_weight']:
        r=by_name[name]
        lines.append(f"| {r['change']} | {r['repeat_target_recall']:.3%} | {r['novel_target_recall']:.3%} |")
    lines+=['','## 开销与核验','',
            '- 调和均值仍是对四个关联系数做逐商品聚合，评分复杂度保持O(FM)，F=4；原共现查表和指标构建成本保持不变。无需新模型、额外索引或新增训练统计。',
            '- 所有候选均使用原版稠密全目录实现，没有把稀疏加速混入本次单处改动；因此质量变化对应唯一改动。',
            '- CPU允许4线程、batch=1；同样300条会话，每候选先核验预测、预热，再随机顺序测3轮，共900次。计时包括原始ID转换、特征、全目录打分、前十及原始ID输出；排除启动与写盘。约0.64ms和0.65ms的差异按基本不变解释。',
            '- 模型数值状态不含Python对象、共享映射和运行库，不能等同进程RSS。除保留对角线增加稀疏非零元素外，其他改动均保持原4.94MB。',
            '- 原版20,000条测试名次逐条匹配旧输出；每个冻结配置40条稀疏缓存/全目录Top-10对照，独立一行源码版本另核验300条真实预测。验证主配置还核验全部20,000条一行源码输出。',
            '- 源代码原文件未改；各候选可直接运行的完整源码和统一diff存于candidate_code，manifest确认每种只新增、删除或替换一行。', '',
            '## 解释范围','',
            '这些是原四指标实现上的探索性比较，测试集沿用此前已查看的集合；本轮参数只由验证结果选择。提升不能推广成所有推荐数据上的普遍优势；本次改变的是灰色关联系数的聚合方式，不是提出新的灰色关联系数理论。','',
            '## 文件与复跑','',
            '- `comparison.csv/json`：18类单改与原版的完整质量、时延和配对统计表。',
            '- `frozen_selection.json`、`validation_search.json`：全部79组配置的验证结果和测试前选择。',
            '- `candidate_code/harmonic_mean.py`、`harmonic_mean.patch`：建议使用的完整代码及一行补丁。',
            '- `candidate_code/all_power.py`、`all_power.patch`：验证MRR选中的主配置源码。',
            '- `single_location_manifest.json`：原版与每种候选的代码改动位置、行数及哈希。',
            '- `test_ranks.npz`、`test_results.json`：全部20,000条会话的各候选名次与结果。',
            '- `primary_predictions.jsonl`：验证主配置的一行源码完整预测。',
            '- `recommended_predictions.jsonl`：建议调和聚合方法的一行源码完整预测，由核验脚本生成。',
            '- `latency.json`：各候选900次计时记录。',
            '- `completion_audit.json`：单处改动、原版一致性和实际输出核验。','',
            '在03_代码与实验目录依次执行：','', '```powershell',
            "$singleChangePython = 'D:\\Envs\\recbole-gnn-py39\\Scripts\\python.exe'",
            '& $singleChangePython experiments/run_grey_one_change.py select',
            '& $singleChangePython experiments/run_grey_one_change.py test',
            '& $singleChangePython experiments/benchmark_grey_one_change.py',
            '& $singleChangePython experiments/report_grey_one_change.py',
            '& $singleChangePython experiments/verify_grey_one_change.py','```','']
    (OUT/'单处改动对比与推荐.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'recommendation':best,'preselected_primary':power,'variants':selected['families']},ensure_ascii=True,indent=2))


if __name__=='__main__':
    main()
