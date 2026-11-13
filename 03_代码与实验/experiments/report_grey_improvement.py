"""Generate reproducible comparison and complexity notes from measured artifacts."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/grey_improved'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    result = read(OUT / 'result.json')
    selection = read(OUT / 'selection.json')
    prepare = read(OUT / 'prepare.json')
    latency = {r['model']: r for r in map(read, sorted((OUT/'latency').glob('*.json')))}
    assert set(latency) == {'SRGNN', 'GCSAN', 'GCEGNN', 'GRA-original', 'GRA-improved'}
    assert len({r['sample_sha256'] for r in latency.values()}) == 1
    measured = []
    for row in result['comparison']:
        r = dict(row)
        speed = latency[r['model']]
        r.update({'response_mean_ms': speed['mean_ms'], 'response_median_ms': speed['median_ms'],
                  'response_p95_ms': speed['p95_ms'],
                  'numeric_state_mb_decimal': speed['numeric_state_bytes']/1e6})
        measured.append(r)
    with (OUT/'comparison.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(measured[0]))
        writer.writeheader()
        writer.writerows(measured)
    (OUT/'comparison.json').write_text(json.dumps(measured, indent=2), encoding='utf-8')
    fast = next(r for r in measured if r['model']=='GRA-improved')
    old = next(r for r in measured if r['model']=='GRA-original')
    gce = next(r for r in measured if r['model']=='GCEGNN')
    speedup = gce['response_mean_ms']/fast['response_mean_ms']
    reduction = 1-fast['numeric_state_mb_decimal']/gce['numeric_state_mb_decimal']
    lines = ['# 灰色关联推荐改进：性能、响应时间与复杂度', '',
             '## 实测结论', '',
             f"在已有 Diginetica 20,000 会话划分上，改进版 GRA 的 Recall@10 为 {fast['recall@10']:.3%}、MRR@10 为 {fast['mrr@10']:.5f}、NDCG@10 为 {fast['ndcg@10']:.5f}，三项均高于本项目已有三个 GNN 配置。",
             f"与 GCE-GNN 相比，Recall 高 {(fast['recall@10']-gce['recall@10'])*100:.3f} 个百分点；单请求平均计算响应时间约快 {speedup:.2f} 倍，模型数值状态占用减少 {reduction:.1%}。这些结论限于本机、该划分与已有模型配置。", '',
             '| 方法 | Recall@10 | MRR@10 | NDCG@10 | 平均响应/ms | 中位响应/ms | P95响应/ms | 模型数值状态/MB |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in measured:
        lines.append(f"| {r['model']} | {r['recall@10']:.3%} | {r['mrr@10']:.5f} | {r['ndcg@10']:.5f} | {r['response_mean_ms']:.3f} | {r['response_median_ms']:.3f} | {r['response_p95_ms']:.3f} | {r['numeric_state_mb_decimal']:.2f} |")
    lines += ['', 'Recall 为真实下一点击进入前十的比例；MRR、NDCG 还奖励较靠前的命中。MB 按 1,000,000 字节计。', '',
              '## 方法如何改进', '',
              '输入会话 → 查训练统计与相似历史会话 → 形成六项评价指标 → 归一化与灰色关联评分 → 精确 Top-10。', '',
              '在原有四项指标上增加两项：', '',
              '1. **相似会话支持度**：仅保存每个训练会话最长已观测前缀的商品集合；通过倒排索引查找与当前历史有交集的会话。以位置加权交集除以历史会话集合长度平方根作为相似度（省略对所有邻居相同的查询范数），取至多100个邻居；候选商品得分为包含该商品的邻居相似度平方之和。当前会话自己的训练前缀不作为邻居，防止自身匹配主导这项指标。',
              '2. **整段历史转移支持度**：将历史中每个商品的训练转移概率按浏览位置加权汇总，补充“只看最后一次点击”的不足。', '',
              '保留的四项是最近点击转移、位置加权共现、位置加权重复点击和 log(1+训练点击次数)。位置权重是 1,…,L 除以其和。共现相似度为训练会话前缀商品共现次数除以两商品会话频次的几何平均；不含自共现。', '',
              '所有指标均按全商品目录做效益型 min-max 归一化；常数指标设零。令 z 为归一化值，加入由验证集选择的幂变换 z^γ，理想参考仍为全1向量。', '',
              'ξ(i,k) = ρ / [1 − z(i,k)^γ + ρ]；score(i) = Σₖ wₖ ξ(i,k)。', '',
              '该数据规模下重复指标同时存在0和1，保证全候选×指标的 Δmin=0、Δmax=1，因此上式与标准两级极值的灰色关联系数公式一致。', '',
              '最终验证集选定参数：', '',
              '| 指标 | 权重 |', '|---|---:|']
    labels = ['最近点击转移', '历史共现', '会话重复点击', '商品热度', '相似会话支持度', '历史转移支持度']
    for label, weight in zip(labels, selection['selected']['weights']):
        lines.append(f'| {label} | {weight:.2f} |')
    lines += ['', f"ρ={selection['selected']['rho']:.2f}，γ={selection['selected']['power']:.2f}，邻居上限100。", '',
              f"共筛选 {selection['screen_trials']} 组配置：先在固定随机抽取的4,000条验证样本上筛选，再在完整20,000条验证样本上比较前12组及原版参考配置。按验证 MRR@10 选择，Recall 只作为并列规则；最优验证 Recall={selection['selected']['full_valid']['recall@10']:.3%}、MRR={selection['selected']['full_valid']['mrr@10']:.5f}。测试前保存配置哈希，测试后未更改参数。", '',
              '## 为什么稀疏实现还能比较全目录 Top-10', '',
              '六项中只有热度是全局静态项；其余指标只在转移、共现、当前浏览和邻居涉及的商品上非零。将这些商品的并集记为 U。U 外商品的局部指标全为0，它们之间的排序只取决于训练热度。', '',
              '每次只需计算 U，再加入全局热度最高的10件商品和内部编号最小的10件商品。所有权重非负、灰色关联系数随效益指标单调递增，因此 U 外未计算商品无法超过保留的前十热度兜底；编号兜底同时覆盖热度权重为0的并列情况。全目录归一化极值也从稀疏支持和预计算热度极值精确取得。', '',
              f"这是针对当前评分公式的精确计算压缩，不是采样候选。验证集120组、测试集100组稀疏/全目录 Top-10 对照均一致。测试时平均只需显式计算 {result['mean_exact_support_items']:.1f} 件商品，P95 为 {result['p95_exact_support_items']:.0f} 件，全目录仍为32,968件。", '',
              '## 响应速度如何测量', '',
              '- 同一台 Windows CPU、Python 3.9.21、PyTorch 2.1.2 CPU，PyTorch统一4线程，单请求 batch=1。',
              '- 相同固定随机300条测试会话；先逐条核对预测与保存输出，再预热；每种方法独立进程，随机顺序测3轮，共900次实际计算；未缓存查询结果。',
              '- 计时包含原始商品ID转换、会话特征或图构建、推荐打分、Top-10与原始ID输出。排除模型启动、磁盘写出、HTTP与排队；这是模型计算响应时间。',
              '- GNN每次仍计算全部商品的得分；GRA使用上文证明的精确稀疏 Top-10。评估目标和推荐输出范围一致。',
              '- 轻量程度采用模型数值数组/张量字节数：GRA计入稀疏统计、会话正排/倒排索引；GCE-GNN计入全局邻接及权重。该数值不含Python对象、共享ID映射、数据集对象和运行库，不能称为整个进程RSS。', '',
              f"改进版相对原 GRA 平均响应提速 {old['response_mean_ms']/fast['response_mean_ms']:.2f} 倍，但数值状态由 {old['numeric_state_mb_decimal']:.2f} MB 增至 {fast['numeric_state_mb_decimal']:.2f} MB，这是增加会话邻域索引的代价；仍低于三个GNN。原版P95为 {old['response_p95_ms']:.3f} ms，改进版为 {fast['response_p95_ms']:.3f} ms，平均提速不代表每个尾延迟指标都改善。", '',
              '## 时间复杂度分析', '',
              '记 M 为商品数，N 为训练会话数，I 为最长训练前缀的总点击数，L/Lmax 为当前/最大会话长度，d 为GNN嵌入维度，U/E 为当前会话图节点/边数，F为评价指标数，b为全局邻居数，f为Transformer前馈层宽度。当前 M=32,968、N=20,000、Lmax=20、d=64、F=6、b=12、f=256。', '',
              'GRA另记 P 为查询商品倒排列表的总访问量，A 为转移/共现/邻居传播中访问的商品记录总量，C 为最终稀疏并集大小，K为会话邻居上限（100）。', '',
              '| 方法 | 单请求主要计算成本 | 主要模型数值存储 |',
              '|---|---|---|',
              '| 原 GRA | O(Σ历史商品共现度 + MF)，再用线性选择取前十 | O(E共现+E转移+M) |',
              '| 改进 GRA | O(P log P + A log A + A log C + CF)，其中邻居商品访问最多 K·Lmax，已计入A | O(E共现+E转移+I+N+M) |',
              '| SR-GNN | O(g(Ud²+Ed)+Ld²+Md)，当前传播步数g=1 | O(Md+d²) |',
              '| GC-SAN | SR局部图成本，再加 O(a(L²d+Ld²+Ldf))；当前自注意力层a=1 | O(Md+d²+df) |',
              '| GCE-GNN | 当前一跳配置 O(Ed+Lmax·b·d²+Ld²+Md) | O(Md+Mb+d²) |', '',
              'GNN表中是前向编码与全目录打分的主要算术成本，另有取前十O(M)和会话图构造开销；本地构图包含唯一化排序，GCE检测双向边的实现还含O(L²)比较。GCE全局分支按Lmax填充，公式使用Lmax。多跳配置需按邻居展开和重复聚合另行计算，不沿用这里的一跳式。', '',
              '改进GRA的log项来自NumPy unique、候选排序及searchsorted；固定前十选择用线性partition。其查询不再必需扫描M个商品，也不执行Md维嵌入点积。但当热门商品倒排很长或共现图稠密时，P与A会增大，不能保证任何数据规模下都更快。', '',
              f"离线方面，GRA共现构建约O(Σ_s L_s²)，正排/倒排索引O(I+N+M)，预排序热度O(M log M)。本次统计和索引构建 {prepare['statistics_and_index_fit_seconds']:.3f} 秒；验证集特征构建、搜索和所选模型写出共 {selection['selection_seconds_including_cache_io']:.3f} 秒。权重选择需要额外 O(T·Q·C·F) 级计算，T为候选配置数、Q为参与验证的查询数，应计入离线成本。", '',
              '三个GNN训练还需按训练样本、实际轮数重复前向/反向、全目录交叉熵与验证，并更新优化器状态；不能只用“一次前向复杂度”代替整个训练成本。', '',
              '## 性能提升体现在哪里', '',
              '| 方法 | 目标已在当前历史出现：Recall@10 | 目标未在当前历史出现：Recall@10 |',
              '|---|---:|---:|']
    for r in measured:
        lines.append(f"| {r['model']} | {r['repeat_target_recall@10']:.3%} | {r['novel_target_recall@10']:.3%} |")
    lines += ['', f"重复目标共 {result['repeat_target_cases']:,} 条，未出现目标共 {result['novel_target_cases']:,} 条。新方法的总指标优势主要来自对历史商品的保留；未浏览商品命中率也比原GRA提高，但低于GCE-GNN。因此可表述为‘本次下一点击任务的质量—速度权衡更好’，不能据此声称新商品发现能力整体超过GNN。这里‘未浏览’不等于训练集从未见过的冷启动商品。", '',
              '## 适用范围', '',
              '- 数据仍是原正式20k子集，训练97,273、验证20,000、测试20,000个前缀样本；按会话留末，而非全局时间隔离。原项目全量商品频次筛选和词表先于划分，此次沿用。',
              '- GNN使用已有单种子、已有超参数和早停权重，未重新调参或重训。GRA此次用验证集调参；对比说明当前实现的实测选择，不是对所有优化充分的GNN作普遍优越性结论。',
              '- 本轮未根据测试结果改权重或选择配置；但上一轮已查看同一测试集表现并据此确定改进方向，因此这是沿用既有测试集的探索性改进比较，尚不构成全新未见测试集上的独立确认。',
              '- 改进同时改变了特征、归一化幂变换、权重与计算实现；这些结果不能单独证明灰色关联变换本身贡献了多少。',
              '- 当前方法可称“基于会话邻域与加权灰色关联的轻量推荐方法”，不含GM灰预测。', '',
              '## 文件与复跑', '',
              '- `comparison.csv/json`：五方法统一指标、响应时间、数值状态占用。',
              '- `selection.json`、`validation_search.json`：验证筛选记录和测试前冻结配置。',
              '- `result.json`、`GRA-improved.jsonl`：完整20,000条测试结果与逐会话Top-10。',
              '- `model_selected/`：可重载统计模型、邻域索引和权重。',
              '- `latency/*.json`：每方法900次计时、相同样本指纹、预测一致性。',
              '- `verification.json`：划分与输出对齐、稀疏/全目录一致性、模型重载核验。', '',
              '在 `03_代码与实验` 目录按顺序执行；计时脚本逐个运行，避免进程争抢CPU：', '',
              '```powershell', "$experimentPython = 'D:\\Envs\\recbole-gnn-py39\\Scripts\\python.exe'",
              '& $experimentPython experiments/prepare_grey_improvement.py',
              '& $experimentPython experiments/tune_grey_improvement.py',
              '& $experimentPython experiments/evaluate_grey_improvement.py',
              "foreach ($methodName in @('GRA-improved','GRA-original','SRGNN','GCSAN','GCEGNN')) {",
              '    & $experimentPython experiments/benchmark_grey_improvement.py --model $methodName', '}',
              '& $experimentPython experiments/report_grey_improvement.py', '```', '',
              '独立复核运行：`& $experimentPython experiments/verify_grey_improvement.py`。审查产物为 `completion_audit.json`。', '',
              '验证特征缓存位于 `validation_features.npz`；如修改特征定义，需重新生成该缓存后调参。', '']
    (OUT/'灰色关联改进与复杂度对比.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'models': measured, 'speedup_vs_gce': speedup,
                      'numeric_state_reduction_vs_gce': reduction}, indent=2))


if __name__ == '__main__':
    main()
