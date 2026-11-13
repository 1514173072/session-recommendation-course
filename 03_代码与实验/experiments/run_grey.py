"""Evaluate fixed GRA on exactly the existing formal splits; reuse GNN predictions."""
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

from grey_session import FEATURES, GreySession, ranking_metrics, self_check

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'RecBole-GNN'))
OUT = ROOT / 'outputs/grey_relation'
MODELS = ['SRGNN', 'GCSAN', 'GCEGNN']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def examples(loader, config):
    feat = loader.dataset.inter_feat
    users = feat[config['USER_ID_FIELD']].numpy()
    targets = feat[config['ITEM_ID_FIELD']].numpy()
    seqs = feat[config['ITEM_ID_FIELD'] + config['LIST_SUFFIX']].numpy()
    lens = feat[config['ITEM_LIST_LENGTH_FIELD']].numpy()
    return [(int(u), s[:int(n)].copy(), int(t)) for u, s, n, t in zip(users, seqs, lens, targets)]


def main():
    os.chdir(ROOT)
    sys.argv = sys.argv[:1]
    from recbole.utils import init_seed
    from recbole_gnn.config import Config
    from recbole_gnn.utils import create_dataset, data_preparation
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    assert self_check()
    prior = {m: json.loads((ROOT / f'outputs/formal/{m}/seed42/result.json').read_text(encoding='utf-8')) for m in MODELS}
    config = Config(model='SRGNN', dataset='diginetica-formal',
                    config_file_list=[str(ROOT / 'configs/formal.yaml')],
                    config_dict={'data_path': str(ROOT / 'data'), 'seed': 42,
                                 'checkpoint_dir': str(OUT / 'unused_checkpoints'), 'use_gpu': False})
    init_seed(config['seed'], config['reproducibility'])
    start = time.perf_counter()
    dataset = create_dataset(config)
    train, valid, test = data_preparation(config, dataset)
    prep_seconds = time.perf_counter() - start
    split_hashes = {}
    for name, loader in [('train', train), ('valid', valid), ('test', test)]:
        identity = torch.stack([loader.dataset.inter_feat[config['USER_ID_FIELD']],
                                loader.dataset.inter_feat[config['ITEM_ID_FIELD']]], dim=1)
        split_hashes[name] = hashlib.sha256(identity.numpy().tobytes()).hexdigest()
    for m in MODELS:
        assert split_hashes == prior[m]['split_session_target_sha256'], (m, split_hashes)
        assert dataset.item_num - 1 == prior[m]['candidate_items']
    train_rows, test_rows = examples(train, config), examples(test, config)
    # Deduplicate nested prefix examples: never count the same observed click many times.
    prefixes = {}
    for user, history, target in train_rows:
        sequence = np.append(history, target)
        if user not in prefixes or len(sequence) > len(prefixes[user]):
            prefixes[user] = sequence
    for user, history, target in train_rows:
        np.testing.assert_array_equal(np.append(history, target), prefixes[user][:len(history) + 1])
    for user, history, _ in test_rows:
        # Last validation target is present in the test query but excluded from fitting.
        np.testing.assert_array_equal(prefixes[user], history[:-1])
    assert len(prefixes) == 20000 and len(test_rows) == 20000
    manifest = json.loads((ROOT / 'outputs/formal/data_manifest.json').read_text(encoding='utf-8'))
    data_file = ROOT / 'data/diginetica-formal/diginetica-formal.inter'
    assert sha(data_file) == manifest['output_sha256']
    print(f'Exact splits verified; fitting {len(prefixes)} unique training prefixes.', flush=True)
    start = time.perf_counter()
    model = GreySession(dataset.item_num).fit(prefixes.values())
    fit_seconds = time.perf_counter() - start
    model.save(OUT / 'grey_model.npz')
    item_tokens = dataset.field2id_token[config['ITEM_ID_FIELD']]
    user_tokens = dataset.field2id_token[config['USER_ID_FIELD']]
    write_json(OUT / 'item_tokens.json', item_tokens.tolist())
    reloaded = GreySession.load(OUT / 'grey_model.npz')
    for _, history, _ in test_rows[:3]:
        before = model.recommend(history)
        after = reloaded.recommend(history)
        np.testing.assert_array_equal(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
    print(f'Train statistics fitted in {fit_seconds:.3f}s; saved/reloaded exactly.', flush=True)
    ranks, rows_by_session, predict_seconds = [], {}, 0.0
    with (OUT / 'GRA.jsonl').open('w', encoding='utf-8') as output:
        for i, (user, history, target) in enumerate(test_rows):
            start = time.perf_counter()
            ids, scores, normalized, coeff = model.recommend(history)
            predict_seconds += time.perf_counter() - start
            match = np.flatnonzero(ids == target)
            rank = int(match[0] + 1) if len(match) else 0
            ranks.append(rank)
            record = {'model': 'GRA', 'session_id': str(user_tokens[user]),
                      'history': [str(item_tokens[x]) for x in history],
                      'target': str(item_tokens[target]), 'target_rank': rank,
                      'history_len': len(history), 'target_in_history': bool(target in history),
                      'recommendations': [
                          {'rank': j + 1, 'item': str(item_tokens[x]), 'score': float(scores[x]),
                           'normalized_features': {f: float(normalized[k, x]) for k, f in enumerate(FEATURES)},
                           'grey_coefficients': {f: float(coeff[k, x - 1]) for k, f in enumerate(FEATURES)}}
                          for j, x in enumerate(ids)]}
            assert len(ids) == 10 and len(set(ids.tolist())) == 10 and np.all(ids > 0)
            assert np.all(np.isfinite(scores[ids]))
            rows_by_session[record['session_id']] = record
            output.write(json.dumps(record, ensure_ascii=False) + '\n')
            if (i + 1) % 2000 == 0:
                print(f'GRA {i + 1}/20000; scoring time {predict_seconds:.2f}s', flush=True)
    result = {'model': 'GRA', 'test_result': ranking_metrics(ranks),
              'statistics_fit_seconds': fit_seconds, 'test_scoring_seconds': predict_seconds,
              'mean_scoring_ms_per_session': predict_seconds / len(ranks) * 1000,
              'data_preparation_seconds': prep_seconds,
              'train_prefix_clicks': int(model.counts.sum()),
              'training_seen_items': int(np.count_nonzero(model.counts)),
              'transition_nnz': model.transitions.nnz, 'cooccurrence_nnz': model.cooccurrence.nnz,
              'test_examples': len(ranks), 'candidate_items': dataset.item_num - 1,
              'rho': model.rho, 'weights': model.weights.tolist(), 'features': FEATURES,
              'hyperparameter_selection': 'Fixed a priori; no validation/test-based tuning.',
              'split_session_target_sha256': split_hashes,
              'split_semantics': 'Within each session: final next-click target for test, penultimate for validation; earlier prefix examples for training.',
              'scope': 'Existing 20k-session formal subset, full catalogue Top-10; not full raw Diginetica or official leaderboard.',
              'created_at': datetime.now().isoformat(timespec='seconds'),
              'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__, 'device': 'cpu'}}
    write_json(OUT / 'result.json', result)
    comparisons, old_predictions, evidence = [], {}, {}
    for m in MODELS:
        path = ROOT / f'outputs/formal/predictions/{m}.jsonl'
        records = [json.loads(line) for line in path.open(encoding='utf-8')]
        assert len(records) == len(rows_by_session)
        assert len({r['session_id'] for r in records}) == len(records)
        all_ranks, seen_ranks, new_ranks = [], [], []
        old_predictions[m] = {}
        for row in records:
            reference = rows_by_session[row['session_id']]
            assert row['history'] == reference['history'], (m, row['session_id'], 'history mismatch')
            assert row['target'] == reference['target']
            rec_ids = [r['item'] for r in row['recommendations']]
            rank = rec_ids.index(row['target']) + 1 if row['target'] in rec_ids else 0
            assert rank == row['target_rank']
            all_ranks.append(rank)
            (seen_ranks if reference['target_in_history'] else new_ranks).append(rank)
            old_predictions[m][row['session_id']] = rank
        measured = ranking_metrics(all_ranks)
        for key in measured:
            assert abs(measured[key] - prior[m]['test_result'][key]) <= 0.000051, (m, key)
        evidence[m] = {'prediction_sha256': sha(path), 'result_sha256': sha(ROOT / f'outputs/formal/{m}/seed42/result.json'),
                       'prediction_rows': len(records), 'history_and_target_match': True,
                       'stored_metrics_recomputed_within_rounding': True}
        comparisons.append({'model': m, **measured,
                            'repeat_target_recall@10': ranking_metrics(seen_ranks)['recall@10'],
                            'new_target_recall@10': ranking_metrics(new_ranks)['recall@10'],
                            'historical_fit_plus_validation_seconds': prior[m]['training_and_validation_seconds']})
    gra_rows = list(rows_by_session.values())
    comparisons.append({'model': 'GRA', **result['test_result'],
                        'repeat_target_recall@10': ranking_metrics([r['target_rank'] for r in gra_rows if r['target_in_history']])['recall@10'],
                        'new_target_recall@10': ranking_metrics([r['target_rank'] for r in gra_rows if not r['target_in_history']])['recall@10'],
                        'historical_fit_plus_validation_seconds': None})
    with (OUT / 'comparison.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    cases = {}
    for row in gra_rows:
        old_ranks = {m: old_predictions[m][row['session_id']] for m in MODELS}
        kind = ('gra_hit_gce_miss' if row['target_rank'] and not old_ranks['GCEGNN'] else
                'gra_miss_gce_hit' if not row['target_rank'] and old_ranks['GCEGNN'] else None)
        if kind and kind not in cases:
            cases[kind] = {'gra': row, 'gnn_target_ranks': old_ranks}
    write_json(OUT / 'cases.json', cases)
    verification = {'all_passed': True, 'analytic_self_checks': True, 'exact_split_hashes': True,
                    'training_prefixes_exclude_validation_and_test_positions': True,
                    'model_save_reload_three_queries_exact': True,
                    'all_top10_legal': True, 'gnn_evidence': evidence,
                    'source_data_sha256': sha(data_file), 'script_sha256': sha(__file__),
                    'model_code_sha256': sha(Path(__file__).with_name('grey_session.py')),
                    'gra_prediction_sha256': sha(OUT / 'GRA.jsonl'),
                    'repeat_target_cases': sum(r['target_in_history'] for r in gra_rows),
                    'new_target_cases': sum(not r['target_in_history'] for r in gra_rows)}
    write_json(OUT / 'verification.json', verification)
    lines = ['# 灰色关联推荐与三个图神经网络的实测比较', '',
             '本次新增固定参数的灰色关联推荐方法（GRA），复用原来三个模型的已保存测试预测，并逐条核对历史序列、真实下一点击和指标。没有重训三个模型。', '',
             '## 数据与比较协议', '',
             '- 数据：已有 Diginetica 正式子集，20,000 个会话、32,968 个候选商品；不是完整原始数据上的官方基准复现。',
             '- 划分：同一会话最后一次点击作为测试标签、倒数第二次作为验证标签，其余较早前缀作为训练样本；97,273 个训练样本、20,000 个验证样本、20,000 个测试样本。',
             '- 所有方法对全部候选商品排序，排除内部 padding=0，允许推荐浏览过的商品。每个样本只有一个真实下一点击。',
             '- 三个模型的训练/验证/测试会话与标签指纹完全一致；保存的测试浏览历史与 GRA 逐条一致。',
             '- GRA 从每个会话最长训练前缀统计一次，避免重复累计重叠前缀；拟合排除验证和测试位置，预测可使用当前完整浏览历史。', '',
             '## 灰色关联方法的实现', '',
             '输入浏览序列 → 计算四项指标 → 全候选商品范围归一化 → 与理想参考向量比较 → 等权汇总灰色关联系数 → Top-10。', '',
             '1. 最近点击转移：训练前缀中，最后点击商品转移到各候选商品的经验概率。',
             '2. 历史共现：训练会话前缀的商品共现次数除以两商品会话频次的几何平均（不含对角自共现），按当前历史位置 1,…,L 加权平均。',
             '3. 会话内重复点击：当前历史中各商品的位置权重之和，位置越靠后权重越大。',
             '4. 商品热度：训练前缀商品点击次数的 log(1+次数)。', '',
             '四项均为效益指标；对每项在全部非 padding 候选上做 min-max 归一化，常数指标设为零。理想参考向量为 (1,1,1,1)。', '',
             'Δ(i,k)=|1−z(i,k)|；ξ(i,k)=(Δmin+ρΔmax)/(Δ(i,k)+ρΔmax)；score(i)=Σₖ wₖξ(i,k)。极值范围是当前查询的全部候选×四项指标。', '',
             '固定 ρ=0.5，四项权重均为 0.25；未按验证或测试结果调参。分数相同时按内部商品编号升序确定顺序。该方法为本项目设计的 GRA 多指标排序基线，包含人工构造的行为特征；未使用 GM(1,1) 灰预测。', '',
             '## 测试结果', '',
             '| 方法 | Recall@10 | MRR@10 | NDCG@10 | 重复目标 Recall@10 | 未见于当前历史的目标 Recall@10 |',
             '|---|---:|---:|---:|---:|---:|']
    for c in comparisons:
        lines.append('| {model} | {recall@10:.5f} | {mrr@10:.5f} | {ndcg@10:.5f} | {repeat_target_recall@10:.5f} | {new_target_recall@10:.5f} |'.format(**c))
    lines.extend(['', 'Recall@10 表示真实下一点击进入前十的比例；MRR@10 对靠前命中奖励更高；NDCG@10 对名次做对数折扣。', '',
                  f"重复目标样本 {verification['repeat_target_cases']:,} 条；未出现在当前历史中的目标样本 {verification['new_target_cases']:,} 条。分组仅描述已有预测，不据此调参。", '',
                  '## 运行成本与解释边界', '',
                  f'- GRA 训练统计构建：{fit_seconds:.3f} 秒；测试打分与取前十：{predict_seconds:.3f} 秒，平均 {result["mean_scoring_ms_per_session"]:.3f} 毫秒/会话（CPU，排除 JSON 写出与公共数据准备）。',
                  '- 三个 GNN 的历史训练加验证耗时保存在 comparison.csv，包含多轮优化与早停验证；与 GRA 一次统计构建的工作内容不同，不据此直接宣称统一倍数加速。',
                  '- GRA 稀疏统计构建包含会话商品两两共现，约 O(Σ L_s²)；一次预测约 O(dM+Σ deg(v))，其中 d=4、M 为商品数，线性选择前十另需 O(M)。',
                  '- 此次只测试一个固定 GRA 设计、同一划分和原有单种子 GNN。结果用于比较本次准确率与方法成本，不能单独证明灰关联机制相对普通加权排序的增益；该归因需要同特征的对照。',
                  '- 原项目的全量商品频次筛选、子集抽样和词表均先于留末划分；本次沿用该协议，不能解释为按全局时间隔离的线上未来评估。', '',
                  '## 文件与复跑', '',
                  '- `result.json`：GRA 实测指标、参数、时长和划分指纹。',
                  '- `comparison.csv`：四方法统一重新核算的测试指标。',
                  '- `GRA.jsonl`：20,000 条逐会话 Top-10，附归一化指标与灰色关联系数。',
                  '- `grey_model.npz`、`item_tokens.json`：可重载统计模型与商品编号映射。',
                  '- `cases.json`：按测试顺序各取第一个 GRA 命中/GCE 未命中和反向例子，用于说明具体成功与失败。',
                  '- `verification.json`：划分、逐条对齐、解析检查、重载和文件哈希。', '',
                  '在 `03_代码与实验` 目录运行：', '', '```powershell',
                  "& 'D:\\Envs\\recbole-gnn-py39\\Scripts\\python.exe' experiments/run_grey.py", '```', ''])
    (OUT / '灰色关联推荐对比结果.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'result': result, 'comparison': comparisons, 'all_checks_passed': True}, ensure_ascii=True, indent=2), flush=True)


if __name__ == '__main__':
    main()
