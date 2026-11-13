"""Extend the specified saved-session demo; cache-mode inference only, no LLM API."""
from pathlib import Path
import hashlib
import json
import re
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TARGET = ROOT / '04_报告与展示/课堂讨论_PPT交付材料/06_实机演示/会话推荐演示.html'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def main():
    original = TARGET.read_text(encoding='utf-8')
    pattern = r'(<script id="demo-data" type="application/json">)(.*?)(</script>)'
    data = json.loads(re.search(pattern, original, re.S)[2])
    assert len(data['rows']) == 20000
    assert all(len(row) in (6, 9) for row in data['rows'])
    # The current demo retains the original three predictions in its first six columns.
    data['rows'] = [row[:6] for row in data['rows']]
    rows = {row[0]: row for row in data['rows']}
    assert len(rows) == 20000
    source_paths = [TARGET,
        ROOT / '06_教师学生推荐性能测试/predictions/gra_original.jsonl',
        ROOT / '06_教师学生推荐性能测试/predictions/gra_beta7.jsonl',
        ROOT / '10_本地2B低延迟优化/results/offline_llm_policy_cache.json',
        ROOT / '03_代码与实验/outputs/grey_relation/grey_model.npz',
        ROOT / '03_代码与实验/outputs/grey_improved/tokens.json',
        ROOT / '06_教师学生推荐性能测试/teacher/prototypes.json']
    hashes = {str(path): digest(path) for path in source_paths}
    for path in source_paths[1:3]:
        seen = set()
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                pred = json.loads(line)
                row = rows[pred['session_id']]
                assert pred['session_id'] not in seen
                seen.add(pred['session_id'])
                assert row[1] == pred['history'] and row[2] == pred['target']
                assert data['rows'][pred['test_index']][0] == row[0]
                row.append([item['item'] for item in pred['recommendations']])
        assert len(seen) == len(rows)

    sys.path.insert(0, str(ROOT / '10_本地2B低延迟优化'))
    import predict
    import numpy as np
    predict.opt.torch.set_num_threads(4)
    # Any accidental model-service invocation must fail rather than create calls.
    def reject_network(*args, **kwargs):
        raise RuntimeError('This build only permits offline cached-policy inference')
    predict.opt.post = reject_network
    recommender = predict.OptimizedRecommender('cache')
    expected = json.loads((ROOT / '10_本地2B低延迟优化/results/cache_comparison.json').read_text(encoding='utf-8'))
    expected_ranks = next(r['ranks'] for r in expected if r['split'] == 'test20000')
    offline_ranks = []
    started = time.perf_counter()
    with (HERE / '离线Qwen_20000条展示预测.jsonl').open('w', encoding='utf-8') as stream:
        for index, row in enumerate(data['rows']):
            pred = recommender.recommend(row[1])
            assert pred['online_2b_calls'] == 0
            ids = [item['item'] for item in pred['recommendations']]
            rank = ids.index(row[2]) + 1 if row[2] in ids else 0
            assert rank == expected_ranks[index]
            offline_ranks.append(rank)
            row.append(ids)
            stream.write(json.dumps({'session_id': row[0], 'history': row[1], 'target': row[2],
                'recommendations': ids, 'target_rank': rank, 'policy_state': pred['policy_state'],
                'policy_label': pred['policy_label'], 'online_2b_calls': 0}, ensure_ascii=False) + '\n')
            if (index + 1) % 5000 == 0:
                print(f'Cached-policy predictions: {index + 1}/20000', flush=True)
    known = set(recommender.tokens[1:])
    for row in data['rows']:
        assert len(row) == 9
        for ids in row[3:]:
            assert len(ids) == 10 and len(set(ids)) == 10
            assert set(ids) <= known
    base_data = json.loads(re.search(pattern, original, re.S)[2])
    assert all(row[:6] == before[:6] for row, before in zip(data['rows'], base_data['rows']))
    data['addedMethods'] = ['灰色关联', '改进灰色关联（β=7）', '离线qwen3.5-2B']
    data['offlineQwenNote'] = '复用已预计算策略，逐会话计算全商品Top10；展示不调用模型服务。'
    prefix = original[:original.index('<script id="demo-data"')]
    prefix = prefix.replace('三模型会话推荐对比', '六种方法会话推荐对比')
    prefix = prefix.replace('三个模型全部命中', '所选方法全部命中').replace('三个模型全部未命中', '所选方法全部未命中')
    prefix = prefix.replace('部分模型命中', '所选方法部分命中')
    prefix = prefix.replace('仅 GCEGNN 命中', '所选方法中仅 GCE-GNN 命中')
    prefix = prefix.replace('其他模型命中、GCEGNN 未命中', '所选其他方法命中、GCE-GNN 未命中')
    extra_css = (HERE / 'extra.css').read_text(encoding='utf-8')
    if extra_css not in prefix:
        prefix = prefix.replace('</style>', extra_css + '\n</style>')
    controls = '''  <fieldset class="method-picker"><legend>选择比较方法</legend><div id="methodChoices"></div></fieldset>
'''
    if 'class="method-picker"' not in prefix:
        prefix = prefix.replace('  <div class="session-control">', controls + '  <div class="session-control">')
    embedded = json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
    js = (HERE / 'demo.js').read_text(encoding='utf-8')
    output = prefix + '<script id="demo-data" type="application/json">' + embedded + '</script>\n<script>\n' + js + '\n</script></body></html>\n'
    assert all(digest(Path(path)) == sha for path, sha in hashes.items())
    TARGET.write_text(output, encoding='utf-8')
    report = {'passed': True, 'sessions': len(rows), 'methods': 6,
        'original_three_predictions_unchanged': True, 'grey_histories_targets_aligned': 20000,
        'offline_ranks_match_existing_20000': True, 'online_llm_calls': 0,
        'all_top10_legal_unique': True, 'offline_metrics': predict.opt.ranking_metrics(offline_ranks),
        'build_prediction_seconds': time.perf_counter() - started,
        'sources_sha256': hashes, 'target': str(TARGET), 'target_sha256': digest(TARGET)}
    save(HERE / 'build_check.json', report)
    print(json.dumps({key: value for key, value in report.items() if key != 'sources_sha256'}, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
