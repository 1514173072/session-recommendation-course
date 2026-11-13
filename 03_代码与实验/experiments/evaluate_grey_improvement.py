"""One full test evaluation after validation configuration is frozen."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from grey_session import ranking_metrics
from grey_session_fast import FastGreySession, FEATURES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/grey_improved'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    selection = json.loads((OUT / 'selection.json').read_text(encoding='utf-8'))
    frozen = digest(OUT / 'model_selected/config.json')
    assert frozen == selection['selected_config_sha256']
    model = FastGreySession.load(OUT / 'model_selected')
    with np.load(OUT / 'splits.npz', allow_pickle=False) as d:
        rows = {k: d['test_' + k] for k in ['users', 'histories', 'lengths', 'targets']}
    tokens = json.loads((OUT / 'tokens.json').read_text(encoding='utf-8'))
    ranks, records, supports = [], {}, []
    predict_seconds = 0
    with (OUT / 'GRA-improved.jsonl').open('w', encoding='utf-8') as f:
        for i, (u, history, length, target) in enumerate(zip(rows['users'], rows['histories'], rows['lengths'], rows['targets'])):
            history = history[:length]
            start = time.perf_counter()
            ids, scores, detail = model.predict(history, int(u), return_detail=True)
            predict_seconds += time.perf_counter() - start
            supports.append(detail['support'])
            match = np.flatnonzero(ids == target)
            rank = int(match[0] + 1) if len(match) else 0
            ranks.append(rank)
            assert len(ids) == 10 and len(set(ids.tolist())) == 10 and np.all(ids > 0)
            row = {'model': 'GRA-improved', 'session_id': str(tokens['users'][u]),
                   'history': [str(tokens['items'][x]) for x in history],
                   'target': str(tokens['items'][target]), 'target_rank': rank,
                   'target_in_history': bool(target in history), 'support_items': detail['support'],
                   'recommendations': [{'rank': j + 1, 'item': str(tokens['items'][x]), 'score': float(scores[j])}
                                       for j, x in enumerate(ids)]}
            records[row['session_id']] = row
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
            if (i+1) % 4000 == 0:
                print(f'TEST {i+1}/20000; scoring {predict_seconds:.2f}s', flush=True)
    rng = np.random.default_rng(20261111)
    checks = rng.choice(len(ranks), 100, replace=False)
    for i in checks:
        args = (rows['histories'][i, :rows['lengths'][i]], int(rows['users'][i]))
        a, av = model.predict(*args)
        b, bv = model.predict(*args, dense=True)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(av, bv, atol=1e-7, rtol=0)
    reloaded = FastGreySession.load(OUT / 'model_selected')
    for i in checks[:3]:
        args = (rows['histories'][i, :rows['lengths'][i]], int(rows['users'][i]))
        a, av = model.predict(*args)
        b, bv = reloaded.predict(*args)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(av, bv)
    comparisons = []
    evidence = {}
    for name, path in [
        ('SRGNN', ROOT / 'outputs/formal/predictions/SRGNN.jsonl'),
        ('GCSAN', ROOT / 'outputs/formal/predictions/GCSAN.jsonl'),
        ('GCEGNN', ROOT / 'outputs/formal/predictions/GCEGNN.jsonl'),
        ('GRA-original', ROOT / 'outputs/grey_relation/GRA.jsonl')]:
        original, seen = [], set()
        repeat, novel = [], []
        for line in path.open(encoding='utf-8'):
            r = json.loads(line)
            ref = records[r['session_id']]
            assert r['history'] == ref['history'] and r['target'] == ref['target']
            assert r['session_id'] not in seen
            seen.add(r['session_id'])
            items = [v['item'] for v in r['recommendations']]
            rank = items.index(r['target'])+1 if r['target'] in items else 0
            assert rank == r['target_rank']
            original.append(rank)
            (repeat if ref['target_in_history'] else novel).append(rank)
        assert len(seen) == len(records) == 20000
        comparisons.append({'model': name, **ranking_metrics(original),
                            'repeat_target_recall@10': ranking_metrics(repeat)['recall@10'],
                            'novel_target_recall@10': ranking_metrics(novel)['recall@10']})
        evidence[name] = {'rows': len(seen), 'history_target_aligned': True, 'sha256': digest(path)}
    newmetrics = ranking_metrics(ranks)
    values = list(records.values())
    comparisons.append({'model': 'GRA-improved', **newmetrics,
                        'repeat_target_recall@10': ranking_metrics([r['target_rank'] for r in values if r['target_in_history']])['recall@10'],
                        'novel_target_recall@10': ranking_metrics([r['target_rank'] for r in values if not r['target_in_history']])['recall@10']})
    result = {'model': 'GRA-improved', 'test_result': newmetrics,
              'test_examples': len(ranks), 'candidate_items': model.item_num-1,
              'selected_config_sha256_before_and_after': frozen,
              'selection_criterion': selection['criterion'], 'test_used_for_selection': False,
              'test_scoring_seconds': predict_seconds,
              'test_mean_scoring_ms': predict_seconds / len(ranks) * 1000,
              'mean_exact_support_items': float(np.mean(supports)),
              'p95_exact_support_items': float(np.percentile(supports, 95)),
              'numeric_state_bytes': model.array_bytes(), 'comparison': comparisons,
              'repeat_target_cases': sum(r['target_in_history'] for r in values),
              'novel_target_cases': sum(not r['target_in_history'] for r in values)}
    assert frozen == digest(OUT / 'model_selected/config.json')
    (OUT / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    verification = {'all_passed': True, 'frozen_config_matches_selection': True,
                    'test_sparse_dense_top10_checks': len(checks),
                    'independent_reload_three_queries_exact': True,
                    'all_top10_legal': True, 'all_comparison_history_targets_aligned': evidence,
                    'predictions_sha256': digest(OUT/'GRA-improved.jsonl'),
                    'code_sha256': {p.name: digest(p) for p in [Path(__file__), Path(__file__).with_name('grey_session_fast.py'), Path(__file__).with_name('tune_grey_improvement.py')]}}
    (OUT / 'verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
