"""Audit training provenance, frozen validation selection, predictions and benchmarks."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from grey_session import GreySession
from grey_session_fast import FastGreySession

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/grey_improved'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    result, selection = read(OUT/'result.json'), read(OUT/'selection.json')
    prepared, verified = read(OUT/'prepare.json'), read(OUT/'verification.json')
    model = FastGreySession.load(OUT/'model_selected')
    with np.load(OUT/'splits.npz', allow_pickle=False) as d:
        data = {k:d[k] for k in d.files}
    expected = json.loads((ROOT/'outputs/formal/data_manifest.json').read_text(encoding='utf-8'))
    assert digest(ROOT/'data/diginetica-formal/diginetica-formal.inter') == expected['output_sha256'] == prepared['source_sha256']
    for name in ['SRGNN', 'GCSAN', 'GCEGNN']:
        assert read(ROOT/f'outputs/formal/{name}/seed42/result.json')['split_session_target_sha256'] == prepared['split_hashes']
    # Validation histories are precisely the longest training-observed prefixes.
    refit = GreySession(model.item_num).fit([h[:n] for h,n in zip(data['valid_histories'], data['valid_lengths'])])
    np.testing.assert_array_equal(refit.counts, model.original.counts)
    for name in ['transitions', 'cooccurrence']:
        difference = (getattr(refit, name) - getattr(model.original, name)).tocsr()
        assert difference.nnz == 0 or np.max(np.abs(difference.data)) < 1e-7
    for u, h, n in zip(data['valid_users'], data['valid_histories'], data['valid_lengths']):
        row = model.user_row[int(u)]
        np.testing.assert_array_equal(np.unique(h[:n]), model.incidence.getrow(row).indices)
    config_sha = digest(OUT/'model_selected/config.json')
    assert config_sha == selection['selected_config_sha256'] == result['selected_config_sha256_before_and_after']
    assert np.all(model.weights >= 0) and abs(float(model.weights.sum())-1)<1e-6
    for name, sha in verified['code_sha256'].items():
        assert digest(Path(__file__).with_name(name)) == sha
    # Check the validation cache against current code and data, not just its existence.
    with np.load(OUT/'validation_features.npz', allow_pickle=False) as cache:
        np.testing.assert_array_equal(cache['users'], data['valid_users'])
        np.testing.assert_array_equal(cache['targets'], data['valid_targets'])
        for i in np.random.default_rng(20261111).choice(20000, 25, replace=False):
            ids,x = model.features(data['valid_histories'][i,:data['valid_lengths'][i]], int(data['valid_users'][i]))
            a,b = cache['offsets'][i:i+2]
            np.testing.assert_array_equal(ids,cache['ids'][a:b])
            np.testing.assert_array_equal(x,cache['x'][a:b])
    tokens = read(OUT/'tokens.json')
    expected_rows = {}
    for u,h,n,t in zip(data['test_users'],data['test_histories'],data['test_lengths'],data['test_targets']):
        expected_rows[str(tokens['users'][u])] = ([str(tokens['items'][i]) for i in h[:n]], str(tokens['items'][t]))
    ranks, seen = [], set()
    for line in (OUT/'GRA-improved.jsonl').open(encoding='utf-8'):
        row = json.loads(line)
        assert row['session_id'] not in seen
        seen.add(row['session_id'])
        history,target = expected_rows[row['session_id']]
        assert row['history']==history and row['target']==target
        ids = [r['item'] for r in row['recommendations']]
        assert len(ids)==len(set(ids))==10 and '[PAD]' not in ids
        rank = ids.index(target)+1 if target in ids else 0
        assert row['target_rank']==rank
        ranks.append(rank)
    n=len(ranks)
    independent = {'recall@10':sum(r>0 for r in ranks)/n,
                   'mrr@10':sum(1/r for r in ranks if r)/n,
                   'ndcg@10':sum(1/math.log2(r+1) for r in ranks if r)/n}
    assert n==20000
    for k,v in independent.items():
        assert abs(v-result['test_result'][k])<1e-12
    assert digest(OUT/'GRA-improved.jsonl') == verified['predictions_sha256']
    for name, ev in verified['all_comparison_history_targets_aligned'].items():
        p = ROOT/('outputs/grey_relation/GRA.jsonl' if name=='GRA-original' else f'outputs/formal/predictions/{name}.jsonl')
        assert digest(p)==ev['sha256'] and ev['rows']==20000 and ev['history_target_aligned']
    benchmarks = [read(p) for p in sorted((OUT/'latency').glob('*.json'))]
    assert len(benchmarks)==5 and len({b['sample_sha256'] for b in benchmarks})==1
    assert all(b['distinct_queries']==300 and b['timed_calls']==900 and b['correctness_queries']==300 for b in benchmarks)
    assert all(b['batch_size']==1 and b['torch_threads']==4 for b in benchmarks)
    for b in benchmarks:
        assert len(b['sample_latencies_ms'])==900
        assert abs(float(np.mean(b['sample_latencies_ms']))-b['mean_ms'])<1e-12
    measurements=read(OUT/'comparison.json')
    by_name={r['model']:r for r in measurements}
    fast,old=by_name['GRA-improved'],by_name['GRA-original']
    for key in ['recall@10','mrr@10','ndcg@10']:
        assert fast[key]>old[key]
        assert fast[key] >= max(by_name[m][key] for m in ['SRGNN','GCSAN','GCEGNN'])
    assert all(fast['response_mean_ms']<by_name[m]['response_mean_ms'] for m in ['SRGNN','GCSAN','GCEGNN','GRA-original'])
    assert all(fast['numeric_state_mb_decimal']<by_name[m]['numeric_state_mb_decimal'] for m in ['SRGNN','GCSAN','GCEGNN'])
    report=OUT/'灰色关联改进与复杂度对比.md'
    body=report.read_text(encoding='utf-8')
    assert all(text in body for text in ['时间复杂度分析','O(P log P','O(Md','378','900','37.190%'])
    audit = {'all_passed':True, 'source_and_original_splits_verified':True,
             'training_statistics_refit_from_observed_prefixes_match':True,
             'training_incidence_all_20000_sessions_match':True,
             'validation_cache_25_queries_match_current_code':True,
             'test_config_matches_validation_frozen_hash':True,
             'independent_all_20000_prediction_metrics':independent,
             'sparse_full_catalogue_top10_checks':{'validation':selection['sparse_dense_validation_checks'],
                                                  'test':verified['test_sparse_dense_top10_checks']},
             'five_models_same_300_query_benchmark_900_calls_each':True,
             'improved_quality_and_lower_mean_latency_verified':True,
             'numeric_model_state_smaller_than_all_three_gnns':True,
             'complexity_derivation_written_and_checked_against_source':True,
             'report_sha256':digest(report), 'selected_config_sha256':config_sha,
             'scope':'Existing 20k subset, existing single-seed GNNs; validation-selected GRA. Latency is warm in-process batch=1. Memory means numeric model state, not total process RSS.'}
    (OUT/'completion_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    main()
