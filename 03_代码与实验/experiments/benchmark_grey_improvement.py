"""Isolated-process batch=1 warm response benchmark on identical raw histories.

Includes token mapping, query graph/features, scoring ALL possible outputs,
Top-10 and conversion back to raw item strings; excludes startup and file I/O.
No HTTP server or queue overhead is included.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import gc

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/grey_improved'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True,
                        choices=['GRA-improved', 'GRA-original', 'SRGNN', 'GCSAN', 'GCEGNN'])
    parser.add_argument('--queries', type=int, default=300)
    parser.add_argument('--rounds', type=int, default=3)
    args = parser.parse_args()
    sys.argv = sys.argv[:1]
    import torch
    torch.set_num_threads(4)
    tokens = json.loads((OUT / 'tokens.json').read_text(encoding='utf-8'))
    token2id = {str(token): i for i, token in enumerate(tokens['items'])}
    with np.load(OUT / 'splits.npz', allow_pickle=False) as d:
        rng = np.random.default_rng(20261111)
        selected = rng.choice(len(d['test_users']), args.queries, replace=False)
        users = d['test_users'][selected].copy()
        raw_queries = [[str(tokens['items'][x]) for x in h[:n]]
                       for h, n in zip(d['test_histories'][selected], d['test_lengths'][selected])]
    sample_sha = hashlib.sha256(json.dumps(list(zip(users.tolist(), raw_queries)), sort_keys=True).encode()).hexdigest()
    load_start = time.perf_counter()
    if args.model == 'GRA-improved':
        from grey_session_fast import FastGreySession
        model = FastGreySession.load(OUT / 'model_selected')
        numeric_bytes = model.array_bytes()
        def predict(raw, user):
            history = np.asarray([token2id[str(x)] for x in raw], dtype=np.int64)
            ids, scores = model.predict(history, int(user))
            return [{'rank': i+1, 'item': str(tokens['items'][x]), 'score': float(scores[i])}
                    for i, x in enumerate(ids)]
        source = OUT / 'GRA-improved.jsonl'
    elif args.model == 'GRA-original':
        from grey_session import GreySession
        model = GreySession.load(ROOT / 'outputs/grey_relation/grey_model.npz')
        numeric_bytes = sum(a.nbytes for a in [model.counts, model.log_popularity, model.weights])
        for matrix in [model.cooccurrence, model.transitions]:
            numeric_bytes += matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
        def predict(raw, user):
            history = np.asarray([token2id[str(x)] for x in raw], dtype=np.int64)
            ids, scores, _, _ = model.recommend(history)
            return [{'rank': i+1, 'item': str(tokens['items'][x]), 'score': float(scores[x])}
                    for i, x in enumerate(ids)]
        source = ROOT / 'outputs/grey_relation/GRA.jsonl'
    else:
        sys.path.insert(0, str(ROOT.parent / 'app'))
        from predictor import SessionPredictor
        adapter = SessionPredictor(args.model)
        model = adapter.model
        tensors = list(model.parameters()) + list(model.buffers())
        for name in getattr(model, 'other_parameter_name', []):
            tensors.append(getattr(model, name))
        unique = {t.data_ptr(): t for t in tensors if torch.is_tensor(t)}
        numeric_bytes = sum(t.numel() * t.element_size() for t in unique.values())
        def predict(raw, user):
            return adapter.predict(raw)
        source = ROOT / f'outputs/formal/predictions/{args.model}.jsonl'
    startup_seconds = time.perf_counter() - load_start
    wanted_sessions = {str(tokens['users'][u]) for u in users}
    prior = {}
    for line in source.open(encoding='utf-8'):
        row = json.loads(line)
        if row['session_id'] in wanted_sessions:
            prior[row['session_id']] = row
    assert len(prior) == len(raw_queries)
    max_score_diff = 0
    for raw, user in zip(raw_queries, users):
        prediction = predict(raw, user)
        expected = prior[str(tokens['users'][user])]['recommendations']
        assert [r['item'] for r in prediction] == [r['item'] for r in expected], (args.model, user)
        diff = max(abs(a['score']-b['score']) for a, b in zip(prediction, expected))
        max_score_diff = max(max_score_diff, diff)
        assert diff < 1e-4, (args.model, user, diff)
    # Explicit warmup after correctness; disable collection equally in all models.
    for raw, user in zip(raw_queries[:30], users[:30]):
        predict(raw, user)
    samples, per_round = [], []
    gc.collect()
    gc.disable()
    try:
        for r in range(args.rounds):
            order = np.random.default_rng(20261111+r).permutation(len(users))
            times = []
            for i in order:
                begin = time.perf_counter_ns()
                prediction = predict(raw_queries[i], users[i])
                elapsed = (time.perf_counter_ns() - begin) / 1e6
                assert len(prediction) == 10
                times.append(elapsed)
            samples.extend(times)
            per_round.append({'mean_ms': float(np.mean(times)), 'median_ms': float(np.median(times)),
                              'p95_ms': float(np.percentile(times, 95))})
            print(f'{args.model} round {r+1}: {per_round[-1]}', flush=True)
    finally:
        gc.enable()
    result = {'model': args.model, 'batch_size': 1, 'device': 'cpu',
              'torch_threads': torch.get_num_threads(), 'distinct_queries': len(users),
              'rounds': args.rounds, 'timed_calls': len(samples),
              'sample_sha256': sample_sha, 'model_startup_seconds_excluded': startup_seconds,
              'mean_ms': float(np.mean(samples)), 'median_ms': float(np.median(samples)),
              'p95_ms': float(np.percentile(samples, 95)), 'p99_ms': float(np.percentile(samples, 99)),
              'numeric_state_bytes': numeric_bytes, 'per_round': per_round,
              'correctness_queries': len(prior), 'max_score_diff_from_saved_predictions': max_score_diff,
              'sample_latencies_ms': samples,
              'scope': 'Warm in-process raw-history -> raw Top-10 latency, batch=1; includes input mapping, query features/graph, ranking and output mapping; excludes model startup, disk I/O, HTTP and queueing.',
              'memory_scope': 'Numeric model arrays/tensors, including sparse indexes and GCE global adjacency; excludes Python objects, shared token maps, dataset objects and runtime libraries.',
              'platform': platform.platform(), 'processor': platform.processor(),
              'python': platform.python_version(), 'torch': torch.__version__,
              'created_unix_time': time.time()}
    directory = OUT / 'latency'
    directory.mkdir(exist_ok=True)
    (directory / f'{args.model}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'sample_latencies_ms'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
