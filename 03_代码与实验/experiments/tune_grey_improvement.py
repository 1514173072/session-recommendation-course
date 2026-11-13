"""Validation-only feature/weight selection. This script never loads test arrays."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from grey_session import ranking_metrics
from grey_session_fast import FastGreySession, FEATURES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/grey_improved'


def load_valid():
    with np.load(OUT / 'splits.npz', allow_pickle=False) as d:
        return {key: d['valid_' + key] for key in ['users', 'histories', 'lengths', 'targets']}


def feature_cache(model, rows, path):
    arrays, item_ids, offsets, targets, target_index, users = [], [], [0], [], [], []
    start = time.perf_counter()
    for i, (user, history, length, target) in enumerate(zip(rows['users'], rows['histories'], rows['lengths'], rows['targets'])):
        ids, x = model.features(history[:length], int(user))
        match = np.flatnonzero(ids == target)
        target_index.append(int(match[0]) + offsets[-1] if len(match) else -1)
        arrays.append(x)
        item_ids.append(ids)
        targets.append(target)
        users.append(user)
        offsets.append(offsets[-1] + len(ids))
        if (i + 1) % 4000 == 0:
            print(f'VALID features {i + 1}/{len(rows["users"])} in {time.perf_counter() - start:.1f}s', flush=True)
    data = {'x': np.concatenate(arrays), 'ids': np.concatenate(item_ids),
            'offsets': np.asarray(offsets), 'targets': np.asarray(targets),
            'target_index': np.asarray(target_index), 'users': np.asarray(users)}
    np.savez_compressed(path, **data)
    return data


def subset(data, rows):
    pieces = [np.arange(data['offsets'][i], data['offsets'][i + 1]) for i in rows]
    index = np.concatenate(pieces)
    offsets = np.concatenate(([0], np.cumsum([len(p) for p in pieces])))
    target_index = np.array([data['target_index'][i] - data['offsets'][i] + offsets[j]
                             if data['target_index'][i] >= 0 else -1 for j, i in enumerate(rows)])
    return {'x': data['x'][index], 'ids': data['ids'][index], 'offsets': offsets,
            'target_index': target_index, 'targets': data['targets'][rows]}


def evaluate(data, coefficients, weights):
    scores = np.sum(coefficients * np.asarray(weights, dtype=np.float32)[None, :], axis=1)
    ti = data['target_index']
    present = ti >= 0
    target_scores = scores[np.maximum(ti, 0)]
    counts = np.diff(data['offsets'])
    repeated_scores = np.repeat(target_scores, counts)
    repeated_ids = np.repeat(data['targets'], counts)
    greater = (scores > repeated_scores) | ((scores == repeated_scores) & (data['ids'] < repeated_ids))
    ranks = 1 + np.add.reduceat(greater.astype(np.int32), data['offsets'][:-1])
    ranks[(~present) | (ranks > 10)] = 0
    return ranking_metrics(ranks)


def weight_options():
    options = [np.array([.25, .25, .25, .25, 0, 0])]
    # Structured combinations retain a transparent repeat / association balance.
    for repeat in [.15, .3, .45]:
        for neighbor in [.15, .35, .55]:
            remaining = 1 - repeat - neighbor
            if remaining >= .05:
                options.append(np.array([remaining * .3, remaining * .5, repeat, remaining * .1, neighbor, remaining * .1]))
    for repeat in [.2, .35, .5]:
        options.append(np.array([(1-repeat)*.4, (1-repeat)*.5, repeat, (1-repeat)*.1, 0, 0]))
        options.append(np.array([(1-repeat)*.2, (1-repeat)*.25, repeat, 0, (1-repeat)*.35, (1-repeat)*.2]))
    rng = np.random.default_rng(20261111)
    for _ in range(48):
        w = rng.dirichlet([1.1, 1.8, 2.0, .25, 2.0, .8])
        options.append(w)
    return [w / w.sum() for w in options]


def main():
    start = time.perf_counter()
    model = FastGreySession.load(OUT / 'model_initial')
    rows = load_valid()
    cache_path = OUT / 'validation_features.npz'
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as d:
            data = {k: d[k] for k in d.files}
    else:
        data = feature_cache(model, rows, cache_path)
    # Match sparse implementation to full-catalogue scoring on validation queries.
    rng = np.random.default_rng(20261111)
    check_rows = rng.choice(len(rows['users']), 60, replace=False)
    equality = []
    for i in check_rows:
        history, user = rows['histories'][i, :rows['lengths'][i]], int(rows['users'][i])
        for weights in [np.array([.25, .25, .25, .25, 0, 0]), np.array([.1,.15,.3,0,.35,.1])]:
            model.weights = weights.astype(np.float32)
            sparse_ids, sparse_scores = model.predict(history, user)
            dense_ids, dense_scores = model.predict(history, user, dense=True)
            np.testing.assert_array_equal(sparse_ids, dense_ids)
            np.testing.assert_allclose(sparse_scores, dense_scores, atol=1e-7, rtol=0)
            equality.append(True)
    screening_rows = rng.choice(len(rows['users']), 4000, replace=False)
    small = subset(data, screening_rows)
    trials = []
    weights = weight_options()
    # All choices are screened ONLY on validation; test arrays are never accessed.
    for power in [.5, 1.0]:
        for rho in [.25, .5, 1.0]:
            coefficients = model.coefficients(small['x'], rho, power)
            for wi, w in enumerate(weights):
                metrics = evaluate(small, coefficients, w)
                trials.append({'rho': rho, 'power': power, 'weight_index': wi,
                               'weights': w.tolist(), 'screen_valid': metrics})
            print(f'VALID screening power={power} rho={rho} done; {time.perf_counter()-start:.1f}s', flush=True)
    trials.sort(key=lambda c: (c['screen_valid']['mrr@10'], c['screen_valid']['recall@10']), reverse=True)
    finalists = trials[:12]
    # Include the original configuration as a validation reference regardless of screening.
    original = next(t for t in trials if t['rho'] == .5 and t['power'] == 1 and t['weight_index'] == 0)
    if original not in finalists:
        finalists.append(original)
    for trial in finalists:
        coeff = model.coefficients(data['x'], trial['rho'], trial['power'])
        trial['full_valid'] = evaluate(data, coeff, trial['weights'])
        print(f"Full VALID MRR={trial['full_valid']['mrr@10']:.6f} Recall={trial['full_valid']['recall@10']:.6f}", flush=True)
    finalists.sort(key=lambda c: (c['full_valid']['mrr@10'], c['full_valid']['recall@10']), reverse=True)
    best = finalists[0]
    model.rho, model.power = best['rho'], best['power']
    model.weights = np.asarray(best['weights'], dtype=np.float32)
    model.save(OUT / 'model_selected')
    selection = {'selected': best, 'features': FEATURES, 'screen_trials': len(trials),
                 'screen_valid_queries': 4000, 'full_valid_queries': 20000,
                 'criterion': 'max full-validation MRR@10; tie-break Recall@10; no test tuning',
                 'neighbours': model.neighbours, 'selection_seconds_including_cache_io': time.perf_counter()-start,
                 'sparse_dense_validation_checks': len(equality),
                 'validation_target_support_coverage': float(np.mean(data['target_index'] >= 0)),
                 'mean_support_items': float(np.diff(data['offsets']).mean()),
                 'selected_config_sha256': hashlib.sha256((OUT/'model_selected/config.json').read_bytes()).hexdigest(),
                 'original_full_valid': original['full_valid']}
    (OUT / 'selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    (OUT / 'validation_search.json').write_text(json.dumps({'screening': trials, 'finalists': finalists}, indent=2), encoding='utf-8')
    print(json.dumps(selection, indent=2), flush=True)


if __name__ == '__main__':
    main()
