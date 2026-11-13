"""Four independent one-parameter GRA experiments on the existing two splits.

No model training, test-based search, candidate sampling or baseline mutation.
Run with the project's recbole-gnn-py39 Python. All paths are project-relative.
"""
import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from grey_session import GreySession, grey_coefficients

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
AMAZON = PROJECT / '15_Amazon_M2算法对比'
OUT = ROOT / 'outputs/grey_four_axes'
PLAN = PROJECT / '05_灰预测与决策方法课程论文/四组灰关联单因素对比试验方案.md'
MODELS = ['SRGNN', 'GCSAN', 'GCEGNN']
GRIDS = {'A': [.1, .25, .5, .75, 1.], 'B': [0, 1, 2, 7, 15],
         'C': [-2, -1, 0, 1, 2], 'D': [0, .25, .5, .75, 1.]}
RESTORE = {'A': .5, 'B': 0, 'C': 0, 'D': 0}
LABELS = {'Base': '原版灰色关联', 'A': 'A 分辨系数调优', 'B': 'B 差异压缩',
          'C': 'C 转移指标权重调优', 'D': 'D 双指标交互聚合',
          'SRGNN': 'SR-GNN', 'GCSAN': 'GC-SAN', 'GCEGNN': 'GCE-GNN'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(ranks):
    ranks = np.asarray(ranks, dtype=np.int64)
    yes = ranks > 0
    rr = np.zeros(len(ranks), dtype=np.float64)
    nd = rr.copy()
    rr[yes] = 1. / ranks[yes]
    nd[yes] = 1. / np.log2(ranks[yes] + 1)
    return {'recall@10': float(yes.mean()), 'mrr@10': float(rr.mean()),
            'ndcg@10': float(nd.mean())}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def configs():
    return [('Base', None)] + [(group, value) for group, values in GRIDS.items()
                              for value in values if value != RESTORE[group]]


def config_key(group, value):
    return 'Base' if group == 'Base' or value == RESTORE.get(group) else f'{group}_{value:g}'


def coeff(z, rho=.5, beta=0):
    delta = np.abs(1.0 - z)
    if beta:
        delta = delta / (1.0 + beta * (1.0 - delta))
    lo, hi = float(delta.min()), float(delta.max())
    return ((lo + rho * hi) / (delta + rho * hi) if hi else np.ones_like(delta))


def base_score(c, dataset):
    if dataset == 'diginetica':
        return np.sum(np.full((c.shape[0], 1), .25, dtype=np.float32) * c, axis=0)
    return c.mean(axis=0)


def weights(f, value):
    w = np.ones(f, dtype=np.float64)
    w[0] = math.exp(value)
    return (w / w.sum()).astype(np.float32)


def candidate_score(z, group, value, dataset, cached_coeff=None):
    if group == 'Base' or value == RESTORE.get(group):
        return base_score(coeff(z) if cached_coeff is None else cached_coeff, dataset)
    if group == 'A':
        return base_score(coeff(z, rho=value), dataset)
    if group == 'B':
        return base_score(coeff(z, beta=value), dataset)
    c = coeff(z) if cached_coeff is None else cached_coeff
    if group == 'C':
        return np.sum(weights(z.shape[0], value)[:, None] * c, axis=0)
    if group == 'D':
        return base_score(c, dataset) - np.float32(value / z.shape[0]) * np.abs(c[0] - c[1])
    raise ValueError(group)


def all_scores(z, dataset):
    c = coeff(z)
    for group, value in configs():
        yield candidate_score(z, group, value, dataset, c)


def masked(scores, history, filter_history):
    s = np.asarray(scores, dtype=np.float32).copy()
    if filter_history:
        s[np.asarray(history, dtype=np.int64) - 1] = -np.inf
    return s


def rank_of(scores, target, history, filter_history):
    s = masked(scores, history, filter_history)
    p = int(target) - 1
    truth = s[p]
    assert np.isfinite(truth), ('excluded target', target)
    rank = 1 + np.count_nonzero(s > truth) + np.count_nonzero(s[:p] == truth)
    return int(rank) if rank <= 10 else 0


def top10(scores, history, filter_history):
    s = masked(scores, history, filter_history)
    cutoff = np.partition(s, len(s) - 10)[len(s) - 10]
    above = np.flatnonzero(s > cutoff)
    equal = np.flatnonzero(s == cutoff)[:10 - len(above)]
    ix = np.r_[above, equal]
    ix = ix[np.lexsort((ix, -s[ix]))]
    assert len(ix) == 10 and np.isfinite(s[ix]).all()
    return ix + 1, s[ix]


def numeric_arrays_bytes(objects):
    arrays = {}
    for value in objects:
        if isinstance(value, np.ndarray):
            arrays[(int(value.__array_interface__['data'][0]), value.nbytes)] = value.nbytes
        elif torch.is_tensor(value):
            arrays[(value.data_ptr(), value.numel() * value.element_size())] = value.numel() * value.element_size()
        elif hasattr(value, 'indptr'):
            for v in [value.data, value.indices, value.indptr]:
                arrays[(int(v.__array_interface__['data'][0]), v.nbytes)] = v.nbytes
    return int(sum(arrays.values()))


class Dataset:
    def __init__(self, name):
        self.name = name
        self.directory = OUT / name
        self.directory.mkdir(parents=True, exist_ok=True)
        begin = time.perf_counter()
        if name == 'diginetica':
            self.model = GreySession.load(ROOT / 'outputs/grey_relation/grey_model.npz')
            d = np.load(ROOT / 'outputs/grey_improved/splits.npz', allow_pickle=False)
            self.tokens = read(ROOT / 'outputs/grey_improved/tokens.json')
            self.rows = {}
            for split in ['valid', 'test']:
                self.rows[split] = [{'history': h[:int(n)].tolist(), 'target': int(t),
                                     'query_id': str(self.tokens['users'][int(u)])}
                                    for u, h, n, t in zip(d[split + '_users'], d[split + '_histories'],
                                                         d[split + '_lengths'], d[split + '_targets'])]
            self.n = self.model.item_num
            self.f = 4
            self.features = ['transition', 'cooccurrence', 'repeat', 'popularity']
            self.numeric_bytes = numeric_arrays_bytes([
                self.model.counts, self.model.log_popularity, self.model.weights,
                self.model.transitions, self.model.cooccurrence])
            self.filter_history = False
            self.prep_kind = 'existing training statistics loading; historical fit time reported separately'
            self.fit_seconds = read(ROOT / 'outputs/grey_relation/result.json')['statistics_fit_seconds']
            self.input_paths = [ROOT / 'outputs/grey_improved/splits.npz',
                                ROOT / 'outputs/grey_improved/tokens.json',
                                ROOT / 'outputs/grey_relation/grey_model.npz',
                                ROOT / 'data/diginetica-formal/diginetica-formal.inter',
                                ROOT / 'configs/formal.yaml', Path(__file__).with_name('grey_session.py')]
            self.reference_gra = ROOT / 'outputs/grey_relation/GRA.jsonl'
        else:
            self.common = load_module('common', AMAZON / 'common.py')
            self.data = self.common.data()
            self.rows = {'valid': self.data['validation'], 'test': self.data['test']}
            for split, rows in self.rows.items():
                for r in rows:
                    r['query_id'] = str(r['source_row'])
            before = time.perf_counter()
            self.model = self.common.Features(self.data)
            self.fit_seconds = time.perf_counter() - before
            self.prep_kind = 'training statistics reconstruction and existing static attributes loading'
            self.n = self.model.n
            self.f = 8
            self.features = list(self.common.FEATURES)
            self.tokens = {'items': self.data['tokens']}
            self.filter_history = True
            self.numeric_bytes = numeric_arrays_bytes([
                self.model.tr, self.model.co, self.model.counts, self.model.pop,
                *self.model.a.values()])
            self.input_paths = [AMAZON / 'run/data.json', AMAZON / 'run/attributes.npz',
                                AMAZON / 'common.py', AMAZON / 'train.py']
            self.reference_gra = AMAZON / 'run/predictions/grey.jsonl'
            prior_hashes = read(AMAZON / 'run/results/test_configuration.json')['sha256']
            for path in self.input_paths:
                assert sha(path) == prior_hashes[str(path)], ('changed Amazon source', str(path))
        self.load_seconds = time.perf_counter() - begin
        assert len({r['query_id'] for r in self.rows['test']}) == len(self.rows['test'])
        for split, rows in self.rows.items():
            assert all(1 <= r['target'] < self.n and 0 < len(r['history']) <= 20 for r in rows)
            if self.filter_history:
                assert all(r['target'] not in r['history'] for r in rows)
        self.input_paths += [self.reference_gra]
        for name in MODELS:
            self.input_paths += [self.checkpoint(name), self.prior_predictions(name), self.training_report(name)]
            prior = read(self.training_report(name))
            assert sha(self.checkpoint(name)) == prior['checkpoint_sha256'], name

    def matrix(self, history):
        full = self.model.normalized_features(history) if self.name == 'diginetica' else self.model.matrix(history)
        z = full[:, 1:]
        assert np.isfinite(z).all() and z.min() >= -1e-6 and z.max() <= 1 + 1e-6
        return z

    def original_prediction(self, history):
        if self.name == 'diginetica':
            return self.model.recommend(history)[0]
        return self.common.topk(self.model.grey_score(history), history)

    def checkpoint(self, name):
        if self.name == 'diginetica':
            return Path(read(self.training_report(name))['checkpoint'])
        return AMAZON / f'run/models/{name}.pt'

    def training_report(self, name):
        return (ROOT / f'outputs/formal/{name}/seed42/result.json' if self.name == 'diginetica'
                else AMAZON / f'run/results/{name}_training.json')

    def prior_predictions(self, name):
        return (ROOT / f'outputs/formal/predictions/{name}.jsonl' if self.name == 'diginetica'
                else AMAZON / f'run/predictions/{name}.jsonl')

    def reference(self, path):
        records = {}
        tokens = {str(t): i for i, t in enumerate(self.tokens['items'])}
        for line in path.open(encoding='utf-8'):
            r = json.loads(line)
            if self.name == 'diginetica':
                query_id = r['session_id']
                history = [tokens[str(x)] for x in r['history']]
                target = tokens[str(r['target'])]
                ids = [tokens[str(v['item'])] for v in r['recommendations']]
                rank = int(r['target_rank'])
            else:
                query_id, history, target, ids, rank = str(r['source_row']), r['history'], r['target'], r['top10'], int(r['rank'])
            assert query_id not in records
            assert rank == (ids.index(target) + 1 if target in ids else 0)
            records[query_id] = {'history': history, 'target': target, 'top10': ids, 'rank': rank}
        result = []
        for row in self.rows['test']:
            prior = records[row['query_id']]
            assert prior['history'] == row['history'] and prior['target'] == row['target']
            result.append(prior)
        assert len(records) == len(result)
        return result


def make_manifest(d):
    path = d.directory / 'manifest.json'
    files = d.input_paths + [Path(__file__), PLAN]
    hashes = {str(p): sha(p) for p in files}
    if path.exists():
        assert hashes == read(path)['sha256'], 'Frozen input or implementation changed; inspect before continuing.'
        return
    save(path, {'dataset': d.name, 'test_queries': len(d.rows['test']),
                'validation_queries': len(d.rows['valid']), 'catalogue_items': d.n - 1,
                'features': d.features, 'filter_history': d.filter_history,
                'grids': GRIDS, 'unique_validation_configurations': len(configs()),
                'selection': 'validation MRR@10, then Recall@10, then NDCG@10; baseline then grid order for exact ties',
                'already_viewed_test_set': True, 'test_used_for_this_round_selection': False,
                'source_gnn_training_seed': 42, 'new_gnn_training': False,
                'numeric_state_bytes': d.numeric_bytes, 'statistics_fit_seconds': d.fit_seconds,
                'statistics_fit_scope': d.prep_kind, 'adapter_loading_seconds': d.load_seconds,
                'sha256': hashes, 'created_unix': time.time(),
                'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                                'torch': torch.__version__, 'platform': platform.platform()}})


def initialize(d):
    make_manifest(d)
    rng = np.random.default_rng(20261113)
    wanted = rng.choice(len(d.rows['test']), 40, replace=False)
    checks = 0
    bounds = [1., 0.]
    for index in wanted:
        row = d.rows['test'][index]
        z = d.matrix(row['history'])
        bounds = [min(bounds[0], float(z.min())), max(bounds[1], float(z.max()))]
        original = d.original_prediction(row['history'])
        for group, value in [('Base', None)] + list(RESTORE.items()):
            ids, _ = top10(candidate_score(z, group, value, d.name), row['history'], d.filter_history)
            np.testing.assert_array_equal(ids, original)
            checks += 1
        if d.name == 'diginetica':
            np.testing.assert_array_equal(coeff(z), grey_coefficients(z))
    # Check the interaction aggregator on coefficient values, independently of features.
    for f in [4, 8]:
        for eta in GRIDS['D']:
            assert (1 - eta) / f >= 0 and (1 + eta) / f >= 0
        for value in GRIDS['C']:
            np.testing.assert_allclose(weights(f, value).sum(), 1, atol=1e-7)
    prior = d.reference(d.reference_gra)
    save(d.directory / 'initial_checks.json', {'passed': True, 'original_restore_checks': checks,
         'sampled_feature_bounds': bounds, 'reference_rows_aligned': len(prior),
         'reference_metrics': metrics([r['rank'] for r in prior]), 'interaction_slopes_nonnegative': True})
    print(f'{d.name} INITIAL CHECKS passed: {checks} restoration checks, {len(prior)} aligned original rows.', flush=True)


def select(d):
    make_manifest(d)
    rows = d.rows['valid']
    grid = configs()
    checkpoint = d.directory / 'validation_progress.npz'
    ranks = np.full((len(rows), len(grid)), -1, dtype=np.int16)
    start = 0
    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as old:
            ranks[:] = old['ranks']
            start = int(old['next_index'])
    begin = time.perf_counter()
    for i in range(start, len(rows)):
        row = rows[i]
        z = d.matrix(row['history'])
        for k, scores in enumerate(all_scores(z, d.name)):
            ranks[i, k] = rank_of(scores, row['target'], row['history'], d.filter_history)
        if (i + 1) % 500 == 0 or i + 1 == len(rows):
            np.savez_compressed(checkpoint, ranks=ranks, next_index=i + 1)
            print(f'{d.name} VALID {i+1}/{len(rows)}; elapsed={time.perf_counter()-begin:.1f}s', flush=True)
    trials = []
    for k, (group, value) in enumerate(grid):
        trials.append({'key': config_key(group, value), 'group': group, 'value': value,
                       **metrics(ranks[:, k])})
    write_csv(d.directory / 'validation_search.csv', trials)
    base = trials[0]
    selected = {}
    for group, values in GRIDS.items():
        candidates = []
        for order, value in enumerate(values):
            r = next(t for t in trials if t['key'] == config_key(group, value))
            candidates.append((r, value, order))
        best, value, _ = max(candidates, key=lambda t: (t[0]['mrr@10'], t[0]['recall@10'],
                  t[0]['ndcg@10'], t[1] == RESTORE[group], -t[2]))
        selected[group] = {'group': group, 'value': value, 'validation': {k: best[k] for k in metrics([0])}}
    frozen = {'dataset': d.name, 'selection_criterion': read(d.directory / 'manifest.json')['selection'],
              'test_used_in_this_round_selection': False, 'prior_test_exposure': True,
              'grids': GRIDS, 'validation_sessions': len(rows), 'unique_configurations': len(grid),
              'baseline': {k: base[k] for k in metrics([0])}, 'selected': selected,
              'validation_search_sha256': sha(d.directory / 'validation_search.csv'),
              'manifest_sha256': sha(d.directory / 'manifest.json'), 'frozen_unix': time.time()}
    path = d.directory / 'frozen_selection.json'
    if path.exists():
        old = read(path)
        assert old['selected'] == selected and old['validation_search_sha256'] == frozen['validation_search_sha256']
    else:
        save(path, frozen)
    print(f'{d.name} FROZEN ' + json.dumps(selected, ensure_ascii=False), flush=True)


def chosen(d):
    freeze = read(d.directory / 'frozen_selection.json')
    assert freeze['manifest_sha256'] == sha(d.directory / 'manifest.json')
    return [('Base', None)] + [(g, freeze['selected'][g]['value']) for g in GRIDS]


def prediction_record(index, row, ids, scores=None):
    ids = [int(x) for x in ids]
    rank = ids.index(row['target']) + 1 if row['target'] in ids else 0
    r = {'query_index': index, 'query_id': row['query_id'], 'history': row['history'],
         'target': row['target'], 'top10': ids, 'rank': rank}
    if scores is not None:
        r['top10_scores'] = [float(x) for x in scores]
    return r


def test_gra(d):
    make_manifest(d)
    selection_sha = sha(d.directory / 'frozen_selection.json')
    cs = chosen(d)
    directory = d.directory / 'predictions'
    directory.mkdir(exist_ok=True)
    prior = d.reference(d.reference_gra)
    ranks = {g: [] for g, _ in cs}
    streams = {g: (directory / (g + '.jsonl')).open('w', encoding='utf-8') for g, _ in cs}
    begin = time.perf_counter()
    restore_full = 0
    try:
        for i, row in enumerate(d.rows['test']):
            z = d.matrix(row['history'])
            c = coeff(z)
            original = None
            for group, value in cs:
                s = candidate_score(z, group, value, d.name, c)
                ids, values = top10(s, row['history'], d.filter_history)
                record = prediction_record(i, row, ids, values)
                streams[group].write(json.dumps(record, ensure_ascii=False) + '\n')
                ranks[group].append(record['rank'])
                if group == 'Base':
                    original = ids
                    assert ids.tolist() == prior[i]['top10'], ('original mismatch', d.name, i)
                    assert record['rank'] == prior[i]['rank']
                # Evaluate the four restoring parameter values on every test query.
                # They dispatch to the common original path, as required by each formula.
            for group, value in RESTORE.items():
                ids, _ = top10(candidate_score(z, group, value, d.name, c), row['history'], d.filter_history)
                np.testing.assert_array_equal(ids, original)
                restore_full += 1
            if (i + 1) % 1000 == 0:
                print(f'{d.name} TEST GRA {i+1}/{len(prior)}; elapsed={time.perf_counter()-begin:.1f}s', flush=True)
    finally:
        for stream in streams.values():
            stream.close()
    assert selection_sha == sha(d.directory / 'frozen_selection.json')
    save(d.directory / 'gra_results.json', {g: {'group': g, 'value': v, **metrics(ranks[g]),
          'test_sessions': len(ranks[g]), 'prediction_sha256': sha(directory / (g + '.jsonl'))} for g, v in cs})
    save(d.directory / 'gra_verification.json', {'passed': True, 'all_original_top10_match_saved': len(prior),
         'full_test_restoring_parameter_checks': restore_full, 'selection_sha256_before_after': selection_sha,
         'new_gra_test_seconds_including_writes_and_restore_checks': time.perf_counter()-begin})
    print(d.name + ' GRA RESULTS ' + json.dumps(read(d.directory / 'gra_results.json'), ensure_ascii=False), flush=True)


def graph_batch(histories, name):
    size = max(map(len, histories))
    nodes, aliases, edges, attrs = [0], [], [], []
    for history in histories:
        h = list(reversed(history)) if name == 'GCEGNN' else history
        unique = sorted(set(h))
        mapping = {x: len(nodes) + i for i, x in enumerate(unique)}
        nodes.extend(unique)
        alias = [mapping[x] for x in h]
        aliases.append(alias + [0] * (size - len(h)))
        pairs = set(zip(alias[:-1], alias[1:]))
        if name == 'GCEGNN':
            for a, b in sorted(pairs | {(b, a) for a, b in pairs}):
                edges.append((a, b))
                attrs.append(3 if (a, b) in pairs and (b, a) in pairs else (1 if (a, b) in pairs else 2))
            for node in sorted(set(alias)):
                edges.append((node, node))
                attrs.append(0)
        else:
            edges.extend(sorted(pairs))
    return {'x': torch.tensor(nodes, dtype=torch.long),
            'edge_index': torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long),
            'edge_attr': torch.tensor(attrs, dtype=torch.long),
            'alias_inputs': torch.tensor(aliases, dtype=torch.long),
            'item_length': torch.tensor([len(h) for h in histories]),
            'item_id': torch.zeros(len(histories), dtype=torch.long)}


def load_gnn(d, name):
    if name in getattr(d, '_gnn_cache', {}):
        return d._gnn_cache[name]
    begin = time.perf_counter()
    if d.name == 'diginetica':
        sys.path.insert(0, str(PROJECT / 'app'))
        from predictor import SessionPredictor
        adapter = SessionPredictor(name)
        assert adapter.dataset.field2id_token[adapter.config['ITEM_ID_FIELD']].tolist() == d.tokens['items']
        model = adapter.model
        del adapter
    else:
        sys.path.insert(0, str(ROOT / 'RecBole-GNN'))
        train = load_module('amazon_four_axes_train', AMAZON / 'train.py')
        model = train.load_model(name, d.data)
    model.eval()
    # AttributeEmbedding.train(False) clears its cache. Compile AFTER the final
    # eval() call so catalogue fusion is prepared once, as in the original M2 run.
    if d.name == 'amazon_m2':
        model.item_embedding.compile()
        assert model.item_embedding.cached is not None
    tensors = list(model.parameters()) + list(model.buffers())
    tensors += [getattr(model, n) for n in getattr(model, 'other_parameter_name', [])]
    tensors += [m.cached for m in model.modules() if torch.is_tensor(getattr(m, 'cached', None))]
    profile = {'loading_and_catalogue_compile_seconds': time.perf_counter()-begin,
               'numeric_state_bytes': numeric_arrays_bytes(tensors),
               'learned_parameters': sum(p.numel() for p in model.parameters()),
               'checkpoint_file_bytes': d.checkpoint(name).stat().st_size,
               'catalogue_embedding_precomputed': torch.is_tensor(getattr(model.item_embedding, 'cached', None)),
               'model_module': type(model).__module__,
               'checkpoint_sha256': sha(d.checkpoint(name))}
    @torch.inference_mode()
    def predict_scores(histories):
        return model.full_sort_predict(graph_batch(histories, name)).reshape(len(histories), d.n).numpy()
    if not hasattr(d, '_gnn_cache'):
        d._gnn_cache = {}
    d._gnn_cache[name] = model, predict_scores, profile
    return d._gnn_cache[name]


def test_gnn(d, name):
    make_manifest(d)
    model, predict, profile = load_gnn(d, name)
    prior = d.reference(d.prior_predictions(name))
    directory = d.directory / 'predictions'
    directory.mkdir(exist_ok=True)
    begin = time.perf_counter()
    rr, same_top10, same_rank = [], 0, 0
    divergences = []
    with (directory / (name + '.jsonl')).open('w', encoding='utf-8') as output:
        for start in range(0, len(prior), 128):
            rows = d.rows['test'][start:start+128]
            scores = predict([r['history'] for r in rows])
            for j, row in enumerate(rows):
                i = start + j
                ids, values = top10(scores[j, 1:], row['history'], d.filter_history)
                r = prediction_record(i, row, ids, values)
                output.write(json.dumps(r, ensure_ascii=False) + '\n')
                rr.append(r['rank'])
                same_top10 += int(ids.tolist() == prior[i]['top10'])
                same_rank += int(r['rank'] == prior[i]['rank'])
                if r['rank'] != prior[i]['rank'] and len(divergences) < 10:
                    divergences.append({'index': i, 'new_rank': r['rank'], 'old_rank': prior[i]['rank']})
            if start % 4096 == 0:
                print(f'{d.name} {name} TEST {min(start+128,len(prior))}/{len(prior)}', flush=True)
    # An independent batch=1 path verifies graph batching and ranking on sampled queries.
    rng = np.random.default_rng(20261113)
    sample = sorted(rng.choice(len(prior), 40, replace=False).tolist())
    one_agree = 0
    for i in sample:
        row = d.rows['test'][i]
        ids, _ = top10(predict([row['history']])[0, 1:], row['history'], d.filter_history)
        rank = ids.tolist().index(row['target']) + 1 if row['target'] in ids else 0
        one_agree += int(rank == rr[i])
    assert one_agree == len(sample), (name, 'single/batch rank mismatch')
    assert same_rank >= len(prior) - 3, (name, 'unexpected divergence from checkpoint prior', divergences)
    old_metrics = metrics([r['rank'] for r in prior])
    save(d.directory / (name + '_results.json'), {'model': name, **metrics(rr), 'test_sessions': len(rr),
         **profile, 'fresh_checkpoint_inference': True, 'new_training': False,
         'quality_evaluation_seconds': time.perf_counter()-begin,
         'previous_top10_exact_rows': same_top10, 'previous_rank_exact_rows': same_rank,
         'previous_metrics': old_metrics, 'divergence_examples': divergences,
         'batch1_batch128_rank_checks': one_agree,
         'prediction_sha256': sha(directory / (name + '.jsonl'))})
    print(d.name + ' ' + name + ' RESULT ' + json.dumps(metrics(rr)), flush=True)
    del predict, model
    gc.collect()


def benchmark(d):
    make_manifest(d)
    cs = chosen(d)
    rng = np.random.default_rng(20261113)
    sample = rng.choice(len(d.rows['test']), 300, replace=False)
    predictors, profiles, models = {}, {}, []
    for group, value in cs:
        def predict(history, g=group, v=value):
            return top10(candidate_score(d.matrix(history), g, v, d.name), history, d.filter_history)[0]
        predictors[group] = predict
        profiles[group] = {'numeric_state_bytes': d.numeric_bytes + (weights(d.f, value).nbytes if group == 'C' else 0) + (0 if group == 'Base' else 8)}
    for name in MODELS:
        model, score, profile = load_gnn(d, name)
        models.append(model)
        def predict(history, fn=score):
            return top10(fn([history])[0, 1:], history, d.filter_history)[0]
        predictors[name] = predict
        profiles[name] = profile
    # Validate the timed batch=1 outputs against this round's fresh complete predictions.
    expected = {}
    for name in predictors:
        expected[name] = {r['query_index']: r['top10'] for r in
                         (json.loads(line) for line in (d.directory / 'predictions' / (name + '.jsonl')).open(encoding='utf-8'))
                         if r['query_index'] in set(sample.tolist())}
    exact_checks = {}
    for name, fn in predictors.items():
        mismatch = []
        for i in sample:
            actual = fn(d.rows['test'][i]['history']).tolist()
            if actual != expected[name][int(i)]:
                mismatch.append(int(i))
        # GNN batch-size arithmetic can change near-equal scores; preserve and quantify it.
        assert not mismatch if name in dict(cs) else len(mismatch) <= 3, (name, 'timing output mismatch', mismatch[:10])
        exact_checks[name] = {'checks': 300, 'exact_top10': 300-len(mismatch), 'different_top10_indices': mismatch}
        for i in sample[:30]:
            fn(d.rows['test'][i]['history'])
    timings = []
    gc.collect()
    gc.disable()
    try:
        for repeat in range(3):
            method_order = rng.permutation(list(predictors))
            order = rng.permutation(sample)
            for name in method_order:
                fn = predictors[name]
                for i in order:
                    begin = time.perf_counter_ns()
                    ids = fn(d.rows['test'][i]['history'])
                    elapsed = (time.perf_counter_ns()-begin)/1e6
                    assert len(ids) == 10
                    timings.append({'method': name, 'round': repeat+1, 'query_index': int(i), 'ms': elapsed})
                print(f'{d.name} BENCH round={repeat+1} {name}', flush=True)
    finally:
        gc.enable()
    write_csv(d.directory / 'latency.csv', timings)
    result = {}
    for name in predictors:
        ts = [r['ms'] for r in timings if r['method'] == name]
        result[name] = {'mean_ms': float(np.mean(ts)), 'median_ms': float(np.median(ts)),
                        'p95_ms': float(np.percentile(ts,95)), 'calls': len(ts), **profiles[name]}
    save(d.directory / 'latency_summary.json', {'methods': result, 'sample_indices': sample.tolist(),
         'correctness': exact_checks, 'batch_size': 1, 'cpu_threads': 4, 'rounds': 3, 'warmup_calls_per_method': 30,
         'scope': 'internal history IDs -> features/query graph -> full catalogue scores -> dataset filter -> Top10; loading, static catalogue preparation, serialization excluded',
         'method_order_randomized_each_round': True, 'created_unix': time.time()})
    print(d.name + ' BENCH COMPLETE', flush=True)


def verify_and_report(d):
    make_manifest(d)
    method_names = ['Base', 'A', 'B', 'C', 'D'] + MODELS
    rows = d.rows['test']
    all_ranks = {}
    check = {}
    for name in method_names:
        ranks, top_ids = [], []
        for i, line in enumerate((d.directory / 'predictions' / (name + '.jsonl')).open(encoding='utf-8')):
            record = json.loads(line)
            assert i == record['query_index'] and record['query_id'] == rows[i]['query_id']
            assert record['history'] == rows[i]['history'] and record['target'] == rows[i]['target']
            ids = record['top10']
            assert len(ids) == len(set(ids)) == 10 and all(1 <= x < d.n for x in ids)
            if d.filter_history:
                assert not set(ids).intersection(record['history'])
            rank = ids.index(record['target']) + 1 if record['target'] in ids else 0
            assert rank == record['rank']
            ranks.append(rank)
            top_ids.append(ids)
        assert len(ranks) == len(rows)
        all_ranks[name] = np.asarray(ranks, dtype=np.int16)
        check[name] = {'rows': len(ranks), 'metrics_recomputed': metrics(ranks), 'top10_legal': True}
    freeze = read(d.directory / 'frozen_selection.json')
    latency = read(d.directory / 'latency_summary.json')
    assert sha(d.directory / 'validation_search.csv') == freeze['validation_search_sha256']
    trial_rows = list(csv.DictReader((d.directory / 'validation_search.csv').open(encoding='utf-8-sig')))
    assert len(trial_rows) == 17
    comparisons = []
    groups = {}
    for name in method_names:
        rank = all_ranks[name]
        result = (read(d.directory / 'gra_results.json')[name] if name not in MODELS
                  else read(d.directory / (name + '_results.json')))
        for key, v in metrics(rank).items():
            assert abs(v-result[key]) < 1e-14
        timing = latency['methods'][name]
        assert timing['calls'] == 900
        label = LABELS[name] + ('＋属性' if d.name == 'amazon_m2' and name in MODELS else '')
        param = '' if name == 'Base' or name in MODELS else freeze['selected'][name]['value']
        comparisons.append({'dataset': d.name, 'method_key': name, 'method': label, 'selected_parameter': param,
            **metrics(rank), 'mean_ms': timing['mean_ms'], 'p95_ms': timing['p95_ms'],
            'numeric_state_mb': timing['numeric_state_bytes']/1e6,
            'recall_gain_pp_vs_base': 100*(metrics(rank)['recall@10']-metrics(all_ranks['Base'])['recall@10']),
            'new_hits_vs_base': int(np.count_nonzero((rank>0)&(all_ranks['Base']==0))),
            'lost_hits_vs_base': int(np.count_nonzero((rank==0)&(all_ranks['Base']>0)))})
        if d.name == 'diginetica':
            flag = np.array([r['target'] in r['history'] for r in rows])
            labels = ['repeat_target', 'novel_target']
        else:
            flag = np.array([d.model.counts[r['target']]>0 for r in rows])
            labels = ['train_seen_target', 'train_unseen_target']
        groups[name] = {label: {'queries': int(mask.sum()), **metrics(rank[mask])}
                       for label, mask in [(labels[0], flag), (labels[1], ~flag)]}
    write_csv(d.directory / 'comparison.csv', comparisons)
    save(d.directory / 'comparison.json', comparisons)
    save(d.directory / 'subgroup_results.json', groups)
    original = d.reference(d.reference_gra)
    np.testing.assert_array_equal(all_ranks['Base'], [r['rank'] for r in original])
    baseline_check = read(d.directory / 'gra_verification.json')
    assert baseline_check['all_original_top10_match_saved'] == len(rows)
    assert baseline_check['full_test_restoring_parameter_checks'] == len(rows)*4
    assert baseline_check['selection_sha256_before_after'] == sha(d.directory / 'frozen_selection.json')
    hashes = {name: sha(d.directory / 'predictions' / (name + '.jsonl')) for name in method_names}
    save(d.directory / 'verification.json', {'passed': True, 'dataset': d.name, 'methods': 8,
         'test_queries_per_method': len(rows), 'aligned_prediction_rows': len(rows)*8,
         'original_prediction_rows_exact': len(rows), 'restore_checks': len(rows)*4,
         'validation_unique_configs': len(trial_rows), 'all_metrics_recomputed': True,
         'frozen_selection_intact': True, 'source_fingerprints_match': True,
         'latency_calls_per_method': 900, 'method_checks': check, 'prediction_sha256': hashes,
         'manifest_sha256': sha(d.directory / 'manifest.json'),
         'selection_sha256': sha(d.directory / 'frozen_selection.json'),
         'limits': 'Existing already-viewed splits; one original GNN training seed; no new independent generalization claim.'})
    lines = [f'# {d.name}：四组灰关联单因素实测', '',
             f'每种方法完整评价{len(rows):,}条测试会话、{d.n-1:,}件候选商品。四组各5档，在完整验证集上按MRR、Recall、NDCG顺序选参；未叠加改动。', '',
             '| 方法 | 参数 | Recall@10 | MRR@10 | NDCG@10 | 平均响应/ms | P95/ms | 数值状态/MB |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in comparisons:
        lines.append(f"| {r['method']} | {r['selected_parameter']} | {r['recall@10']*100:.3f}% | {r['mrr@10']:.5f} | {r['ndcg@10']:.5f} | {r['mean_ms']:.3f} | {r['p95_ms']:.3f} | {r['numeric_state_mb']:.3f} |")
    lines += ['', 'Recall表示前十命中率，MRR与NDCG反映命中位置。质量评价与响应计时分别执行；计时采用CPU4线程、batch=1、固定300个请求、预热30次、3轮各900次，包含指标/图构建、全目录评分、过滤与Top10，不含加载与静态目录预计算。数值状态是数组/张量大小，不是进程运行内存。', '',
              '## 改动带来的命中变化', '', '| 变体 | 新增命中 | 失去命中 | Recall变化/百分点 |', '|---|---:|---:|---:|']
    for r in comparisons[1:5]:
        lines.append(f"| {r['method']} | {r['new_hits_vs_base']} | {r['lost_hits_vs_base']} | {r['recall_gain_pp_vs_base']:+.3f} |")
    lines += ['', '## 证据与范围', '',
              '- GRA原版完整Top10与旧预测逐条一致；四组恢复原版参数在全部测试请求上核对。',
              '- 三个GNN读取原有检查点，重新进行完整测试推理；本轮未重训。新旧排序一致性及batch=1与128核对记录见各模型results.json。',
              '- Amazon模型名中的“＋属性”表示实际使用已有属性融合；Dig允许再访，Amazon统一过滤历史商品。两数据集的绝对指标不直接比较难度。',
              '- 两套现有测试集已有查看记录，本轮属于扩展性探索；参数只由本轮验证集选择，不据此声称跨划分稳定性或方法首创。',
              '- validation_search.csv、frozen_selection.json、predictions/、latency.csv与verification.json分别保存搜索、冻结、完整预测、计时及核验。', '']
    (d.directory / '实测结果.md').write_text('\n'.join(lines), encoding='utf-8')
    print(d.name + ' VERIFICATION passed. ' + json.dumps(comparisons, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['diginetica','amazon_m2'], required=True)
    parser.add_argument('--stage', choices=['init','select','test','gnn','benchmark','verify','all'], default='all')
    parser.add_argument('--model', choices=MODELS)
    args = parser.parse_args()
    sys.argv = sys.argv[:1]
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    d = Dataset(args.dataset)
    if args.stage in ['init','all']:
        initialize(d)
    if args.stage in ['select','all']:
        select(d)
    if args.stage in ['test','all']:
        test_gra(d)
    if args.stage in ['gnn','all']:
        for model in ([args.model] if args.model else MODELS):
            test_gnn(d, model)
    if args.stage in ['benchmark','all']:
        benchmark(d)
    if args.stage in ['verify','all']:
        verify_and_report(d)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
