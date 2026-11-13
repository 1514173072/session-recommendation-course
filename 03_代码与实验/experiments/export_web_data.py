"""Assemble the web dashboard data from formal results, logs and predictions.

Also writes outputs/formal/comparison.csv so the report/acceptance record and the
web page share one source of truth.
"""
import csv
import json
import os
from pathlib import Path
import re
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / 'RecBole-GNN'
sys.path.insert(0, str(REPO))

MODELS = ['SRGNN', 'GCSAN', 'GCEGNN']
LABELS = {'SRGNN': 'SR-GNN', 'GCSAN': 'GC-SAN', 'GCEGNN': 'GCE-GNN'}
ANSI = re.compile(r'\x1b\[[0-9;]*m')


def parse_curves(log_path):
    losses, mrrs = [], []
    for line in log_path.read_text(encoding='utf-8', errors='replace').splitlines():
        line = ANSI.sub('', line)
        m = re.search(r'epoch (\d+) training \[time: [^,]+, train loss: ([\d.]+)\]', line)
        if m:
            losses.append((int(m.group(1)) + 1, float(m.group(2))))
        m = re.search(r'epoch (\d+) evaluating \[time: [^,]+, valid_score: ([\d.]+)\]', line)
        if m:
            mrrs.append((int(m.group(1)) + 1, float(m.group(2))))
    mrr_map = dict(mrrs)
    return [{'epoch': e, 'loss': loss, 'valid_mrr': mrr_map.get(e)} for e, loss in losses]


def local_graph(history):
    nodes = []
    seen = {}
    for item in history:
        if item not in seen:
            seen[item] = len(nodes)
            nodes.append({'id': item, 'count': 0})
        nodes[seen[item]]['count'] += 1
    edges = {}
    for a, b in zip(history[:-1], history[1:]):
        key = (a, b)
        edges[key] = edges.get(key, 0) + 1
    return {
        'nodes': nodes,
        'edges': [{'source': a, 'target': b, 'value': v} for (a, b), v in edges.items()],
    }


def pick_cases(by_model):
    sessions = set(by_model['SRGNN']) & set(by_model['GCSAN']) & set(by_model['GCEGNN'])
    success, failure, disagreement = [], [], []
    for sid in sessions:
        r = {m: by_model[m][sid]['target_rank'] for m in MODELS}
        gce = r['GCEGNN']
        others = [r['SRGNN'], r['GCSAN']]
        if gce == 1 and all(x == 0 or x > 3 for x in others):
            success.append(sid)
        elif gce == 0 and any(x == 1 for x in others):
            failure.append(sid)
        elif (gce > 0 and all(x == 0 for x in others)) or (gce == 0 and any(x == 1 for x in others)):
            disagreement.append(sid)

    def diverse(candidates, n=6):
        buckets = {0: [], 1: [], 2: []}
        for sid in sorted(candidates):
            h = by_model['GCEGNN'][sid]['history_len']
            buckets[0 if h <= 9 else 1 if h <= 14 else 2].append(sid)
        picked = []
        for i in range(n):
            for b in range(3):
                if buckets[b]:
                    sid = buckets[b].pop(0)
                    if sid not in picked:
                        picked.append(sid)
            if len(picked) >= n:
                break
        return picked[:n]

    cases = []
    used = set()
    for kind, pool in [('success', success), ('disagreement', disagreement), ('failure', failure)]:
        pool = [sid for sid in pool if sid not in used]
        picked = diverse(pool, 6)
        used.update(picked)
        for sid in picked:
            cases.append((kind, sid))
    return cases


def main():
    outputs = ROOT / 'outputs/formal'
    metrics = []
    curves = {}
    for model in MODELS:
        result = json.loads((outputs / model / 'seed42/result.json').read_text(encoding='utf-8'))
        t = result['test_result']
        metrics.append({
            'model': LABELS[model],
            'key': model,
            'recall': t['recall@10'],
            'mrr': t['mrr@10'],
            'ndcg': t['ndcg@10'],
            'hit': t['hit@10'],
            'precision': t['precision@10'],
            'time': result['training_and_validation_seconds'],
            'params': result['parameters'],
            'valid_mrr': result['best_valid_score'],
        })
        curves[model] = parse_curves(outputs / model / 'seed42/train.log')

    with (outputs / 'comparison.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'recall@10', 'mrr@10', 'ndcg@10', 'hit@10', 'precision@10',
                         'training_and_validation_seconds', 'parameters', 'best_valid_mrr@10'])
        for m in metrics:
            writer.writerow([m['key'], f"{m['recall']:.6f}", f"{m['mrr']:.6f}", f"{m['ndcg']:.6f}",
                             f"{m['hit']:.6f}", f"{m['precision']:.6f}", f"{m['time']:.3f}",
                             m['params'], f"{m['valid_mrr']:.6f}"])

    by_model = {m: {} for m in MODELS}
    for model in MODELS:
        path = outputs / 'predictions' / f'{model}.jsonl'
        with path.open(encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                by_model[model][row['session_id']] = row

    selected = pick_cases(by_model)
    result0 = json.loads((outputs / 'GCEGNN/seed42/result.json').read_text(encoding='utf-8'))
    manifest = json.loads((outputs / 'data_manifest.json').read_text(encoding='utf-8'))
    meta = {
        'title': '下一次点击，预测得准吗？',
        'subtitle': '基于图神经网络的电商会话推荐 · 20k 真实会话闭环实验',
        'dataset': 'Diginetica（RecBole not-merged）',
        'protocol': '完整数据 item 5-core → 抽取 20,000 个 5–20 次点击会话 → 会话级留一划分',
        'seed': 42,
        'epochs': result0['epochs_requested'],
        'counts': {
            'sessions': result0['raw_session_count'],
            'items': result0['candidate_items'],
            'train': result0['train_examples'],
            'valid': result0['valid_examples'],
            'test': result0['test_examples'],
        },
        'funnel': {
            'source_sessions': manifest['source_sessions'],
            'source_items': manifest['source_items'],
            'source_interactions': manifest['source_interactions'],
            'eligible_sessions': manifest['eligible_sessions'],
            'selected_sessions': manifest['selected_sessions'],
            'selected_items': manifest['selected_items'],
            'selected_interactions': manifest['selected_interactions'],
        },
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'limits': [
            '单种子、单次实验；不宣称统计显著性',
            '商品 ID 已匿名化，无法还原真实名称或图片',
            '20k 会话子集，非官方排行榜完整复现',
            '未做简单基线与 GCE 去全局分支消融',
        ],
    }

    # Global neighbours from the trained GCE-GNN checkpoint, for the graph panel.
    global_map = {}
    try:
        import numpy as np
        import torch
        from recbole.utils import init_seed, init_logger
        from recbole_gnn.config import Config
        from recbole_gnn.utils import create_dataset, data_preparation, get_model

        torch.set_num_threads(4)
        gce_dir = outputs / 'GCEGNN' / 'seed42'
        config = Config(
            model='GCEGNN',
            dataset='diginetica-formal',
            config_file_list=[str(ROOT / 'configs/formal.yaml')],
            config_dict={'data_path': str(ROOT / 'data'), 'seed': 42,
                         'checkpoint_dir': str(gce_dir / 'checkpoints'), 'use_gpu': False},
        )
        init_seed(config['seed'], config['reproducibility'])
        init_logger(config)
        dataset = create_dataset(config)
        train, _, _ = data_preparation(config, dataset)
        gce = get_model('GCEGNN')(config, train.dataset)
        state = torch.load(Path(result0['checkpoint']), map_location=config['device'])
        gce.load_state_dict(state['state_dict'])
        gce.load_other_parameter(state.get('other_parameter'))
        gce.eval()
        adj = gce.global_adj.cpu()
        weight = gce.global_weight.cpu()

        def neighbours(item):
            try:
                idx = int(dataset.token2id(config['ITEM_ID_FIELD'], [item])[0])
            except Exception:
                return []
            out = []
            for j in range(adj.shape[1]):
                nid = int(adj[idx, j].item())
                if nid > 0:
                    out.append({'id': str(dataset.id2token(config['ITEM_ID_FIELD'], nid)),
                                'weight': float(weight[idx, j].item())})
            return out
    except Exception as exc:  # keep the dashboard usable if the global panel fails
        print('global neighbours unavailable:', exc, flush=True)

        def neighbours(item):
            return []

    cases = []
    for kind, sid in selected:
        base = by_model['GCEGNN'][sid]
        history = base['history']
        case = {
            'id': sid,
            'kind': kind,
            'history': history,
            'target': base['target'],
            'history_len': base['history_len'],
            'repeat_count': base['repeat_count'],
            'target_popularity_rank': base['target_popularity_rank'],
            'graph': local_graph(history),
            'global_neighbours': neighbours(history[-1]) if history else [],
            'models': {},
        }
        for model in MODELS:
            row = by_model[model][sid]
            case['models'][model] = {
                'rank': row['target_rank'],
                'recommendations': row['recommendations'],
            }
        cases.append(case)

    best = max(metrics, key=lambda m: m['recall'])
    srgnn = next(m for m in metrics if m['key'] == 'SRGNN')
    summary = {
        'best_model': best['model'],
        'best_recall': best['recall'],
        'gain_vs_srgnn_pct': (best['recall'] - srgnn['recall']) / srgnn['recall'] * 100.0,
        'time_ratio_vs_srgnn': best['time'] / srgnn['time'],
        'n_cases': len(cases),
    }

    web = {
        'meta': meta,
        'metrics': metrics,
        'summary': summary,
        'curves': curves,
        'cases': cases,
        'latency': {},
    }
    for model in MODELS:
        p = outputs / 'predictions' / f'{model}-latency.json'
        if p.exists():
            web['latency'][model] = json.loads(p.read_text(encoding='utf-8'))

    web_dir = ROOT.parent / 'web'
    web_dir.mkdir(parents=True, exist_ok=True)
    (web_dir / 'data.json').write_text(json.dumps(web, ensure_ascii=False, indent=2), encoding='utf-8')
    payload = json.dumps(web, ensure_ascii=False).replace('</', '<\\/')
    (web_dir / 'data.js').write_text('window.__DATA__ = ' + payload + ';\n', encoding='utf-8')
    print(json.dumps({'comparison': [m['model'] for m in metrics], 'cases': len(cases),
                      'summary': summary}, ensure_ascii=True, indent=2), flush=True)


if __name__ == '__main__':
    main()
