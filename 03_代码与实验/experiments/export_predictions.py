"""Export test-set Top-10 predictions for one formal checkpoint.

Outputs one JSONL row per test session, with raw item IDs, target rank, history
length, repeat-click count and target popularity rank.  This is the data source
for the web dashboard and for post-hoc case selection.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / 'RecBole-GNN'
sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['SRGNN', 'GCSAN', 'GCEGNN'], required=True)
    parser.add_argument('--dataset', default='diginetica-formal')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int, default=0, help='0 means all test sessions')
    args = parser.parse_args()

    sys.argv = sys.argv[:1]
    os.chdir(ROOT)
    import numpy as np
    import torch
    from recbole.utils import init_seed, init_logger
    from recbole_gnn.config import Config
    from recbole_gnn.utils import create_dataset, data_preparation, get_model

    torch.set_num_threads(4)
    output = ROOT / 'outputs/formal' / args.model / f'seed{args.seed}'
    prior = json.loads((output / 'result.json').read_text(encoding='utf-8'))
    config = Config(
        model=args.model,
        dataset=args.dataset,
        config_file_list=[str(ROOT / 'configs/formal.yaml')],
        config_dict={
            'data_path': str(ROOT / 'data'),
            'seed': args.seed,
            'checkpoint_dir': str(output / 'checkpoints'),
            'use_gpu': False,
        },
    )
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    dataset = create_dataset(config)
    train, valid, test = data_preparation(config, dataset)

    model = get_model(args.model)(config, train.dataset).to(config['device'])
    state = torch.load(Path(prior['checkpoint']), map_location=config['device'])
    model.load_state_dict(state['state_dict'])
    model.load_other_parameter(state.get('other_parameter'))
    model.eval()

    counts = torch.bincount(train.dataset.inter_feat[config['ITEM_ID_FIELD']], minlength=dataset.item_num)
    order = torch.argsort(counts, descending=True)
    popularity_rank = {int(item): i + 1 for i, item in enumerate(order.tolist()) if item > 0}

    out_dir = ROOT / 'outputs/formal/predictions'
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    latency = None
    with (out_dir / f'{args.model}.jsonl').open('w', encoding='utf-8') as f:
        for batch_idx, batch in enumerate(test):
            interaction = batch[0] if isinstance(batch, tuple) else batch
            interaction = interaction.to(config['device'])
            if latency is None:
                with torch.no_grad():
                    model.full_sort_predict(interaction)
                start = time.perf_counter()
                with torch.no_grad():
                    scores = model.full_sort_predict(interaction)
                latency = (time.perf_counter() - start) * 1000.0 / interaction[config['ITEM_ID_FIELD']].shape[0]
            with torch.no_grad():
                scores = model.full_sort_predict(interaction).reshape(-1, dataset.item_num)
            scores = scores.clone()
            scores[:, 0] = -float('inf')
            values, ids = torch.topk(scores, k=10, dim=-1)

            lengths = interaction[config['ITEM_LIST_LENGTH_FIELD']]
            histories = interaction[config['ITEM_ID_FIELD'] + config['LIST_SUFFIX']]
            targets = interaction[config['ITEM_ID_FIELD']]
            sessions = interaction[config['USER_ID_FIELD']]
            for row in range(scores.shape[0]):
                n = int(lengths[row].item())
                history = histories[row, :n].cpu().numpy()
                if args.model == 'GCEGNN':
                    history = history[::-1].copy()
                target = int(targets[row].item())
                top_ids = ids[row].cpu().tolist()
                top_scores = values[row].cpu().tolist()
                rank = top_ids.index(target) + 1 if target in top_ids else 0
                record = {
                    'model': args.model,
                    'session_id': str(dataset.id2token(config['USER_ID_FIELD'], int(sessions[row].item()))),
                    'history': [str(x) for x in dataset.id2token(config['ITEM_ID_FIELD'], history)],
                    'target': str(dataset.id2token(config['ITEM_ID_FIELD'], target)),
                    'target_rank': rank,
                    'history_len': n,
                    'repeat_count': n - len(set(history.tolist())),
                    'target_popularity_rank': popularity_rank.get(target, 0),
                    'recommendations': [
                        {'rank': i + 1, 'item': str(dataset.id2token(config['ITEM_ID_FIELD'], item)), 'score': float(score)}
                        for i, (item, score) in enumerate(zip(top_ids, top_scores))
                    ],
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                rows_written += 1
                if args.limit and rows_written >= args.limit:
                    break
            if args.limit and rows_written >= args.limit:
                break
            if batch_idx % 20 == 0:
                print(f'{args.model}: {rows_written} rows', flush=True)

    latency_path = out_dir / f'{args.model}-latency.json'
    latency_path.write_text(json.dumps({'model': args.model, 'ms_per_session': latency}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'model': args.model, 'rows': rows_written, 'latency_ms_per_session': latency}, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
