"""Export exact existing splits and fit training-only sparse structures."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy import sparse
import torch

from grey_session import GreySession
from grey_session_fast import FastGreySession

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/grey_improved'


def main():
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / 'RecBole-GNN'))
    sys.argv = sys.argv[:1]
    from recbole_gnn.config import Config
    from recbole_gnn.utils import create_dataset, data_preparation
    from recbole.utils import init_seed
    from run_grey import examples
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    config = Config(model='SRGNN', dataset='diginetica-formal',
                    config_file_list=[str(ROOT / 'configs/formal.yaml')],
                    config_dict={'data_path': str(ROOT / 'data'), 'seed': 42,
                                 'checkpoint_dir': str(OUT / 'unused_checkpoints'), 'use_gpu': False})
    init_seed(42, True)
    dataset = create_dataset(config)
    train, valid, test = data_preparation(config, dataset)
    split_hashes = {}
    all_rows = {}
    for name, loader in [('train', train), ('valid', valid), ('test', test)]:
        feat = loader.dataset.inter_feat
        identity = torch.stack([feat[config['USER_ID_FIELD']], feat[config['ITEM_ID_FIELD']]], dim=1)
        split_hashes[name] = hashlib.sha256(identity.numpy().tobytes()).hexdigest()
        all_rows[name] = examples(loader, config)
    for m in ['SRGNN', 'GCSAN', 'GCEGNN']:
        prior = json.loads((ROOT / f'outputs/formal/{m}/seed42/result.json').read_text(encoding='utf-8'))
        assert prior['split_session_target_sha256'] == split_hashes
    prefixes = {}
    for user, history, target in all_rows['train']:
        seq = np.append(history, target)
        if user not in prefixes or len(seq) > len(prefixes[user]):
            prefixes[user] = seq
    for user, history, _ in all_rows['valid']:
        np.testing.assert_array_equal(prefixes[user], history)
    for user, history, _ in all_rows['test']:
        np.testing.assert_array_equal(prefixes[user], history[:-1])
    payload = {}
    for split in ['valid', 'test']:
        rows = all_rows[split]
        histories = np.zeros((len(rows), 20), dtype=np.int64)
        for i, (_, history, _) in enumerate(rows):
            histories[i, :len(history)] = history
        payload.update({split + '_users': np.array([r[0] for r in rows]),
                        split + '_targets': np.array([r[2] for r in rows]),
                        split + '_lengths': np.array([len(r[1]) for r in rows]),
                        split + '_histories': histories})
    np.savez_compressed(OUT / 'splits.npz', **payload)
    item_tokens = dataset.field2id_token[config['ITEM_ID_FIELD']].tolist()
    user_tokens = dataset.field2id_token[config['USER_ID_FIELD']].tolist()
    (OUT / 'tokens.json').write_text(json.dumps({'items': item_tokens, 'users': user_tokens}), encoding='utf-8')
    start = time.perf_counter()
    original = GreySession(dataset.item_num).fit(prefixes.values())
    users = np.asarray(list(prefixes), dtype=np.int64)
    rr, cc = [], []
    for row, sequence in enumerate(prefixes.values()):
        unique = np.unique(sequence)
        rr.extend([row] * len(unique))
        cc.extend(unique.tolist())
    incidence = sparse.csr_matrix((np.ones(len(rr), dtype=np.float32), (rr, cc)),
                                  shape=(len(prefixes), dataset.item_num))
    model = FastGreySession(original, incidence, users)
    fit_seconds = time.perf_counter() - start
    model.save(OUT / 'model_initial')
    verification = {'split_hashes': split_hashes, 'exact_prefix_checks': True,
                    'sessions': len(prefixes), 'items': dataset.item_num - 1,
                    'train_prefix_clicks': int(original.counts.sum()),
                    'statistics_and_index_fit_seconds': fit_seconds,
                    'raw_array_bytes_initial': model.array_bytes(),
                    'source_sha256': hashlib.sha256((ROOT / 'data/diginetica-formal/diginetica-formal.inter').read_bytes()).hexdigest()}
    (OUT / 'prepare.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == '__main__':
    main()
