"""Formal multi-seed runner for SRGNN / GCSAN / GCEGNN and the GCE local-only ablation.

The official model implementations are imported unchanged.  The only code-level
change is the ``--variant local`` subclass, which removes the global branch at
forward time while keeping the same dataset, split and local graph.
"""
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / 'RecBole-GNN'
sys.path.insert(0, str(REPO))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, 'item'):
        return value.item()
    if isinstance(value, (str, float, int, bool)) or value is None:
        return value
    return str(value)


def make_local_gce():
    import torch.nn.functional as F
    from recbole_gnn.model.sequential_recommender.gcegnn import GCEGNN

    class GCEGNNLocal(GCEGNN):
        """GCE-GNN with the global branch removed; local graph + fusion retained."""

        def __init__(self, config, dataset):
            config['build_global_graph'] = False
            super().__init__(config, dataset)
            self.other_parameter_name = []

        def forward(self, x, edge_index, edge_attr, alias_inputs, item_seq_len):
            mask = alias_inputs.gt(0).unsqueeze(-1)
            h = self.item_embedding(x)
            h_local = self.local_agg(h, edge_index, edge_attr)
            h_local = F.dropout(h_local, self.dropout_local, training=self.training)
            h_local = h_local[alias_inputs]
            return self.fusion(h_local, mask)

    return GCEGNNLocal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['SRGNN', 'GCSAN', 'GCEGNN'], required=True)
    parser.add_argument('--dataset', default='diginetica-formal')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--variant', choices=['base', 'local'], default='base')
    parser.add_argument('--tag', default='')
    args = parser.parse_args()
    if args.variant == 'local' and args.model != 'GCEGNN':
        raise SystemExit('--variant local is only defined for GCEGNN.')

    sys.argv = sys.argv[:1]
    os.chdir(ROOT)
    import numpy as np
    import torch
    import recbole
    import torch_geometric
    from recbole.utils import init_seed, init_logger
    from recbole_gnn.config import Config
    from recbole_gnn.utils import create_dataset, data_preparation, get_model, get_trainer

    torch.set_num_threads(4)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but this environment does not provide a CUDA device.')

    label = args.model + ('-local' if args.variant == 'local' else '')
    if args.tag:
        label += '-' + args.tag
    output = ROOT / 'outputs/formal' / label / f'seed{args.seed}'
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / 'checkpoints'
    config = Config(
        model=args.model,
        dataset=args.dataset,
        config_file_list=[str(ROOT / 'configs/formal.yaml')],
        config_dict={
            'data_path': 'data',
            'epochs': args.epochs,
            'seed': args.seed,
            'checkpoint_dir': str((checkpoint_dir).relative_to(ROOT)),
            'use_gpu': args.device == 'cuda',
        },
    )
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    log_path = output / 'train.log'
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    config_text = str(config)
    (output / 'effective_config.txt').write_text(config_text, encoding='utf-8')

    start = time.perf_counter()
    dataset = create_dataset(config)
    train, valid, test = data_preparation(config, dataset)
    prep_seconds = time.perf_counter() - start
    split_fingerprints = {}
    for name, loader in [('train', train), ('valid', valid), ('test', test)]:
        identity = torch.stack(
            [loader.dataset.inter_feat[config['USER_ID_FIELD']], loader.dataset.inter_feat[config['ITEM_ID_FIELD']]],
            dim=1,
        )
        split_fingerprints[name] = hashlib.sha256(identity.numpy().tobytes()).hexdigest()

    init_seed(config['seed'], config['reproducibility'])
    if args.variant == 'local':
        model_class = make_local_gce()
    else:
        model_class = get_model(args.model)
    model = model_class(config, train.dataset).to(config['device'])
    trainer_class = get_trainer(config['MODEL_TYPE'], args.model)
    trainer = trainer_class(config, model)

    start = time.perf_counter()
    valid_score, valid_result = trainer.fit(train, valid, saved=True, show_progress=False)
    training_seconds = time.perf_counter() - start
    checkpoint = Path(trainer.saved_model_file)

    start = time.perf_counter()
    test_result = trainer.evaluate(test, load_best_model=True, show_progress=False)
    test_seconds = time.perf_counter() - start

    result = {
        'model': args.model,
        'variant': args.variant,
        'dataset': args.dataset,
        'scope': 'Formal course-project experiment on a 20k-session disjoint subset; not an official leaderboard reproduction.',
        'epochs_requested': args.epochs,
        'seed': args.seed,
        'device': str(config['device']),
        'raw_session_count': dataset.user_num - 1,
        'candidate_items': dataset.item_num - 1,
        'train_examples': len(train.dataset),
        'valid_examples': len(valid.dataset),
        'test_examples': len(test.dataset),
        'split_session_target_sha256': split_fingerprints,
        'best_valid_score': valid_score,
        'best_valid_result': valid_result,
        'test_result': test_result,
        'data_preparation_seconds': prep_seconds,
        'training_and_validation_seconds': training_seconds,
        'test_seconds': test_seconds,
        'checkpoint': checkpoint.resolve().relative_to(ROOT.resolve()).as_posix(),
        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'parameters': sum(p.numel() for p in model.parameters()),
        'log_file': log_path.relative_to(ROOT).as_posix(),
        'environment': {
            'python': platform.python_version(),
            'torch': torch.__version__,
            'numpy': np.__version__,
            'recbole': recbole.__version__,
            'pyg': torch_geometric.__version__,
            'upstream_commit': subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
        },
    }
    (output / 'result.json').write_text(json.dumps(json_safe(result), ensure_ascii=False, indent=2), encoding='utf-8')
    logging.getLogger().removeHandler(file_handler)
    file_handler.close()
    trainer.tensorboard.close()
    print(json.dumps(json_safe(result), ensure_ascii=True, indent=2), flush=True)


if __name__ == '__main__':
    main()
