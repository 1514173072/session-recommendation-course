"""Official RecBole-GNN training plus checkpoint round-trip and real-session prediction."""
import argparse
import hashlib
import json
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['SRGNN', 'GCSAN', 'GCEGNN'], required=True)
    parser.add_argument('--dataset', default='diginetica-smoke')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--inference-only', action='store_true')
    args = parser.parse_args()
    # RecBole reads sys.argv itself. Give it the explicit Config arguments below.
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
    output = ROOT / 'outputs' / args.dataset / args.model
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / 'checkpoints'
    config = Config(model=args.model, dataset=args.dataset,
        config_file_list=[str(ROOT / 'configs/smoke.yaml')],
        config_dict={'data_path': str(ROOT / 'data'), 'epochs': args.epochs,
                     'checkpoint_dir': str(checkpoint_dir), 'use_gpu': args.device == 'cuda'})
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    config_text = str(config)
    (output / 'effective_config.txt').write_text(config_text, encoding='utf-8')
    start = time.perf_counter()
    dataset = create_dataset(config)
    train, valid, test = data_preparation(config, dataset)
    prep_seconds = time.perf_counter() - start
    split_fingerprints = {}
    for name, loader in [('train', train), ('valid', valid), ('test', test)]:
        identity = torch.stack([loader.dataset.inter_feat[config['USER_ID_FIELD']],
                                loader.dataset.inter_feat[config['ITEM_ID_FIELD']]], dim=1)
        split_fingerprints[name] = hashlib.sha256(identity.numpy().tobytes()).hexdigest()
    init_seed(config['seed'], config['reproducibility'])
    model_class = get_model(args.model)
    model = model_class(config, train.dataset).to(config['device'])
    trainer_class = get_trainer(config['MODEL_TYPE'], args.model)
    trainer = trainer_class(config, model)
    if args.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    if args.inference_only:
        prior = json.loads((output / 'result.json').read_text(encoding='utf-8'))
        assert prior['split_session_target_sha256'] == split_fingerprints, 'Data split differs from saved training run.'
        checkpoint = Path(prior['checkpoint'])
        state = torch.load(checkpoint, map_location=config['device'])
        model.load_state_dict(state['state_dict'])
        model.load_other_parameter(state.get('other_parameter'))
        valid_result, valid_score, training_seconds = prior['best_valid_result'], prior['best_valid_score'], 0.0
    else:
        start = time.perf_counter()
        valid_score, valid_result = trainer.fit(train, valid, saved=True, show_progress=False)
        if args.device == 'cuda':
            torch.cuda.synchronize()
        training_seconds = time.perf_counter() - start
        checkpoint = Path(trainer.saved_model_file)
    start = time.perf_counter()
    test_result = trainer.evaluate(test, load_best_model=not args.inference_only, show_progress=False)
    test_seconds = time.perf_counter() - start
    if args.inference_only:
        assert dict(test_result) == prior['test_result'], 'Reloaded test metrics differ from the training run.'
    interaction, _, _, _ = next(iter(test))
    interaction = interaction.to(config['device'])
    model.eval()
    with torch.no_grad():
        before = model.full_sort_predict(interaction).reshape(-1, dataset.item_num)
    # Rebuild from the same TRAINING dataset, especially for GCEGNN's global graph.
    reloaded = model_class(config, train.dataset).to(config['device'])
    state = torch.load(checkpoint, map_location=config['device'])
    reloaded.load_state_dict(state['state_dict'])
    reloaded.load_other_parameter(state.get('other_parameter'))
    reloaded.eval()
    if args.device == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        after = reloaded.full_sort_predict(interaction).reshape(-1, dataset.item_num)
    if args.device == 'cuda':
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - start
    assert torch.isfinite(after).all(), 'Non-finite prediction scores.'
    max_difference = (before - after).abs().max().item()
    assert torch.allclose(before, after, atol=1e-6, rtol=1e-5), max_difference
    scores = after.clone()
    scores[:, 0] = -float('inf')  # Padding is never a recommendation.
    values, ids = torch.topk(scores, k=10, dim=-1)
    assert (ids > 0).all()
    assert len(torch.unique(ids[0])) == 10
    cases = []
    for row in range(min(10, after.shape[0])):
        n = int(interaction[config['ITEM_LIST_LENGTH_FIELD']][row].item())
        history = interaction[config['ITEM_ID_FIELD'] + config['LIST_SUFFIX']][row, :n].cpu().numpy()
        if args.model == 'GCEGNN':
            history = history[::-1].copy()  # Convert model's reversed sequence back to browsing order.
        target = int(interaction[config['ITEM_ID_FIELD']][row].item())
        raw_ids = dataset.id2token(config['ITEM_ID_FIELD'], ids[row].cpu().numpy()).tolist()
        cases.append({'session_id': str(dataset.id2token(config['USER_ID_FIELD'], int(interaction[config['USER_ID_FIELD']][row].item()))),
                      'history_item_ids': dataset.id2token(config['ITEM_ID_FIELD'], history).tolist(),
                      'true_next_item_id': str(dataset.id2token(config['ITEM_ID_FIELD'], target)),
                      'target_in_top10': bool((ids[row] == target).any().item()),
                      'recommendations': [{'rank': i+1, 'item_id': item, 'model_score': float(values[row, i].item())} for i, item in enumerate(raw_ids)]})
    result = {'model': args.model, 'dataset': args.dataset,
              'scope': 'Small-scale runtime verification; not official leaderboard reproduction.',
              'epochs_requested': args.epochs,
              'seed': config['seed'], 'device': str(config['device']),
              'raw_session_count': dataset.user_num - 1, 'candidate_items': dataset.item_num - 1,
              'train_examples': len(train.dataset), 'valid_examples': len(valid.dataset), 'test_examples': len(test.dataset),
              'split_session_target_sha256': split_fingerprints,
              'best_valid_score': valid_score, 'best_valid_result': valid_result, 'test_result': test_result,
              'data_preparation_seconds': prep_seconds, 'training_and_validation_seconds': training_seconds,
              'test_seconds': test_seconds, 'prediction_batch_size': after.shape[0],
              'prediction_batch_seconds': inference_seconds,
              'checkpoint': str(checkpoint), 'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'checkpoint_reload_passed': True, 'max_reload_score_difference': max_difference,
              'top10_validation_passed': True,
              'parameters': sum(p.numel() for p in model.parameters()),
              'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
                              'recbole': recbole.__version__, 'pyg': torch_geometric.__version__,
                              'upstream_commit': subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()}}
    if args.device == 'cuda':
        result['peak_gpu_allocated_bytes'] = torch.cuda.max_memory_allocated()
    (output / ('inference_check.json' if args.inference_only else 'result.json')).write_text(json.dumps(json_safe(result), ensure_ascii=False, indent=2), encoding='utf-8')
    if args.inference_only:
        # Interaction.__len__ includes graph-sized tensors. Record actual score rows instead.
        # Refresh prediction diagnostics from this measured independent-process check.
        prior['prediction_batch_size'] = after.shape[0]
        prior['prediction_batch_seconds'] = inference_seconds
        prior['prediction_measurement'] = 'Independent checkpoint reload check; see inference_check.json.'
        prior['independent_process_reload_passed'] = True
        (output / 'result.json').write_text(json.dumps(json_safe(prior), ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'recommendations.json').write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding='utf-8')
    trainer.tensorboard.close()
    print(json.dumps(json_safe(result), ensure_ascii=True, indent=2), flush=True)

if __name__ == '__main__':
    main()
