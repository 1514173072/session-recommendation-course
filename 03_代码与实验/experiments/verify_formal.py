"""Independent-process reload check and Top-10 legality check for formal models."""
import json
import os
from pathlib import Path
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / 'RecBole-GNN'
sys.path.insert(0, str(REPO))
MODELS = ['SRGNN', 'GCSAN', 'GCEGNN']


def main():
    os.chdir(ROOT)
    import torch
    import numpy as np
    from recbole.utils import init_seed, init_logger
    from recbole_gnn.config import Config
    from recbole_gnn.utils import create_dataset, data_preparation, get_model, get_trainer

    torch.set_num_threads(4)
    summary = {'checked_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'models': {}, 'all_passed': True}
    for model_name in MODELS:
        out = ROOT / 'outputs/formal' / model_name / 'seed42'
        prior = json.loads((out / 'result.json').read_text(encoding='utf-8'))
        config = Config(
            model=model_name,
            dataset='diginetica-formal',
            config_file_list=[str(ROOT / 'configs/formal.yaml')],
            config_dict={'data_path': str(ROOT / 'data'), 'seed': 42,
                         'checkpoint_dir': str(out / 'checkpoints'), 'use_gpu': False},
        )
        init_seed(config['seed'], config['reproducibility'])
        init_logger(config)
        dataset = create_dataset(config)
        train, valid, test = data_preparation(config, dataset)
        model = get_model(model_name)(config, train.dataset)
        state = torch.load(Path(prior['checkpoint']), map_location='cpu')
        model.load_state_dict(state['state_dict'])
        model.load_other_parameter(state.get('other_parameter'))
        trainer = get_trainer(config['MODEL_TYPE'], model_name)(config, model)
        test_result = dict(trainer.evaluate(test, load_best_model=False, show_progress=False))
        expected = dict(prior['test_result'])
        diffs = {k: abs(float(test_result[k]) - float(expected[k])) for k in expected}
        reload_passed = all(v < 1e-12 for v in diffs.values())

        pred_path = ROOT / 'outputs/formal/predictions' / f'{model_name}.jsonl'
        rows = 0
        top10_ok = True
        bad_reason = ''
        with pred_path.open(encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                rows += 1
                recs = row['recommendations']
                items = [r['item'] for r in recs]
                if len(recs) != 10 or len(set(items)) != 10 or '0' in items:
                    top10_ok = False
                    bad_reason = f"session {row['session_id']}: invalid Top-10 {items}"
                    break
                if row['target_rank'] < 0 or row['target_rank'] > 10:
                    top10_ok = False
                    bad_reason = f"session {row['session_id']}: bad target rank {row['target_rank']}"
                    break
                if not all(isinstance(r['score'], (int, float)) for r in recs):
                    top10_ok = False
                    bad_reason = f"session {row['session_id']}: non-numeric score"
                    break
        top10_ok = top10_ok and rows == prior['test_examples']

        passed = reload_passed and top10_ok
        summary['models'][model_name] = {
            'checkpoint': prior['checkpoint'],
            'test_result_reloaded': test_result,
            'max_metric_difference': max(diffs.values()),
            'reload_passed': reload_passed,
            'prediction_rows': rows,
            'expected_rows': prior['test_examples'],
            'top10_passed': top10_ok,
            'top10_error': bad_reason,
            'passed': passed,
        }
        summary['all_passed'] = summary['all_passed'] and passed
        trainer.tensorboard.close()
        print(f'{model_name}: reload={reload_passed} top10={top10_ok} rows={rows}', flush=True)

    summary['scope'] = 'Independent-process reload and Top-10 legality check for the 20k formal experiment.'
    (ROOT / 'outputs/formal/verification_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=True, indent=2), flush=True)


if __name__ == '__main__':
    main()
