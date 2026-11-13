"""Repeat the unchanged three-model experiment and verify independently."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
TAG = 'rerun20261112'
OUT = ROOT / 'outputs' / TAG
MODELS = ('SRGNN', 'GCSAN', 'GCEGNN')


def write(data):
    (OUT / 'progress.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    state = {'started_at': datetime.now().isoformat(timespec='seconds'), 'tag': TAG,
             'status': 'running', 'steps': [], 'configuration': {'seed': 42, 'epochs_max': 20, 'device': 'cpu', 'threads': 4}}
    environment = os.environ.copy()
    environment['PYTHONIOENCODING'] = 'utf-8'
    environment['PYTHONUTF8'] = '1'
    tasks = []
    for model in MODELS:
        tasks.append((model + '-train', ['experiments/run_formal.py', '--model', model, '--epochs', '20', '--seed', '42', '--tag', TAG]))
        tasks.append((model + '-predictions', ['experiments/export_predictions.py', '--model', model, '--tag', TAG]))
    tasks.append(('independent-verification', ['experiments/verify_formal.py', '--tag', TAG]))
    try:
        for name, args in tasks:
            state['active_step'] = name
            state['updated_at'] = datetime.now().isoformat(timespec='seconds')
            write(state)
            print('START ' + name, flush=True)
            with (OUT / (name + '.log')).open('w', encoding='utf-8') as handle:
                result = subprocess.run([sys.executable] + args, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT)
            state['steps'].append({'name': name, 'exit_code': result.returncode, 'finished_at': datetime.now().isoformat(timespec='seconds')})
            write(state)
            if result.returncode:
                raise RuntimeError(name + ' failed; see ' + name + '.log')
            print('DONE ' + name, flush=True)
        comparison = []
        for model in MODELS:
            new = json.loads((ROOT / 'outputs/formal' / (model + '-' + TAG) / 'seed42/result.json').read_text(encoding='utf-8'))
            old = json.loads((ROOT / 'outputs/formal' / model / 'seed42/result.json').read_text(encoding='utf-8'))
            comparison.append({'model': model, 'test_result': new['test_result'],
                               'difference_from_previous': {key: new['test_result'][key] - old['test_result'][key] for key in old['test_result']},
                               'same_data_split': new['split_session_target_sha256'] == old['split_session_target_sha256']})
        assert all(row['same_data_split'] for row in comparison)
        report = {'completed_at': datetime.now().isoformat(timespec='seconds'), 'tag': TAG, 'models': comparison,
                  'dataset_sha256': hashlib.sha256((ROOT / 'data/diginetica-formal/diginetica-formal.inter').read_bytes()).hexdigest(),
                  'scope': 'Fresh training, full-test prediction export and independent checkpoint evaluation; same single seed 42.'}
        (OUT / 'rerun_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        state['status'] = 'complete'
        state['active_step'] = None
        state['completed_at'] = report['completed_at']
        write(state)
        print('ALL COMPLETE', flush=True)
    except Exception as error:
        state['status'] = 'failed'
        state['error'] = str(error)
        write(state)
        raise


if __name__ == '__main__':
    main()
