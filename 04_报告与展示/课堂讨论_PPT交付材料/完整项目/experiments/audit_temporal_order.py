"""Compare exported histories with exact-second and framework float32 ordering."""
from collections import defaultdict
import csv
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
from recbole.data.dataset.dataset import Dataset

ROOT=Path(__file__).resolve().parents[1]
TAG='rerun20261112'


def main():
    raw_path=ROOT/'data/diginetica-formal/diginetica-formal.inter'
    pred_path=ROOT/'outputs/formal'/('predictions-'+TAG)/'SRGNN.jsonl'
    sessions=defaultdict(list)
    with raw_path.open(encoding='utf-8') as handle:
        for row in csv.DictReader(handle,delimiter='\t'):
            sessions[row['session_id:token']].append((float(row['timestamp:float']),row['item_id:token']))
    report={'sessions':len(sessions),'exact_history_mismatches':[],'exact_target_mismatches':[],'float32_mismatches':[]}
    with pred_path.open(encoding='utf-8') as handle:
        rows=0
        for line in handle:
            row=json.loads(line);rows+=1
            raw=sessions[row['session_id']]
            exact=[item for time,item in sorted(raw,key=lambda x:x[0])]
            converted=[item for time,item in sorted(raw,key=lambda x:np.float32(x[0]))]
            if row['history']!=exact[:-1]:report['exact_history_mismatches'].append(row['session_id'])
            if row['target']!=exact[-1]:report['exact_target_mismatches'].append(row['session_id'])
            if row['history']!=converted[:-1] or row['target']!=converted[-1]:report['float32_mismatches'].append(row['session_id'])
    assert rows==len(sessions)==20000 and not report['float32_mismatches']
    report['interpretation']='RecBole converts float fields to float32 before sequential ordering. Nearby epoch-second timestamps can become equal. Histories and labels follow the actual training/evaluation dataset order.'
    report['dataset_conversion_code']=inspect.getsource(Dataset._dataframe_to_interaction)
    report['sources']=[{'path':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in (raw_path,pred_path)]
    (ROOT/'outputs'/TAG/'temporal_order_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('TEMPORAL AUDIT:',len(report['exact_history_mismatches']),'history differences;',len(report['exact_target_mismatches']),'target differences; float32-order match passed',flush=True)


if __name__=='__main__':
    main()
