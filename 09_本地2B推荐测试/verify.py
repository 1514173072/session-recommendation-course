"""Recompute all ranks and metrics from saved real Ollama responses."""
from pathlib import Path
import csv,hashlib,json,math,sys
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(HERE))
from run import MODEL,OPTIONS,parse,sha,load_grey

def main():
    source=ROOT/'07_云端模型八组测试'
    frozen=json.loads((HERE/'frozen_before_requests.json').read_text(encoding='utf-8'))
    cases=json.loads((source/'cases.json').read_text(encoding='utf-8'))
    records=json.loads((HERE/'results/predictions.json').read_text(encoding='utf-8'))
    summary=json.loads((HERE/'results/summary.json').read_text(encoding='utf-8'))
    assert frozen['cases_sha256']==sha(source/'cases.json')
    assert frozen['teacher_sha256']==sha(ROOT/'06_教师学生推荐性能测试/teacher/prototypes.json')
    assert frozen['run_script_sha256']==sha(HERE/'run.py')
    assert len(records)==60 and len({(r['condition'],r['case_index']) for r in records})==60
    grey=load_grey()
    tokens=json.loads((ROOT/'03_代码与实验/outputs/grey_improved/tokens.json').read_text(encoding='utf-8'))['items']
    attempts=0;valid=0
    for row in records:
        i=row['case_index'];condition=row['condition'];case=cases[i]
        prompt_path=source/f'prompts/{condition}_{i:02d}.txt'
        assert sha(prompt_path)==frozen['prompts'][prompt_path.name]
        prompt=prompt_path.read_text(encoding='utf-8')
        evidence=json.loads(prompt.split('Session evidence:\n')[1].splitlines()[0])
        assert set(evidence)=={'history','length','unique_ratio','last_product_repeated','maximum_product_frequency_ratio','last_product_training_popularity_normalized','last_product_transition_entropy'}
        assert evidence['history']==case['history']
        numeric=[]
        for a in row['attempts']:
            attempts+=1
            assert a['prompt_sha256']==hashlib.sha256(prompt.encode('utf-8')).hexdigest()
            disk=json.loads((HERE/f'requests/{condition}_{i:02d}_a{a["attempt"]}.json').read_text(encoding='utf-8'))
            assert disk==a
            if 'response' in a:
                parsed=parse(a['response']);assert parsed['success']==a['success']
                if parsed['success']:assert parsed['coefficients']==a['coefficients']
            numeric.append(a['success'])
        assert row['success']==any(numeric)
        assert sum(numeric)<=1 and (not row['success'] or numeric[-1])
        assert math.isclose(row['api_wall_seconds'],sum(a['wall_seconds'] for a in row['attempts']))
        actual_rank=0
        if row['success']:
            valid+=1
            c=np.asarray(row['attempts'][-1]['coefficients'],dtype=np.float32)
            z=grey.normalized_features(np.asarray(case['internal_history'],dtype=np.int64))
            scores=(z*c[:4,None]).sum(axis=0)
            for j,(a,b) in enumerate([(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]):scores+=c[4+j]*z[a]*z[b]
            scores[0]=-np.inf
            ids=np.lexsort((np.arange(len(scores)),-scores))[:10]
            assert [str(tokens[x]) for x in ids]==row['recommendations']
            hits=np.flatnonzero(ids==case['internal_target'])
            actual_rank=int(hits[0]+1) if len(hits) else 0
        assert actual_rank==row['target_rank']
    for s in summary:
        rr=[r for r in records if r['condition']==s['condition']]
        assert len(rr)==30
        ranks=[r['target_rank'] for r in rr]
        assert math.isclose(s['recall@10'],sum(r>0 for r in ranks)/30)
        assert math.isclose(s['mrr@10'],sum(1/r if r else 0 for r in ranks)/30)
        assert math.isclose(s['ndcg@10'],sum(1/math.log2(r+1) if r else 0 for r in ranks)/30)
        assert math.isclose(s['mean_api_wall_seconds'],sum(r['api_wall_seconds'] for r in rr)/30)
    export_path=ROOT/'08扩展对比/本地2B对比/本地2B与已有模型对比.csv'
    with export_path.open(encoding='utf-8-sig',newline='') as f:export=list(csv.DictReader(f))
    assert len(export)==13
    for s in summary:
        e=next(r for r in export if r['模型']==MODEL and r['条件']==('有教师' if s['condition']=='teacher' else '无教师'))
        assert float(e['Recall@10'])==s['recall@10'] and float(e['MRR@10'])==s['mrr@10']
    report={'all_passed':True,'unique_test_cases':30,'conditions':2,'case_results':60,'valid_outputs':valid,
         'actual_api_attempts':attempts,'source_cases_and_prompts_unchanged':True,'teacher_source_unchanged':True,
         'no_heldout_target_field_in_prompts':True,'all_predictions_recomputed_from_saved_outputs':True,
         'metrics_and_times_recomputed':True,'exported_csv_rows':13,'export_sha256':sha(export_path),
         'scripts':{p.name:sha(p) for p in [HERE/'run.py',Path(__file__)]}}
    (HERE/'results/completion_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
