"""Audit every real request and recompute all240 full-catalogue recommendations."""
from pathlib import Path
import csv,hashlib,json,math,sys
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'06_教师学生推荐性能测试'))
from common import load_grey,numpy_score,stable_topk
from run import parse

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    torch.set_num_threads(4)
    manifest=json.loads((HERE/'manifest.json').read_text(encoding='utf-8'))
    frozen=json.loads((HERE/'frozen_before_requests.json').read_text(encoding='utf-8'))
    cases=json.loads((HERE/'cases.json').read_text(encoding='utf-8'))
    summaries=json.loads((HERE/'results/summary.json').read_text(encoding='utf-8'))
    predictions=json.loads((HERE/'results/predictions.json').read_text(encoding='utf-8'))
    assert sha(HERE/'manifest.json')==frozen['manifest_sha256']
    assert sha(HERE/'cases.json')==frozen['cases_sha256']
    assert sha(ROOT/'06_教师学生推荐性能测试/teacher/prototypes.json')==frozen['teacher_sha256']
    assert sha(Path.home()/'.dsh/profiles/desktop/cordis.patch.yml')==frozen['desktop_config_sha256']
    for model,h in frozen['patches'].items():assert sha(HERE/f'patches/{model}.yml')==h
    tokens=json.loads((ROOT/'03_代码与实验/outputs/grey_improved/tokens.json').read_text(encoding='utf-8'))
    with np.load(ROOT/'06_教师学生推荐性能测试/cache/splits.npz',allow_pickle=False) as z:
        split={k:z[k] for k in z.files}
    expected_indices=np.random.default_rng(20261112).choice(len(split['test_users']),30,replace=False).tolist()
    assert [c['test_index'] for c in cases]==expected_indices
    for c in cases:
        i=c['test_index']
        assert c['internal_history']==split['test_histories'][i,:split['test_lengths'][i]].tolist()
        assert c['internal_target']==split['test_targets'][i]
        assert c['target']==str(tokens['items'][c['internal_target']])
    proto=json.loads((ROOT/'06_教师学生推荐性能测试/teacher/prototypes.json').read_text(encoding='utf-8'))
    proto_by={r['group']:r for r in proto['prototypes']}
    for name,h in frozen['prompts'].items():
        p=HERE/'prompts'/name
        assert sha(p)==h
        s=p.read_text(encoding='utf-8')
        evidence=json.loads(s.split('Session evidence:\n')[1].splitlines()[0])
        assert set(evidence)=={'history','length','unique_ratio','last_product_repeated',
            'maximum_product_frequency_ratio','last_product_training_popularity_normalized',
            'last_product_transition_entropy'}
        condition,index=name[:-4].split('_');c=cases[int(index)]
        assert evidence['history']==c['history']
        if condition=='teacher':
            guide=json.loads(s.split('warrants it:\n')[1].splitlines()[0])
            ref=proto_by[c['group']]
            assert guide=={k:ref[k] for k in ['weights','interactions','rationale']}
        else:assert 'Teacher guidance derived' not in s
    assert len(predictions)==240
    keys={(p['model'],p['condition'],p['case_index']) for p in predictions}
    assert keys=={(m,c,i) for m in manifest['models'] for c in ['plain','teacher'] for i in range(30)}
    grey=load_grey()
    features={i:grey.normalized_features(np.asarray(c['internal_history'],dtype=np.int64)) for i,c in enumerate(cases)}
    groups={};attempts=0;valid=0;all_finished=True
    for row in predictions:
        model,condition,index=row['model'],row['condition'],row['case_index']
        case=cases[index]
        assert row['session_id']==case['session_id'] and row['history']==case['history'] and row['target']==case['target']
        prompt_path=HERE/f'prompts/{condition}_{index:02d}.txt'
        request_sha=hashlib.sha256(prompt_path.read_text(encoding='utf-8').encode('utf-8')).hexdigest()
        assert row['prompt_sha256']==request_sha
        assert row['prompt_file_sha256']==sha(prompt_path)
        assert 1<=len(row['attempts'])<=2
        for a in row['attempts']:
            attempts+=1
            saved=json.loads((HERE/'requests'/(a['stem']+'.meta.json')).read_text(encoding='utf-8'))
            assert saved['model']==model and saved['condition']==condition and saved['case_index']==index
            assert saved['patch_sha256']==frozen['patches'][model]
            assert saved['prompt_sha256']==request_sha
            assert saved['wall_seconds_including_harness_startup']>0
            parsed=parse(saved)
            assert parsed['success']==a['success']
            assert parsed['final_text']==a['final_text']
            assert parsed['thinking_characters']==a['thinking_characters']
            if parsed['success']:
                np.testing.assert_array_equal(parsed['coefficients'],a['coefficients'])
                assert saved['model_step_seconds']>0
                ee=saved['event_timestamps']
                start=next(x['elapsed_seconds'] for x in ee if x['phase']=='step_start')
                end=[x['elapsed_seconds'] for x in ee if x['phase']=='step_end'][-1]
                assert abs(end-start-saved['model_step_seconds'])<1e-9
        selected=next((i for i,a in enumerate(row['attempts']) if a['success']),None)
        assert row['selected_attempt_index']==selected
        assert row['success']==(selected is not None)
        expected_wall=sum(a['wall_seconds_including_harness_startup'] for a in row['attempts'][:selected+1]) if selected is not None else sum(a['wall_seconds_including_harness_startup'] for a in row['attempts'])
        assert abs(row['recommendation_wall_seconds']-expected_wall)<1e-9
        for a in row['attempts']:assert set(a.get('tool_names',[]))<={'exit_plan_mode'}
        if row['success']:
            valid+=1
            np.testing.assert_array_equal(row['coefficients'],row['attempts'][selected]['coefficients'])
            assert row['successful_model_step_seconds']==row['attempts'][selected]['model_step_seconds']
            coeff=np.asarray(row['coefficients'],dtype=np.float32)
            scores=numpy_score(features[index],coeff);scores[0]=-np.inf
            ids=stable_topk(scores)
            expected=[str(tokens['items'][x]) for x in ids]
            assert expected==[r['item'] for r in row['recommendations']]
            assert len(expected)==len(set(expected))==10
            rank=expected.index(case['target'])+1 if case['target'] in expected else 0
            assert rank==row['target_rank']
            for item,rr in zip(ids,row['recommendations']):assert abs(float(scores[item])-rr['score'])<1e-7
        else:
            assert row['target_rank']==0 and not row['recommendations']
        groups.setdefault((model,condition),[]).append(row)
    assert len(groups)==8 and all(len(rr)==30 for rr in groups.values())
    checks={}
    for summary in summaries:
        key=(summary['model'],summary['condition']);rr=groups[key]
        hits=sum(r['target_rank']>0 for r in rr)
        mrr=sum(1/r['target_rank'] for r in rr if r['target_rank'])/30
        ndcg=sum(1/math.log2(r['target_rank']+1) for r in rr if r['target_rank'])/30
        for k,v in [('recall@10',hits/30),('mrr@10',mrr),('ndcg@10',ndcg)]:
            assert abs(summary[k]-v)<1e-12
        assert summary['valid_outputs']==sum(r['success'] for r in rr)
        assert summary['actual_attempts']==sum(len(r['attempts']) for r in rr)
        steps=[r['successful_model_step_seconds'] for r in rr if r['success']]
        if steps:assert abs(summary['mean_model_step_seconds']-np.mean(steps))<1e-9
        assert abs(summary['mean_all_attempt_wall_seconds']-np.mean([r['total_attempt_wall_seconds'] for r in rr]))<1e-9
        assert abs(summary['mean_recommendation_wall_seconds']-np.mean([r['recommendation_wall_seconds'] for r in rr]))<1e-9
        checks[' / '.join(key)]={'queries':30,'valid_outputs':summary['valid_outputs'],
            'independent_recall':hits/30,'independent_mrr':mrr,'actual_attempts':summary['actual_attempts']}
    dest=ROOT/'08扩展对比/云端八组对比'
    with (dest/'云端八组对比.csv').open(encoding='utf-8-sig',newline='') as f:csv_rows=list(csv.DictReader(f))
    assert len(csv_rows)==8
    by={(s['model'],s['condition']):s for s in summaries}
    for r in csv_rows:
        s=by[(r['模型ID'],'teacher' if r['教师指导']=='有' else 'plain')]
        assert abs(float(r['Recall@10'])-s['recall@10'])<1e-12
        assert abs(float(r['平均总调用耗时_s'])-s['mean_all_attempt_wall_seconds'])<1e-9
    export=json.loads((HERE/'figures/export_audit.json').read_text(encoding='utf-8'))
    assert export['visual_checked']
    assert sha(dest/'云端八组对比.png')==export['figure_sha256']
    assert sha(dest/'云端八组对比.csv')==export['csv_sha256']
    audit={'all_passed':True,'groups':8,'queries_per_group':30,'planned_cases':240,
        'actual_attempts':attempts,'valid_final_outputs':valid,'same_cases_all_groups':True,
        'all_full_catalogue_recommendations_recomputed':True,'prompts_exclude_test_targets':True,
        'frozen_teacher_prior_matches_training_source':True,'prompt_file_and_wire_newlines_verified':True,
        'desktop_Harness_configuration_unchanged':True,'group_checks':checks,
        'timing_scope':frozen['timing_scope'],'cloud_weight_training':False,
        'numeric_validity_separate_from_tool_protocol':True,
        'prediction_uses_first_actual_valid_output':True,
        'figure_and_csv_checked':True,'script_sha256':{p.name:sha(p) for p in HERE.glob('*.py')}}
    (HERE/'results/completion_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
