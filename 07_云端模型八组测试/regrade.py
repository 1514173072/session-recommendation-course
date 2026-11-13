"""Separate numeric-output validity from benign plan-tool protocol compliance.

No API calls, no new model outputs, no tuning on targets. Select the FIRST
actually usable coefficient vector; retain all retries and their physical cost.
"""
import json,hashlib,shutil
from pathlib import Path
from run import HERE,parse,summarize

def main():
    old=json.loads((HERE/'results/predictions.json').read_text(encoding='utf-8'))
    for name in ['summary.json','predictions.json']:
        dst=HERE/'pilot'/('strict_protocol_'+name)
        if not dst.exists():shutil.copy2(HERE/'results'/name,dst)
    results=[]
    for row in old:
        attempts=[]
        for a in row['attempts']:
            meta=json.loads((HERE/'requests'/(a['stem']+'.meta.json')).read_text(encoding='utf-8'))
            attempts.append({**meta,**parse(meta)})
        selected=next((i for i,a in enumerate(attempts) if a['success']),None)
        row.update(attempts=attempts,success=selected is not None,selected_attempt_index=selected,
            coefficients=attempts[selected].get('coefficients') if selected is not None else None,
            total_attempt_wall_seconds=sum(a['wall_seconds_including_harness_startup'] for a in attempts),
            recommendation_wall_seconds=sum(a['wall_seconds_including_harness_startup'] for a in attempts[:selected+1]) if selected is not None else sum(a['wall_seconds_including_harness_startup'] for a in attempts),
            successful_model_step_seconds=attempts[selected]['model_step_seconds'] if selected is not None else None)
        (HERE/'results'/f"{row['model']}_{row['condition']}_{row['case_index']:02d}.json").write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf-8')
        results.append(row)
    cases=json.loads((HERE/'cases.json').read_text(encoding='utf-8'))
    summary=summarize(results,cases)
    note={'reason':'Originally the parser rejected some valid numeric outputs solely for plan-control tool calls. Numeric recommendation validity and protocol compliance are now reported separately.',
        'selection':'first completed schema/range-valid actual output, independent of target/rank',
        'new_API_calls':0,'prior_outputs_preserved':'pilot/strict_protocol_predictions.json',
        'post_success_diagnostic_attempts':sum(len(r['attempts'])-r['selected_attempt_index']-1 for r in results if r['success']),
        'latency':'time until first usable output includes preceding failures; all physical attempt cost separately retained',
        'valid_numeric_outputs':sum(r['success'] for r in results)}
    (HERE/'results/evaluation_definition.json').write_text(json.dumps(note,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
