"""Sequential evaluation/benchmark/report runner; each benchmark is isolated."""
import subprocess
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent

def run(script,*args):
    label=script.removesuffix('.py')+('_'+args[-1] if args else '')
    print('START '+label,flush=True)
    with (HERE/(label+'.log')).open('w',encoding='utf-8') as f:
        p=subprocess.run([sys.executable,'-u',str(HERE/script),*args],cwd=HERE,stdout=f,stderr=subprocess.STDOUT)
    if p.returncode:
        print((HERE/(label+'.log')).read_text(encoding='utf-8')[-6000:],flush=True)
        raise SystemExit(p.returncode)
    print('DONE '+label,flush=True)

if __name__=='__main__':
    run('evaluate.py')
    # Fixed deterministic order; no simultaneous training/benchmark processes.
    for model in ['gra_original','student_teacher','GCSAN','teacher_rules','GCEGNN','student_plain','gra_beta7','SRGNN']:
        run('benchmark.py','--model',model)
    run('report.py')
    run('verify.py')
