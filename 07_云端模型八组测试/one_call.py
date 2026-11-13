"""One real Harness request; credentials stay inside the existing Harness home."""
from pathlib import Path
import argparse,json,os,subprocess,time,hashlib,threading

HERE=Path(__file__).resolve().parent
CLI=r'D:\dsh\resources\runtime\cli\bin\dsh.cmd'
AUTH=r'D:\dsh-model-check\headless-route-auth.patch.yml'

def call(model,condition,index,attempt=0,timeout=100):
    prompt=(HERE/f'prompts/{condition}_{index:02d}.txt').read_text(encoding='utf-8')
    stem=f'{model}_{condition}_{index:02d}_a{attempt}'
    output=HERE/'requests'/stem
    cmd=[CLI,'--profile','headless','--patch',AUTH,'--patch',str(HERE/f'patches/{model}.yml'),'--json','-']
    env=os.environ.copy()
    # dsh's own adapter supplies its native client/session headers.
    # No user-agent spoofing and no direct-key HTTP calls.
    start=time.perf_counter()
    proc=subprocess.Popen(cmd,cwd=HERE,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0,env=env)
    timed_out=False
    stdout_parts=[];stderr_parts=[];events=[]
    def read_stdout():
        for line in iter(proc.stdout.readline,b''):
            stdout_parts.append(line)
            try:
                obj=json.loads(line.decode('utf-8'))
                events.append({'elapsed_seconds':time.perf_counter()-start,'type':obj.get('type'),
                    'phase':obj.get('phase')})
            except (ValueError,UnicodeDecodeError):pass
    def read_stderr():
        stderr_parts.append(proc.stderr.read())
    threads=[threading.Thread(target=read_stdout),threading.Thread(target=read_stderr)]
    for thread in threads:thread.start()
    proc.stdin.write(prompt.encode('utf-8'));proc.stdin.close()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out=True
        # Terminate only this specific request process and its children.
        import psutil
        try:
            parent=psutil.Process(proc.pid)
            for child in parent.children(recursive=True):
                try:child.kill()
                except psutil.NoSuchProcess:pass
            parent.kill()
        except psutil.NoSuchProcess:pass
        proc.wait()
    for thread in threads:thread.join()
    stdout=b''.join(stdout_parts);stderr=b''.join(stderr_parts)
    elapsed=time.perf_counter()-start
    Path(str(output)+'.ndjson').write_bytes(stdout)
    Path(str(output)+'.stderr.txt').write_bytes(stderr)
    record={'model':model,'condition':condition,'case_index':index,'attempt':attempt,
        'wall_seconds_including_harness_startup':elapsed,'exit_code':proc.returncode,
        'timed_out':timed_out,'prompt_sha256':hashlib.sha256(prompt.encode('utf-8')).hexdigest(),
        'stdout_bytes':len(stdout),'stderr_bytes':len(stderr),'stem':stem}
    record['patch_sha256']=hashlib.sha256((HERE/f'patches/{model}.yml').read_bytes()).hexdigest()
    starts=[e['elapsed_seconds'] for e in events if e['phase']=='step_start']
    ends=[e['elapsed_seconds'] for e in events if e['phase']=='step_end']
    record['model_step_seconds']=ends[-1]-starts[0] if starts and ends else None
    record['event_timestamps']=events
    Path(str(output)+'.meta.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    return record

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--condition',default='plain')
    p.add_argument('--index',type=int,default=0)
    args=p.parse_args()
    print(json.dumps(call(args.model,args.condition,args.index),ensure_ascii=False,indent=2),flush=True)
