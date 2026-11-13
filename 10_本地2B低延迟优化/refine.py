"""Small token batches improve recurrent-prefix reuse; validation-only choice."""
import json,time
import numpy as np
import optimize as opt
original_prompt=opt.prompt

def prompt(variant,h,c):
 if variant!='raw12':return original_prompt(variant,h,c)
 length=0 if len(h)<=3 else (1 if len(h)<=7 else 2)
 repeat=int(c[2]);popular=int(c[4]>=0.23069553)
 # Few-shot table is fixed by the existing training teacher's group definitions.
 table='\n'.join(f'{i//4} {(i//2)%2} {i%2}: {opt.LABELS[i]}' for i in range(12))
 return 'Session policy lookup. Columns: length_bin repeated_last popular_last. Output the matching letter.\n'+table+f'\n{length} {repeat} {popular}:'

opt.prompt=prompt

def main():
 opt.torch.set_num_threads(4)
 data=np.load(opt.ROOT/'06_教师学生推荐性能测试/cache/splits.npz')
 freeze=json.loads((opt.HERE/'frozen_before_selection.json').read_text(encoding='utf-8'))
 grey=opt.load_grey();ctx=opt.Context(grey)
 indices=freeze['valid_indices'];variants=['examples12','raw12']
 opt.save(opt.HERE/'refine_before_selection.json',{'variants':variants,'valid_indices':indices,
       'script_sha256':opt.sha(__file__),'base_script_sha256':opt.sha(opt.HERE/'optimize.py'),
       'selection':'highest valid MRR; tie -> lower end-to-end mean','runtime_config':(opt.HERE/'runtime_config.txt').read_text(encoding='utf-8-sig')})
 results=[]
 for variant in variants:
  h=opt.history(data,'valid',indices[-1]);warm=opt.infer(variant,h,ctx.get(h),'refine_warmup');opt.save(opt.HERE/f'requests/{variant}_refine_warmup.json',warm)
  s,_=opt.evaluate_batch(variant,'valid',indices,data,grey,ctx,opt.HERE/f'requests/{variant}_refine_validation.jsonl')
  results.append(s);print(json.dumps(s,ensure_ascii=False),flush=True)
 chosen=max(results,key=lambda r:(r['mrr@10'],-r['mean_end_to_end_ms']))['variant']
 opt.save(opt.HERE/'results/refined_selection_before_test.json',{'selected':chosen,'validation':results,'config_sha256':opt.sha(opt.HERE/'refine_before_selection.json')})
 h=opt.history(data,'valid',indices[-1]);opt.infer(chosen,h,ctx.get(h),'final_test_warmup')
 test,rows=opt.evaluate_batch(chosen,'test',freeze['test_indices'],data,grey,ctx,opt.HERE/'requests/refined_test1000.jsonl')
 h=opt.history(data,'valid',indices[-1]);opt.infer(chosen,h,ctx.get(h),'final_old30_warmup')
 old,_=opt.evaluate_batch(chosen,'test',freeze['original_test_indices'],data,grey,ctx,opt.HERE/'requests/refined_original30.jsonl')
 opt.save(opt.HERE/'results/refined_summary.json',{'selected':chosen,'validation':results,'test1000':test,'original30':old})
 print(json.dumps({'selected':chosen,'test1000':test,'original30':old},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
