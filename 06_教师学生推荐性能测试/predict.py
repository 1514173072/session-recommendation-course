"""Run a saved local student on raw Diginetica item IDs."""
import argparse
import json
from common import *

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',default='student_teacher',choices=['student_plain','student_teacher','teacher_rules'])
    p.add_argument('--items',nargs='+',required=True)
    args=p.parse_args()
    torch.set_num_threads(4)
    tokens=json.loads((CODE/'outputs/grey_improved/tokens.json').read_text(encoding='utf-8'))
    lookup={str(x):i for i,x in enumerate(tokens['items'])}
    if any(str(x) not in lookup or lookup[str(x)]==0 for x in args.items):
        p.error('Unknown or padding item ID')
    model=Recommender(args.model)
    history=np.asarray([lookup[str(x)] for x in args.items],dtype=np.int64)
    ids,scores,coeff=model.recommend(history)
    print(json.dumps({'model':args.model,'history':args.items,'weights':coeff[:4].tolist(),
        'interactions':coeff[4:].tolist(),'recommendations':[{'rank':i+1,'item':str(tokens['items'][x]),
        'score':float(scores[x])} for i,x in enumerate(ids)]},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
