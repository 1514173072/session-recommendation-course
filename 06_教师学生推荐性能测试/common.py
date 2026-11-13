"""Shared data definitions; no change to any pre-existing recommender."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
CODE = HERE.parent / '03_代码与实验'
sys.path.insert(0, str(CODE / 'experiments'))
from grey_session import GreySession, stable_topk, ranking_metrics

PAIRS = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
CONTEXT_NAMES = ['log_length','unique_ratio','last_repeated','max_repeat_ratio',
                 'last_training_popularity','last_transition_entropy']

def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def load_grey():
    return GreySession.load(CODE / 'outputs/grey_relation/grey_model.npz')

def numeric_bytes(model):
    result = sum(x.nbytes for x in [model.counts, model.log_popularity, model.weights])
    for x in [model.transitions,model.cooccurrence]:
        result += x.data.nbytes + x.indices.nbytes + x.indptr.nbytes
    return result

class Context:
    def __init__(self, model):
        self.model = model
        self.pop = model.log_popularity / max(float(model.log_popularity.max()), 1e-12)
        self.entropy = np.zeros(model.item_num, dtype=np.float32)
        for i in range(model.item_num):
            p = model.transitions.data[model.transitions.indptr[i]:model.transitions.indptr[i+1]]
            if len(p)>1:
                self.entropy[i] = float(-(p*np.log(p)).sum()/np.log(len(p)))
        self.pop_threshold = float(np.median(self.pop[model.counts > 0]))
    def get(self, history):
        ids, count = np.unique(history, return_counts=True)
        last_count = int(count[ids==history[-1]][0])
        return np.asarray([np.log1p(len(history))/np.log(21),len(ids)/len(history),
                           float(last_count>1),float(count.max())/len(history),
                           self.pop[history[-1]],self.entropy[history[-1]]], dtype=np.float32)
    def group(self, history, c=None):
        if c is None: c=self.get(history)
        length = 'short' if len(history)<=3 else ('medium' if len(history)<=7 else 'long')
        return f'{length}_repeat{int(c[2])}_popular{int(c[4]>=self.pop_threshold)}'

class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6,16),nn.ReLU(),nn.Linear(16,10))
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)
    def forward(self, c):
        raw = self.net(c)
        return torch.cat([torch.softmax(raw[:,:4],dim=-1),0.5*torch.tanh(raw[:,4:])],dim=-1)

def torch_score(z, coeff):
    score = (z*coeff[:,None,:4]).sum(-1)
    for k,(a,b) in enumerate(PAIRS):
        score = score + coeff[:,None,4+k]*z[:,:,a]*z[:,:,b]
    return score

def numpy_score(z, coeff):
    # Same float32 accumulation as torch_score, with feature axis first.
    score = (z*coeff[:4,None]).sum(axis=0)
    for k,(a,b) in enumerate(PAIRS):
        score = score + coeff[4+k]*z[a]*z[b]
    return score

def load_teacher():
    obj = json.loads((HERE/'teacher/prototypes.json').read_text(encoding='utf-8'))
    return {r['group']: np.array(r['weights']+r['interactions'],dtype=np.float32)
            for r in obj['prototypes']}

def load_student(name):
    model=Student()
    state=torch.load(HERE/f'models/{name}.pt',map_location='cpu')
    model.load_state_dict(state['state_dict'])
    model.eval()
    return model

class Recommender:
    def __init__(self,name):
        self.name=name
        self.grey=load_grey()
        self.context=Context(self.grey) if name not in ['gra_original','gra_beta7'] else None
        self.student=load_student(name) if name in ['student_plain','student_teacher'] else None
        self.teacher=load_teacher() if name=='teacher_rules' else None
        self.state_bytes=numeric_bytes(self.grey)
        if self.student is not None:
            self.state_bytes+=sum(x.numel()*x.element_size() for x in self.student.parameters())
        if self.teacher is not None:
            self.state_bytes+=sum(x.nbytes for x in self.teacher.values())
        # Include the actually retained precomputed context numeric arrays.
        if name not in ['gra_original','gra_beta7']:
            self.state_bytes+=self.context.pop.nbytes+self.context.entropy.nbytes
    @torch.inference_mode()
    def coefficients(self,history,c=None):
        if c is None: c=self.context.get(history)
        if self.teacher is not None:
            return self.teacher[self.context.group(history,c)]
        return self.student(torch.from_numpy(c).unsqueeze(0))[0].numpy()
    def recommend(self,history):
        if self.name=='gra_original':
            ids,scores,_,_=self.grey.recommend(history)
            return ids,scores,None
        z=self.grey.normalized_features(history)
        if self.name=='gra_beta7':
            delta=np.abs(1.0-z[:,1:])
            delta=delta/(1.0+7*(1.0-delta))
            lo,hi=float(delta.min()),float(delta.max())
            coef=(lo+0.5*hi)/(delta+0.5*hi) if hi else np.ones_like(delta)
            scores=np.full(self.grey.item_num,-np.inf,dtype=np.float32)
            scores[1:]=coef.mean(axis=0)
            return stable_topk(scores),scores,None
        coeff=self.coefficients(history)
        scores=numpy_score(z,coeff)
        scores[0]=-np.inf
        return stable_topk(scores),scores,coeff
