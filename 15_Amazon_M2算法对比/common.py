from pathlib import Path
from collections import Counter
import hashlib,json,time
import numpy as np
from scipy import sparse

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'run'
FEATURES=['transition','cooccurrence','text','brand','color','material','price','popularity']
def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def data():return read(RUN/'data.json')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def topk(scores,h,k=10):
    scores=np.asarray(scores,dtype=np.float32).copy();scores[0]=-np.inf;scores[h]=-np.inf
    cutoff=np.partition(scores,len(scores)-k)[len(scores)-k]
    above=np.flatnonzero(scores>cutoff);ties=np.flatnonzero(scores==cutoff)[:k-len(above)]
    ids=np.r_[above,ties];return ids[np.lexsort((ids,-scores[ids]))]
def ranks(scores,targets,histories):
    scores=np.asarray(scores);scores[:,0]=-np.inf
    for s,h in zip(scores,histories):s[h]=-np.inf
    truth=scores[np.arange(len(targets)),targets,None]
    ids=np.arange(scores.shape[1])[None,:]
    rank=1+np.sum(scores>truth,axis=1)+np.sum((scores==truth)&(ids<np.asarray(targets)[:,None]),axis=1)
    return np.where(rank<=10,rank,0).astype(np.int16)
def metrics(r):
    r=np.asarray(r);yes=r>0;rr=np.zeros(len(r));nd=rr.copy()
    rr[yes]=1/r[yes];nd[yes]=1/np.log2(r[yes]+1)
    return {'recall@10':float(yes.mean()),'mrr@10':float(rr.mean()),'ndcg@10':float(nd.mean())}
def state(c,t):return int(c[0]>t[0])+2*int(c[1]>0)+4*int(c[2]>=t[2])+8*int(c[3]<=t[3])

class Features:
    def __init__(self,d,base=False):
        self.a={k:v for k,v in np.load(RUN/'attributes.npz').items()};self.n=len(d['tokens'])
        rows=d['fit'] if base else d['train'];rr=[];cc=[];ur=[];uc=[];counts=np.zeros(self.n,dtype=np.float32)
        for i,row in enumerate(rows):
            seq=row['history']+[row['target']];rr.extend(seq[:-1]);cc.extend(seq[1:])
            unique=set(seq);ur.extend([i]*len(unique));uc.extend(unique);np.add.at(counts,seq,1)
        tr=sparse.coo_matrix((np.ones(len(rr),dtype=np.float32),(rr,cc)),shape=(self.n,self.n)).tocsr()
        mx=np.asarray(tr.max(axis=1).toarray()).ravel();self.tr=sparse.diags(1/np.maximum(mx,1)).dot(tr).tocsr()
        x=sparse.csr_matrix((np.ones(len(ur),dtype=np.float32),(ur,uc)),shape=(len(rows),self.n))
        co=(x.T@x).tocsr();co.setdiag(0);co.eliminate_zeros()
        inv=1/np.sqrt(np.maximum(np.asarray(x.sum(axis=0)).ravel(),1))
        co=sparse.diags(inv).dot(co).dot(sparse.diags(inv)).tocsr()
        mx=np.asarray(co.max(axis=1).toarray()).ravel();self.co=sparse.diags(1/np.maximum(mx,1e-8)).dot(co).tocsr()
        self.counts=counts;self.pop=np.log1p(counts);self.pop/=max(float(self.pop.max()),1)
    def context(self,h):
        b=self.a['brand'][h];b=b[b>0]
        concentration=max(Counter(b.tolist()).values())/len(b) if len(b) else 0
        p=self.a['log_price'][h];p=p[~self.a['missing'][h]]
        return [len(h),int(h[-1] in h[:-1]),concentration,float(np.std(p)) if len(p)>1 else 0]
    def compute(self,h,needed=range(8)):
        h=np.asarray(h,dtype=np.int64);needed=set(needed)
        w=np.arange(1,len(h)+1,dtype=np.float32);w/=w.sum();out={}
        if 0 in needed:out[0]=self.tr.getrow(h[-1]).toarray().ravel()
        if 1 in needed:out[1]=np.asarray(self.co[h].multiply(w[:,None]).sum(axis=0)).ravel()
        if 2 in needed:
            p=w@self.a['text'][h];p/=max(float(np.linalg.norm(p)),1e-8);out[2]=np.maximum(self.a['text']@p,0)
        for i,key in [(3,'brand'),(4,'color'),(5,'material')]:
            if i in needed:
                ids=self.a[key];v=np.bincount(ids[h],weights=w,minlength=int(ids.max())+1).astype(np.float32);v[0]=0;out[i]=v[ids]
        if 6 in needed:
            valid=~self.a['missing'][h]
            if valid.any():
                center=np.average(self.a['log_price'][h][valid],weights=w[valid]);v=np.exp(-np.abs(self.a['log_price']-center)/self.a['price_scale']);v[self.a['missing']]=0;out[6]=v
            else:out[6]=np.zeros(self.n,dtype=np.float32)
        if 7 in needed:out[7]=self.pop
        return out
    def matrix(self,h):
        z=self.compute(h);return np.stack([z[i] for i in range(8)])
    def grey_score(self,h,beta=0):
        z=self.matrix(h);diff=np.abs(1-z[:,1:])
        if beta:diff/=1+beta*(1-diff)
        lo,hi=diff.min(),diff.max();coef=(lo+.5*hi)/(diff+.5*hi) if hi else np.ones_like(diff)
        score=np.zeros(self.n,dtype=np.float32);score[1:]=coef.mean(axis=0);return score
    def policy_score(self,h,w,compiled=True):
        w=np.asarray(w,dtype=np.float32)
        if not compiled:return (self.matrix(h)*w[:,None]).sum(axis=0)
        active=np.flatnonzero(w);z=self.compute(h,active);score=np.zeros(self.n,dtype=np.float32)
        for i in active:score+=w[i]*z[i]
        return score
def benchmark(predict,rows):
    sample=rows[:200]
    for r in sample[:20]:predict(r['history'])
    times=[]
    for _ in range(3):
        for r in sample:
            start=time.perf_counter_ns();predict(r['history']);times.append((time.perf_counter_ns()-start)/1e6)
    return {'mean_ms':float(np.mean(times)),'p95_ms':float(np.percentile(times,95)),'calls':len(times),
            'scope':'CPU4, batch1, internal history -> features/graph -> full catalogue scores -> history mask -> Top10; model loading and static catalogue preparation excluded'}
