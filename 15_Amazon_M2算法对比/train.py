import argparse,copy,json,sys,time
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from common import ROOT,RUN,data,save,metrics,ranks,sha
sys.path.insert(0,str(ROOT.parent/'03_代码与实验/RecBole-GNN'))
from recbole_gnn.model.sequential_recommender.srgnn import SRGNN
from recbole_gnn.model.sequential_recommender.gcsan import GCSAN
from recbole_gnn.model.sequential_recommender.gcegnn import GCEGNN

class AttributeEmbedding(nn.Module):
    def __init__(self,ids,a,seen):
        super().__init__();self.ids=ids;self.cached=None
        for key in ['text','brand','color','material','numeric']:self.register_buffer(key,torch.from_numpy(a[key]))
        self.register_buffer('available',torch.tensor(seen,dtype=torch.float32).reshape(-1,1))
        self.brand_embedding=nn.Embedding(int(self.brand.max())+1,8,padding_idx=0)
        self.color_embedding=nn.Embedding(int(self.color.max())+1,4,padding_idx=0)
        self.material_embedding=nn.Embedding(int(self.material.max())+1,4,padding_idx=0)
        self.projection=nn.Linear(50,32,bias=False);self.gate=nn.Parameter(torch.tensor(-1.0))
    def train(self,mode=True):self.cached=None;return super().train(mode)
    def fused(self):
        side=torch.cat([self.text,self.brand_embedding(self.brand),self.color_embedding(self.color),self.material_embedding(self.material),self.numeric],1)
        w=self.ids.weight*self.available+torch.sigmoid(self.gate)*self.projection(side)
        mask=torch.ones(w.shape[0],1);mask[0]=0;return w*mask
    @property
    def weight(self):return self.fused() if self.cached is None else self.cached
    def forward(self,ids):return torch.nn.functional.embedding(ids,self.weight,padding_idx=0)
    def compile(self):
        with torch.no_grad():self.cached=self.fused().detach()

def training_rows(d):
    rows=[]
    for row in d['train']:
        seq=row['history']+[row['target']]
        for i in range(1,len(seq)):rows.append({'history':seq[:i],'target':seq[i]})
    return rows

def batch(rows,name):
    size=max(len(r['history']) for r in rows);nodes=[0];aliases=[];edges=[];attrs=[]
    for row in rows:
        h=list(reversed(row['history'])) if name=='GCEGNN' else row['history']
        unique=sorted(set(h));mapping={x:len(nodes)+i for i,x in enumerate(unique)};nodes.extend(unique)
        alias=[mapping[x] for x in h];aliases.append(alias+[0]*(size-len(h)));pairs=set(zip(alias[:-1],alias[1:]))
        if name=='GCEGNN':
            # Match official GCEGraphDataset: reversed history, backward=1, forward=2.
            for a,b in sorted(pairs|{(b,a) for a,b in pairs}):
                edges.append((a,b));attrs.append(3 if (a,b) in pairs and (b,a) in pairs else (1 if (a,b) in pairs else 2))
            for node in set(alias):edges.append((node,node));attrs.append(0)
        else:edges.extend(sorted(pairs))
    return {'x':torch.tensor(nodes,dtype=torch.long),
            'edge_index':torch.tensor(edges,dtype=torch.long).t().contiguous() if edges else torch.empty((2,0),dtype=torch.long),
            'edge_attr':torch.tensor(attrs,dtype=torch.long),'alias_inputs':torch.tensor(aliases,dtype=torch.long),
            'item_length':torch.tensor([len(r['history']) for r in rows]),'item_id':torch.tensor([r.get('target',1) for r in rows])}
def cfg():return {'USER_ID_FIELD':'user_id','ITEM_ID_FIELD':'item_id','LIST_SUFFIX':'_list','ITEM_LIST_LENGTH_FIELD':'item_length',
    'NEG_PREFIX':'neg_','MAX_ITEM_LIST_LENGTH':20,'device':torch.device('cpu'),'embedding_size':32,'hidden_size':32,
    'step':1,'loss_type':'CE','n_layers':1,'n_heads':1,'inner_size':64,'hidden_dropout_prob':.1,'attn_dropout_prob':.1,
    'hidden_act':'gelu','layer_norm_eps':1e-12,'weight':.5,'reg_weight':1e-5,'initializer_range':.02,
    'leakyrelu_alpha':.2,'dropout_local':.1,'dropout_global':.1,'dropout_gcn':.1,'build_global_graph':True,'sample_num':8,'hop':1}
def make_model(name,d):
    rows=training_rows(d);hist=torch.zeros(len(rows),20,dtype=torch.long)
    for i,r in enumerate(rows):
        h=r['history'][::-1] if name=='GCEGNN' else r['history'];hist[i,:len(h)]=torch.tensor(h)
    dataset=SimpleNamespace(num=lambda field:len(d['tokens']),field2seqlen={'item_id_list':20},
        inter_feat={'item_id_list':hist,'item_id':torch.tensor([r['target'] for r in rows])})
    model={'SRGNN':SRGNN,'GCSAN':GCSAN,'GCEGNN':GCEGNN}[name](cfg(),dataset)
    seen=np.zeros(len(d['tokens']),dtype=np.float32)
    for r in d['train']:seen[r['history']+[r['target']]]=1
    a={k:v for k,v in np.load(RUN/'attributes.npz').items()};model.item_embedding=AttributeEmbedding(model.item_embedding,a,seen)
    return model
@torch.inference_mode()
def evaluate(model,name,rows):
    model.eval();model.item_embedding.compile();all_ranks=[]
    for start in range(0,len(rows),128):
        chunk=rows[start:start+128];scores=model.full_sort_predict(batch(chunk,name)).numpy()
        all_ranks.extend(ranks(scores,[r['target'] for r in chunk],[r['history'] for r in chunk]).tolist())
    return metrics(all_ranks),all_ranks
def load_model(name,d):
    torch.manual_seed(42);m=make_model(name,d);ck=torch.load(RUN/f'models/{name}.pt',map_location='cpu')
    m.load_state_dict(ck['state_dict']);m.eval();m.item_embedding.compile();return m
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['SRGNN','GCSAN','GCEGNN'],required=True);p.add_argument('--epochs',type=int,default=15);args=p.parse_args()
    name=args.model;torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42);d=data();rows=training_rows(d)
    model=make_model(name,d);opt=torch.optim.Adam(model.parameters(),lr=.003,weight_decay=1e-5)
    cached=[batch(rows[i:i+256],name) for i in range(0,len(rows),256)]
    print(name,json.dumps({'training_prefixes':len(rows),'batches':len(cached),'validation_sessions':len(d['validation'])}),flush=True)
    start=time.perf_counter();best=-1;patience=0;trajectory=[]
    for epoch in range(args.epochs):
        model.train();losses=[];t=time.perf_counter()
        for i in np.random.permutation(len(cached)):
            b=cached[i];opt.zero_grad();model.item_embedding.cached=model.item_embedding.fused()
            forward=[b['x'],b['edge_index']]
            if name=='GCEGNN':forward.append(b['edge_attr'])
            output=model.forward(*forward,b['alias_inputs'],b['item_length']);scores=output@model.item_embedding.weight.T
            unavailable=model.item_embedding.available[:,0].eq(0);scores=scores.masked_fill(unavailable[None,:],-1e9)
            loss=torch.nn.functional.cross_entropy(scores,b['item_id'])
            if name=='GCSAN':loss=loss+model.reg_weight*model.reg_loss(model.item_embedding.weight).squeeze()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(float(loss.detach()))
        valid,_=evaluate(model,name,d['validation']);record={'epoch':epoch+1,'loss':float(np.mean(losses)),**valid,'seconds':time.perf_counter()-t}
        trajectory.append(record);print(name,json.dumps(record),flush=True)
        save(RUN/f'results/{name}_progress.json',{'trajectory':trajectory})
        if valid['mrr@10']>best+1e-7:best=valid['mrr@10'];best_epoch=epoch+1;best_state=copy.deepcopy(model.state_dict());patience=0
        else:patience+=1
        if patience>=4:break
    model.load_state_dict(best_state);valid,before=evaluate(model,name,d['validation'])
    torch.save({'state_dict':best_state,'best_epoch':best_epoch,'data_sha256':sha(RUN/'data.json'),'attributes_sha256':sha(RUN/'attributes.npz')},RUN/f'models/{name}.pt')
    reloaded=load_model(name,d);after,after_ranks=evaluate(reloaded,name,d['validation']);assert before==after_ranks and valid==after
    save(RUN/f'results/{name}_training.json',{'model':name,'best_epoch':best_epoch,'validation':valid,'trajectory':trajectory,
         'training_seconds':time.perf_counter()-start,'training_prefix_examples':len(rows),'parameters':sum(p.numel() for p in model.parameters()),
         'independent_reload_validation_exact':True,'module':type(model).__module__,'gnn_source_sha256':sha(PathSource(name)),
         'cold_handling':'Content-only train-unseen IDs; training-seen softmax; full-catalogue evaluation',
         'checkpoint_sha256':sha(RUN/f'models/{name}.pt')})
def PathSource(name):return ROOT.parent/f'03_代码与实验/RecBole-GNN/recbole_gnn/model/sequential_recommender/{name.lower()}.py'
if __name__=='__main__':main()
