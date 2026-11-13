import csv,re,hashlib,json
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
from common import ROOT,RUN,save,sha

RAW=ROOT.parent/'14_Amazon_M2推荐数据/data/raw'
TOKEN=re.compile(r'''['"]([A-Z0-9]{10})['"]''')
def hash64(text):return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8],'big')
def main():
    if (RUN/'data.json').exists():raise RuntimeError('Data already frozen')
    products=[]
    with (RAW/'products_train.csv').open(encoding='utf-8',newline='') as stream:
        for row in csv.DictReader(stream):
            if row['locale']=='ES':products.append(row)
    products.sort(key=lambda r:r['id']);tokens=['[PAD]']+[r['id'] for r in products];lookup={x:i for i,x in enumerate(tokens)}
    splits={'train':[],'validation':[],'test':[]};fit=[];held=[];groups={k:set() for k in splits}
    with (RAW/'sessions_train.csv').open(encoding='utf-8',newline='') as stream:
        for index,row in enumerate(csv.DictReader(stream)):
            if row['locale']!='ES':continue
            full=TOKEN.findall(row['prev_items']);target=row['next_item'];key='|'.join(full+[target])
            digest=hash64('42|'+key);bucket=digest%10000
            part='train' if bucket<8000 else ('validation' if bucket<9000 else 'test')
            record={'source_row':index,'history':[lookup[x] for x in full[-20:]],'target':lookup[target]}
            assert record['history'] and record['target'] not in record['history']
            splits[part].append(record);groups[part].add(digest)
            if part=='train':
                (held if hash64('calibration|'+key)%10==0 else fit).append(record)
    assert not(groups['train']&groups['test'] or groups['train']&groups['validation'] or groups['test']&groups['validation'])
    rng=np.random.default_rng(43);calibration=[held[int(i)] for i in rng.choice(len(held),min(1024,len(held)),replace=False)]
    policy_validation=[splits['validation'][int(i)] for i in rng.choice(len(splits['validation']),1024,replace=False)]
    seen=set(x for row in splits['train'] for x in row['history']+[row['target']]);seen_ids=sorted(seen)
    docs=['missing']+[(r['title']+' '+r['desc'][:500]).strip() or 'missing' for r in products]
    vectorizer=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=3,max_features=12000,sublinear_tf=True)
    vectorizer.fit([docs[i] for i in seen_ids]);tfidf=vectorizer.transform(docs)
    svd=TruncatedSVD(n_components=32,n_iter=3,random_state=42);svd.fit(tfidf[seen_ids]);text=normalize(svd.transform(tfidf)).astype(np.float32);text[0]=0
    arrays={'text':text};labels={}
    for key in ['brand','color','material']:
        values=['']+[r[key].strip().casefold() for r in products];vocab={v:i+1 for i,v in enumerate(sorted(set(values)-{''}))}
        arrays[key]=np.asarray([vocab.get(v,0) for v in values],dtype=np.int64);labels[key]=values
    prices=np.full(len(tokens),np.nan,dtype=np.float32)
    for i,r in enumerate(products,1):
        try:
            value=float(r['price'])
            if np.isfinite(value) and value>0:prices[i]=value
        except ValueError:pass
    missing=~np.isfinite(prices);lp=np.log1p(np.where(missing,0,prices))
    trainp=lp[seen_ids];trainp=trainp[~missing[seen_ids]];center=float(np.median(trainp));scale=max(float(np.subtract(*np.percentile(trainp,[75,25]))),.1)
    arrays.update(missing=missing,log_price=lp,price_scale=np.float32(scale),numeric=np.stack([np.where(missing,0,np.clip((lp-center)/scale,-5,5)),missing.astype(np.float32)],axis=1).astype(np.float32))
    arrays['numeric'][0]=0
    np.savez_compressed(RUN/'attributes.npz',**arrays)
    d={**splits,'fit':fit,'calibration':calibration,'policy_validation':policy_validation,
       'tokens':tokens,'titles':['[PAD]']+[r['title'] for r in products],**labels}
    save(RUN/'data.json',d)
    manifest={'locale':'ES','seed':42,'sessions':sum(len(s) for s in splits.values()),'catalogue_items':len(products),
       'splits':{k:len(v) for k,v in splits.items()},'train_seen_items':len(seen),'feature_fit_sessions':len(fit),
       'calibration_held_out_sessions':len(held),'policy_calibration_samples':len(calibration),'policy_validation_samples':len(policy_validation),
       'test_cold_targets':sum(r['target'] not in seen for r in splits['test']),'max_history':20,
       'duplicate_full_sequences_grouped':True,'cross_split_full_sequence_overlap':0,'global_temporal_or_user_split':False,
       'text_fit_training_catalogue_only':True,'svd_explained_variance':float(svd.explained_variance_ratio_.sum()),
       'data_sha256':sha(RUN/'data.json'),'attributes_sha256':sha(RUN/'attributes.npz'),
       'raw_sha256':{name:sha(RAW/name) for name in ['products_train.csv','sessions_train.csv']}}
    save(RUN/'manifest.json',manifest);print(json.dumps(manifest,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=4):main()
