
'use strict';
const DATA=JSON.parse(document.getElementById('experiment-data').textContent);
const MODELS=new Map(DATA.models.map(m=>[m.id,m]));
const RECORDS=new Map(DATA.records.map(r=>[r.scope+'|'+r.id,r]));
const DEFAULT_IDS=['gra','gra_improved','srgnn','gcsan','gcegnn','qwen_teacher','qwen_offline'];
let scope='30', selected=new Set(DEFAULT_IDS), sortKey=null, sortDirection=1;
const byId=id=>document.getElementById(id);
const escapeHtml=v=>String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmtMs=v=>v==null?'—':v.toLocaleString('en-US',{minimumFractionDigits:3,maximumFractionDigits:3});
const getRecord=id=>RECORDS.get(scope+'|'+id);
function teacher(id){if(id==='rules'||id==='mlp_teacher'||id==='qwen_offline'||id==='qwen_sync'||id==='qwen_teacher'||id.endsWith('_teacher'))return '有';if(id==='mlp'||id==='qwen_plain'||id.endsWith('_plain'))return '无';return '—';}
function shortMode(m){if(m.id==='qwen_offline')return '离线 LLM 策略预计算 · 在线不运行2B';if(m.id==='qwen_plain'||m.id==='qwen_teacher')return '在线 LLM 动态策略生成 · 每次运行2B';if(m.id==='qwen_sync')return '过程方案 · 每次运行2B';if(m.group==='云端LLM')return '云端调用 · 每次运行LLM';return m.group==='深度学习'?'本地CPU · 深度学习':'本地CPU';}
function activeRows(){let rr=DATA.models.filter(m=>selected.has(m.id)&&getRecord(m.id)).map(m=>({...getRecord(m.id),model:m}));if(sortKey){rr.sort((a,b)=>{const av=sortKey==='name'?a.model.name:a[sortKey],bv=sortKey==='name'?b.model.name:b[sortKey];if(av==null)return bv==null?0:1;if(bv==null)return -1;return (typeof av==='string'?av.localeCompare(bv,'zh-CN'):av-bv)*sortDirection;});}return rr;}
function renderScopes(){byId('scopeButtons').innerHTML=DATA.scopes.map(s=>`<button class="scope-button" data-scope="${s.id}" aria-pressed="${scope===s.id}">${escapeHtml(s.name)}</button>`).join('');byId('scopeDescription').textContent=DATA.scopes.find(s=>s.id===scope).description;}
function renderChoices(){const groups=['基础方法','深度学习','本地LLM'];byId('choices').innerHTML=groups.map(group=>`<div class="group"><div class="group-title">${group}</div>${DATA.models.filter(m=>m.group===group).map(m=>{let r=getRecord(m.id),checked=selected.has(m.id)&&!!r;return `<label class="choice ${checked?'checked':''} ${r?'':'disabled'}"><input type="checkbox" data-id="${m.id}" aria-label="${escapeHtml(m.name)}" ${checked?'checked':''} ${r?'':'disabled'}><span class="choice-text"><span class="choice-name">${escapeHtml(m.name)}</span>${!r?'<span class="unavailable"> · 未测试</span>':''}</span></label>`}).join('')}</div>`).join('');}
function renderTable(){const rr=activeRows(),best={};for(const k of ['recall','mrr','ndcg','mean_ms','p95_ms']){const values=rr.map(r=>r[k]).filter(v=>v!=null);best[k]=values.length?(k.endsWith('_ms')?Math.min(...values):Math.max(...values)):null;}
byId('selectedCount').textContent=`已选 ${rr.length} 项`;
const n=DATA.scopes.find(s=>s.id===scope).n;byId('sampleSize').textContent=n.toLocaleString('en-US')+' 条';byId('sampleSub').textContent=`${rr.length} 种方法 · 同一批真实会话`;
const qr=rr.filter(r=>r.recall===best.recall),tr=rr.filter(r=>r.mean_ms===best.mean_ms);
byId('bestRecall').textContent=rr.length?(best.recall*100).toFixed(2)+'%':'—';byId('bestRecallName').textContent=qr.length?qr[0].model.name+(qr.length>1?` 等${qr.length}项`:''):'请勾选方法';
byId('bestLatency').textContent=rr.length?fmtMs(best.mean_ms)+' ms':'—';byId('bestLatencyName').textContent=tr.length?tr[0].model.name:'请勾选方法';
byId('tableCaption').textContent=`相同${n.toLocaleString('en-US')}条会话 · 全商品Top10 · ${rr.length}种方法`;
if(!rr.length){byId('rows').innerHTML='<tr><td colspan="7" class="empty"><strong>选择要比较的算法</strong><small>在左侧勾选方法，结果将同时显示在这里。</small></td></tr>';}else{byId('rows').innerHTML=rr.map(r=>{const m=r.model;let t=teacher(m.id);const cell=(k,text)=>`<td class="${r[k]!=null&&Math.abs(r[k]-best[k])<1e-12?'best':''}" title="${k==='p95_ms'&&r[k]==null?escapeHtml(r.p95_note||'未记录同口径P95'):r[k]!=null&&Math.abs(r[k]-best[k])<1e-12?'当前选择中的最优值':''}">${text}</td>`;return `<tr data-method="${m.id}"><td title="${escapeHtml(r.timing)}"><div class="modelcell"><span class="modeldot" style="background:${m.color}"></span><div><div class="model-name">${escapeHtml(m.name)}</div><div class="model-sub">${escapeHtml(shortMode(m))}${r.valid!=null&&r.valid<r.n?` · 有效输出${r.valid}/${r.n}`:''}</div></div></div></td><td><span class="teacher ${t==='有'?'yes':''}">${t}</span></td>${cell('recall',(r.recall*100).toFixed(2)+'%')}${cell('mrr',r.mrr.toFixed(5))}${cell('ndcg',r.ndcg.toFixed(5))}${cell('mean_ms',fmtMs(r.mean_ms))}${cell('p95_ms',fmtMs(r.p95_ms))}</tr>`}).join('');}
document.querySelectorAll('th[data-sort]').forEach(th=>{const active=th.dataset.sort===sortKey;th.setAttribute('aria-sort',active?(sortDirection===1?'ascending':'descending'):'none');th.querySelector('.arrow').textContent=active?(sortDirection===1?'↑':'↓'):'↕';});
const cloud=rr.some(r=>r.model.group==='云端LLM'),online=rr.some(r=>['qwen_plain','qwen_teacher'].includes(r.id)),sync=rr.some(r=>r.id==='qwen_sync'),offline=rr.some(r=>r.id==='qwen_offline');
let notes=[];if(offline)notes.push('<strong>离线qwen3.5-2B：</strong>已引入强模型关联分析，在线查策略表，不运行2B。');
if(online)notes.push('<strong>原版在线qwen3.5-2B：</strong>有、无强模型关联分析两组各完成30条；其他样本范围尚无结果。');
if(cloud)notes.push('<strong>云端耗时：</strong>包含Harness启动、网络和生成，最多4路并发；与本地CPU/GPU计时范围不同。');
if(scope==='20000')notes.push('<strong>计时范围：</strong>基线为历史900次预热CPU测量；离线Qwen为本次完整20,000条测量。');
if(scope==='1000')notes.push('<strong>计时范围：</strong>原始灰关联与GNN为本次同1,000条的实测；改进灰关联沿用历史900次计时。');
if(scope==='30'&&rr.some(r=>r.id==='gra_improved'))notes.push('<strong>改进灰关联：</strong>质量来自同30条保存预测，耗时沿用历史900次预热测量。');
if(sync)notes.push('<strong>单Token同步：</strong>表中为连续请求预热均值，空闲后首次请求可能更慢。');
byId('notes').innerHTML=notes.map(x=>'<p>'+x+'</p>').join('');
byId('timingNote').textContent='全部耗时统一为毫秒，模型加载不计入。— 表示缺少同口径P95记录。悬停方法名称可查看计时范围；点击列标题可排序。';
}
function render(){renderScopes();renderChoices();renderTable();}
byId('scopeButtons').addEventListener('click',e=>{const b=e.target.closest('[data-scope]');if(!b)return;scope=b.dataset.scope;render();});
byId('choices').addEventListener('change',e=>{if(!e.target.matches('input[data-id]'))return;const id=e.target.dataset.id;e.target.checked?selected.add(id):selected.delete(id);renderChoices();renderTable();});
byId('selectAll').addEventListener('click',()=>{for(const m of DATA.models)if(getRecord(m.id))selected.add(m.id);render();});
byId('clearAll').addEventListener('click',()=>{selected.clear();render();});
byId('reset').addEventListener('click',()=>{selected=new Set(DEFAULT_IDS);sortKey=null;sortDirection=1;render();});
document.querySelector('thead').addEventListener('click',e=>{const b=e.target.closest('button[data-key]');if(!b)return;const k=b.dataset.key;if(sortKey===k)sortDirection*=-1;else{sortKey=k;sortDirection=(k==='name'||k.endsWith('_ms'))?1:-1;}renderTable();});
render();
