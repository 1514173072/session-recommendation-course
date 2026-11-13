'use strict';
const DATA=JSON.parse(document.getElementById('demo-data').textContent),$=id=>document.getElementById(id);
const NAMES=['SR-GNN','GC-SAN','GCE-GNN','灰色关联','改进灰色关联（β=7）','离线qwen3.5-2B'];
const CLASSES=['sr','gc','gce','gra','gra-improved','offline-qwen'];
const selected=new Set(NAMES.map((_,i)=>i));
const ranks=row=>row.slice(3).map(ids=>ids.indexOf(row[2])+1);
const byId=new Map(DATA.rows.map(row=>[row[0],row]));
const baseOptions=new Map([...$('type').options].map(option=>[option.value,option.textContent]));
function matches(row,type){
  if(!selected.size)return false;
  const r=ranks(row),hits=[...selected].filter(i=>r[i]>0).length;
  return type==='all'||type==='allHit'&&hits===selected.size||type==='partial'&&hits>0&&hits<selected.size||type==='allMiss'&&hits===0||type==='gceOnly'&&selected.has(2)&&r[2]>0&&hits===1||type==='gceMiss'&&selected.has(2)&&!r[2]&&hits>0;
}
function updateCategories(){
  for(const option of $('type').options){
    let count=0;for(const row of DATA.rows)if(matches(row,option.value))count++;
    option.textContent=baseOptions.get(option.value)+'（'+count.toLocaleString()+'）';
    option.disabled=['gceOnly','gceMiss'].includes(option.value)&&!selected.has(2);
  }
  if($('type').selectedOptions[0].disabled)$('type').value='all';
}
NAMES.forEach((name,index)=>{
  const label=document.createElement('label');label.className='method-option';
  const input=document.createElement('input');input.type='checkbox';input.checked=true;input.value=String(index);
  input.addEventListener('change',()=>{
    input.checked?selected.add(index):selected.delete(index);
    updateCategories();applyFilters($('session').value);
  });
  label.append(input,document.createTextNode(name));$('methodChoices').append(label);
});
function applyFilters(preferred){
  const query=$('sessionSearch').value.trim(),type=$('type').value;
  const matching=DATA.rows.filter(row=>matches(row,type)&&(!query||row[0].includes(query)));
  const shown=matching.slice(0,300),chosen=matching.find(row=>row[0]===preferred);
  if(chosen&&!shown.includes(chosen))shown.unshift(chosen);
  $('session').replaceChildren(...shown.map(row=>new Option('会话 '+row[0]+' · '+row[1].join(' → '),row[0])));
  if(chosen)$('session').value=chosen[0];
  $('count').textContent='已选 '+selected.size+' 种方法 · 符合条件 '+matching.length.toLocaleString()+' / '+DATA.rows.length.toLocaleString()+' 个会话'+(matching.length>shown.length?' · 列出前 300 个，可按会话 ID 搜索':'');
  $('session').disabled=!shown.length;render();
}
function render(){
  const row=byId.get($('session').value),k=Number($('topk').value);
  $('empty').hidden=!!row;$('sessionDetail').hidden=!row;$('models').hidden=!row;
  $('empty').textContent=selected.size?'没有符合条件的测试会话，请调整筛选条件。':'请至少勾选一种方法。';
  if(!row)return;
  $('history').replaceChildren();
  row[1].forEach((id,i)=>{
    if(i){const arrow=document.createElement('span');arrow.className='arrow';arrow.textContent='→';$('history').append(arrow)}
    const chip=document.createElement('span');chip.className='chip';chip.textContent=id;$('history').append(chip);
  });
  $('length').textContent=row[1].length+' 次点击';$('truth').textContent=row[2];
  $('answerNote').textContent=(DATA.orderNoteIds||[]).includes(row[0])?'本例存在时间戳精度造成的排序差异，按项目处理顺序留出答案':'留出的答案，仅用于检验推荐结果';
  $('models').replaceChildren(...NAMES.flatMap((name,m)=>{
    if(!selected.has(m))return [];
    const ids=row[m+3].slice(0,k),rank=ids.indexOf(row[2])+1;
    const card=document.createElement('article');card.className='model '+CLASSES[m];
    const head=document.createElement('div');head.className='model-head';
    const h=document.createElement('h2');h.className='model-title';h.textContent=name;
    const verdict=document.createElement('div');verdict.className='verdict '+(rank?'hit':'miss');
    verdict.textContent=rank?'真实商品排第 '+rank+' 名 · 命中':'真实商品未进入前 '+k+' 名';head.append(h,verdict);
    const table=document.createElement('table');table.innerHTML='<thead><tr><th>排名</th><th>推荐商品 ID</th><th>真实目标</th></tr></thead>';
    const body=document.createElement('tbody');
    ids.forEach((id,i)=>{
      const tr=document.createElement('tr');if(id===row[2])tr.className='target';
      for(const text of [i+1,id,id===row[2]?'命中':'—']){const td=document.createElement('td');td.textContent=text;tr.append(td)}body.append(tr);
    });
    table.append(body);card.append(head,table);return [card];
  }));
  $('footer').textContent='Diginetica · '+DATA.rows.length.toLocaleString()+' 个测试会话 · GNN沿用原页面保存预测，新增方法使用相同浏览记录。真实下一次点击仅用于检验。离线Qwen复用预计算策略；页面展示已保存Top10，不调用模型服务。';
}
$('type').addEventListener('change',()=>applyFilters($('session').value));
$('sessionSearch').addEventListener('input',()=>applyFilters($('session').value));
$('session').addEventListener('change',render);$('topk').addEventListener('change',render);
updateCategories();applyFilters('10114');
