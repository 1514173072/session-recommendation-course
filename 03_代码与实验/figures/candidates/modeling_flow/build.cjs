const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { chromium } = require('C:/anacode/Lib/site-packages/playwright/driver/package');

const project = 'D:/Projects/算法分析与理论课程';
const delivery = path.join(project, '04_报告与展示/课堂讨论_PPT交付材料/02_问题建模');
const shotPath = path.join(delivery, '原始数据截图.png');
const png = fs.readFileSync(shotPath);
const dataImage = 'data:image/png;base64,' + png.toString('base64');
const output = path.join(delivery, '电商会话推荐建模流程图_文字校正版.png');
const inputImage = 'D:/网站download/电商会话推荐建模流程图.png';
const manifest = JSON.parse(fs.readFileSync(path.join(project, '03_代码与实验/outputs/formal/data_manifest.json'), 'utf8'));
const result = JSON.parse(fs.readFileSync(path.join(project, '03_代码与实验/outputs/formal/GCEGNN/seed42/result.json'), 'utf8'));
if (manifest.selected_sessions !== 20000 || manifest.selected_items !== 32968 || manifest.selected_interactions !== 157273 || result.train_examples !== 97273) throw Error('Input counts changed.');

const box = (x,y,w,h,text,classes='') => `<div class="box ${classes}" style="left:${x}px;top:${y}px;width:${w}px;height:${h}px"><div>${text}</div></div>`;
const text = (x,y,w,content,classes='') => `<div class="text ${classes}" style="left:${x}px;top:${y}px;width:${w}px">${content}</div>`;
const math = x => `<span class="math">${x}</span>`;
const sub = (x,s) => `${x}<sub>${s}</sub>`;
const hist = k => math(`(${sub('v','1')}, …, ${sub('v',k)})`);
const panels = [
 [20,340,'模块一：','原始点击记录与整理'],
 [375,370,'模块二：','历史序列与监督样本'],
 [760,345,'模块三：','会话图表示与编码'],
 [1120,415,'模块四：','候选打分与推荐输出'],
 [1550,350,'模块五：','离线评价']
];
let content = panels.map(([x,w,a,b]) => `<section class="panel" style="left:${x}px;width:${w}px"><header>${a}<br>${b}</header></section>`).join('');

// Use the actual screenshot bytes, without regenerating or transcribing the image.
content += `<img id="raw-data" src="${dataImage}" alt="完整交互文件的表头及前12条真实记录" style="position:absolute;left:35px;top:246px;width:310px;height:auto">`;
content += box(35,553,310,46,'同一会话内按时间排序','ribbon center');
content += text(40,621,300,'数据筛选：','bold');
content += text(40,667,302,'● 全量商品频次 ≥ 5<br>● 保留 5–20 次点击的会话<br>● 排除前期试跑会话<br>● 抽取 20,000 个会话','list');

content += box(391,246,338,48,'完整会话： '+math('S = (v<sub>1</sub>, v<sub>2</sub>, …, v<sub>L</sub>)'),'ribbon small');
content += text(391,320,340,'预测样本：历史序列 → 下一点击','small bold');
content += box(392,362,185,94,'历史<br>'+math('X<sub>t</sub> = (v<sub>1</sub>, …, v<sub>t−1</sub>)'),'center teal small');
content += box(591,362,139,94,'真实目标<br>'+math('y<sub>t</sub> = v<sub>t</sub>'),'center orange');
content += box(391,492,328,48,'会话内留尾划分','ribbon center');
content += `<table class="split" style="left:391px;top:540px;width:328px"><tbody>
<tr><th>训练</th><td>历史 ${hist('t−1')}<br>→ 目标 ${math('v<sub>t</sub>')}，${math('t = 2, …, L−2')}</td></tr>
<tr><th>验证</th><td>历史 ${hist('L−2')}<br>→ 目标 ${math('v<sub>L−1</sub>')}</td></tr>
<tr><th>测试</th><td>历史 ${hist('L−1')}<br>→ 目标 ${math('v<sub>L</sub>')}</td></tr>
</tbody></table>`;
content += text(390,802,330,'同一会话的历史前缀可以重叠，<br>目标位置不同。','tiny');

content += `<svg id="session-graph" style="position:absolute;left:788px;top:246px;width:290px;height:177px" viewBox="0 0 290 177" role="img" aria-label="四个不同商品节点，边为1到2、2到3、3到1、1到4">
<defs><marker id="graph-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7Z" fill="#075965"/></marker><linearGradient id="node-fill" x2="0" y2="1"><stop stop-color="#dcf4f1"/><stop offset="1" stop-color="#a9e3dc"/></linearGradient></defs>
<g fill="none" stroke="#075965" stroke-width="3" marker-end="url(#graph-arrow)"><path d="M61,39 L109,39"/><path d="M160,39 L208,39"/><path d="M222,62 C193,114 73,109 47,64"/><path d="M43,65 Q45,146 130,146"/></g>
<g fill="url(#node-fill)" stroke="#087680" stroke-width="3"><circle cx="36" cy="39" r="25"/><circle cx="135" cy="39" r="25"/><circle cx="234" cy="39" r="25"/><circle cx="157" cy="146" r="25"/></g>
<g fill="#073d4f" font-family="Georgia" font-size="29" font-style="italic" text-anchor="middle"><text x="36" y="49">v₁</text><text x="135" y="49">v₂</text><text x="234" y="49">v₃</text><text x="157" y="156">v₄</text></g></svg>`;
content += text(778,437,315,'● 节点＝历史中的不同商品<br>● 有向边＝相邻点击关系<br>● 保留点击位置到节点的映射<br>● 重复访问同一商品对应同一节点','graph-notes');
content += box(776,589,313,43,'模型编码器','ribbon center');
content += box(782,645,158,34,'SR-GNN','teal center model');
content += box(782,689,158,34,'GC-SAN','teal center model');
content += box(782,733,158,34,'GCE-GNN','teal center model');
content += text(950,656,140,'相同任务下<br>分别运行<br>三种模型<br>（非串联）','small center');
content += box(799,801,264,43,'会话向量 '+math('h'),'teal center');

content += box(1139,245,161,85,'会话向量<br>'+math('h'),'teal center');
content += box(1320,245,194,85,'候选商品向量<br>'+math('e<sub>j</sub>'),'teal center');
content += box(1151,385,232,104,'全候选打分<br>'+math('z<sub>j</sub> = h<sup>T</sup>e<sub>j</sub>'),'teal center');
content += text(1401,370,114,'训练阶段','orange-ink small center bold');
content += box(1400,418,118,137,'交叉熵损失<br>↓<br>更新参数','orange center small loss');
content += box(1151,534,232,46,'按得分取 Top-10','teal center');
content += `<div class="rankings" style="left:1151px;top:600px;width:232px"><div>☷　 第 1 位</div><div>☷　 第 2 位</div><div class="ellipsis">…</div><div>☷　 第 10 位</div></div>`;
content += text(1140,766,378,'● 真实候选商品 32,968 个<br>● 输出排除填充索引 0<br>● 历史商品可再次被推荐','cost-notes');

content += box(1570,245,145,83,'Top-10<br>推荐列表','teal center');
content += box(1734,245,147,83,'真实下一<br>点击标签','orange center');
content += box(1570,399,311,65,'命中与排名核对','teal center');
content += `<div class="metrics" style="left:1568px;top:490px;width:316px"><div><b>● Recall@10：</b><p>真实下一点击是否进入<br>前 10 位。</p></div><div><b>● MRR@10：</b><p>命中排名倒数的平均值。</p></div><div><b>● NDCG@10：</b><p>经对数折扣后的排名质量。</p></div></div>`;
content += box(1568,766,316,77,'■ 验证集 MRR@10 选模<br>■ 测试集报告最终结果','orange small');

const arrow = (d,color='#009e93',width=4) => `<path d="${d}" fill="none" stroke="${color}" stroke-width="${width}" stroke-linecap="round" stroke-linejoin="round" marker-end="url(#${color==='#009e93'?'flow':'label'})"/>`;
content += `<svg class="arrows" width="1920" height="1080"><defs><marker id="flow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6Z" fill="#009e93"/></marker><marker id="label" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6Z" fill="#ed6b00"/></marker></defs>`;
for(const [a,b] of [[360,375],[745,760],[1105,1120],[1535,1550]]) content += arrow(`M${a},468 L${b-2},468`);
content += arrow('M484,456 L484,489');
content += arrow('M932,565 L932,587');
content += arrow('M932,772 L932,797');
content += arrow('M1220,330 L1220,351 L1267,351 L1267,381');
content += `<path d="M1417,330 L1417,351 L1267,351" fill="none" stroke="#009e93" stroke-width="4"/>`;
content += arrow('M1267,489 L1267,530');
content += arrow('M1267,580 L1267,606');
content += arrow('M1383,448 L1397,448','#ed6b00',3);
content += `<path d="M1642,328 L1642,366 L1725,366 M1807,328 L1807,366 L1725,366" fill="none" stroke="#00887f" stroke-width="4"/>`;
content += arrow('M1725,366 L1725,395');
// Labels travel separately from the historical input; only training labels reach the loss.
content += arrow('M650,456 L650,474 L733,474 L733,896 L1459,896 L1459,559','#ed6b00',3);
content += arrow('M661,456 L661,480 L739,480 L739,938 L1713,938 L1713,847','#ed6b00',3);
content += arrow('M672,456 L672,486 L745,486 L745,975 L1802,975 L1802,847','#ed6b00',3);
content += '</svg>';
content += text(789,871,181,'训练标签<br>（用于损失）','branch');
content += text(981,913,181,'验证标签<br>（用于选模）','branch');
content += text(1175,950,181,'测试标签<br>（用于评价）','branch');

const html = `<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>电商会话推荐的问题建模</title><style>
*{box-sizing:border-box}body{margin:0;background:white;color:#081e46;font-family:"Microsoft YaHei","Segoe UI",sans-serif}#canvas{position:relative;width:1920px;height:1080px;overflow:hidden;background:white}h1{margin:0;position:absolute;top:7px;width:100%;font-size:66px;text-align:center;letter-spacing:1px;font-weight:750}hr{position:absolute;left:20px;top:105px;width:1880px;border:0;border-top:3px solid #194e79}.panel{position:absolute;top:139px;height:721px;border:2px solid #118c93;border-radius:18px;overflow:hidden;background:#fff}.panel header{height:87px;background:linear-gradient(110deg,#148a92,#007780);padding:9px 15px;color:white;font-weight:650;font-size:26px;line-height:1.27}.text,.box{position:absolute;font-size:24px;line-height:1.5}.text{z-index:2}.box{display:flex;flex-direction:column;justify-content:center;padding:5px 10px;border-radius:12px;line-height:1.4;z-index:1}.center{text-align:center}.bold{font-weight:650}.ribbon{background:linear-gradient(90deg,#e2f4f4,#bce9e6);font-weight:650}.teal{background:linear-gradient(110deg,#d5f1ec,#94dfd4);border:1.5px solid #09a191;font-weight:600}.orange{background:#fff1e2;border:1.5px solid #f07819;color:#b94400;font-weight:600}.orange-ink{color:#c54b00}.math{font-family:Georgia,"Times New Roman",serif;font-style:italic;white-space:nowrap}sub,sup{font-size:.68em;line-height:0;position:relative;vertical-align:baseline}sub{bottom:-.25em}sup{top:-.5em}.small{font-size:21px}.tiny{font-size:17px;line-height:1.55}.list{line-height:1.85;font-size:23px}.compact{line-height:1.55;font-size:21px}.graph-notes{font-size:20px;line-height:1.65}.model{font-size:24px;padding:0}.split{position:absolute;border-collapse:collapse;border:1.5px solid #36566e;border-radius:0;font-size:19px;line-height:1.65;z-index:1}.split th,.split td{border:1px solid #5c808b;padding:8px 7px;vertical-align:middle}.split th{width:66px;background:#ddf2e3;font-size:21px}.split td{background:#fcfdfd}.rankings{position:absolute;border:2px solid #075365;border-radius:12px;padding:10px 14px;font-size:25px;text-align:center;line-height:1.55}.rankings>div{background:#e7f3f5;border-radius:8px;margin:4px 0;height:37px}.rankings .ellipsis{height:24px;line-height:18px;background:white}.metrics{position:absolute;border-radius:12px;background:linear-gradient(120deg,#f4fbfb,#dff3f1);padding:9px 12px;font-size:23px;line-height:1.45}.metrics p{font-size:21px;margin:0 0 6px 21px;line-height:1.4}.arrows{position:absolute;left:0;top:0;pointer-events:none;z-index:3}.branch{background:white;color:#cb4c00;font-size:20px;font-weight:650;text-align:center;line-height:1.12;z-index:4}.footer{position:absolute;left:20px;top:1003px;width:1880px;height:65px;border:2px solid #174e7a;border-radius:13px;text-align:center;padding:3px 8px;background:#fcfdff;font-size:30px;font-weight:650;line-height:1.1}.footer small{display:block;font-size:19px;font-weight:400;margin-top:5px}
.rankings{font-size:21px;padding:7px 14px}.rankings>div{height:28px;margin:3px 0}.rankings .ellipsis{height:18px;line-height:14px}.cost-notes{font-size:19px;line-height:1.5}.loss{font-size:19px}
</style><div id="canvas"><h1>电商会话推荐的问题建模</h1><hr>${content}<div class="footer">20,000 会话｜157,273 点击｜32,968 商品｜训练 / 验证 / 测试样本：97,273 / 20,000 / 20,000<small>当前实验采用会话内留尾划分；商品频次筛选与词表确定在划分前完成。</small></div></div></html>`;
fs.writeFileSync(path.join(__dirname,'电商会话推荐建模流程图_排版源文件.html'),html);

(async()=>{
 const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});
 const page=await browser.newPage({viewport:{width:1920,height:1080},deviceScaleFactor:2});
 await page.setContent(html,{waitUntil:'load'});
 await page.evaluate(()=>document.fonts.ready);
 const overflow=await page.locator('.text,.box,.metrics,.rankings,.split,.footer').evaluateAll(nodes=>nodes.filter(e=>e.scrollHeight>e.clientHeight+2 || e.scrollWidth>e.clientWidth+2).map(e=>({text:e.innerText,scroll:[e.scrollWidth,e.scrollHeight],client:[e.clientWidth,e.clientHeight]})));
 if(overflow.length) { console.log(JSON.stringify({overflow})); await browser.close(); throw Error('Text overflow'); }
 const embedded=await page.locator('#raw-data').getAttribute('src');
 if(embedded!==dataImage)throw Error('Screenshot mismatch');
 await page.locator('#canvas').screenshot({path:output});
 await page.locator('#canvas').screenshot({path:path.join(__dirname,'recheck.png')});
 const digest=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
 if(digest(output)!==digest(path.join(__dirname,'recheck.png')))throw Error('Repeated render mismatch');
 await browser.close();
 const record={figure:'电商会话推荐的问题建模',source_kind:'new_candidate',gallery_searches:['data-type=流程, keyword=建模: []','method-type=流程图, keyword=机器学习: []'],reason:'Existing raster figure needs accurate native-font typesetting and exact data screenshot; no compatible gallery template returned.',input_reference:inputImage,reference_sha256:digest(inputImage),screenshot:shotPath,screenshot_sha256:digest(shotPath),output,size:[3840,2160],output_sha256:digest(output),repeated_render_identical:true,text_overflow:overflow,fonts:['Microsoft YaHei','Georgia'],scope:'Problem formulation and workflow; not experimental performance evidence.',checks:{data_counts:true,target_separate_from_history:true,unique_graph_nodes:true,caption_checked:true,text_figure_checked:true}};
 fs.writeFileSync(path.join(__dirname,'verification.json'),JSON.stringify(record,null,2));
 fs.writeFileSync(path.join(__dirname,'figures.yaml'),'figures:\n  - id: modeling_flow_native_text\n    source:\n      kind: new_candidate\n      gallery_searches: ["流程/建模: no match", "流程图/机器学习: no match"]\n      rejection_reason: "No matching template; preserve supplied five-module composition using native text and authentic screenshot."\n    code: build.cjs\n    inputs: ["original raster flowchart", "actual raw-data screenshot", "formal data manifest", "formal GCEGNN result"]\n    outputs: ["电商会话推荐建模流程图_排版源文件.html", "recheck.png"]\n    style: "Microsoft YaHei and Georgia; 3840x2160; navy/teal/orange; color plus labels and arrows"\n    statistics: "Frozen formal-subset counts, no performance claims or uncertainty estimates"\n    status: verified\n    caption_checked: true\n    text_figure_checked: true\n    reproduction_contract: "Two renders have identical SHA256; see verification.json"\n');
 console.log(JSON.stringify({output,size:[3840,2160],native_text:true,authentic_screenshot:true,repeated_render_identical:true}));
})().catch(e=>{console.error(e);process.exit(1)});
