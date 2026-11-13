"""Export the eight completed groups as a table image, CSV and short notes."""
from pathlib import Path
import csv,json,hashlib,shutil
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt,font_manager
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
DEST=ROOT/'08扩展对比/云端八组对比'
ART=HERE/'figures/candidates'
MODELS=['mimo-v2.6-flash','qwen3.8-flash','glm-5.3-flash','deepseek-v4.1-flash']
NAMES=['MiMo-V2.6-Flash','Qwen3.8 Flash','GLM-5.3-Flash','DeepSeek V4.1 Flash']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    DEST.mkdir(parents=True,exist_ok=True);ART.mkdir(parents=True,exist_ok=True)
    summary=json.loads((HERE/'results/summary.json').read_text(encoding='utf-8'))
    pred=json.loads((HERE/'results/predictions.json').read_text(encoding='utf-8'))
    by={(r['model'],r['condition']):r for r in summary}
    rows=[]
    for model,name in zip(MODELS,NAMES):
        for cond in ['plain','teacher']:
            r=by[(model,cond)]
            assert r['queries']==30
            rows.append({'模型':name,'模型ID':model,'教师指导':'无' if cond=='plain' else '有',
                '会话数':r['queries'],'有效输出数':r['valid_outputs'],'有效输出率':r['valid_output_rate'],
                'Recall@10':r['recall@10'],'MRR@10':r['mrr@10'],'NDCG@10':r['ndcg@10'],
                '平均总调用耗时_s':r['mean_all_attempt_wall_seconds'],
                '平均首个有效输出耗时_s':r['mean_recommendation_wall_seconds'],
                '平均模型步骤耗时_s':r['mean_model_step_seconds'],
                'P95模型步骤耗时_s':r['p95_model_step_seconds'],'实际尝试次数':r['actual_attempts'],
                '有思考输出的有效请求数':r['thinking_response_count'],
                '调用计划工具的有效请求数':r['requests_with_tool_calls'],
                '严格协议通过数':r['strict_protocol_count'],
                '非缓存输入Tokens':r['total_input_tokens'],'输出Tokens':r['total_output_tokens']})
    csv_path=DEST/'云端八组对比.csv'
    with csv_path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fontpath=r'C:\Windows\Fonts\msyh.ttc'
    font_manager.fontManager.addfont(fontpath)
    font=font_manager.FontProperties(fname=fontpath).get_name()
    plt.rcParams.update({'font.family':font,'axes.unicode_minus':False,'savefig.facecolor':'white'})
    headers=['模型','教师','Recall@10','MRR@10','NDCG@10','平均调用（s）','有效输出']
    cells=[[r['模型'],r['教师指导'],f"{r['Recall@10']*100:.2f}%",f"{r['MRR@10']:.5f}",
            f"{r['NDCG@10']:.5f}",f"{r['平均首个有效输出耗时_s']:.3f}",f"{r['有效输出数']}/30"] for r in rows]
    fig,ax=plt.subplots(figsize=(15.5,4.7),dpi=180);fig.subplots_adjust(left=.015,right=.985,top=.98,bottom=.02)
    ax.axis('off')
    # Fixed cell geometry avoids Table's iterative bbox adjustment across draws.
    from matplotlib.patches import Rectangle
    ax.set_xlim(0,1);ax.set_ylim(0,1)
    widths=[.245,.075,.13,.13,.13,.17,.12]
    edges=np.r_[0,np.cumsum(widths)]
    height=1/9
    for r,values in enumerate([headers]+cells):
        for c,value in enumerate(values):
            face='#e8eef2' if r==0 else ('#ffffff' if ((r-1)//2)%2==0 else '#f3f6f8')
            ax.add_patch(Rectangle((edges[c],1-(r+1)*height),widths[c],height,
                facecolor=face,edgecolor='#d6dee4',linewidth=.8,clip_on=False))
            ax.text((edges[c]+edges[c+1])/2,1-(r+.5)*height,value,
                ha='center',va='center',fontsize=13,
                fontweight='bold' if r==0 else 'normal',color='#23323c')
    candidate=ART/'云端八组对比.png'
    # Table cell layout is resolved at the final output DPI before hashing.
    fig.canvas.draw();fig.canvas.draw()
    for n in range(2):
        fig.savefig(candidate,dpi=180,metadata={'Software':'matplotlib; frozen8group Harness pilot'})
        current=sha(candidate)
        if n==0:first=current
        else:assert first==current
    plt.close(fig);shutil.copy2(candidate,DEST/candidate.name)
    paired=[]
    for model,name in zip(MODELS,NAMES):
        plain={p['case_index']:p for p in pred if p['model']==model and p['condition']=='plain'}
        guided={p['case_index']:p for p in pred if p['model']==model and p['condition']=='teacher'}
        wins=sum(bool(guided[i]['target_rank']) and not plain[i]['target_rank'] for i in range(30))
        losses=sum(bool(plain[i]['target_rank']) and not guided[i]['target_rank'] for i in range(30))
        paired.append({'model':model,'name':name,'new_hits':wins,'lost_hits':losses,'net_hits':wins-losses,
            'recall_gain_points':(by[(model,'teacher')]['recall@10']-by[(model,'plain')]['recall@10'])*100,
            'mrr_gain':by[(model,'teacher')]['mrr@10']-by[(model,'plain')]['mrr@10']})
    (HERE/'results/paired.json').write_text(json.dumps(paired,ensure_ascii=False,indent=2),encoding='utf-8')
    notes=['# 八组测试说明','','|项目|设置|','|---|---|',
        '|对比|四个模型，各测无教师、有教师两种，共八组|',
        '|样本|相同30个随机抽取的真实测试会话；每组30条请求|',
        '|输入|匿名浏览历史、会话状态和四类推荐特征定义；不提供真实下一点击|',
        '|输出|四项权重和六项交互系数；本地程序对全部32,968个商品评分并输出Top-10|',
        '|无教师|模型独立决定评分参数|',
        '|有教师|在相同提示词中加入已有训练侧教师原型；模型权重不变|',
        '|教师来源|当前对话助手依据训练数据编写的12类关联指导|',
        '|调用|通过本机DeepSeek Harness和OpenCode Go；最多4个并发请求|',
        '|思考设置|GLM入口强制思考，使用low；其余请求off，实际思考输出单独记录|',
        '|平均调用耗时|从启动Harness到首个合法参数输出，含网络、客户端启动和此前失败重试；无合法输出时计入全部失败耗时；全部实际尝试成本另列CSV|',
        '|工具行为|Qwen少量请求调用计划工具后返回合法参数；数值有效性与严格协议遵从分开记录；未执行文件、shell或网页操作|',
        '|模型步骤耗时|Harness的step_start至step_end，包含模型请求、网络和生成，排除客户端启动|',
        '|失败处理|最多重试一次；两次均无有效输出则按未命中计，不补造推荐|',
        '|范围|30会话试测，非原先20,000会话全测试；本轮不加入课程论文|',
        '', '|模型|教师新增命中|教师丢失命中|净变化|Recall变化（百分点）|MRR变化|',
        '|---|---:|---:|---:|---:|---:|']
    for p in paired:
        notes.append(f"|{p['name']}|{p['new_hits']}|{p['lost_hits']}|{p['net_hits']:+d}|{p['recall_gain_points']:+.2f}|{p['mrr_gain']:+.5f}|")
    (DEST/'测试说明.md').write_text('\n'.join(notes)+'\n',encoding='utf-8')
    audit={'figure_sha256':current,'two_renders_same_hash':True,'csv_sha256':sha(csv_path),
        'source_summary_sha256':sha(HERE/'results/summary.json'),'rows':8,'font':font,
        'size_inches':[15.5,4.7],'dpi':180,'image_style':'table only, no title or footnotes',
        'visual_checked':False,'text_figure_checked':True}
    (HERE/'figures/export_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(rows,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
