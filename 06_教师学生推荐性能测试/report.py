"""Readable comparison; no artificial API timing or claim of teacher gain."""
import csv
import json
from common import *

NAMES={'gra_original':'原始灰关联','gra_beta7':'差异压缩灰关联 β=7',
       'teacher_rules':'对话教师原型规则（本地执行）','student_plain':'MLP：真实点击监督',
       'student_teacher':'MLP：真实点击＋教师指导','SRGNN':'SR-GNN',
       'GCSAN':'GC-SAN','GCEGNN':'GCE-GNN'}

def main():
    test=json.loads((HERE/'results/test_summary.json').read_text(encoding='utf-8'))
    train=json.loads((HERE/'results/training_summary.json').read_text(encoding='utf-8'))
    lat={k:json.loads((HERE/f'results/latency/{k}.json').read_text(encoding='utf-8')) for k in test}
    rows=[]
    for k in ['gra_original','gra_beta7','SRGNN','GCSAN','GCEGNN','teacher_rules','student_plain','student_teacher']:
        rows.append({'model':k,'name':NAMES[k],**test[k]['test'],
            'mean_ms':lat[k]['mean_ms'],'p95_ms':lat[k]['p95_ms'],
            'numeric_state_bytes':lat[k]['numeric_state_bytes'],
            'process_rss_after_load_bytes':lat[k]['process_rss_after_load_bytes'],
            'process_rss_after_requests_bytes':lat[k]['process_rss_after_requests_bytes'],
            'trainable_parameters':lat[k]['trainable_parameters']})
    write_json(HERE/'results/comparison.json',rows)
    with (HERE/'results/comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    # Paired student wins/losses and genuine case examples.
    records={}
    for k in ['student_plain','student_teacher','teacher_rules','gra_beta7']:
        records[k]=[json.loads(line) for line in (HERE/f'predictions/{k}.jsonl').open(encoding='utf-8')]
    wins=[];losses=[];rankwins=ranklosses=0
    cases={}
    for i,(a,b) in enumerate(zip(records['student_plain'],records['student_teacher'])):
        assert a['session_id']==b['session_id']
        ra,rb=a['target_rank'],b['target_rank']
        if rb and not ra: wins.append(i)
        if ra and not rb: losses.append(i)
        if rb and (not ra or rb<ra): rankwins+=1
        if ra and (not rb or ra<rb): ranklosses+=1
        kind='teacher_student_gain' if rb and not ra else ('teacher_student_failure' if ra and not rb else None)
        if kind and kind not in cases:
            cases[kind]={'plain':a,'teacher_student':b,'teacher_rules':records['teacher_rules'][i],
                         'gra_beta7':records['gra_beta7'][i]}
    paired={'new_hits':len(wins),'lost_hits':len(losses),'net_hits':len(wins)-len(losses),
            'rank_improved':rankwins,'rank_worsened':ranklosses,'queries':20000}
    write_json(HERE/'results/paired_students.json',paired)
    write_json(HERE/'results/cases.json',cases)
    delta=(test['student_teacher']['test']['recall@10']-test['student_plain']['test']['recall@10'])*100
    delta_mrr=test['student_teacher']['test']['mrr@10']-test['student_plain']['test']['mrr@10']
    direction='提高' if delta>=0 else '降低'
    lines=['# 教师关联指导与轻量学生：性能比较', '',
        f'有教师 MLP 相比无教师 MLP，Recall@10 {direction} {abs(delta):.3f} 个百分点，MRR@10 变化 {delta_mrr:+.6f}；新增命中 {len(wins)} 个、丢失命中 {len(losses)} 个。',
        '', '本次为独立性能测试，不纳入灰预测课程论文。数值来自真实全商品预测和本轮本地计时。', '',
        '## 实际实现', '',
        '- 教师：当前对话助手阅读训练侧分组证据后编写 12 类情境原型，包含四个权重和六个交互系数；界面模型名由用户报告为 GPT-6.1 Sol，未通过独立 API 核验。',
        '- 12 个原型覆盖长度×最后商品重复×热度，十二类均有实际训练证据。2,000 条指导由原型确定性赋值，不是 2,000 次独立大模型调用。',
        '- 学生：6→16→10 的条件 MLP，282 参数，FP32 数值权重 1,128 字节。输出四项 softmax 权重与六项有界交互系数，以四特征加权和及二次交互完成排序。',
        '- 两个学生均使用 97,273 个训练前缀、相同初始化/负样本/批次顺序/30轮预算；教师学生另有 2,000 条指导的系数监督。每五轮在固定 2,000 验证样本上按 MRR 选择。',
        '- 训练行为特征采用按会话五折交叉拟合，避免当前训练目标自身计入特征；验证、测试与在线推荐复用全部训练统计。', '',
        '## 推荐质量和本地响应时间', '',
        '|方法|Recall@10|MRR@10|NDCG@10|平均 ms|P95 ms|数值状态 MB|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'|{r["name"]}|{r["recall@10"]:.5f}|{r["mrr@10"]:.5f}|{r["ndcg@10"]:.5f}|{r["mean_ms"]:.3f}|{r["p95_ms"]:.3f}|{r["numeric_state_bytes"]/1e6:.3f}|')
    lines += ['', '**教师规则这一行是已生成参数原型的本地执行，不是大模型在线推理延迟。** 本次没有调用付费 API，也没有可复现的大模型逐请求在线计时，不能据此计算相对 API 的提速倍数。', '',
        '## 内存与离线训练', '',
        '|方法|加载后进程 RSS MB|请求后 RSS MB|', '|---|---:|---:|']
    for r in rows:
        lines.append(f'|{r["name"]}|{r["process_rss_after_load_bytes"]/1e6:.1f}|{r["process_rss_after_requests_bytes"]/1e6:.1f}|')
    lines += ['', 'RSS 包含 Python、PyTorch、映射及可能保留的数据集对象；数值状态只统计算法保留的数组与张量。该参考实现为复用既有统计接口保留 counts/weights 等字段。MLP 自身的 1.1 KiB 不能代表完整进程内存。', '',
        '|学生|选中轮次|优化时间 s|训练及验证选择时间 s|', '|---|---:|---:|---:|']
    for k,v in train.items(): lines.append(f'|{NAMES[k]}|{v["selected_epoch"]}|{v["optimization_seconds"]:.2f}|{v["training_and_selection_seconds"]:.2f}|')
    lines += ['', '教师分析发生在当前对话中，未建立独立调用计时；数据准备与学生训练分别留有日志。GNN 使用已有 checkpoint 和已核对预测，未重新训练。', '',
        '## 计时与比较范围', '',
        '- 全测试：20,000 个会话，32,968 个商品；允许历史商品，排除 padding；相同预测目标和历史。',
        '- 本轮计时：CPU、PyTorch 4 线程、batch=1；相同 300 条原始商品编号请求，预热后每种方法三轮，共 900 次。包含编号转换、特征/图构建、全商品评分、Top-10 和输出转换，排除启动/文件读写/HTTP/排队。',
        '- 原灰关联与 β=7 的全测试指标须与原结果一致；GNN 逐条核对已有预测，重新测量本地延迟。',
        '- 旧测试集已在历史探索中查看；本轮教师规则与学生选择未使用测试目标。这是单数据、单种子探索，不是新的独立确认实验。',
        '- 新方法含六个上下文和交互项，不能将与灰关联的全部差异归因于教师。教师的净作用以两个同结构学生的对照为准。', '',
        '## 真实案例', '', '完整历史、匿名商品编号、两学生评分参数及推荐列表见 results/cases.json。匿名编号不代表已恢复商品名称。']
    for kind,case in cases.items():
        a,b=case['plain'],case['teacher_student']
        lines += ['',f'### {"指导后新增命中" if kind=="teacher_student_gain" else "指导后丢失命中"}', '',
            f'会话 {a["session_id"]}；历史 {" → ".join(a["history"])}；真实下一商品 {a["target"]}。',
            f'无教师排名：{a["target_rank"] or "前十未命中"}；有教师排名：{b["target_rank"] or "前十未命中"}。']
    (HERE/'性能比较报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines[:25]),flush=True)

if __name__=='__main__': main()
