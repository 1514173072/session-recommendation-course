"""Create traceable presentation figures from actual training and test outputs."""
import argparse
import csv
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('SRGNN', 'GCSAN', 'GCEGNN')
COLORS = ('#0072B2', '#D58A00', '#009E73')
MARKERS = ('o', 's', '^')
STYLES = ('-', '--', '-.')
FONT = 'Microsoft YaHei'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def curve_from_log(path):
    text = path.read_text(encoding='utf-8')
    losses = {int(epoch): float(loss) for epoch, loss in re.findall(r'epoch (\d+) training \[time: [\d.]+s, train loss: ([\d.]+)\]', text)}
    valid = {int(epoch): float(score) for epoch, score in re.findall(r'epoch (\d+) evaluating \[time: [\d.]+s, valid_score: ([\d.]+)\]', text)}
    assert losses and set(losses) == set(valid), path
    return {'epochs': [e + 1 for e in sorted(losses)], 'loss_sum': [losses[e] for e in sorted(losses)],
            'valid_mrr10': [valid[e] for e in sorted(losses)]}


def configure():
    plt.rcParams.update({'font.family': FONT, 'axes.unicode_minus': False, 'font.size': 14,
                         'axes.titlesize': 17, 'axes.labelsize': 15, 'xtick.labelsize': 12,
                         'ytick.labelsize': 12, 'legend.fontsize': 12, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.edgecolor': '#B6C0CE',
                         'axes.labelcolor': '#35455C', 'text.color': '#24334A',
                         'xtick.color': '#607089', 'ytick.color': '#607089',
                         'axes.axisbelow': True, 'pdf.fonttype': 42, 'ps.fonttype': 42})


def base(title, subtitle):
    return plt.figure(figsize=(12.8, 7.2), facecolor='white')


def foot(fig, text):
    # Explanations are kept in the accompanying Markdown, outside the image.
    pass


def legend(fig):
    handles = [plt.Line2D([], [], color=c, marker=m, linestyle=s, lw=2, label=name)
               for name, c, m, s in zip(MODELS, COLORS, MARKERS, STYLES)]
    fig.legend(handles=handles, ncol=3, loc='lower center', bbox_to_anchor=(.51, .02), frameon=False)


def comparison_summary(inputs):
    fig = plt.figure(figsize=(18, 10.125), facecolor='white')
    grid = fig.add_gridspec(2, 6, left=.065, right=.975, bottom=.12, top=.90, hspace=.42, wspace=.95)
    quality = fig.add_subplot(grid[0, :2])
    time_axis = fig.add_subplot(grid[0, 2:4])
    params_axis = fig.add_subplot(grid[0, 4:])
    length_axis = fig.add_subplot(grid[1, :3])
    popularity_axis = fig.add_subplot(grid[1, 3:])
    width = .23
    metrics = ('recall@10', 'mrr@10', 'ndcg@10')
    for m, (model, color) in enumerate(zip(MODELS, COLORS)):
        values = [inputs['models'][model]['result']['test_result'][key] for key in metrics]
        bars = quality.bar(np.arange(3)+(m-1)*width, values, width, color=color,
                           hatch=('', '//', '..')[m], edgecolor='white')
        for bar, value in zip(bars, values):
            quality.text(bar.get_x()+bar.get_width()/2, value+.008, f'{value*100:.2f}', ha='center', fontsize=9)
    quality.set_xticks(np.arange(3), ['Recall@10', 'MRR@10', 'NDCG@10'])
    quality.set(title='推荐效果', ylim=(0, .42))
    quality.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    for axis, key, divisor, title, unit in (
            (time_axis, 'training_and_validation_seconds', 60, '训练及验证耗时', '分钟'),
            (params_axis, 'parameters', 1e6, '模型参数量', '百万个参数')):
        values = [inputs['models'][model]['result'][key]/divisor for model in MODELS]
        bars = axis.bar(MODELS, values, color=COLORS, width=.56)
        for bar, value, hatch in zip(bars, values, ('', '//', '..')):
            bar.set_hatch(hatch);bar.set_edgecolor('white')
            axis.text(bar.get_x()+bar.get_width()/2, value+max(values)*.035, f'{value:.2f}', ha='center', fontsize=12)
        axis.set(title=title, ylabel=unit, ylim=(0, max(values)*1.25))
    for axis, key, title, xlabel in (
            (length_axis, 'length_groups', '不同会话长度', '会话长度'),
            (popularity_axis, 'popularity_groups', '不同商品流行度', '真实目标商品的训练频次排名')):
        group = inputs[key]
        counts = group['statistics'][MODELS[0]]['sample_counts']
        x = np.arange(len(group['bins']))
        for m, (model, color) in enumerate(zip(MODELS, COLORS)):
            values = group['statistics'][model]['recall10']
            bars = axis.bar(x+(m-1)*width, values, width, color=color,
                            hatch=('', '//', '..')[m], edgecolor='white')
            for bar, value in zip(bars, values):
                axis.text(bar.get_x()+bar.get_width()/2, value+.01, f'{value*100:.1f}', ha='center', fontsize=9)
        axis.set_xticks(x, [label+'\nn='+format(n, ',') for (label, _, _), n in zip(group['bins'], counts)])
        axis.set(title=title, xlabel=xlabel, ylabel='Recall@10', ylim=(0, .66))
        axis.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    for axis in fig.axes:
        axis.grid(axis='y', color='#E6EBF2')
        axis.tick_params(labelsize=11)
        axis.title.set_fontsize(15)
        axis.xaxis.label.set_fontsize(12)
        axis.yaxis.label.set_fontsize(12)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color, label=model, hatch=hatch, ec='white')
               for model, color, hatch in zip(MODELS, COLORS, ('', '//', '..'))]
    fig.legend(handles=handles, ncol=3, loc='upper center', bbox_to_anchor=(.52, .99), frameon=False, fontsize=14)
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', default='rerun20261112')
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    if not args.preview:
        verification_path = ROOT / 'outputs/formal' / ('verification_summary-' + args.tag + '.json' if args.tag else 'verification_summary.json')
        assert json.loads(verification_path.read_text(encoding='utf-8'))['all_passed']
    configure()
    out = ROOT / ('cache/ppt_preview' if args.preview else 'figures/candidates/ppt_20261112')
    out.mkdir(parents=True, exist_ok=True)
    delivery = ROOT.parent / 'PPT展示图片'
    if not args.preview:
        delivery.mkdir(parents=True, exist_ok=True)
    inputs = {'tag': args.tag, 'models': {}, 'sources': [], 'scope': '20k sessions, full-sort Top10, seed42, one run; no uncertainty intervals.'}
    predictions = {}
    temporal_audit = ROOT / 'outputs/rerun20261112/temporal_order_audit.json'
    if temporal_audit.exists():
        inputs['temporal_order_audit'] = json.loads(temporal_audit.read_text(encoding='utf-8'))
        inputs['sources'].append({'path': temporal_audit.relative_to(ROOT).as_posix(), 'sha256': sha(temporal_audit)})
    for model in MODELS:
        label = model + ('-' + args.tag if args.tag else '')
        directory = ROOT / 'outputs/formal' / label / 'seed42'
        result_path = directory / 'result.json'
        result = json.loads(result_path.read_text(encoding='utf-8'))
        log_path = directory / 'train.log'
        pred_path = ROOT / 'outputs/formal' / ('predictions-' + args.tag if args.tag else 'predictions') / (model + '.jsonl')
        with pred_path.open(encoding='utf-8') as handle:
            rows = [json.loads(line) for line in handle]
        assert len(rows) == result['test_examples'] == 20000
        assert len({row['session_id'] for row in rows}) == 20000
        ranks = [row['target_rank'] for row in rows]
        raw_metrics = {'recall@10': np.mean([r > 0 for r in ranks]),
                       'mrr@10': np.mean([1/r if r else 0 for r in ranks]),
                       'ndcg@10': np.mean([1/math.log2(r+1) if r else 0 for r in ranks])}
        assert all(abs(float(value) - result['test_result'][key]) <= .000051 for key, value in raw_metrics.items()), model
        predictions[model] = rows
        curve = curve_from_log(log_path)
        assert abs(max(curve['valid_mrr10']) - result['best_valid_score']) < 1e-8
        inputs['models'][model] = {'result': result, 'curve': curve, 'metrics_from_prediction_rows': raw_metrics}
        for path in (result_path, log_path, pred_path):
            inputs['sources'].append({'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path)})
    for model in MODELS[1:]:
        assert [(r['session_id'], r['history'], r['target']) for r in predictions[model]] == [(r['session_id'], r['history'], r['target']) for r in predictions[MODELS[0]]]
    captions = []
    records = []

    def save(fig, name, task, caption):
        if name[:2] in ('02', '03', '04', '05') and name != '02_三模型对比汇总表':
            plt.close(fig)
            return
        png = out / (name + '.png')
        fig.savefig(png, dpi=300, metadata={'Software': 'Matplotlib'})
        size_inches = [float(value) for value in fig.get_size_inches()]
        plt.close(fig)
        if not args.preview:
            shutil.copy2(png, delivery / png.name)
        records.append({'id': name[:2], 'claim_id': task, 'task': task, 'source': {'kind': 'new_candidate'},
                        'outputs': [png.relative_to(ROOT).as_posix()],
                        'caption': caption, 'style': {'font': FONT, 'size_inches': size_inches, 'png_dpi': 300,
                                                       'colors': list(COLORS), 'markers': list(MARKERS), 'linestyles': list(STYLES)},
                        'status': 'pending_visual_review', 'caption_checked': False, 'text_figure_checked': False})
        captions.append((name + '.png', caption))
        print('WROTE ' + name, flush=True)

    fig = base('模型训练过程', '同一数据划分、seed 42；验证集 MRR@10 选择最佳权重')
    left, right = fig.subplots(1, 2)
    fig.subplots_adjust(left=.09, right=.96, bottom=.18, top=.91, wspace=.28)
    for model, color, marker, style in zip(MODELS, COLORS, MARKERS, STYLES):
        curve = inputs['models'][model]['curve']
        left.plot(curve['epochs'], curve['loss_sum'], color=color, marker=marker, ls=style, lw=2, ms=5)
        right.plot(curve['epochs'], curve['valid_mrr10'], color=color, marker=marker, ls=style, lw=2, ms=5)
        best = int(np.argmax(curve['valid_mrr10']))
        right.scatter(curve['epochs'][best], curve['valid_mrr10'][best], s=160, facecolors='none', edgecolors=color, linewidths=1.6, zorder=5)
    left.set(title='训练损失', xlabel='训练轮次', ylabel='每轮批次损失之和', ylim=(0, None))
    right.set(title='验证集推荐质量', xlabel='训练轮次', ylabel='MRR@10', ylim=(0, .20))
    right.yaxis.set_major_formatter(PercentFormatter(1))
    for axis in (left, right):
        axis.grid(axis='y', color='#E6EBF2')
        axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    legend(fig)
    foot(fig, '空心圆标出各模型最佳验证轮次；最多训练 20 轮，采用提前停止。单种子运行，无误差带。')
    save(fig, '01_模型训练过程', 'training_trace', '左侧为每轮批次交叉熵损失之和，右侧为验证集 MRR@10；空心圈为日志中最高验证分数。下降的训练损失与验证质量应结合阅读。')

    fig = base('三个模型的测试集推荐效果', '20,000 个测试会话；每次在全部候选商品中排序，保留 Top-10')
    axes = fig.subplots(1, 3)
    fig.subplots_adjust(left=.07, right=.97, top=.91, bottom=.12, wspace=.33)
    for axis, metric, title in zip(axes, ('recall@10', 'mrr@10', 'ndcg@10'), ('Recall@10', 'MRR@10', 'NDCG@10')):
        values = [inputs['models'][model]['result']['test_result'][metric] for model in MODELS]
        bars = axis.bar(MODELS, values, color=COLORS, width=.58)
        for bar, value, hatch in zip(bars, values, ('', '//', '..')):
            bar.set_hatch(hatch);bar.set_edgecolor('white')
            axis.text(bar.get_x()+bar.get_width()/2, value+.01, f'{value*100:.2f}%', ha='center', fontsize=12, weight='bold')
        axis.set(title=title, ylim=(0, .42))
        axis.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        axis.grid(axis='y', color='#E6EBF2')
    foot(fig, '指标越高越好。每个会话只有一个真实下一次点击，因此本实验 Recall@10 等于 Hit@10；单种子，无误差条。')
    save(fig, '02_三模型推荐效果对比', 'test_quality', '三项指标来自本轮 result.json，并由全部测试预测逐条重新计算核对。Recall 是前十命中比例，MRR/NDCG 同时考虑目标排名。仅描述本次划分与种子。')

    fig = base('训练计算代价与模型规模', '同一台电脑、CPU、每进程 4 个 PyTorch 线程；三个模型依次运行')
    axes = fig.subplots(1, 2)
    fig.subplots_adjust(left=.09, right=.96, top=.91, bottom=.12, wspace=.31)
    times = [inputs['models'][m]['result']['training_and_validation_seconds']/60 for m in MODELS]
    params = [inputs['models'][m]['result']['parameters']/1e6 for m in MODELS]
    for axis, values, title, ylabel in zip(axes, (times, params), ('实测训练及验证耗时', '模型参数量'), ('分钟', '百万个参数')):
        bars = axis.bar(MODELS, values, color=COLORS, width=.58)
        axis.set(title=title, ylabel=ylabel, ylim=(0, max(values)*1.25))
        for bar, value, hatch in zip(bars, values, ('', '//', '..')):
            bar.set_hatch(hatch);bar.set_edgecolor('white')
            axis.text(bar.get_x()+bar.get_width()/2, value+max(values)*.035, f'{value:.2f}', ha='center', weight='bold')
        axis.grid(axis='y', color='#E6EBF2')
    foot(fig, '耗时包含各模型实际训练轮次和验证，不包含数据准备、模型初始化、全局图构建及最终测试；不是渐近复杂度。')
    save(fig, '03_训练计算代价与参数量', 'measured_cost', '耗时为训练脚本 trainer.fit 的墙钟时间；模型初始化前的图构建不在该计时段。参数量是实际实例的参数数目。提前停止轮次可能不同，不作每轮效率或大 O 复杂度等同解释。')

    def grouped_plot(name, title, subtitle, bins, mask_fn, xlabel, task, caption):
        fig = base(title, subtitle)
        axis = fig.add_subplot(111)
        fig.subplots_adjust(left=.10, right=.96, top=.94, bottom=.20)
        x = np.arange(len(bins));width=.23
        counts=[];stats={}
        for label, low, high in bins:
            mask = [mask_fn(r, low, high) for r in predictions[MODELS[0]]]
            counts.append(sum(mask))
            assert counts[-1] > 0
        assert sum(counts) == 20000
        for index, (model, color) in enumerate(zip(MODELS, COLORS)):
            values=[];hits=[]
            for label, low, high in bins:
                group=[r for r in predictions[model] if mask_fn(r, low, high)]
                hit=sum(r['target_rank']>0 for r in group)
                hits.append(hit);values.append(hit/len(group))
            stats[model]={'hit_counts':hits, 'sample_counts':counts, 'recall10':values}
            bars=axis.bar(x+(index-1)*width, values, width, color=color, label=model, hatch=('', '//', '..')[index], edgecolor='white')
            for bar,value in zip(bars,values):
                axis.text(bar.get_x()+bar.get_width()/2,value+.008,f'{value*100:.1f}%',ha='center',fontsize=10)
        axis.set_xticks(x,[label+'\n'+'n='+format(n,',') for (label,_,_),n in zip(bins,counts)])
        axis.set(ylabel='Recall@10',xlabel=xlabel,ylim=(0,.65))
        axis.yaxis.set_major_formatter(PercentFormatter(1,decimals=0));axis.grid(axis='y',color='#E6EBF2')
        axis.legend(ncol=3,frameon=False,loc='upper right')
        foot(fig,'n 为该组测试会话数，分母为组内会话总数。单次测试结果，无误差条；分组比较描述关联。')
        inputs[task]={'bins':[list(b) for b in bins], 'statistics':stats}
        save(fig,name,task,caption)

    grouped_plot('04_不同会话长度的推荐表现','浏览记录长度与推荐表现','按模型输入中的点击次数分组；各模型在相同会话上比较',
                 [('4–6 次',4,6),('7–10 次',7,10),('11–14 次',11,14),('15–19 次',15,19)],
                 lambda r,lo,hi:lo<=r['history_len']<=hi,'会话长度','length_groups',
                 '按测试输入长度分组计算 Recall@10，图上 n 为每组样本量。各组分母不同；该图不能解释为增加点击记录必然提高准确率。')
    grouped_plot('05_不同商品流行度的推荐表现','商品流行度与推荐表现','按真实目标商品在训练集中的点击频次排名分组',
                 [('前 100 名',1,100),('101–1,000 名',101,1000),('1,001–5,000 名',1001,5000),('5,000 名以后',5001,1000000)],
                 lambda r,lo,hi:lo<=r['target_popularity_rank']<=hi,'真实目标商品的训练频次排名','popularity_groups',
                 '商品流行度来自预测导出脚本在训练样本中统计的目标商品频次排名。每组以对应测试会话数为分母；低频组表现用于讨论长尾局限，不等同于新商品冷启动实验。')

    table_rows = []
    for metric in ('recall@10', 'mrr@10', 'ndcg@10'):
        table_rows.append(['推荐效果', metric.upper().replace('RECALL', 'Recall'), '%', 20000,
                           *[inputs['models'][m]['result']['test_result'][metric]*100 for m in MODELS]])
    table_rows.append(['计算代价', '训练及验证耗时', '分钟', '',
                       *[inputs['models'][m]['result']['training_and_validation_seconds']/60 for m in MODELS]])
    table_rows.append(['计算代价', '参数量', '个', '',
                       *[inputs['models'][m]['result']['parameters'] for m in MODELS]])
    for category, key in (('会话长度', 'length_groups'), ('商品流行度', 'popularity_groups')):
        group = inputs[key]
        for index, (label, _, _) in enumerate(group['bins']):
            table_rows.append([category, label+' · Recall@10', '%', group['statistics'][MODELS[0]]['sample_counts'][index],
                               *[group['statistics'][m]['recall10'][index]*100 for m in MODELS]])
    columns = ['类别', '指标或分组', '单位', '会话数', *MODELS]
    csv_path = out/'02_三模型对比汇总表.csv'
    with csv_path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows([[*row[:4], *[round(value, 6) if isinstance(value, float) else value for value in row[4:]]] for row in table_rows])
    if not args.preview:
        shutil.copy2(csv_path, delivery/csv_path.name)
    image_rows = []
    previous_category = None
    for row in table_rows:
        category, label, unit, count, *values = row
        shown_category = category if category != previous_category else ''
        previous_category = category
        image_rows.append([shown_category, label, unit, f'{count:,}' if count != '' else '—',
                           *[f'{int(value):,}' if unit == '个' else f'{value:.2f}' for value in values]])
    table_figure = plt.figure(figsize=(16, 8.4), facecolor='white')
    table_axis = table_figure.add_axes([.025, .025, .95, .95])
    table_axis.axis('off')
    table = table_axis.table(cellText=image_rows, colLabels=columns, cellLoc='center', colLoc='center',
                             colWidths=[.105, .29, .065, .10, .147, .147, .146], bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(12.5)
    category_index = -1
    prior = None
    for row_index in range(len(image_rows)+1):
        if row_index and table_rows[row_index-1][0] != prior:
            category_index += 1
            prior = table_rows[row_index-1][0]
        for col_index in range(len(columns)):
            cell = table[row_index, col_index]
            cell.set_edgecolor('#DFE6EF');cell.set_linewidth(.7)
            cell.set_facecolor('#EEF2F7' if row_index == 0 else ('#FFFFFF' if category_index%2 == 0 else '#F7F9FC'))
            if row_index == 0:
                cell.get_text().set_weight('bold')
                if col_index >= 4:
                    cell.get_text().set_color(COLORS[col_index-4])
            if col_index == 1 and row_index:
                cell.get_text().set_ha('left');cell.PAD = .05
    save(table_figure, '02_三模型对比汇总表', 'model_comparison_table',
         '表格汇总三个模型的推荐效果、训练及验证耗时、参数量、会话长度及商品流行度分组 Recall@10。百分比数值的单位为 %；会话数为对应组的分母。CSV 保存数值列，最多保留六位小数；PNG 显示两位小数。耗时不含图构建与最终测试，分组表现是描述性结果。')
    records[-1]['outputs'].append(csv_path.relative_to(ROOT).as_posix())

    row = next(r for r in predictions['SRGNN'] if r['session_id']=='10114')
    assert row['history']==['13669','8719','13669','9841']
    fig = base('从浏览序列构造会话图', '真实会话 10114；绿色节点表示重复浏览的商品，箭头表示相邻点击方向')
    axis=fig.add_axes([.04,.04,.92,.92]);axis.set(xlim=(0,1),ylim=(0,1));axis.axis('off')
    for i,item in enumerate(row['history']):
        x=.13+i*.22
        axis.add_patch(FancyBboxPatch((x-.06,.80),.12,.10,boxstyle='round,pad=.015',facecolor='#EEF3FB',edgecolor='#CDDDF4'))
        axis.text(x,.85,item,ha='center',va='center',fontfamily='Consolas',fontsize=17,color='#31558A')
        if i<3:axis.add_patch(FancyArrowPatch((x+.08,.85),(x+.14,.85),arrowstyle='-|>',mutation_scale=15,color='#9AAABD'))
    positions={'13669':(.5,.45),'8719':(.20,.20),'9841':(.80,.20)}
    counts=Counter(row['history']);edges=Counter(zip(row['history'][:-1],row['history'][1:]))
    for (source,target),count in edges.items():
        radius=.18 if (target,source) in edges else 0
        axis.add_patch(FancyArrowPatch(positions[source],positions[target],connectionstyle=f'arc3,rad={radius}',arrowstyle='-|>',mutation_scale=22,lw=2,color='#71839D',shrinkA=38,shrinkB=38))
    for item,(x,y) in positions.items():
        axis.add_patch(Circle((x,y),.064,facecolor='#E9F7EF' if counts[item]>1 else '#EEF3FB',edgecolor='#009E73' if counts[item]>1 else '#89A7CD',lw=2))
        axis.text(x,y,item,ha='center',va='center',fontsize=17,fontfamily='Consolas')
    foot(fig,'这里展示原始点击转移图。模型会进一步按各自实现构造边、反向边或边类型；真实目标单独留出。')
    inputs['session_graph']={'session_id':row['session_id'],'history':row['history'],'target':row['target'],
                             'node_click_counts':dict(counts),'edges':[{'source':a,'target':b,'count':v} for (a,b),v in edges.items()]}
    save(fig,'06_真实会话的序列转图','sequence_to_graph','用真实会话说明序列如何形成去重节点和有向相邻转移。13669 出现两次、形成双向访问；图示为原始转移图，不代替各模型完整图构建或架构。')

    (out/'figure_inputs.json').write_text(json.dumps(inputs,ensure_ascii=False,indent=2),encoding='utf-8')
    manifest={'scope':inputs['scope'],'data':(out/'figure_inputs.json').relative_to(ROOT).as_posix(),
              'code':'experiments/export_ppt_figures.py','gallery_queries':[{'data_type':'时间序列','keyword':'训练','matches':0},
              {'method_type':'模型比较','keyword':'性能','matches':0},{'keyword':'收敛','reviewed':'分面收敛曲线.md（外部模板查询记录）'},
              {'keyword':'曲线','result':'查到因果、超参数稳健性等专用契约，与本次单种子训练和模型测试不同'},
              {'keyword':'热图','result':'查到时滞与时空场矩阵等专用契约，不匹配本次三模型分组表'}],
              'rejection_reason':'已查看分面收敛曲线的 Markdown、PNG 和数据契约。它要求双参数分面和样本量网格；本任务只有训练轮次、单种子指标和会话组统计。其他专用契约也不匹配。新建基础折线、分组柱形和原始会话图，仅保存在项目候选池。',
              'figures':records}
    (out/'figures.yaml').write_text(yaml.safe_dump(manifest,allow_unicode=True,sort_keys=False),encoding='utf-8')
    if not args.preview:
        lines=['# PPT 图片使用说明','','本轮来源：2026-11-12 完整重新训练、测试及权重独立重载。全部结果采用同一 20,000 会话子集、seed 42。',
               '','仅保留 PNG 图片；对比汇总表同时提供 CSV。训练过程与会话图为 3840 × 2160，汇总表为 4800 × 2520。网页案例为高清结果区域截图。',
               '','## 对应汇报段落','','| 图片 | 建议放置段落 |','| --- | --- |',
               '| 01 训练过程 | 算法设计与对比实验：损失、验证曲线及最佳轮次选择 |',
               '| 02 对比汇总表（PNG、CSV） | 对比实验、计算代价与分组表现 |',
               '| 06 序列转图 | 问题建模 |',
               '| 07–10 页面截图 | 实机演示、模型比较与失败案例 |',
               '','## 图表及用途','']
        for filename,caption in captions:
            lines.extend(['### '+filename,'',caption,''])
        lines.extend(['## 讲解口径','','本轮只有一个随机种子，图表描述本次实验；不主张统计显著性、普遍优劣或正式榜单复现。',
                      '','训练曲线与测试图来自本轮同一组权重；独立测试重载用于检查保存与重新使用。',
                      '','本任务允许再次点击已浏览的商品，模型也允许推荐历史商品。示例会话的真实目标 13669 已在历史中出现，这表示回访。',
                      '','原始数值、日志哈希和复现代码保存在“完整项目/figures/candidates/ppt_20261112”和“完整项目/experiments/export_ppt_figures.py”。',
                      '','## 数据时间顺序说明','',
                      '框架使用 float32 时间戳排序。对原始秒级时间戳核对发现，20,000 个会话中有 14 个的部分点击顺序存在并列精度差异，其中 1 个影响留出的末次点击；完整记录见“完整项目/outputs/rerun20261112/temporal_order_audit.json”。本轮保留原处理方式以对照历史结果。指标按实际模型数据集中的顺序和标签计算；案例截图选择不受此差异影响的会话。'])
        notes_path = delivery/'图片使用说明.md'
        previous = notes_path.read_text(encoding='utf-8') if notes_path.exists() else ''
        case_notes = '\n\n## 网页案例截图'+previous.split('## 网页案例截图', 1)[1] if '## 网页案例截图' in previous else ''
        notes_path.write_text('\n'.join(lines)+case_notes+'\n',encoding='utf-8')
    print('FIGURES COMPLETE',flush=True)


if __name__=='__main__':
    main()
