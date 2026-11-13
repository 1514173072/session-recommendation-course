"""Display existing measured outputs; does not train or evaluate models again."""
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures/candidates'
MODELS = ['SRGNN', 'GCSAN', 'GCEGNN']
LABELS = ['SR-GNN', 'GC-SAN', 'GCE-GNN']

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = {}
    curves = {}
    results = {}
    cases = {}
    for model in MODELS:
        log = ROOT / 'outputs/console' / f'{model}-train.log'
        text = log.read_text(encoding='utf-8')
        losses = [(int(e)+1, float(v)) for e, v in re.findall(r'epoch (\d+) training \[time: [^,]+, train loss: ([\d.]+)\]', text)]
        mrr = [(int(e)+1, float(v)) for e, v in re.findall(r'epoch (\d+) evaluating \[time: [^,]+, valid_score: ([\d.]+)\]', text)]
        assert [e for e, _ in losses] == [1, 2, 3]
        assert [e for e, _ in mrr] == [1, 2, 3]
        curves[model] = {'epochs': [x[0] for x in losses], 'loss_sum': [x[1] for x in losses], 'valid_mrr10': [x[1] for x in mrr]}
        folder = ROOT / 'outputs/diginetica-smoke' / model
        results[model] = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
        cases[model] = {x['session_id']: x for x in json.loads((folder / 'recommendations.json').read_text(encoding='utf-8'))}
        for p in [log, folder/'result.json', folder/'recommendations.json']:
            source[str(p.relative_to(ROOT))] = digest(p)
    snapshot = {'curves': curves, 'source_sha256': source, 'scope': '2000 real sessions, 3 epochs, seed=42, CPU; single-run smoke verification.'}
    (OUT / 'training_curves.input.json').write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding='utf-8')
    font = FontProperties(family='Microsoft YaHei').get_name()
    plt.rcParams.update({'font.family': font, 'axes.unicode_minus': False, 'font.size': 12,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.axisbelow': True, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.5))
    colors = ['#0072B2', '#D55E00', '#009E73']
    for model, label, color, marker, style in zip(MODELS, LABELS, colors, ['o', 's', '^'], ['-', '--', '-.']):
        d = curves[model]
        axes[0].plot(d['epochs'], d['loss_sum'], color=color, marker=marker, linestyle=style, linewidth=2.2, markersize=7, label=label)
        axes[1].plot(d['epochs'], d['valid_mrr10'], color=color, marker=marker, linestyle=style, linewidth=2.2, markersize=7, label=label)
    axes[0].set_title('A  训练损失变化', loc='left', pad=13, fontsize=14)
    axes[0].set_ylabel('每轮批次损失之和')
    axes[0].set_ylim(0, 1600)
    axes[1].set_title('B  验证集推荐排序表现', loc='left', pad=13, fontsize=14)
    axes[1].set_ylabel('MRR@10（越高越好）')
    axes[1].set_ylim(0, .18)
    for ax in axes:
        ax.set_xticks([1, 2, 3])
        ax.set_xlabel('训练轮次')
        ax.grid(axis='y', color='#E2E8F0', linewidth=.8)
        ax.set_xlim(.85, 3.15)
    fig.suptitle('三模型首次运行：真实训练日志', x=.065, y=.98, ha='left', fontsize=18, weight='bold')
    fig.text(.065, .90, 'Diginetica · 2,000 个真实会话 · CPU · 每模型 3 轮 · seed=42', fontsize=12, color='#475569')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(.5, .09), ncol=3, frameon=False)
    fig.text(.065, .035, '单次小规模验证；损失用于观察各模型内部变化。仅 3 轮训练，尚不能判断充分收敛。', fontsize=10, color='#475569')
    fig.subplots_adjust(top=.78, bottom=.26, left=.085, right=.98, wspace=.30)
    fig.savefig(OUT / 'training_curves.png', dpi=180, facecolor='white', metadata={'Software': 'matplotlib'})
    fig.savefig(OUT / 'training_curves.pdf', facecolor='white', metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    lines = ['# 当前可展示结果', '', '本页只整理已有实测输出，没有追加训练。范围：2,000个真实会话，每模型CPU训练3轮，单一随机种子。', '',
             '## 1. 测试集推荐效果与计算成本', '',
             '每个测试会话只有一个真实下一点击商品，因此Recall@10也等于命中会话占比。MRR@10和NDCG@10同时反映命中商品的排名。', '',
             '| 模型 | 命中会话/2000 | Recall@10 | MRR@10 | NDCG@10 | 训练及验证/秒 |',
             '|---|---:|---:|---:|---:|---:|']
    for m, label in zip(MODELS, LABELS):
        r = results[m]; t = r['test_result']
        lines.append(f"| {label} | {round(t['recall@10']*r['test_examples'])} | {t['recall@10']:.4f} | {t['mrr@10']:.4f} | {t['ndcg@10']:.4f} | {r['training_and_validation_seconds']:.2f} |")
    lines += ['', '训练及验证时间包含3轮拟合、验证与检查点保存，不含环境安装和数据准备。该单次CPU用时只用于本轮展示。', '',
              '## 2. 训练过程', '', '![三模型训练曲线](figures/candidates/training_curves.png)', '',
              '三个模型的训练损失均下降，验证MRR均上升。损失包含各模型各自定义的项，图A用于观察模型内部趋势；图B使用相同的验证指标。尚不能凭3个点断言模型充分收敛。', '',
              '## 3. 同一真实会话的模型推荐差异', '',
              '从已导出的前10个测试案例中选取以下两个案例用于说明行为差异；它们不用于估计总体性能。商品为匿名ID，案例均含重复点击。', '']
    for sid in ['4344', '6254']:
        common = cases['SRGNN'][sid]
        lines += [f'### 会话 {sid}', '', '历史浏览：`'+' → '.join(common['history_item_ids'])+'`。', '',
                  '实际下一点击：`'+common['true_next_item_id']+'`。', '',
                  '| 模型 | 推荐首位商品 | 真实下一商品在Top-10中的排名 |', '|---|---|---|']
        for m, label in zip(MODELS, LABELS):
            c = cases[m][sid]
            assert c['history_item_ids'] == common['history_item_ids'] and c['true_next_item_id'] == common['true_next_item_id']
            rank = next((str(x['rank']) for x in c['recommendations'] if x['item_id']==c['true_next_item_id']), '未进入')
            lines.append(f"| {label} | {c['recommendations'][0]['item_id']} | {rank} |")
        lines.append('')
    lines += ['会话4344中，GCE-GNN把真实下一商品排在第1位；会话6254中，GC-SAN命中第1位，而GCE-GNN未命中Top-10。整体指标较高的模型也会在具体会话上失败。', '',
              '## 来源', '', '- 原始测试指标与用时：outputs/diginetica-smoke/<模型>/result.json。',
              '- 真实推荐：outputs/diginetica-smoke/<模型>/recommendations.json。',
              '- 训练日志：outputs/console/<模型>-train.log。',
              '- 图形输入快照和来源SHA256：figures/candidates/training_curves.input.json。',
              '- 正式完整数据实验与交互网页尚未实施；本页可用于阶段性展示。', '']
    (ROOT / '当前可展示结果.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Exported measured results, learning curves, and two real session comparisons.')

if __name__ == '__main__':
    main()
