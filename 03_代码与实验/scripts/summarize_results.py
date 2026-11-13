"""Summarize actual completed runs and verify comparable data splits."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    models = ['SRGNN', 'GCSAN', 'GCEGNN']
    results = [json.loads((ROOT / 'outputs/diginetica-smoke' / m / 'result.json').read_text(encoding='utf-8')) for m in models]
    assert all(r['checkpoint_reload_passed'] and r['top10_validation_passed'] for r in results)
    assert all(r.get('independent_process_reload_passed') for r in results)
    assert all(r['split_session_target_sha256'] == results[0]['split_session_target_sha256'] for r in results)
    data = json.loads((ROOT / 'outputs/data_manifest.json').read_text(encoding='utf-8'))
    rows = [{'model': r['model'], **r['test_result'], 'training_and_validation_seconds': r['training_and_validation_seconds'],
             'parameters': r['parameters'], 'reload_passed': r['checkpoint_reload_passed']} for r in results]
    with (ROOT / 'outputs/comparison.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    report = ['# 首次运行验收记录', '', '题目：基于图神经网络的电商会话推荐算法研究与实现。', '',
              '三个模型均实际完成训练、验证、测试、权重保存与新建模型重载，以及真实测试会话Top-10推荐。独立进程重载后，测试指标与原训练进程完全一致。', '',
              f"完整下载数据：{data['raw_sessions']:,}个会话、{data['raw_items']:,}个商品、{data['raw_interactions']:,}条交互。",
              f"本轮训练：{data['smoke_sessions']:,}个真实会话、{data['smoke_items']:,}个商品、{data['smoke_interactions']:,}条原始交互；每模型3轮，CPU，随机种子42。", '',
              '| 模型 | Recall@10 | MRR@10 | NDCG@10 | 训练及验证/秒 | 权重重载 |',
              '|---|---:|---:|---:|---:|---|']
    for r in results:
        t = r['test_result']
        report.append(f"| {r['model']} | {t['recall@10']:.4f} | {t['mrr@10']:.4f} | {t['ndcg@10']:.4f} | {r['training_and_validation_seconds']:.2f} | 通过 |")
    report += ['', '## 结果解释与运行范围', '',
               '这些数值验证真实数据上的软件运行链路。子集、过滤和训练预算不同于官方排行榜，尚未完成正式复现或多随机种子比较，不用本表宣称模型优劣。',
               '训练/验证/测试的会话ID与目标商品ID序列指纹在三个模型间一致；全候选评价排除padding商品。重载比较使用同一训练数据重建GCE-GNN全局图。',
               '推荐示例位于各模型recommendations.json，包含真实历史、下一商品标签、排名与模型得分；得分不解释为购买概率。', '',
               '## 文件与复跑', '',
               '- 源码：RecBole-GNN/，官方提交 '+results[0]['environment']['upstream_commit']+'。',
               '- 数据来源与SHA256：outputs/data_manifest.json。官方S3入口返回403，使用官方RecSysDatasets链接的Google Drive公开备份。',
               '- 环境：D:/Envs/recbole-gnn-py39，精确依赖见requirements-lock.txt。CPU首次验证避开较慢的CUDA大包下载。',
               '- 配置：configs/smoke.yaml，各模型effective_config.txt保存实际参数。',
               '- 训练与重载验证：powershell -ExecutionPolicy Bypass -File ./run_smoke.ps1。',
               '- 独立进程重载推荐：powershell -ExecutionPolicy Bypass -File ./run_smoke.ps1 -InferenceOnly。',
               '- 每模型权重与结果：outputs/diginetica-smoke/<模型>/。完整控制台日志：outputs/console/。',
               '- 后续正式实验与Streamlit界面尚未包含在本轮首次运行验收中。', '']
    (ROOT / '首次运行验收.md').write_text('\n'.join(report), encoding='utf-8')
    summary = {'models_completed': models, 'same_split_passed': True, 'checkpoint_roundtrip_passed': True,
               'independent_process_reload_passed': True,
               'valid_top10_passed': True, 'full_data_downloaded': True, 'scope': '2000 real sessions, 3 epochs, CPU; runtime verification only.'}
    (ROOT / 'outputs/verification_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(rows, ensure_ascii=True, indent=2))

if __name__ == '__main__':
    main()
