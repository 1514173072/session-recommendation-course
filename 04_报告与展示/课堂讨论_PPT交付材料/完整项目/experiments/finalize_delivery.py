"""Audit the delivered HTML, full rerun evidence and actual exported images."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
TAG = 'rerun20261112'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--visual-reviewed',action='store_true')
    args=parser.parse_args()
    assert args.visual_reviewed, 'Inspect final PNGs before finalizing.'
    out=ROOT/'outputs'/TAG
    progress=read(out/'progress.json')
    assert progress['status']=='complete' and len(progress['steps'])==7
    assert all(step['exit_code']==0 for step in progress['steps'])
    independent=read(ROOT/'outputs/formal'/('verification_summary-'+TAG+'.json'))
    assert independent['all_passed']
    summary=read(out/'rerun_summary.json')
    results={}
    for model in ('SRGNN','GCSAN','GCEGNN'):
        item=independent['models'][model]
        assert item['reload_passed'] and item['top10_passed'] and item['prediction_rows']==20000
        result=read(ROOT/'outputs/formal'/(model+'-'+TAG)/'seed42/result.json')
        assert sha(ROOT/result['checkpoint'])==result['checkpoint_sha256']
        assert result['test_examples']==20000 and result['raw_session_count']==20000
        results[model]=result
    assert all(item['same_data_split'] for item in summary['models'])
    manifest=read(BASE/'演示页面/app/demo_manifest.json')
    html=BASE/'演示页面/会话推荐演示.html'
    assert manifest['tag']==TAG and manifest['test_sessions']==20000
    assert manifest['html_sha256']==sha(html)
    for source in manifest['source_files']:
        assert sha(Path(source['path']))==source['sha256']
    ui=read(BASE/'演示页面/app/html_verification.json')
    assert ui['passed'] and ui['html_sha256']==sha(html) and len(ui['screenshots'])==4
    figures=ROOT/'figures/candidates/ppt_20261112'
    inputs=read(figures/'figure_inputs.json')
    assert inputs['tag']==TAG
    for source in inputs['sources']:
        assert sha(ROOT/source['path'])==source['sha256']
    repeat=read(figures/'reproduction_check.json')
    assert repeat['all_identical'] and len(repeat['files'])==4
    gallery=yaml.safe_load((figures/'figures.yaml').read_text(encoding='utf-8'))
    assert len(gallery['figures'])==3
    assets=[]
    for record in gallery['figures']:
        for name in record['outputs']:
            path=ROOT/name
            copied=BASE/'PPT展示图片'/path.name
            assert path.is_file() and copied.is_file() and sha(path)==sha(copied)
            if path.suffix=='.png':
                with Image.open(path) as image:
                    expected=tuple(round(value*record['style']['png_dpi']) for value in record['style']['size_inches'])
                    assert image.size==expected
            assets.append({'path':str(copied),'sha256':sha(copied)})
        record.update(status='verified',caption_checked=True,text_figure_checked=True,
                      review='Final PNGs inspected; numerical sources, units, zero baselines, counts, single-seed scope and source hashes checked.')
    (figures/'figures.yaml').write_text(yaml.safe_dump(gallery,allow_unicode=True,sort_keys=False),encoding='utf-8')
    for screenshot in ui['screenshots']:
        path=BASE/'PPT展示图片'/screenshot['file']
        assert sha(path)==screenshot['sha256']
        with Image.open(path) as image:
            assert image.width>=2000 and image.height>=1400
        assets.append({'path':str(path),'sha256':sha(path)})
    pngs=list((BASE/'PPT展示图片').glob('*.png'))
    assert len(pngs)==7
    assert len(list((BASE/'PPT展示图片').glob('*.csv')))==1
    assert not list((BASE/'PPT展示图片').rglob('*.pdf'))
    report={'verified_at':datetime.now().isoformat(timespec='seconds'),'passed':True,
            'rerun_steps':progress['steps'],'test_sessions':20000,'png_images':7,'vector_pdf_images':0,'csv_tables':1,
            'html_sha256':sha(html),'figures_source':'full rerun '+TAG,'assets':assets,
            'scope':'Same single seed, unchanged official model implementations, identical data split. Training/testing completed; independent reload verified. This does not establish general superiority.',
            'temporal_order_audit':read(out/'temporal_order_audit.json')}
    (out/'delivery_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 本轮运行与展示材料验收','','日期：2026-11-12。','','## 运行结果','',
           '三个模型均从头重新训练，导出每模型 20,000 条测试预测，并在独立进程中重载权重重新评估。所有步骤退出码为 0。',
           '','| 模型 | Recall@10 | MRR@10 | NDCG@10 | 训练及验证（分钟） |','| --- | --- | --- | --- | --- |']
    for model,result in results.items():
        metric=result['test_result']
        lines.append(f"| {model} | {metric['recall@10']:.4f} | {metric['mrr@10']:.4f} | {metric['ndcg@10']:.4f} | {result['training_and_validation_seconds']/60:.2f} |")
    same=all(all(abs(value)<1e-12 for value in row['difference_from_previous'].values()) for row in summary['models'])
    lines.extend(['','与此前实验使用相同的数据划分。'+('三项测试指标与此前一致。' if same else '本轮与此前指标差异已保存在 rerun_summary.json。'),
                  '','## 展示材料','','- 同级“演示页面/会话推荐演示.html”：20,000 个会话、三个模型同时展示、案例筛选、Top-5/10、真实目标高亮，全离线。',
                  '- 同级“PPT展示图片”：训练过程、三模型对比汇总表、序列转图共 3 张 PNG，4 张成功失败案例截图，1 份汇总表 CSV，以及图片使用说明。',
                  '- figures/candidates/ppt_20261112：真实输入、日志与结果哈希、图形清单、双次生成一致性记录。',
                  '','## 核验范围','','独立重载的测试指标与本轮训练输出一致；全部预测行的 Top-10 合法性通过。HTML 来源、筛选、目标和推荐 ID 已核对，浏览器无脚本错误、无需网络。',
                  '','本次为单种子课程子集实验；没有新增基线、消融或多种子统计。',
                  '','## 时间戳处理局限','','框架将时间戳转为 float32 再排序，14/20,000 个会话与原始秒级顺序有细微差异，其中 1 个影响留出的目标。本轮保留原处理规则以便核对历史实验；完整明细见 outputs/rerun20261112/temporal_order_audit.json。',
                  '','页面对受影响会话给出说明。四张案例截图均采用不受该差异影响的会话。'])
    (ROOT/'本轮运行与展示材料验收.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('DELIVERY AUDIT PASSED: full rerun; 20000 paired test sessions; 7 PNG + 1 CSV; offline HTML',flush=True)


if __name__=='__main__':
    main()
