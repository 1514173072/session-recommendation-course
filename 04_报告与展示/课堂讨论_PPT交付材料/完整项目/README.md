# 电商会话推荐完整项目

包含 RecBole-GNN 源码、Diginetica 数据、实验配置、训练脚本、三个模型的权重、日志与测试预测。

## 演示与 PPT 图片

直接打开同级“演示页面/会话推荐演示.html”，选择测试会话，同时查看真实点击和三个模型的推荐结果。页面可独立离线使用。

同级“PPT展示图片”保存训练曲线、模型对比、分组分析、会话图和演示截图。

## 本轮重新运行

在此目录执行：

```powershell
& '.venv\Scripts\python.exe' experiments/rerun_for_delivery.py
```

脚本依次重新训练 SRGNN、GCSAN、GCEGNN，导出全部测试预测，再在独立进程中重载权重验证。采用原配置、20,000 会话、最多 20 轮、seed 42、CPU、4 个线程；输出使用 `rerun20261112` 标记，保留此前正式实验。

运行记录：`outputs/rerun20261112/`。模型权重、训练日志：`outputs/formal/模型名-rerun20261112/seed42/`。完整测试预测：`outputs/formal/predictions-rerun20261112/`。

图表生成脚本：`experiments/export_ppt_figures.py`；输入快照和原图：`figures/candidates/ppt_20261112/`。

## 主要文件

| 路径 | 内容 |
| --- | --- |
| `RecBole-GNN/` | GitHub 原项目与模型实现 |
| `data/` | 完整交互数据及实验子集 |
| `configs/` | 训练和评估配置 |
| `experiments/`、`scripts/` | 数据准备、训练、预测、验证及图表生成 |
| `outputs/` | 运行记录、模型权重、日志、指标与预测 |
| `figures/candidates/` | 本项目图表、数值输入与复现记录 |
| `requirements-*.txt` | 运行环境依赖 |

当前电脑使用 `.venv\Scripts\python.exe`。迁移运行环境时需要 Python 3.9，按本机路径调整 `install_environment.ps1` 后安装。现有实验为课程子集上的单种子实验。