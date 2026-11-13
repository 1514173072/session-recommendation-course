# 基于图神经网络的电商会话推荐算法研究与实现

已下载官方 RecBole-GNN，完成 SRGNN、GCSAN、GCEGNN 的首次真实数据运行验证，以及 20,000 个真实会话上的正式技术闭环。

- 正式闭环与复跑命令：[正式闭环验收](正式闭环验收.md)
- 交互实验台：[../app/README.md](../app/README.md)，运行地址 http://127.0.0.1:8503
- 首次运行验收：[首次运行验收](首次运行验收.md)
- 灰色关联推荐对照实验：[实测比较与复跑说明](outputs/grey_relation/灰色关联推荐对比结果.md)（2026-11-11，沿用同一20k划分，三个GNN使用已有逐会话预测重新核算）
- 灰色关联改进：[质量、单请求响应与复杂度对比](outputs/grey_improved/灰色关联改进与复杂度对比.md)（2026-11-11，验证集选权重；Recall@10=37.190%，同机平均响应0.455ms）
- 灰色关联单处改动：[18类79配置逐一比较](outputs/grey_single_change/单处改动对比与推荐.md)（2026-11-11；建议仅将最终算术聚合改为调和聚合，Recall@10=34.880%，无新增调参，计算开销基本不变）
- 灰色关联系数内部改动：[一行差异压缩与实测](outputs/grey_coefficient_change/灰色关联系数内部改动与实测.md)（2026-11-11；保留原等权算术平均，只在关联系数函数中加入差异压缩，Recall@10=35.540%，平均0.709ms）
- 静态结果总览：[../web/index.html](../web/index.html)
- 首次 smoke 结果：[当前可展示结果](当前可展示结果.md)

## 2026-11-02 正式闭环

| 模型 | Recall@10 | MRR@10 | NDCG@10 | 训练+验证/秒 |
|---|---:|---:|---:|---:|
| SR-GNN | 0.2818 | 0.1340 | 0.1688 | 416.7 |
| GC-SAN | 0.2952 | 0.1387 | 0.1756 | 603.7 |
| GCE-GNN | 0.3314 | 0.1590 | 0.1998 | 731.2 |

数据为完整 Diginetica 先做 item 5-core、再抽 20,000 个 5–20 次点击会话；排除 smoke；单种子 seed=42。三模型均完成训练、测试、保存、独立进程重载和 Top-10 推理，重载后指标完全一致。详细边界、命令和产物见 [正式闭环验收](正式闭环验收.md)。

## 当前交付

- `RecBole-GNN/`：官方 Git 源码，未修改模型实现。
- `data/diginetica-not-merged/`：完整官方原子交互文件，993,483条记录。
- `data/diginetica-smoke/`：固定种子抽取的2,000个真实会话，15,813条交互，9,109个商品。
- `data/diginetica-formal/`：正式 20,000 会话、32,968 商品、157,273 交互。
- `outputs/diginetica-smoke/<模型>/`：smoke 权重、配置、测试指标、重载核验与推荐示例。
- `outputs/formal/<模型>/seed42/`：正式权重、配置、测试指标、训练日志与重载结果。
- `outputs/formal/predictions/<模型>.jsonl`：20,000 条测试会话的 Top-10 预测。
- `outputs/formal/comparison.csv`、`outputs/formal/verification_summary.json`：正式汇总与验收状态。
- `experiments/`：正式数据准备、训练、预测导出、重载校验与网页数据生成脚本。
- `../app/`：交互式推荐实验台（`main.py` + 实时推理 `predictor.py`）。
- `../web/`：离线静态结果总览（`index.html` + 本地 ECharts + `data.js`）。
- `outputs/comparison.csv`、`outputs/verification_summary.json`：smoke 汇总与验收状态。
- `outputs/console/`、`log/`、`log_tensorboard/`：控制台与训练日志。
- `requirements-lock.txt`：当前环境精确版本；环境路径 `D:/Envs/recbole-gnn-py39`。

首次 smoke 每模型 CPU 训练 3 轮，统一数据划分，验证链路为“训练→验证选权重→独立测试→保存重载→Top-10推荐”。正式闭环使用 20k 会话、20 轮上限、seed=42，详见 [正式闭环验收](正式闭环验收.md)。

## 直接复跑

在本目录打开 PowerShell：

```powershell
powershell -ExecutionPolicy Bypass -File ./run_smoke.ps1
```

只加载已有权重并重新评价/推荐：

```powershell
powershell -ExecutionPolicy Bypass -File ./run_smoke.ps1 -InferenceOnly
```

只运行一个模型：

```powershell
powershell -ExecutionPolicy Bypass -File ./run_smoke.ps1 -Models SRGNN -Epochs 3
```

单模型Python入口也可用：

```powershell
& 'D:\Envs\recbole-gnn-py39\Scripts\python.exe' ./scripts/run_model.py --model GCEGNN --inference-only
```

## 正式闭环复跑

在本目录打开 PowerShell，按顺序执行（无“一键脚本”）：

```powershell
$py = 'D:\Envs\recbole-gnn-py39\Scripts\python.exe'

& $py experiments/prepare_formal.py --sessions 20000 --seed 2026
& $py experiments/run_formal.py --model SRGNN  --epochs 20 --seed 42
& $py experiments/run_formal.py --model GCSAN  --epochs 20 --seed 42
& $py experiments/run_formal.py --model GCEGNN --epochs 20 --seed 42
& $py experiments/export_predictions.py --model SRGNN  --seed 42
& $py experiments/export_predictions.py --model GCSAN  --seed 42
& $py experiments/export_predictions.py --model GCEGNN --seed 42
& $py experiments/verify_formal.py
& $py experiments/export_web_data.py
```

结果与边界见 [正式闭环验收](正式闭环验收.md)。

## 数据与环境重建

原数据和子集均已准备好。需要重新生成子集时执行：

```powershell
& 'D:\Envs\recbole-gnn-py39\Scripts\python.exe' ./scripts/prepare_data.py
```

数据下载先尝试RecBole自带S3地址，403时切换到官方RecSysDatasets链接的公开Google Drive备份。原始压缩包、来源与SHA256在`cache/`和`outputs/data_manifest.json`，不依赖登录。

环境重建入口：`install_environment.ps1`，使用已有Python 3.9创建D盘独立venv。PyTorch与PyG扩展从官方地址获取；常规PyPI依赖使用阿里云镜像。要严格重建当前依赖，可在同版Python的隔离环境中安装`requirements-lock.txt`并添加PyG轮子索引`https://data.pyg.org/whl/torch-2.1.0+cpu.html`。

## 实验边界

首次 smoke 采用 2,000 个长度 5 至 20 的真实会话，训练/验证/测试例数为 9,813/2,000/2,000。正式闭环采用完整数据先做 item 5-core、再抽 20,000 个会话，训练/验证/测试例数为 97,273/20,000/20,000；单种子 seed=42、20 轮上限、早停 5。三模型测试指标与重载校验见 [正式闭环验收](正式闭环验收.md)。

正式闭环未做多种子、最后点击/热门基线、GCE 去全局分支消融；因此不宣称统计显著，也不把三模型差异单独归因于全局分支。

推荐输出为匿名商品ID和模型得分，不是商品名称或购买概率。GCE-GNN按官方方式逆序建图，输出示例已还原为浏览顺序。全局图使用训练集构建；三个模型的数据划分指纹一致。

未完成的CUDA临时下载仍在cache/tmp中；自动审批策略拦截了删除操作，未提供更具体原因。当前CPU环境及验收不受影响。
