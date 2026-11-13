# 四组灰关联单因素试验

执行范围：Diginetica已有20k会话子集与Amazon-M2 ES已有划分；原版GRA、四组独立变体、三个已有GNN检查点。原始数据、旧实验、模型权重保持不变。

## 运行与复核

在`03_代码与实验`目录运行，两个`all`命令依次执行，保证最终响应计时没有其他实验并行占用CPU：

```powershell
& 'D:\Envs\recbole-gnn-py39\Scripts\python.exe' -u experiments/grey_four_axes.py --dataset diginetica --stage all
& 'D:\Envs\recbole-gnn-py39\Scripts\python.exe' -u experiments/grey_four_axes.py --dataset amazon_m2 --stage all
& 'D:\Envs\recbole-gnn-py39\Scripts\python.exe' -X utf8 experiments/verify_grey_four_axes.py
```

验证阶段每500个请求保存一次进度；完成后重跑会从已保存的精确验证名次重新核算指标和选参。源文件、数据、原GRA、GNN检查点或方案改变时，指纹检查会拒绝静默混用旧结果。需要新一轮研究时另建输出版本，而非删除冻结记录。

本轮开发时修正过验证名次为int16时NDCG的对数运算精度：统一转为int64后以float64汇总。验证名次数组未改变，四组选择未改变；之后的完整测试、GNN重新推理与计时按修正实现执行。原开发元数据存于D盘CodexOutputs的`development_precision_correction`，不作为正式结果。

计时复核还修正了Amazon GNN适配层的目录缓存：属性融合缓存需在最后一次`eval()`之后编译，否则每次请求会重复融合整个商品目录。正式结果按恢复缓存后的实现重新完成测试和计时，数值状态计入预计算目录向量。先前未缓存的开发计时已保存在`development_cache_correction`，不作为算法对比结果。

## 文件

- `两数据集实测汇总.md`、`两数据集完整对比.csv`：独立核验通过后生成的最终16方法结果。
- `independent_audit.json`：两数据集全量预测、指标、验证选参和计时的另外一份核对记录。
- `gnn_source_snapshot.json`：实际导入的模型、图适配与公共层源码哈希；Amazon源码与原训练记录匹配。
- `diginetica/`、`amazon_m2/`：各自的冻结输入、17配置完整验证记录、四组参数、8方法完整预测、响应原始记录、分组结果和核验文件。

完成判据：两个数据集各自`verification.json`为passed，且最终`independent_audit.json`同时通过两数据集。程序启动成功或小批样本通过不等于完成全量测试。

## 结果口径

每组固定5档参数，只以本轮完整验证MRR为主要选参依据，Recall和NDCG辅助处理并列。保留恢复原版的参数选项，若选中该选项则如实报告没有获得有利改动。

响应计时为CPU4线程、batch=1、相同300请求、预热30次、3轮各900次；包含特征/图构建、全目录评分、过滤和Top10，排除模型加载、静态目录预计算和写盘。GRA与GNN数值状态MB指实际数组/张量及少量配置参数，不等同进程运行内存或模型文件下载体积。

Dig为四指标、允许再访；Amazon为八指标、过滤历史商品，三个GNN使用已有属性融合。两个数据集各自内部比较，不将绝对命中率差异解释为数据复杂度的直接效应。两套测试集此前均已查看，结果属于既有划分上的扩展性探索。
