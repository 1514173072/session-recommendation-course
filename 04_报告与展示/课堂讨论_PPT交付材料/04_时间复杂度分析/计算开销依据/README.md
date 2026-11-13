## 1. 文件及用途

三个模型分别存放在 `SRGNN`、`GCSAN`、`GCEGNN` 子文件夹中，均取自完整项目的 `rerun20261112`、seed 42 运行记录，路径已改为相对路径，实验数值保留原值。

| 文件 | 内容 | 汇报用途 |
| --- | --- | --- |
| 各模型目录中的 `effective_config.txt` | 实际运行时的完整配置，包括表示维度、序列长度、批量大小、模型结构和设备设置 | 确认理论复杂度分析与实测耗时所采用的条件 |
| 各模型目录中的 `result.json` | 训练及验证耗时、测试评价耗时、参数量、样本数和评价指标 | 核对实际耗时对比表中的训练与测试数据 |
| 各模型目录中的 `模型名-latency.json` | `ms_per_session`，单位为毫秒/会话 | 核对实际耗时对比表中的批量前向均摊值 |
| [export_predictions.py](export_predictions.py) | 模型加载、预测导出、前向计时、填充商品屏蔽和 Top-10 提取代码 | 说明推理耗时的测量范围 |

每个模型对应的文件：

| 模型 | 实际配置 | 训练与测试记录 | 前向计时记录 |
| --- | --- | --- | --- |
| SR-GNN | [effective_config.txt](SRGNN/effective_config.txt) | [result.json](SRGNN/result.json) | [SRGNN-latency.json](SRGNN/SRGNN-latency.json) |
| GC-SAN | [effective_config.txt](GCSAN/effective_config.txt) | [result.json](GCSAN/result.json) | [GCSAN-latency.json](GCSAN/GCSAN-latency.json) |
| GCE-GNN | [effective_config.txt](GCEGNN/effective_config.txt) | [result.json](GCEGNN/result.json) | [GCEGNN-latency.json](GCEGNN/GCEGNN-latency.json) |

记录中的相对路径以“完整项目”目录为基准；上一级材料的 Markdown 链接仍相对于各自所在文件夹。

## 2. 建议阅读顺序

1. 阅读上一级 [时间复杂度分析.md](../时间复杂度分析.md)，了解符号、推导和三模型对比。
2. 查看各模型的 `effective_config.txt`，确认表示维度、最大序列长度，以及 GCE-GNN 邻居数、GC-SAN 注意力层数等设置。
3. 对照 [三模型实际耗时对比表.csv](../三模型实际耗时对比表.csv)，查阅各模型的 `result.json` 和前向计时记录。
4. 阅读 `export_predictions.py` 的 `main()`，重点查看 `if latency is None` 分支及其后的 `torch.topk` 调用。

## 3. 时间指标与代码对应关系

| 时间指标 | 原始字段 | 测量范围 |
| --- | --- | --- |
| 训练及验证（秒） | `result.json` 的 `training_and_validation_seconds` | `run_formal.py` 中 `trainer.fit()` 的总耗时，包括每轮验证和选模，采用实际停止轮数 |
| 测试集评价（秒） | `result.json` 的 `test_seconds` | `trainer.evaluate(test, load_best_model=True)` 的调用耗时，包含最佳权重加载与完整测试评价 |
| 批量前向均摊（毫秒/会话） | 前向计时记录的 `ms_per_session` | 首个测试批次预热一次后，再测量一次 `full_sort_predict()`，将该批次耗时除以批次样本数 |

前向均摊值不包含模型初始化与加载、数据准备、Top-10 提取、预测文件写入和页面展示，也不是逐条在线请求的响应时间。它来自一次批量前向测量，适合作为本轮运行的参考记录。

三模型均采用 CPU、4 线程、seed 42，测试样本数为 20,000。理论复杂度说明规模增长规律，实测时间描述这组配置下的执行成本，二者结合讲解。

训练计时代码可在第三部分已有的 [run_formal.py](../../03_算法设计思想与伪代码描述/实现代码/run_formal.py) 中查阅。这些副本用于阅读和说明，执行仍使用完整项目。
