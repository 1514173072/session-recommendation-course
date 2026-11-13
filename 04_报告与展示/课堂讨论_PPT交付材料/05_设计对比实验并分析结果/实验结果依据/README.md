## 1. 文件及用途

三个模型分别存放在 `SRGNN`、`GCSAN`、`GCEGNN` 子文件夹中。模型结果、日志与预测对应完整项目的 `rerun20261112`、seed 42 实验，路径已改为相对路径，实验数值和完整预测保留原值。

| 文件 | 内容 | 汇报用途 |
| --- | --- | --- |
| 各模型目录中的 `result.json` | 验证与测试指标、样本数、参数量、耗时和权重信息 | 核对总体结果及效果与成本的比较 |
| 各模型目录中的 `train.log` | 每轮训练损失、验证结果、选模与停止记录 | 解释训练过程及最佳权重的选择 |
| 各模型目录中的 `模型名.jsonl` | 每模型 20,000 条完整测试预测，一行对应一个测试会话 | 核对真实目标、推荐列表、命中排名，以及按会话长度和商品流行度分组的依据 |
| [data_manifest.json](data_manifest.json) | 数据来源、筛选条件、抽样种子、样本规模与数据哈希 | 对应实验设计中的数据准备过程 |
| [figure_inputs.json](figure_inputs.json) | 指标、训练曲线、分组命中数、分组样本数及来源哈希 | 核对汇总表和分组分析的计算输入 |
| [verification_summary-rerun20261112.json](verification_summary-rerun20261112.json) | 独立进程重载指标、预测行数及 Top-10 合法性检查结果 | 说明保存的权重可重载，预测输出通过已有核验 |

| 模型 | 总体结果 | 训练日志 | 完整测试预测 |
| --- | --- | --- | --- |
| SR-GNN | [result.json](SRGNN/result.json) | [train.log](SRGNN/train.log) | [SRGNN.jsonl](SRGNN/SRGNN.jsonl) |
| GC-SAN | [result.json](GCSAN/result.json) | [train.log](GCSAN/train.log) | [GCSAN.jsonl](GCSAN/GCSAN.jsonl) |
| GCE-GNN | [result.json](GCEGNN/result.json) | [train.log](GCEGNN/train.log) | [GCEGNN.jsonl](GCEGNN/GCEGNN.jsonl) |

记录中的相对路径以“完整项目”目录为基准；上一级材料的 Markdown 链接仍相对于各自所在文件夹。

## 2. 建议阅读顺序

1. 先阅读上一级 [设计对比实验并分析结果.md](../设计对比实验并分析结果.md)，了解实验设置与结论。
2. 查看 `data_manifest.json`，确认抽样及数据规模；训练集、验证集和测试集的前缀样本数见各模型的 `result.json`。
3. 将各模型 `result.json` 的 `test_result` 与 [三模型对比汇总表 CSV](../02_三模型对比汇总表.csv) 中的推荐指标对应。
4. 查看 `train.log`，结合已有训练过程图解释验证集选模；查看 `figure_inputs.json` 中的 `length_groups`、`popularity_groups`，理解分组结果。
5. 需要展示单条案例时，按 `session_id` 查找各模型的 JSONL 记录，对照历史、真实目标和 Top-10 推荐。
6. 查看 `verification_summary-rerun20261112.json`，了解此前已完成的独立重载与预测检查范围。

## 3. 结果字段与分析对应关系

### 总体指标与分组统计

| 位置或字段 | 含义 | 对应分析 |
| --- | --- | --- |
| `result.json → test_result` | 最佳验证权重在测试集上的评价指标 | 总体推荐效果 |
| `result.json → best_valid_result` | 选中权重对应的验证集结果 | 选模说明，与最终测试结果分别使用 |
| `figure_inputs.json → length_groups` | 按测试历史长度划分的组及统计量 | 不同会话长度的 Recall@10 比较 |
| `figure_inputs.json → popularity_groups` | 按真实目标的训练频次排名划分的组及统计量 | 不同目标商品流行度的 Recall@10 比较 |
| 分组统计中的 `hit_counts`、`sample_counts`、`recall10` | 组内命中数、样本数及命中比例 | 核对分组分母与命中率；百分数为命中比例乘以 100 |

总体表沿用 `result.json` 中框架输出的指标精度。根据 JSONL 的排名重新计算时，未经舍入的值可能存在末位差异，应按相同精度比较。

### 单条预测记录

| JSONL 字段 | 含义 |
| --- | --- |
| `session_id` | 测试会话 ID，可用于三模型之间对应同一条样本 |
| `history` | 模型实际使用的历史商品 ID 序列，导出时按时间正序展示 |
| `target` | 真实下一点击商品 ID |
| `target_rank` | 真实目标在 Top-10 中的排名；0 表示没有进入前 10 位 |
| `history_len` | 历史点击次数，包含重复点击，用于长度分组 |
| `target_popularity_rank` | 真实目标按训练样本目标出现次数得到的频次排名，用于流行度分组 |
| `repeat_count` | 历史点击次数减去不同商品数量，用于描述重复浏览 |
| `recommendations` | 按得分降序排列的 10 个商品，含排名、商品 ID 和模型得分 |

推荐 `score` 是模型打分，不是点击概率；不同模型的原始分数不宜直接比较。实验效果应比较相同测试样本上的命中与排名指标。

核验记录说明权重加载、指标一致性及预测结构通过了已有检查，不等同于多种子显著性检验或官方排行榜结果复现。当前材料对应固定课程子集上的一次种子设置。
