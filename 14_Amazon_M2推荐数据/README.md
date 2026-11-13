# Amazon-M2推荐数据

后续算法测试：[ES站点统一实测结果](../15_Amazon_M2算法对比/实测结果.md)。原始CSV保留不变。

- [数据说明](数据说明.md)：真实规模、字段、任务和实际商品标题样例。
- [下载清单](下载清单.json)：来源、ZIP完整性与SHA-256。
- [逐行核查](数据核查.json)：按站点统计、属性缺失和商品匹配。
- [真实会话样例](真实会话样例.json)：原始历史与下一商品，映射可读标题。

原始文件在 `data/raw/`：保留完整下载ZIP，只解压 `products_train.csv` 和 `sessions_train.csv`。本次未抽样、未训练模型。

复核已有CSV：

```powershell
& 'D:\Envs\recbole-gnn-py39\Scripts\python.exe' 'D:\Projects\算法分析与理论课程\14_Amazon_M2推荐数据\inspect_data.py'
```

数据ZIP约418MB，两个CSV合计约848MB，下载包与两个CSV合计约1.27GB。2026-11-13已将约394MB的下载临时分片移入系统回收站，原始ZIP和两个CSV继续保留。清理记录见 `../90_管理记录/清理记录_20261113.json`。
