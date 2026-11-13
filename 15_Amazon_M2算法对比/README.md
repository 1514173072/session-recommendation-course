# Amazon-M2算法对比

使用完整ES站点：71,191条训练会话、8,886条验证会话、8,970条测试会话、42,503件候选商品。训练会话内部前缀增强后共260,486个训练样本。

- [实验方案](实验方案.md)
- `run/manifest.json`：数据规模、划分和原始文件指纹。
- `run/results/*_progress.json`：各GNN的训练进度与验证指标。
- `run/logs/`：运行日志。

已完成统一测试，9种主方法及消融共80,730条预测核查通过：

- [实测结果](实测结果.md)
- [算法对比CSV](run/results/算法对比.csv)
- [核查记录](run/results/verification.json)
- [逐条预测](run/predictions)

方法包括属性增强灰关联、仅改差异曲线的灰关联、原RecBole-GNN三种图模型，以及本地Qwen3.5-2B离线策略预计算。另保留热门、固定策略和原始Qwen选策作为消融。

复跑入口：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File 'D:\Projects\算法分析与理论课程\15_Amazon_M2算法对比\run_all.ps1'
```

正在训练时不要再次启动相同模型。入口可在已完成模型与冻结结果存在时继续后续阶段；修改数据或算法配置应建立新运行目录，避免覆盖既有实验。

训练可并行，统一单请求推理计时在所有训练和本地LLM服务退出后进行。离线Qwen不更新模型权重，线上请求不执行2B；实际请求和模型响应另保留。
