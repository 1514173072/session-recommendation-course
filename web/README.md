# 可视化网页

> 这是离线静态结果总览，适合不启动服务器时快速查看。需要编辑会话、实时推理和三模型同屏对比，请使用交互实验台 `app/main.py`（http://127.0.0.1:8503）。

双击 `index.html` 即可打开。页面离线加载，不需要服务器或网络。

- `index.html`：页面结构
- `style.css`：版式与配色
- `app.js`：交互与 ECharts 图表
- `data.js`：由 `03_代码与实验/experiments/export_web_data.py` 生成的数据
- `assets/echarts.min.js`：本地 ECharts

数据来源：

- `03_代码与实验/outputs/formal/comparison.csv`
- `03_代码与实验/outputs/formal/<模型>/seed42/result.json`
- `03_代码与实验/outputs/formal/<模型>/seed42/train.log`
- `03_代码与实验/outputs/formal/predictions/<模型>.jsonl`

重新生成数据：

```powershell
& 'D:\Envs\recbole-gnn-py39\Scripts\python.exe' 03_代码与实验/experiments/export_web_data.py
```

页面内容：

1. 实验全景：数据漏斗与 KPI
2. 三模型对照：效果 vs 训练时间、三项指标
3. 学习过程：验证 MRR、训练损失
4. 会话回放：18 个真实测试会话，可切换三个模型
5. 会话图：局部会话图 / GCE 全局邻居
6. 失败与边界：真实失败案例与实验边界

商品 ID 为 Diginetica 匿名编码；页面展示的是真实行为与真实排名，不含商品名称或图片。
