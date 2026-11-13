# 交互式会话推荐实验台

这是一个实时交互页面，不是静态汇报页。选择或编辑浏览序列后，直接加载正式 checkpoint 推理，显示三个模型的 Top-10、得分、真实下一商品命中排名。

## 启动

如果当前环境还没有 Streamlit：

```powershell
& 'D:\Envs\recbole-gnn-py39\Scripts\python.exe' -m pip install streamlit==1.40.0
```

在项目根目录执行：

```powershell
& 'D:\Envs\recbole-gnn-py39\Scripts\python.exe' -m streamlit run app/main.py --server.port 8503 --server.address 127.0.0.1 --server.headless true
```

打开：**http://127.0.0.1:8503**

首次点击“运行当前模型”时会加载模型，约需 1–2 分钟；之后同一模型会缓存。

## 视觉皮肤

页面内置三套强风格皮肤，可在左侧 `SKIN` 切换，也可用 URL 直接指定：

- `http://127.0.0.1:8503/?skin=ACID`：酸性黑 / 荧光绿（默认）
- `http://127.0.0.1:8503/?skin=BLUEPRINT`：深蓝 / 青色工程图
- `http://127.0.0.1:8503/?skin=PAPER`：纸本 / 橙色印刷

## 页面能做什么

- **推荐工作台**：编辑商品 ID 序列，运行 SR-GNN / GC-SAN / GCE-GNN，实时看 Top-10 条形图、得分和命中排名。
- **三模型同屏对比**：同一输入分别运行三个模型，查看目标排名、首位推荐和分模型 Top-10。
- **测试集检索**：按命中/未命中、历史长度、重复点击筛选 20,000 个测试会话；查看分组 Recall，并载入工作台。
- **模型对比**：正式指标表、指标柱状图、验证曲线、按历史长度分组的 Recall。
- **会话图 / 全局邻居**：局部会话图与 GCE-GNN 的全局邻居。

## 边界

- 单种子、单次实验；没有多种子、简单基线和 GCE 消融。
- 商品 ID 为 Diginetica 匿名编码；得分为模型分数，不是购买概率。
- 只有正式子集词表中的商品 ID 可以输入；不在词表中的 ID 会提示错误。
- 自定义编辑序列后，原测试会话的真实标签会被清除，避免误用。

## 文件

- `app/main.py`：Streamlit 页面
- `app/predictor.py`：真实推理适配层，已与离线 `predictions/*.jsonl` 校验一致
- `03_代码与实验/outputs/formal/...`：正式 checkpoint、指标和预测
- `web/index.html`：离线静态结果总览（不使用服务器时的备选）
