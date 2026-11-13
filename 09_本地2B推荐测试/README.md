# qwen3.5-2B

正式方法名称：**在线 LLM 动态策略生成（Online LLM Policy Generation）**。使用固定的预训练权重，每次推荐通过本地LLM生成评分系数；未针对本任务微调权重。

用户于2026-11-12确定保留本方案，展示名称统一为`qwen3.5-2B`。每次推荐都向本地2B输入会话、生成系数并接收返回结果。原始技术ID为`qwen3.5:2b-q4_K_M`；无教师、有教师两组记录均保留。

用户于2026-11-12要求部署本地2B模型并运行推荐测试。本实验独立于课程论文，沿用云端八组的同30条会话及两套提示词。

| 文件或目录 | 内容 |
|---|---|
| `run.py` | 预热、逐条调用本地模型、保存真实响应、全商品推荐及导出 |
| `verify.py` | 从原始响应重算Top10和指标，检查样本、提示词与结果对应 |
| `frozen_before_requests.json` | 模型、参数、样本和提示词指纹 |
| `deployment` | Ollama版本、模型元数据、预热和显卡实际加载信息 |
| `requests` | 每次真实调用的完整响应与耗时 |
| `results` | 逐条推荐、汇总和检查结果 |

运行程序位于 `D:\CodexTools\ollama`，模型位于 `D:\Models\Ollama`。程序、模型、临时目录、独立用户目录均放在D盘；未设置开机自启或修改系统PATH。

启动服务：

```powershell
& 'D:\CodexTools\ollama\启动本地服务.ps1'
```

以下为历史运行或断点续跑命令。2026-11-12用户要求删除OpenCode四模型结果后，原脚本的混合导出及13行CSV核验依赖已撤下；执行前需适配本地结果导出，不直接整段重跑。现有本地请求、预测和指标保持原样，当前性能页面由11目录构建。

```powershell
& 'D:\Envs\recbole-gnn-py39\Scripts\python.exe' 'D:\Projects\算法分析与理论课程\09_本地2B推荐测试\run.py'
& 'D:\Envs\recbole-gnn-py39\Scripts\python.exe' 'D:\Projects\算法分析与理论课程\09_本地2B推荐测试\verify.py'
```

测试采用本地常驻模型、单路串行、2048上下文、温度0、关闭思考输出；不使用额外JSON语法约束。教师为既有训练侧原型加入提示词，模型权重保持下载版本。

本地结果沿用原云端实验的同30条会话。云端8组结果已于2026-11-12删除，公共会话和提示词继续保留；当前本地结果用于30条会话的小样本性能比较。

官方来源：[Windows独立CLI](https://docs.ollama.com/windows)、[本地Chat API](https://docs.ollama.com/api/chat)、[Qwen3.5-2B Q4模型](https://ollama.com/library/qwen3.5:2b-q4_K_M)。
