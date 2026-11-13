# 离线qwen3.5-2B

正式方法名称：**离线 LLM 策略预计算（Offline LLM Policy Precomputation）**。使用固定的本地LLM提前为有限会话状态选择候选策略并缓存；在线查表后重新计算商品排名。该方案未训练学生模型，未进行知识蒸馏或参数微调。

用户于2026-11-12确定：策略预计算与在线查表方案正式显示为`离线qwen3.5-2B`。原09方案继续保留，显示为`qwen3.5-2B`；本目录的单Token同步优化保留为过程方案。

使用同一份Qwen3.5-2B Q4权重，提供同步单Token判别和离线策略预计算两条路径。课程论文和原09实验保持原位。

| 入口 | 作用 |
|---|---|
| `predict.py` | 可直接使用的推荐器，支持cache与sync |
| `启动优化服务.ps1` | 启动CUDA本地llama-server，仅监听127.0.0.1:11435 |
| `optimize.py` | 第一轮验证集选择与真实同步调用 |
| `refine.py` | 短输入和上下文缓存调整，验证集选定最终方案 |
| `cache_compare.py` | 本地2B实际生成12类状态的策略选择表，评估在线查表路径 |
| `baselines.py` | 原灰关联、固定教师策略和三种GNN的相同样本实测 |
| `report_verify.py` | 独立重算同步预测和全部20000条缓存结果，导出CSV与说明 |
| `requests` | 每次实际本地2B响应、提示词、配置、耗时和Top10 |
| `results` | 冻结选择、策略表、指标、逐条耗时和核对记录 |

缓存模式：先由本地2B预计算状态→策略，在线按会话重新计算全部商品Top10。在线命中不执行2B。

```powershell
& 'D:\Envs\recbole-gnn-py39\Scripts\python.exe' 'D:\Projects\算法分析与理论课程\10_本地2B低延迟优化\predict.py' --mode cache --items 33890 4926 4759 51340 8091 82732
```

同步模式：每次真实运行完整2B模型输出一个策略标签。

```powershell
& 'D:\Projects\算法分析与理论课程\10_本地2B低延迟优化\启动优化服务.ps1' -BatchSize 32
& 'D:\Envs\recbole-gnn-py39\Scripts\python.exe' 'D:\Projects\算法分析与理论课程\10_本地2B低延迟优化\predict.py' --mode sync --items 33890 4926 4759 51340 8091 82732
```

模型和GPU依赖复用D:\CodexTools\ollama，未新增模型下载；启动脚本只修改当前启动进程的环境。使用同步服务前释放Ollama中同一份2B权重，避免4GB显存中重复加载。脚本已处理这一点。

交付检查后已停止优化服务以释放显存；缓存模式可直接使用，同步模式先运行启动脚本。同步延迟表是连续请求下的预热测量；最后一次空闲后的接口检查耗时约355毫秒，后续实际部署需将首请求与持续服务延迟分开。该次响应与缓存模式的推荐名单逐项相同，记录在results/cli_check.json。

交付结果位于 `08扩展对比\本地2B低延迟优化`。同步方案的几十毫秒与缓存方案的约0.6毫秒分别计时，加载和离线预计算不计入在线延迟。

首次直接启动未显式加载CUDA，CPU诊断输出保存在pilot_cpu.json；修正为实际CUDA DLL路径后，25/25层加载到GPU。最终结果全部来自修正后的CUDA运行；原始记录保留以便追溯。

官方接口来源：[llama.cpp本地服务](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)、[Ollama后端加载实现](https://github.com/ollama/ollama/blob/main/llm/llama_server.go)。
