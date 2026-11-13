# 云端四模型：实验结果已清理

2026-11-12按用户要求删除四个OpenCode模型的八组测试结果、请求输出、试测记录、日志和图表，以及08目录中的云端对比交付文件。混合CSV中的8行云端结果也已移除。

`cases.json`和`prompts`仍由本地Qwen实验引用，继续保留。配置及代码作为历史实现留存，不自动重新运行云端实验。本地Qwen、离线Qwen、灰关联及GNN结果保持原样。

以下为清理前的历史实验说明，所列云端结果文件已不存在。

|项目|内容|
|---|---|
|模型|MiMo-V2.6-Flash、Qwen3.8 Flash、GLM-5.3-Flash、DeepSeek V4.1 Flash|
|分组|各模型无教师/有教师两组，共八组，每组相同30个真实会话|
|有教师含义|提示词加入训练侧教师关联指导；本轮不修改云端模型权重|
|推荐方式|模型输出四权重和六交互系数，本地对全商品评分|
|接口|本机Harness、OpenCode Go、已有认证引用；无工具评分请求|
|运行|prepare.py → run.py → export.py → verify.py|
|制图核对|export.py之后查看图片，核对数值并在figures/export_audit.json记录visual_checked，再运行verify.py|
|环境|D:\Envs\recbole-gnn-py39\Scripts\python.exe；制图使用C:\anacode\python.exe|
|续跑|再次运行run.py，复用已完成请求和结果；不重新运行prepare.py以免改写冻结设置|
|交付|08扩展对比/云端八组对比，含表格图片、CSV、说明|
|原始证据|requests中的NDJSON/元数据、prompts、frozen_before_requests.json|
|判定记录|数值有效性与计划工具行为分开统计，使用首次合法实际返回；修正说明见results/evaluation_definition.json|
|计时|同时保存模型步骤耗时和包含Harness启动的实际调用总耗时|
|范围|固定30会话探索性试测，不属于课程论文|

早期连通及设置调试记录位于pilot，不混入正式八组统计。
