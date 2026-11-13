# 教师指导的轻量会话推荐

独立性能实验，**不加入课程论文**。

本实现借鉴 Augur 的教师—学生思想，以当前对话分析训练证据、生成结构化关联指导，再训练本地小网络。

先读 [实验计划](实验计划.md)，完成后查看 [性能比较报告](性能比较报告.md)。

## 复跑顺序

环境：`D:\Envs\recbole-gnn-py39\Scripts\python.exe`，Python3.9 / torch2.1.2+cpu。
在本目录依次执行：

```powershell
& 'D:\Envs\recbole-gnn-py39\Scripts\python.exe' prepare.py
# 教师原型已人工分析冻结，位于 teacher/prototypes.json；prepare 不调用任何大模型。
& 'D:\Envs\recbole-gnn-py39\Scripts\python.exe' train.py
& 'D:\Envs\recbole-gnn-py39\Scripts\python.exe' evaluate.py
# benchmark.py --model 分别运行下列八种模型；每次使用独立进程。
# gra_original gra_beta7 teacher_rules student_plain student_teacher SRGNN GCSAN GCEGNN
& 'D:\Envs\recbole-gnn-py39\Scripts\python.exe' benchmark.py --model student_teacher
& 'D:\Envs\recbole-gnn-py39\Scripts\python.exe' report.py
& 'D:\Envs\recbole-gnn-py39\Scripts\python.exe' verify.py
```

## 文件

- `teacher/training_evidence.json`：真实训练侧分组统计和示例。
- `teacher/prototypes.json`：当前对话助手依据证据编写的十二类指导及理由。
- `teacher/guidance.jsonl`：2,000 条原型赋值指导，非逐条独立 LLM 调用。
- `models/*.pt`：两个 282 参数学生。
- `results/frozen_before_test.json`：测试前冻结指纹。
- `predictions/*.jsonl`：全测试的真实推荐列表。
- `results/comparison.csv`：推荐质量、响应时间和内存数值。
- `results/cases.json`：指导后成功与失败的真实案例。
- `results/completion_audit.json`：数据、重载、全测试预测及计时核验。

教师原型的本地执行时间与大模型调用时间分别处理，本实验没有大模型 API 的实测延迟。
