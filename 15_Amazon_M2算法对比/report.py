from common import ROOT,RUN,read
def main():
    rows=read(RUN/'results/comparison.json');m=read(RUN/'manifest.json');p=read(RUN/'results/frozen_policy.json');a=read(RUN/'results/verification.json')
    assert a['passed'];by={r['key']:r for r in rows}
    table=['| 方法 | Recall@10 | MRR@10 | NDCG@10 | 平均/ms | P95/ms |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for key in ['popularity','grey','grey_improved','SRGNN','GCSAN','GCEGNN','fixed_policy','qwen_raw','qwen_offline']:
        r=by[key];table.append(f'| {r["method"]} | {r["recall@10"]:.2%} | {r["mrr@10"]:.5f} | {r["ndcg@10"]:.5f} | {r["mean_ms"]:.3f} | {r["p95_ms"]:.3f} |')
    q=by['qwen_offline'];fixed=by['fixed_policy'];grey=by['grey'];raw=by['qwen_raw']
    fallback_note='最终α=0，校准版线上评分完全回退为全局固定策略；应以原始Qwen选策消融评价实际LLM选择，不能把该回退策略称为LLM路由的成功。' if p['selected_alpha']==0 else '校准版包含分组与全局策略，实际LLM原始选择的独立结果见消融行。'
    text=f'''# Amazon-M2统一实测结果

已在完整西班牙站完成灰关联、三个GNN和离线LLM策略预计算的统一实验。数据共{m['sessions']:,}条会话、{m['catalogue_items']:,}件候选商品，未将训练池缩为2000名用户。

## 数据与任务

输入是原始prev_items的末20件商品及已知目录属性，目标是原始next_item。按完整原始序列的固定哈希分组，训练{m['splits']['train']:,}条、验证{m['splits']['validation']:,}条、测试{m['splits']['test']:,}条。完全相同的原始序列没有跨集合重叠；截断后的重叠计数见核查记录。缺少用户标识和原始时间戳，因此本次不是用户隔离或全局时间隔离实验。

所有方法在完整{m['catalogue_items']:,}商品目录排序，统一过滤padding和历史商品，无随机采样负例。测试冷目标{m['test_cold_targets']:,}条，占{m['test_cold_targets']/m['splits']['test']:.2%}。数据使用发布方训练CSV进行本地划分，并非直接复现官方排行榜划分。

## 统一测试结果

{chr(10).join(table)}

Recall@10是目标进入前10名的会话比例；MRR@10是命中位置倒数的平均；NDCG@10使用对数位置折扣。三者越高越好。单目标情况下Recall@10等于Hit Rate@10。

计时：CPU4线程、batch1，预热后固定200条测试历史各重复3次，共600次。计入内部ID历史到特征/图构建、完整目录评分、历史过滤和Top10。排除模型加载、静态商品向量预计算、原始CSV解析和界面网络开销。所有方法均使用相同样本。西班牙站规模、划分和候选集与原Diginetica实验不同，数值不直接横比。

训练进程曾并行执行，日志内训练墙钟包含共享CPU的影响，不作为隔离训练效率比较。统一推理计时在训练与任务本地LLM服务结束后单独执行。

## 算法实际执行了什么

- 灰关联：训练转移、共现、文本相似度、品牌、颜色、材质、价格及热度8项归一化指标，理想值1，ρ=0.5，等权算术平均。
- 改进灰关联：仅把差异d替换为 `d/[1+β(1−d)]`，验证选择β={p['selected_beta']}。
- GNN：直接导入原RecBole-GNN的SR-GNN、GC-SAN、GCE-GNN；统一融合ID32、训练商品拟合的字符TF-IDF/SVD32、品牌8、颜色4、材质4、价格及缺失标记。节点、全局邻居和候选评分使用同一融合层。训练会话内部前缀增强，交叉熵、Adam 0.003、batch256，最多15轮，验证MRR连续4轮未提升早停。训练未见商品采用属性表示，训练softmax排除无正例的未见商品，测试仍包含全部目录。
- 离线qwen3.5-2B：本地Q4模型真实参与离线选策，模型权重未更新。16种状态由历史长度、最后商品是否重复、品牌集中度和价格离散度构造。候选权重经训练留出证据评分，以样本量收缩减少低支持状态波动；Qwen从4个选项输出一个字母，再经数值检查与验证校准。最终全局混合α={p['selected_alpha']}，按非零权重计算特征。

离线Qwen执行{p['actual_policy_llm_calls']}次策略调用和1次预热，策略请求累计{p['policy_llm_wall_seconds']:.3f}秒；这是LLM请求时间，另有模型加载、候选校准及验证开销。在线每条推荐调用2B次数为0，单请求耗时属于预计算策略评分器。没有蒸馏或微调，也没有额外云端教师API调用。

完整策略准备过程（读取数据、统计、候选证据、LLM请求及验证校准）实测{p['full_policy_preparation_seconds']:.1f}秒；原生服务模型加载由启动阶段另行完成。策略准备与部分GNN训练并行，该离线墙钟受共享CPU影响，不作为隔离运行效率结论。

训练侧校准从训练会话另留{m['calibration_held_out_sessions']:,}条，行为证据统计只使用其余{m['feature_fit_sessions']:,}条；固定抽取{m['policy_calibration_samples']:,}条生成候选证据。策略校准和β选择使用固定{m['policy_validation_samples']:,}条验证会话；GNN使用完整验证集早停。测试集在全部模型和策略冻结后统一评价。

## 固定策略对照与结论边界

离线Qwen的Recall为{q['recall@10']:.2%}、MRR为{q['mrr@10']:.5f}、平均耗时{q['mean_ms']:.3f}ms；不使用LLM的固定策略对应为{fixed['recall@10']:.2%}、{fixed['mrr@10']:.5f}、{fixed['mean_ms']:.3f}ms。须同时比较这三项，不能把训练校准、数值回退和稀疏执行的收益全部归因于语言模型。原始Qwen选择、回退及候选证据保留在qwen_requests和results。

未经数值回退或全局混合覆盖的原始Qwen选策Recall为{raw['recall@10']:.2%}、MRR为{raw['mrr@10']:.5f}。{fallback_note}

本次是单站点单种子、本地划分实测；未开展多种子或独立站点确认，不能据此声称普遍优越或统计显著。GNN使用一组基础超参数和验证早停，策略有独立的候选搜索预算，未开展等预算的系统超参数优化。商品属性是发布目录快照。文本是词项降维向量，不是大型预训练语义编码器。不同方法的失败预测均保存在逐条预测中。

## 可查看文件

- [算法对比CSV](run/results/算法对比.csv)
- [逐条预测](run/predictions)
- [验证记录](run/results/verification.json)：{a['aligned_prediction_rows']:,}条预测重算、冻结输入指纹、三个模型重载和全部Qwen稀疏/完整执行一致性检查通过。
- [实验方案](实验方案.md)、[数据说明](../14_Amazon_M2推荐数据/数据说明.md)
- `run/models/`模型权重；`run/logs/`训练日志；`run/qwen_requests/`实际本地请求响应。
'''
    (ROOT/'实测结果.md').write_text(text,encoding='utf-8');print(ROOT/'实测结果.md')
if __name__=='__main__':main()
