# 历史实验辅助代码

这些辅助文件来自实际200步run的冻结配置。它们复用项目Agent、环境和logger，不实现新的奖励或训练算法。

- prepare_data.py：固定split与缓存转换，保留原ID/hash约束。
- experiment_metrics.py：原生logging hook补充工具统计，保留原生记录。
- observed_agent.py＋run_observed_agent.sh：每会话委托原行为，仅记录工具与固定case。
- run_retriever.py：单GPU的连续Flat分片部署，保留同一检索协议。

依赖来自既有训练/应用/检索环境；未增加项目requirements。固定代码版本、生成预算和运行条件见上级文档。此分支只做归档，未用这些导出文件启动新GPU作业。

公开版本省略部署专用的sandbox环境项；完整原始运行副本保持私有。
