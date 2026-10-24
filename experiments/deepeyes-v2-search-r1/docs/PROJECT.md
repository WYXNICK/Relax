# 项目概述：贡献什么，如何证明有效

## 背景与目标

项目为 [Relax](https://github.com/redai-studio/Relax)，本次贡献对应[开源贡献者计划第二期 Issue #321](https://github.com/redai-studio/Relax/issues/321)的 Task 10，提交入口为 [PR #342](https://github.com/redai-studio/Relax/pull/342)。本文只整理 DeepEyes-V2 搜索后端和这次训练验证；不混入另一个认领任务或旧版 DeepEyes 的实现。

DeepEyes-V2 是能调用代码执行、图片处理和搜索工具的视觉 Agent。Task 10 要求把占位文本 `search()` 换成可配置后端，支持确定性离线 mock、Search-R1 retriever 和外部搜索 API；统一返回结构，并在异常时沿用环境的 `"Error"` 约定。

Reviewer 进一步要求：确认 DeepEyes-V2 接入真实 Search-R1 retriever 后能跑端到端训练，并提供训练日志、效果曲线和性能曲线。本轮实验正是对此做验证。

## 贡献范围

| 层次     | 已完成内容                                                                                                | 关键价值                                   |
| -------- | --------------------------------------------------------------------------------------------------------- | ------------------------------------------ |
| 搜索后端 | mock / retriever / external 可插拔配置                                                                    | 保留无网络运行能力，同时支持真实服务       |
| 协议适配 | Search-R1 POST `queries/topk/return_scores`；转换成 `elapsed_time/data`，结果含 `title/link/snippet/date` | Agent 不需要理解各服务的私有返回格式       |
| 接口配置 | URL、top-k、超时、重试；外部 GET/POST、请求字段与嵌套响应映射；认证从环境变量读取                         | 减少硬编码和重复业务逻辑                   |
| 失败语义 | 超时、服务错误、非法返回转为非终止的 `"Error"`；限制重试范围；安全错误信息                                | 单次工具失败不会直接击穿 Agent 进程        |
| 验证     | 后端回归、真实服务预检、短训练保存/恢复、200 更新正式实验、固定留出评估                                   | 从接口正确性推进到真实训练链路验证         |
| 实验工程 | 冻结资产/config、工具与 Judge 指标、TensorBoard、最佳模型保护、独立 HDFS 归档                             | 可以复盘和核验结果，而不只给一张 reward 图 |

这里的“Search-R1 训练”指**DeepEyes-V2 策略模型接入 Search-R1 retriever 后的 GRPO 训练**。E5 检索编码器和 FAISS 索引作为固定工具服务，不在这次实验中更新权重。图片搜索仍使用 DeepEyes-V2 图片缓存；本次替换的是文本搜索后端。

## 端到端链路

```text
固定问题 / 图片
  → DeepEyes-V2 策略模型生成动作
  → Agent 环境解析动作
  → 文本 search() → Search-R1 HTTP → E5 + Wiki18 FAISS → 统一返回结果
  → 图片搜索缓存 / Apptainer Jupyter 代码执行（按模型动作调用）
  → 工具 observation 回到 Agent → 生成最终回答
  → 项目原生格式检查 + Qwen3-8B Judge → reward
  → 同一问题 8 条轨迹计算 GRPO advantage → 策略更新
  → 权重同步到 SGLang → 下一轮 rollout
```

8 张 H100 80GB 由训练和 rollout 同步共用，另外 2 张运行 retriever 与 Judge。检索和 Judge 是固定辅助服务，不参与策略模型反向传播。

## 可以展示的核心成果

- 200 次有效更新，51,200 条接受训练轨迹；固定 Search 留出 256 题，每题 8 次采样，五个评估点均完整。
- 最佳 checkpoint100：Search 答案准确率 41.26% → 63.53%，+22.27 个百分点；格式合格率 63.23% → 97.17%。
- 同一验证条件下，平均文本检索调用 3.128 → 1.079 次/轨迹，减少 65.5%。这是调用成本变化，不能直接解释成延迟下降 65.5%。
- 训练文本检索累计 36,962 次，失败 11 次（约 0.030%）；接受训练样本 scorer 异常为 0。该统计不是所有生成请求或所有工具的零故障声明。
- 普通步中位耗时 168.54 秒，有效 token 吞吐中位数 303.94 tokens/GPU/s；按完整7200秒重新核算，已保存的64个两小时SM窗口均值均超过20%。

项目执行复用了 Relax 的训练、奖励、保存/恢复和原生 router 能力。PR 的搜索实现、实验配置与运行记录属于不同层次，不能把框架已有的 GRPO/Megatron/SGLang 宣称为个人从零实现。对外描述以“实现搜索后端并完成端到端训练验证、数据与评估设计、运行排障和结果分析”为主。
