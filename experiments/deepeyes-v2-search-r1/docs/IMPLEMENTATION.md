# 实现说明

## 源码入口

| 模块         | 代码                                                                               | 作用                                           |
| ------------ | ---------------------------------------------------------------------------------- | ---------------------------------------------- |
| 搜索后端     | [search_backends.py](../../../examples/deepeyes_v2_agentic/app/search_backends.py) | mock、Search-R1和外部API三种适配，统一返回结构 |
| 环境工具接口 | [search_utils.py](../../../examples/deepeyes_v2_agentic/app/search_utils.py)       | 复用现有工具接口和非终止错误契约               |
| 后端测试     | [协议测试](../../../tests/examples/deepeyes_v2_agentic/test_search_backends.py)    | 请求、响应、错误和重试行为                     |
| 数据准备     | [prepare_data.py](../implementation/prepare_data.py)                               | 固定去重、图片组隔离和训练/验证划分            |
| 指标记录     | [experiment_metrics.py](../implementation/experiment_metrics.py)                   | 使用原生日志hook记录工具统计，不改奖励         |
| 行为观察     | [observed_agent.py](../implementation/observed_agent.py)                           | 委托原Agent行为，记录工具调用和固定案例        |
| 训练配方     | [训练脚本](../../../scripts/training/multimodal/run-qwen36-deepeyes-search-r1.sh)  | 保留实验使用的训练参数                         |

个人实现重点是搜索适配、数据与观测工具、端到端实验验证。GRPO、模型训练、推理、评分、保存和恢复复用了Relax已有能力。

## 如何复查

先按[实验设计](EXPERIMENT.md)查看固定版本、数据规则、超参数与评估协议，再结合[结果分析](RESULTS.md)核对五个评估点。恢复与补评估的方法统一放在[训练过程](PROCESS.md)，原生截图和数据表通过[证据索引](EVIDENCE.md)查找。

本分支服务于源码学习和实验复盘，未在新的运行环境重新训练。公开辅助代码省略环境专用配置；完整原始运行副本保存在开发机。
