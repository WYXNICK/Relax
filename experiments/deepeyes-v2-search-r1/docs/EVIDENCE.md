# 结果与证据索引

## 常用入口

| 需要查看       | 文件                                                                                                                                                |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| 五轮完整评估   | [结果CSV](../data/evaluation.csv)、[样本完整性](../data/evaluation-completion.csv)                                                                  |
| 逐步训练指标   | [200步宽表](../data/training-200-updates.csv)、[原生标量压缩文件](../data/scalars.csv.gz)                                                           |
| 汇总统计       | [summary.json](../data/summary.json)、[关键结果](../data/verified-facts.json)                                                                       |
| 数据与模型版本 | [资产版本](../data/assets.csv)、[文件hash](../data/data-files.csv)、[划分摘要](../evidence/data-split-manifest.json)                                |
| 固定案例统计   | [匿名16题、五阶段](../data/fixed-16-cases.csv)                                                                                                      |
| 训练日志       | [前30步](../evidence/training-first30.log)、[最终一步](../evidence/training-final.log)                                                              |
| 超参数与版本   | [实验配置](../configs/experiment.json)、[依赖版本](../configs/environment.json)                                                                     |
| 恢复与最终评估 | [恢复记录](../evidence/resume.json)、[参数对照](../evidence/evaluation-parameters.json)、[完成计数](../evidence/evaluation-completion.json)         |
| 原生截图       | [效果](../figures/01-training-evaluation-hd.png)、[性能](../figures/02-performance-hd.png)、[日志](../figures/03-training-log-steps-000-009-hd.png) |
| 文件完整性     | [校验清单](../artifact-manifest.json)、[公开材料说明](../evidence/export-provenance.json)                                                           |
| 开源贡献状态   | [PR #342合并记录](../evidence/contribution-status.json)                                                                                             |

## 统计口径

- 最终模型对应200次有效更新、51,200条接受训练轨迹；恢复前未保存的尾段不重复计入。
- Search每轮256题×8=2048，五轮完整。VStar每轮计划1528；早期缺失采样及两种分母均保留在结果CSV中。
- raw_reward有400条原生writer事件，但只有200个唯一step；逐步宽表按tag和step取最后值。
- best100按原始Search验证答案准确率选择，不使用平滑峰值、综合reward或acc/max。
- 固定案例使用case_01至case_16，匿名编号在五阶段保持一致；它们是预先选定的案例，不是按成功结果挑选。

公开材料保留数值与必要的方法说明，日志仅摘录训练指标并移除进程身份。原始部署记录、完整标识清单及详细诊断材料保存在开发机，不进入公开分支。

验证公开文件：

```bash
python experiments/deepeyes-v2-search-r1/tools/verify_artifacts.py
```

该命令只校验文件hash、样本计数和截图尺寸，不执行训练或评估。
