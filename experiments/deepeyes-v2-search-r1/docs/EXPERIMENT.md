# 实验设计与复现

## 固定配方

本轮基于 Relax DeepEyes-V2 的默认训练配方，沿用核心训练超参数；**并非完全原封不动运行默认脚本**。没有把旧 DeepEyes 的小模型配置套到 V2。

| 项目       | 正式执行值                                                                                                         |
| ---------- | ------------------------------------------------------------------------------------------------------------------ |
| 代码基准   | `e5e1df624e8b0ea635b6f2f8a5c41c83c33958e0`；运行目录保存实际 recipe、参数核验和工作树 patch                        |
| 策略模型   | Qwen3.6-35B-A3B，HF revision `995ad96eacd98c81ed38be0c5b274b04031597b0`                                            |
| Judge      | Qwen3-8B，revision `b968826d9c46dd6066d109eabc6255188de91218`，非思考、无量化；正式新 run 重建 baseline            |
| 文本检索   | Search-R1 HTTP，Wiki18 + E5-base-v2 + FAISS Flat；top-k=3，超时10秒，最大重试2次，退避0.5秒，trust_env=false       |
| 图片检索   | 上游图片搜索缓存，转换后23,999个图片；Search留出集缓存覆盖256/256                                                  |
| RL         | GRPO；每批32题×8条轨迹，global batch256、microbatch1；seed与rollout_seed均1234                                     |
| Optimizer  | Adam，恒定lr1e-6，betas0.9/0.98，weight decay0.1；CPU optimizer offload＋overlap                                   |
| 策略约束   | KL loss coef0，entropy coef0，clip0.2 / high0.28 / c3，项目原生TIS                                                 |
| 生成限制   | context16384、每轮response4096、max turns9；rollout temperature1；评估top-p0.7、每题8条                            |
| 并行       | 训练TP4 / PP1 / CP1 / EP8 / ETP1，full recompute；4个SGLang引擎，每个TP2                                           |
| 实际执行   | synchronous colocate、max_staleness0；DeepEP flex、balance-data、静态显存比例0.60、max_tokens_per_gpu16384、NVLS=0 |
| 预算       | 正式200次更新，训前baseline，完成50/100/150/200次时保存和评估                                                      |
| 最后补评估 | 同一checkpoint199 BF16导出；不创建optimizer，无权重更新；并发prompt组8→4；17项采样/奖励/数据/引擎参数核对一致      |

Judge 的选择做过固定 60 个正反例预检：原1.5B为49/60、8B非思考58/60，输出格式8B为60/60；随后8个原生scorer请求全部通过。它是有限的兼容性/质量预检，不是对 Judge 总体准确率的估计。[原始验证说明](JUDGE.md)保留两例误判与范围。

## 数据准备

原始资产为 `honglyhly/DeepEyesV2_RL`，固定 revision 见 [assets.csv](../data/assets.csv)。复用 V2 转换工具及统一 Agent prompt，不在本轮另做原项目 cold-start SFT。

1. Search原始4,856行，按question＋image去重120行，剩4,736行。
2. seed1234按图片组划分：4,480行训练、256行留出；留出不按模型答题结果筛选。
3. perception/reason各选1,000个合格训练样本；选择前排除与评估图片字节重复的2/63行。
4. 正式训练共6,480题；VStar191题用于辅助检查。固定16个Search题用于追踪行为，而非事后挑选成功样例。
5. 保存所有ID、图片组规则、文件大小、SHA256、缓存覆盖。图片隔离为字节一致性隔离，没有做感知近重复检测。

派生文件：search_train约1.402GB、search_holdout约0.085GB、VStar约0.113GB、perception约0.072GB、reason约0.028GB，合计约1.70GB。[精确大小与hash](../data/data-files.csv)。原始数据约10.73GB、Wiki18压缩语料约5.12GB、索引约64.56GB；语料解包、Arrow缓存与临时索引拼接另占空间。

模型准备大小约：策略模型完整HF目录71.93GB，Judge16.40GB，E5约0.439GB。MoE名称中的A3B是激活参数规模口径，不能据此只为模型分配3B权重存储。

完整原生恢复checkpoint实测约492.65GB，含模型与optimizer/RNG/恢复元信息，另保留数据游标；final200推理HF导出约72GB。best＋final两份原生checkpoint约985.30GB。准备时索引双份、临时保存与上传pin会增加峰值；历史执行预算为训练本地约2TiB、HDFS逻辑约3TiB，并非现在实测剩余空间。此复盘目录不复制大模型权重。

## 奖励与指标

Search复用项目公式：`reward = 0.8 × answer_accuracy × (1 - search_penalty) + 0.2 × format_valid`，有文本或图片搜索时`search_penalty=0.1`。是否调用搜索影响一次惩罚；不是每调用一次固定累加0.1。其余任务使用各自原生scorer。

主模型按固定 Search 留出集的 **mean sampled answer accuracy** 选型，分母为256×8=2048；不是“八次里只要一次答对”的pass@8，也不是8倍独立问题。`acc/max=1`不能用来选择模型。辅助检查格式、调用成本、工具错误及VStar，不以综合reward替代答案准确率。

VStar早期部分会话缺失，因此同时保留有效样本分母和计划分母；对齐趋势用计划1528条，缺失视为未答对。VStar是辅助集，不宣称本次普遍视觉能力提升。

## 与默认示例的差异

| 保留的核心配方                                                             | 按本任务配置/调整                                              |
| -------------------------------------------------------------------------- | -------------------------------------------------------------- |
| Qwen3.6-35B-A3B、GRPO、lr1e-6、32×8/global256、上下文/轮数、主要并行与重算 | 文本mock→真实Search-R1；训练加入Search并做隔离留出             |
| 原生奖励/scorer、同步更新与权重同步、原生checkpoint恢复                    | Judge1.5B→8B，独立新baseline；默认2000→200更新                 |
| 原生训练控制链路                                                           | 默认eval500且跳过baseline→baseline＋每50；默认save200→每50     |
| 全局采样/优化目标未为利用率修改                                            | DeepEP/长度均衡、静态池0.60、NVLS=0；恢复时使用原生SlimeRouter |

实际执行以 [冻结recipe](../../../scripts/training/multimodal/run-qwen36-deepeyes-search-r1.sh)、[初始参数核验](../evidence/runtime-argument-check.json)、[恢复参数核验](../evidence/resume-runtime-argument-check.json)为准。早期 `preparation/code-snapshot` 和 `environment.json` 可能保留旧保存间隔、旧run时间或旧状态；它们是历史材料，不覆盖正式完成回执。

## 复现顺序

具体代码、资产和环境准备步骤见[实现与复现](IMPLEMENTATION.md)。正式运行前先用独立短pilot验证生成、评分、更新、同步、保存和恢复；pilot不计入正式预算。

随后冻结recipe与run目录，先baseline再按固定预算训练，每50步评估；同一条件比较候选并保护best/final。验收同时核对训练步数、评估样本数和成功状态，不能只看进程exit0。本次归档整理没有重新做远端权重取回或复训验收。
