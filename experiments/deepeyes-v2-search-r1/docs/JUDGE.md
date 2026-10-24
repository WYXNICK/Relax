# Qwen3-8B Judge选择

Relax示例默认Qwen2.5-1.5B-Instruct；DeepEyesV2上游[Judge说明](https://github.com/Visual-Agent/DeepEyesV2#judge-server)也给出8B选择。本次不是直接因参数更大就假定更准，而是先固定60个明确正反例，再验证原生scorer。

| 候选                  | 合计判对 | 标准输出格式 |
| --------------------- | -------- | ------------ |
| Qwen2.5-1.5B-Instruct | 49/60    | 53/60        |
| Qwen3-8B非思考        | 58/60    | 60/60        |
| Qwen3-8B思考          | 57/60    | 60/60        |

60例由36个明确正反例与24个预先选定固定案例子集组成；范围有限，不能视作Judge总体准确率。非思考模式仍存在两例语义等价误判；同题真实案例子集上1.5B和8B均22/24。非思考模式该组延迟中位约0.234秒，P95约1.290秒；缓存、首请求与CUDA graph配置影响延迟，不能由此声称正式吞吐提升。

随后直接调用未修改的原生perception/search/reason/vstar四类scorer，各一正一反，8/8符合预期，包含原生8192 max_tokens请求。正式Judge固定非思考、无量化，TP1、context32768、static memory fraction0.40、max-running-requests32。

切换Judge后从初始策略模型新建正式run和baseline；旧1.5B运行保持独立，未拼接到8B曲线。[原预检统计](../evidence/judge-preflight.json)。该选择改善了本轮协议兼容性和有限预检表现，不保证无误判，也不把Judge升级单独当作模型能力提升的因果证据。
