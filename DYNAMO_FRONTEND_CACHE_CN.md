# Dynamo 前端缓存实验说明

> 实验分支：`igniteServe/dynamo-frontend-cache`
>
> 基线提交：`e77091cf2e`（并发 piecewise CUDA Graph 实验分支）
>
> 验证日期：2026-08-31

## 1. 问题与结论

在 S4 已命中 Inductor、Triton best-config、piecewise 元数据和 KV profile
缓存时，Qwen3-1.7B 的 `Model Compilation` 中仍有约 7.6 秒用于
TorchDynamo Python bytecode tracing。cProfile 显示重复 Decoder Layer 的逐层
展开是主要来源：`Qwen3DecoderLayer.forward` 被跟踪 28 次，累计约 6.54 秒。

本分支实现一个 exact-config Dynamo frontend cache。首次启动仍执行真实
TorchDynamo，并持久化完整 FX 结构、必要的 symbolic-shape 元数据、转换后的
Python bytecode 和依赖源码哈希。后续兼容启动直接恢复这些产物，再进入原有
vLLM piecewise backend，不再逐层解释模型 Python 代码。

三轮 A/B 的中位数结果如下。两组使用同一个模型、S4 资源、Inductor cache、
CUDA Graph profile 和固定 greedy 请求，仅切换前端缓存开关。

| 指标 | 原 S4 warm cache | 前端缓存命中 | 减少 |
|---|---:|---:|---:|
| Dynamo 前端 / 完整恢复 | 7.5834 s | 0.9700 s | 6.6134 s（87.2%） |
| Model Compilation | 9.6471 s | 1.9293 s | 7.7178 s（80.0%） |
| EngineCore Total Init | 13.1720 s | 5.4266 s | 7.7454 s（58.8%） |
| LLM 构造 `init_seconds` | 13.8674 s | 6.1250 s | 7.7424 s（55.8%） |

三轮的四条 greedy 输出完全一致。冷 page-cache 额外测试中，前端恢复先于
权重加载到达第一次 kernel execution；该 kernel 正确等待 1.524 秒，权重加载
完成后再执行，输出仍与基线一致。

## 2. 缓存内容

一个完整 bundle 以同一 `rank + pid` 标识，由以下文件组成：

- `dynamo_frontend_<rank_pid>.json`：原始/转换 bytecode 的结构签名及生成全局量。
- `dynamo_frontend_<rank_pid>_transformed_code.marshal`：Dynamo 转换后的 code object。
- `dynamo_fx_<rank_pid>.fxstruct.pkl`：不含权重值的完整 FX 图结构。
- `dynamo_fx_<rank_pid>.direct_symbols.graphpickle`：两个直接 `SymInt` 及 ShapeEnv。
- `dynamo_fx_<rank_pid>.json`：node 到 free-symbol 的依赖、运行环境和源码哈希。

Qwen3-1.7B 的 bundle 约 456 KiB，包含 3295 个 FX node 和 231 个
placeholder；它不序列化模型 Parameter、真实输入 tensor 或 CUDA runtime
对象。早期把完整 example inputs 交给 `GraphPickler` 的尝试会把 Parameter
值写入约 6.89 GB 文件，已经明确废弃。

默认目录由以下语义键确定：

- Python、PyTorch、CUDA 版本；
- vLLM runtime/config/compiler hash；
- wrapper 类和原始 forward bytecode 的结构 hash；
- rank、data-parallel rank 和 backend prefix。

命中后还会逐一校验本次 Dynamo tracing 涉及的源码文件 SHA-256。当前
Qwen3-1.7B 记录了 16 个 PyTorch/vLLM 源文件。

## 3. 使用方式

启用 S4 和前端缓存只需增加一个开关：

```bash
export VLLM_ENABLE_STARTUP_OPTIMIZATIONS=1
export VLLM_ENABLE_DYNAMO_FRONTEND_CACHE=1
export VLLM_STARTUP_RESOURCE_ROOT=/path/to/vllm_startup_resources
```

缓存默认位于：

```text
$VLLM_STARTUP_RESOURCE_ROOT/dynamo_frontend/<semantic-key>/rank_<r>_<dp>/<prefix>/
```

第一次启动该目录不完整时会执行真实 Dynamo tracing 并生成 bundle；后续启动
自动恢复。调试时可以用 `VLLM_DYNAMO_FRONTEND_CACHE_DIR` 显式指定一个
bundle 目录。

## 4. 实现位置

- `vllm/compilation/wrapper.py`
  - 构造语义键并选择同一 rank/PID 的完整 bundle。
  - 校验原始 bytecode、运行环境和 traced source hash。
  - 恢复 transformed bytecode、FX graph、ShapeEnv 和生成全局量。
  - 损坏或不兼容时 fail closed，回退真实 Dynamo tracing。
- `vllm/compilation/backends.py`
  - 持久化无权重的 FX 结构、直接 SymInt 和 node/free-symbol 依赖。
  - replay 时继续使用原有 split、canonical cache 和 alias 装配逻辑。
- `vllm/compilation/decorators.py`
  - 在第一次 `compiled_callable` 调用前尝试 frontend replay。
- `vllm/compilation/cuda_piecewise_backend.py`
  - 第一个 compiled piece 执行前等待 finalized weights。
- `vllm/compilation/weight_compile_overlap.py`
  - 提供进程内 weights-ready barrier，并传播权重加载异常。
- `vllm/v1/engine/core.py`
  - 在当前单 rank 默认并行路径中 arm/signal/close 上述 barrier。

## 5. 恢复开销细分

一次代表性命中的 0.9634 秒恢复时间如下：

| 子阶段 | 开销 |
|---|---:|
| runtime/source 校验 | 0.0052 s |
| FX 结构反序列化 | 0.1735 s |
| ShapeEnv / direct-symbol 恢复 | 0.0803 s |
| vLLM backend | 0.7032 s |
| transformed code/global 安装 | 0.0002 s |

backend 的 0.7032 秒进一步包括：FX split 0.2158 秒、piecewise 输入元数据
恢复 0.1076 秒、4 个 canonical cache 加载和 25 个 follower alias 装配
0.3648 秒。因此 `GraphPickler.loads()` 已不是主要瓶颈；继续优化它的理论上限
约 80 ms。下一步若要继续压缩，应持久化兼容的 post-split backend artifact，
或减少 canonical callable 的父进程加载成本。

## 6. 正确性与失效测试

- 配置隔离：把 `max_model_len` 从 2048 改为 1024 后，semantic key 从
  `edf3236c7c5e94da722b` 变为 `0f02cac3690ffcdcc3f8`，旧图不会错命中。
- 源码失效：篡改缓存中 `qwen3.py` 的期望 SHA-256 后，系统拒绝 replay，
  执行 7.99 秒真实 tracing，推理输出正确。
- 文件损坏：截断 FX payload 后，恢复异常被捕获并回退真实 tracing，进程不会
  因缓存损坏退出。
- bundle 一致性：FX、ShapeEnv、bytecode 和 metadata 必须具有同一
  `rank + pid`；目录中的不完整或不同 PID 残留文件不会被混合装载。
- 权重边界：结构恢复可与权重 I/O 并行，但第一次 compiled kernel 必须等待
  `weights_ready`；加载异常会通过 barrier 传播。

完整日志位于：

```text
A100_0606/dynamo_frontend_cache_20260831_v2/logs/
```

关键日志包括 `final_baseline_{1,2,3}.log`、`final_replay_{1,2,3}.log`、
`correctness_sync_replay_coldweights.log`、`invalidation_config_1024.log`、
`invalidation_source_hash.log`、`invalidation_corrupt_bundle.log` 和
`replay_breakdown.log`。

## 7. 已探索但未采用的 PyTorch 区域编译

PyTorch 2.7 提供实验性
`torch._higher_order_ops.invoke_subgraph.mark_compile_region`。本实验分别验证了
Decoder Layer 和 MLP 两种区域：

1. 整层区域在 HOP 的 alias/mutation 安全分析中失败。该分析会用
   FakeTensor/ProxyTensor 再次跟踪 attention，底层 CUDA 算子要求 shape 为
   concrete `SymIntArrayRef`，无法处理当前动态 token 维度。
2. 仅 MLP 区域能够形成 `invoke_subgraph`，但 Dynamo 前端增至 9.22 秒；随后
   backend functionalization 无法把嵌套 `GraphModule` 编码为 FX node 参数，
   抛出 `NotImplementedError`。

绕过 alias/mutation 检查可能产生静默状态错误，不能作为可接受实现。若未来
继续这条路线，需要在 PyTorch 中完善带 symbolic shape 的 HOP 安全分析，并
让 vLLM/Inductor 的 piecewise 编译和跨进程 wire serializer原生理解
`invoke_subgraph`，而不是简单关闭保护逻辑。

## 8. 当前边界

- 已正式验证 TP=PP=DP=1、Qwen3-1.7B、FP16、`max_model_len=2048`。
- exact-config cache 不等同于通用 Dynamo guard cache；配置或源码变化会回退
  tracing，而不是尝试不安全的近似复用。
- 尚未验证 TP/PP、多模态、LoRA、量化和 speculative decoding。
- 当前实现位于 vLLM 适配层。PyTorch 2.7 的
  `torch.compiler.save_cache_artifacts()` 只覆盖 Inductor、autotune、
  AOTAutograd 和 PGO，不持久化完整 Dynamo frontend/guard state；将本方案
  上移为通用 PyTorch frontend cache 需要新增正式的缓存 artifact 协议。
