# IgniteServe 自动启动合同生成器

本分支 `igniteServe/automatic-startup-contract` 从原生顺序启动路径生成候选
启动 DAG。发现阶段不会读取现有 25 节点人工 DAG、seed 依赖、资源标签或已有
`StartupContext.put/require` 事件；现有 `off/shadow/execute` 行为保持不变。

## 运行模式

- `VLLM_STARTUP_DAG_MODE=discover`：执行 `off` 对应的顺序路径，并以
  `engine_startup.root` 记录完整 EngineCore 初始化。
- `VLLM_STARTUP_DAG_MODE=execute-generated`：加载最终合同和 adapter。所有
  检查均在第一个 action 执行前完成；失败会说明原因并回退顺序路径。
- `off/shadow/execute`：保留原行为，其中 `execute` 仍是现有人工 DAG。

## 离线工作流

1. 分别在 D0、D1、D2 缓存状态采集 root trace。light/audit、`sys.setprofile`、
   py-spy 和 Nsight 分开运行，避免多个 profiler 相互污染。
2. `discover` 聚合根区间、调用树、原始调用顺序、I/O、GPU 证据和严格兼容性
   信息，生成 `discovery/startup_discovery.json`。
3. `extract` 对顺序入口执行 AST/use-def 分析，生成候选节点、状态边、adapter
   stub 和源码区域映射。仅闭合的 bound-method call 自动成为可执行 action；
   条件、循环、异常处理和不透明生命周期保持为 `opaque_region_N`。
4. 第二轮顺序启动使用 AST 生成的临时方法包装候选源码区域。包装只加入
   `with recorder.stage(name)`/NVTX，不启用全局 line tracing，也不改变原顺序。
5. `enrich` 将节点成本、audit I/O、Nsight/CUPTI 资源以及动态 RAW 状态边写入
   D0/D1/D2 variant。
6. 第一层 `verify-serial --execution-kind source_instrumentation` 要求每个源码区域
   恰好执行一次，并比较输出前缀、KV Cache 内存不变量和 artifact 集合。它只
   证明候选切分覆盖原始顺序源码，不能作为最终 adapter 的执行正确性证据。
7. 人工解决 adapter、别名和不透明约束后，`perturb`/`validate` 对每条低置信度
   顺序边在独立进程中运行三次。只有实际发生换序或重叠、指纹一致且 P50/最大
   启动时间回退不超过 5% 时，边才标记为 `safe_remove`。
8. 人工补齐 action 后，必须用生成 adapter 依原顺序运行 D0/D1/D2，并执行
   `verify-serial --execution-kind generated_adapter`。随后完成删边矩阵；`finalize`
   只接受这一层串行等价证据、完整删边验证和全部 mandatory 审核。

## 主要命令

```bash
export PYTHONPATH=/data/wenbiao/igniteserve_offline_analysis
PY=/home/wenbiao/miniconda3/envs/vllm_my_A100/bin/python
TOOL=tools/startup_offline_analysis.py

$PY $TOOL discover \
  --trace <root-trace.jsonl> \
  --call-profile <pyspy.speedscope.json> \
  --nsight-summary <nsight-summary.json> \
  --source vllm/v1/engine/core.py \
  --output <root>/discovery/startup_discovery.json

$PY $TOOL extract \
  --source vllm/v1/engine/core.py \
  --discovery <root>/discovery/startup_discovery.json \
  --output-root <root>/extraction

$PY $TOOL enrich \
  --contract <root>/extraction/candidate/startup_contract.candidate.v0.json \
  --variant-trace D0=<trace> --variant-trace D1=<trace> \
  --variant-trace D2=<trace> \
  --variant-match 'D0={"cache_state":"D0"}' \
  --variant-match 'D1={"cache_state":"D1"}' \
  --variant-match 'D2={"cache_state":"D2"}' \
  --review-dir <root>/extraction/manual_review \
  --output <root>/extraction/candidate/startup_contract.candidate.v1.json
```

完整 Qwen3-14B 驱动：

```bash
$PY A100_0606/automatic_startup_contract_qwen14/scripts/\
run_automatic_contract_discovery.py \
  --model /data/wenbiao/models/Qwen3-14B \
  --repeats 3 \
  --output-root A100_0606/automatic_startup_contract_qwen14/<timestamp>
```

## 人工审核点

- `manual_review/action_bindings.json`：实现每个 opaque adapter stub，并将决定改为
  `resolved`。
- `manual_review/state_aliases.json`：确认多写者、别名和原地修改；必要时补状态边。
- `manual_review/opaque_constraints.json`：确认 allocator、C++/CUDA、进程及 collective
  的安全约束。
- `manual_review/failed_edge_tests.json`：处理 `inconclusive` 删边实验。
- `manual_review/node_coarsening.json`：合并建议，不会自动应用。

审核未完成时，`finalize` 只写 `blocking_items.json`，不会生成伪完整合同。

建议审核顺序如下：

1. 处理 `action_bindings.json` 中的 opaque stub，并修改 candidate adapter。
2. 处理 `state_aliases.json`，补齐别名、条件定义和原地修改对应的状态边。
3. 处理 `opaque_constraints.json`，确认 allocator、CUDA/C++ 生命周期及同步边。
4. 以 `VLLM_STARTUP_DAG_MAX_WORKERS=1` 执行 reviewed candidate，完成
   `generated_adapter` 串行等价验证。
5. 按 `validation/edge_validation_plan.json` 对每个候选边独立运行三次，应用
   `safe_remove/required/inconclusive` 结果。
6. 审核 `failed_edge_tests.json`，可选处理 `node_coarsening.json`，最后 finalize。

## Qwen3-14B 实测

正式产物位于：

```text
A100_0606/automatic_startup_contract_qwen14/formal_20260902_0255/
```

| 缓存状态 | root light 中位数 | 节点级 light 中位数 |
|---|---:|---:|
| D0：模型、编译及启动资源冷 | 70.1710 s | 66.1133 s |
| D1：模型页缓存热，其余冷 | 71.2276 s | 64.7947 s |
| D2：模型、编译及启动资源热 | 39.8443 s | 34.0963 s |

本次从零发现获得 806 个过滤后的调用候选，并从 EngineCore 顺序源码提取 14 个
候选节点，其中 7 个可自动绑定、7 个为 opaque；D0/D1/D2 各有 21 条候选边，
其中共 39 个低置信度边进入后续删边计划，并生成 3 个仅供审核的节点合并建议。
源代码区域的串行覆盖和语义指纹已通过，但 generated adapter 尚未经过人工补齐，
因此 finalizer 正确拒绝生成最终合同。

测试还保留了一次未计入结果的 D2 失败：并行加载相同热缓存时，Triton launcher
偶发出现“期望 20 个参数、实际 14 个”的 ABI 错配；从同一 seed 重新克隆后连续
启动成功。该样本位于失败归档目录，后续应单独修复并行缓存物化的模块装载竞态。

## 在线加载

```bash
CUDA_VISIBLE_DEVICES=0 \
VLLM_STARTUP_DAG_MODE=execute-generated \
VLLM_STARTUP_CACHE_STATE=D2 \
VLLM_STARTUP_CONTRACT_PATH=/absolute/path/startup_contract.json \
VLLM_STARTUP_ADAPTER_PATH=/absolute/path/startup_adapter.py \
python -m vllm.entrypoints.openai.api_server ...
```

合同精确校验源码哈希、模型架构、dtype、PyTorch/CUDA、GPU compute capability、
GPU 型号、TP/PP/DP 和关键调度配置。D0/D1/D2 variant 当前由
`VLLM_STARTUP_CACHE_STATE` 显式给出；缺失或不匹配时安全回退顺序路径。

## 测试

```bash
PYTHONPATH=$PWD $PY -m pytest -q \
  tests/v1/engine/test_startup_offline_analysis.py \
  tests/v1/engine/test_automatic_startup_contract.py
```
