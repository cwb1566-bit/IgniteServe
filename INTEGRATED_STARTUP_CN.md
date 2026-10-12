# Mixed capture 与自动启动合同整合分支

分支：`IgniteServe/integrated-startup-contract-mixed`。
合并父版本：`igniteServe/mixed-piecewise-capture@1a68457884` 与
`igniteServe/automatic-startup-contract@798f96cfff`。
配套 PyTorch：`igniteServe/concurrent-piecewise-cudagraph-capture@ec04483`。

## 保留的能力与入口

本分支保留 mixed 版本的运行时、分布式启动优化、persistent shell、
Dynamo frontend cache，以及自动合同分支的发现、生成、审核和验证工具。

`VLLM_STARTUP_DAG_MODE` 默认仍为 `off`，可选：

| 值 | 行为 |
|---|---|
| off | 原有顺序初始化入口 |
| shadow | 记录已有人工 DAG，沿原有路径执行 |
| execute | 执行已有人工 DAG；支持 deferred executor 的 TP/PP 路径保留 |
| discover | 走原有 off 初始化路径并采集根级/候选阶段证据，不读已有人工 DAG |
| execute-generated | 校验并执行已审核的最终合同及 adapter |

discover 的 off 基线保留引擎原本的权重加载线程等内部行为；它不承诺所有
底层操作均无并发。其含义是不会用已有人工 DAG 启动。

自动构图和 generated-adapter 的验证范围仍是单卡 EngineCore；继承了
分布式启动代码不代表从零构图已验证多 rank。mixed/prefill 后台 capture
同样保留单卡 text-only piecewise 及修改版 PyTorch 的能力检查。

## 默认捕获策略

* worker adaptive controller 与后台 concurrent capture 默认启用。
* 在线 `VLLM_ADAPTIVE_CUDAGRAPH_MIN_HITS` 默认1。
* 离线预捕获先取命中累计99%的最短前缀，再过滤命中小于10和不可捕获的
  shape，补入已注册的1、2；不补齐被过滤的尾部。
* 显式 TOP_K、强制全图及关闭开关继续有效。
* shadow 校验默认关闭。

建议在启动配置中显式设置 `VLLM_ENABLE_ADAPTIVE_CUDAGRAPH=1`，使
EngineCore 的旧 idle/SLO 开关与 worker controller 一致。
详细参数见 `MIXED_CAPTURE_DEFAULTS.md`。

HotspotAnalyzer 合入了按 engine generation 保存分片、原子汇总CSV的逻辑，
同一 generation 重复导出不会重复累加。正式实验仍应使用独立的热点目录，
保留离线原始 profile；不同工作负载不要共用可写目录。

## 发现与最终合同执行

```bash
export VLLM_STARTUP_DAG_MODE=discover
export VLLM_STARTUP_OFFLINE_ANALYSIS_DIR=/analysis/root_D0
export VLLM_STARTUP_ANALYSIS_PYTHON_CALLS=1
# 然后运行正常的单卡模型启动入口；py-spy/Nsight另轮采集。
```

工具入口：`tools/startup_offline_analysis.py`；工作流仍为 discover、extract、
节点级插桩、enrich、人工审核、generated-adapter串行等价、逐边验证、finalize。
参见 `AUTOMATIC_STARTUP_CONTRACT_CN.md`。

```bash
export VLLM_STARTUP_DAG_MODE=execute-generated
export VLLM_STARTUP_CACHE_STATE=D2
export VLLM_STARTUP_CONTRACT_PATH=/contracts/model/startup_contract.json
export VLLM_STARTUP_ADAPTER_PATH=/contracts/model/startup_adapter.py
unset VLLM_STARTUP_ALLOW_CANDIDATE_VALIDATION
```

历史automatic实验中的合同、adapter及stage map绑定旧的源码和环境，不能
直接作为本分支的已验证产物。合并改变了 EngineCore 源码与行号；必须重新
生成映射、核对action/语义边界并完成相应验证。不得只替换hash来跳过验证。
旧合同被拒绝时会在action开始前回退；action开始后的异常不会重新顺序启动。
历史14节点/19或17边的结果仍属于原automatic分支，不是本分支重新测量结果。

## 合并处理和验证

唯一Git冲突位于人工DAG执行结束：先flush离线recorder，再返回
stage_seconds、worker overlap结果和plan路径，保留两边行为。

测试覆盖：自动合同发现/提取/最终化/兼容性、状态与资源验证、删边实验，
mixed捕获生命周期与输入契约、热点跨generation导出和图匹配；新增测试
覆盖EngineCore原有模式转发、discover插桩及失败flush、generated预检回退、
action失败不重启、deferred分布式入口，以及合并源码的无seed候选提取。

本次使用修改版 `torch 2.7.0+ignite.concurrentcapture` 运行CPU/合成测试，
未启动真实模型或重跑GPU性能矩阵。两个原分支既有GPU结果不能代替本分支
的联合真实模型验收。
