# Mixed capture 与自动启动合同整合分支

分支：`IgniteServe/integrated-startup-contract-mixed`。
合并父版本：`igniteServe/mixed-piecewise-capture@1a68457884` 与
`igniteServe/automatic-startup-contract@798f96cfff`。
配套 PyTorch：`igniteServe/concurrent-piecewise-cudagraph-capture@ec04483`。

## 保留的能力与入口

本分支保留 mixed 版本的运行时、分布式启动优化、persistent shell、
Dynamo frontend cache，以及自动合同分支的发现、生成、审核和验证工具。

未开启主开关时，`VLLM_STARTUP_DAG_MODE` 默认为 `off`；开启 `VLLM_ENABLE_STARTUP_OPTIMIZATIONS=1` 后，未显式设置的 DAG 模式默认 `execute`。九项一键默认及覆盖规则见 `STARTUP_ASSETS_CN.md`。可选模式：

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

主开关现在为未设置的 `VLLM_ENABLE_ADAPTIVE_CUDAGRAPH` 填入1，使
EngineCore 的 idle/SLO 开关与 worker controller 一致；单项仍可显式覆盖。
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

合并时使用修改版 `torch 2.7.0+ignite.concurrentcapture` 运行CPU/合成测试，
当时未启动真实模型。2026-10-12随后完成了本分支的Qwen3-4B单卡API冒烟，
不等同于重跑完整GPU性能矩阵，也不覆盖旧自动合同的重新审核与验证。

## 本地构建产物准备

新Git工作目录不包含已编译的 `.so`、构建时生成的FlashAttention Python包
及 `_version.py`。仅设置 `PYTHONPATH` 不能代替构建；缺少FlashAttention
会在模型类检查时触发 `No module named vllm.vllm_flash_attn.layers`。

在本服务器可复用可信、相同原生源码的已有构建：

```bash
conda activate vllm_concurrent_capture
cd /data/wenbiao/igniteserve_integrated_startup
python -m pip install --target "$PWD/.deps/build-tools" setuptools-scm
python tools/prepare_local_runtime.py \
  --from-checkout /home/wenbiao/vllm_my_A100
```

准备工具逐文件检查 `csrc/`、`cmake/` 和 `CMakeLists.txt`，补齐11个本地
产物并生成本工作目录的版本文件；遇到原生源码差异、缺失产物或现有产物
不一致时拒绝覆盖。版本生成依赖只装入本目录，不修改共享Conda环境。
这只是本地开发辅助工具，不替代发布wheel的正式构建；CUDA和ABI兼容仍
需要真实模型测试确认。

## 单卡手动启动

```bash
conda activate vllm_concurrent_capture
cd /data/wenbiao/igniteserve_integrated_startup
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 VLLM_ENABLE_STARTUP_OPTIMIZATIONS=1 \
python -m vllm.entrypoints.openai.api_server \
  --model /data/wenbiao/models/Qwen3-4B --max-model-len 2048 --port 8000
```

上述命令执行人工DAG，不是 `execute-generated`。热点表缺失或不兼容时
预捕获完整注册集合。建议额外指定独立的 `VLLM_CUDA_GRAPH_HOTSPOT_DIR`；
模型、调度参数或软件版本改变后使用新目录并重新采集，不应改写旧manifest
绕过检查。导出端现在也校验manifest，拒绝将新统计写入不兼容目录。

## 权重加载与编译重叠

`VLLM_ENABLE_WEIGHT_COMPILE_OVERLAP=1` 不再按模型架构名称设白名单；
主开关默认启用，显式设置为0仍可关闭完整DAG路径的重叠。
编译模式、非量化、无LoRA/推测解码以及auto/safetensors加载格式检查保持不变。
真实调优和首次编译结果执行前仍等待权重就绪；移除架构名单不等于任意自定义
loader已获正确性验证。自定义loader若替换参数存储或改变张量属性，应显式关闭
重叠并先验证。旧off路径的加载线程不由此开关统一控制。

## 真实模型冒烟记录

产物：`A100_0606/integrated_api_smoke_20261012/`，保留每次尝试的环境、
服务器日志、请求响应、结果和telemetry。使用GPU0，未停止其他GPU的任务。

* 修复构建产物缺失后，健康检查、模型列表和实际生成API通过。
* 日志确认人工DAG及Full-FX子进程编译路径，首次37个子图物化为4次编译、
  33次结构复用。后续启动命中缓存，不能把其时间称为无缓存冷启动时间。
* `attempt06` 从真实短测采集的热点表选中 `[1,2]`，按需捕获8个新shape：
  prefill为8/24，mixed为96/48/144/184，decode为16/4。
  每个shape捕获37个piece；观察到prefill和decode动态图replay。
* 冒烟仅确认请求完成、非空生成及捕获/回放事件，不构成与原生vLLM的
  数值等价证明或长期延迟稳定性结论。
* 修复后 `attempt07` 的16次生成均成功，预捕获 `[1,2,16]` 后完成7个
  新shape的后台捕获，覆盖prefill/mixed/decode。`attempt08` 重测上述
  最简命令，3次生成通过；不兼容热点表的文件hash在运行前后保持不变。
  所属进程组和8000端口均释放，GPU0显存回到0 MiB。
* 73项CPU/合成测试通过，包括构建产物准备、旧热点表写入保护及合并功能。
* 退出时仍有上游NCCL显式销毁提示；所属进程退出后另行核验GPU0显存释放，
  不以退出码代替资源检查。
