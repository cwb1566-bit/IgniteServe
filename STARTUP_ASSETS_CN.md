# 统一启动资产仓库

适用于 `IgniteServe/integrated-startup-contract-mixed`。

## 一键启动优化

构造 `VllmConfig` 时，`VLLM_ENABLE_STARTUP_OPTIMIZATIONS=1` 为下列未设置的子项填入默认值。显式子项始终优先，例如 `VLLM_STARTUP_DAG_MODE=off/discover/execute-generated` 或某项 `=0` 不会被覆盖。

| 功能 | 子项及默认值 |
|---|---|
| 已有人工 DAG | `VLLM_STARTUP_DAG_MODE=execute` |
| Dynamo 前端缓存 | `VLLM_ENABLE_DYNAMO_FRONTEND_CACHE=1` |
| 子进程并行编译 | `VLLM_ENABLE_SUBPROCESS_COMPILE=1` |
| 自适应 CUDA 图管理 | `VLLM_ENABLE_ADAPTIVE_CUDAGRAPH=1` |
| 编译池快速关闭 | `TORCHINDUCTOR_FULL_FX_FAST_SHUTDOWN=1` |
| 权重分片预取 | `VLLM_ENABLE_WEIGHT_SHARD_PIPELINE=1` |
| 权重加载与编译重叠 | `VLLM_ENABLE_WEIGHT_COMPILE_OVERLAP=1` |
| 两阶段编译 worker 预热 | `VLLM_ENABLE_TWO_PHASE_FULL_FX_PREWARM=1` |
| 编译器身份异步预热 | `VLLM_ENABLE_ASYNC_COMPILER_HASH_PREWARM=1` |

只在主开关开启时填默认值；关闭或不设置主开关时，不改写原有环境。环境配置是进程级的，解析后供引擎及子进程继承；不要在同一进程内通过反复修改主开关切换实例配置，应使用独立进程。

上述配置不会绕过硬件、模型或执行器能力检查，不会自动批准生成合同，也不会开启 persistent shell。已存在的显式 FILE/DIR 路径继续有效。没有兼容冻结热点表时仍回退默认预捕获图集合。

权重加载/编译重叠不再按模型架构名称限制；仍检查编译模式、非量化、无 LoRA/推测解码及 auto/safetensors 加载格式，并保留权重就绪屏障。自定义 loader 若替换参数存储或改变张量属性，应先关闭重叠并单独验证。

服务器当前账户的默认仓库已通过 `~/.config/igniteserve/config.json` 设置为 `~/igniteserve_assets`。手动启动：

```bash
conda activate vllm_concurrent_capture
cd ~/data/igniteserve_integrated_startup
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 VLLM_ENABLE_STARTUP_OPTIMIZATIONS=1 \
python -m vllm.entrypoints.openai.api_server \
  --model /data/wenbiao/models/Qwen3-4B --max-model-len 2048 \
  --host 127.0.0.1 --port 8000
```

需要改用其他资产仓库时设置 `VLLM_STARTUP_RESOURCE_ROOT`。确认所选 GPU 空闲；既有 shell 若残留子项 `=0` 或其他 DAG 模式，需先取消相应覆盖。

## 目录与读取

```text
<root>/
  model_info/<key>.json
  kv_profile/<key>.json
  best_config/<key>.json
  input_metadata/<key>.json
  transformed_code/<key>/
  dynamo_frontend/<key>/
  compile/<key>/
  hotspots/<key>/
  startup_contract/<key>/
  startup_cost/<key>.json
  runs/<time_ns>_<pid>/
```

目录按需生成；没有启用或没有产生的资产不会创建空包。多文件包内部保留后端需要的 rank/prefix 子目录。

公共实现只有两层：`vllm/compilation/startup_resources.py` 负责路径，`startup_assets.py` 负责 hash、校验、文件锁及原子写入。各阶段仍在原来的 lookup/save 处判断语义兼容性，不增加数据库、注册中心或常驻管理进程。

读取顺序：计算当前身份 → 定位资产 → 比较 schema/identity/checksum → 执行该阶段的专用校验 → 命中则使用，否则按原流程重新生成。checksum 只证明完整性，不证明代码可信或执行正确。

| 资产 | 主要入口 | 额外校验或限制 |
|---|---|---|
| 模型能力 | `model_executor/models/registry.py` | 类与源码内容；不能识别的外部动态类不缓存 |
| 显存探测 | `v1/worker/kv_profile_cache.py` | 模型/运行/硬件身份及当前权重占用、空闲显存；用当前预算重新计算 KV 空间，不直接使用历史剩余 bytes |
| 调优配置 | `v1/worker/best_config_cache.py` | kernel、实际输入签名、候选集和设备；仍要求唯一 launcher 匹配 |
| 输入元数据 | `compilation/backends.py` | 当前图及当前输入证据与历史契约匹配；恢复后再比较。未知符号、无法恢复的 alias/offset/layout 回退 FakeTensor 数据流 |
| 转换代码 | `compilation/transformed_code_cache.py` | old/new code 与文件 hash；只复用展示文件，不替代 tracing |
| Dynamo 前端 | `compilation/wrapper.py` | 入口输入、运行身份、源码及完整 bundle；仍为显式启用的实验路径，未支持的重定位重新 tracing |
| 编译产物 | `compilation/backends.py`、`compiler_interface.py` | 图/输入契约、配置分区与底层 Inductor guards；拒绝的 handle 重新编译，不能变成空执行函数 |
| 热点统计 | `v1/worker/cudagraph_profile.py` | 运行身份与文件完整性；正式运行只写 runs，不修改冻结表 |
| 启动合同 | `v1/engine/startup_analysis/generated.py` | 仅 final、兼容且完整的合同；保留人工审核门槛。不能用缓存状态环境标签代替实际探测来授权跳过节点 |
| 调度成本 | `v1/engine/startup_orchestrator.py` | 状态签名与编排源码；仅为调度估计，不作为依赖正确性的证明 |

输入恢复与 canonical 复用分别验证；历史节点编号或名称不构成等价证明。恢复保留 stride；未支持的数据关系宁可未命中，也不猜测连续布局或符号值。

## 根目录

优先级：`VLLM_STARTUP_RESOURCE_ROOT` > 用户配置 > `$XDG_CACHE_HOME/igniteserve/assets`（默认 `~/.cache/igniteserve/assets`）。路径规范化后，软链接与真实目录使用同一仓库。

单次指定：

```bash
export VLLM_STARTUP_RESOURCE_ROOT=/data/wenbiao/igniteserve_assets
```

不设置环境变量时，可在 `~/.config/igniteserve/config.json` 设置默认值（配置目录支持 `XDG_CONFIG_HOME`）：

```json
{"startup_resource_root": "/data/wenbiao/igniteserve_assets"}
```

本次改造不自动修改用户全局配置，也不移动或删除旧缓存。旧 FILE/DIR、显式 compilation cache_dir 入口继续优先；外部目录不在仓库导出范围内。旧数据缺少新校验证据时重新生成，不静默升级成可信资产。

## 查看与迁移

在项目环境中执行：

```bash
python tools/startup_assets.py inspect
python tools/startup_assets.py verify
python tools/startup_assets.py export /tmp/startup_assets.zip
VLLM_STARTUP_RESOURCE_ROOT=/new/assets python tools/startup_assets.py import /tmp/startup_assets.zip --trusted
```

导出不包含 runs；从独立快照校验后打包。导入先检查归档路径、checksum 和包完整性，不执行 pickle/marshal/adapter；目标已有不同文件时拒绝覆盖。含编译代码、Dynamo 或 adapter 的导入必须显式 `--trusted`，只应用于可信来源。

编译映射中的仓库内 handle 保存为相对路径；导出包含仓库内完整编译目录。但这不承诺任意后端二进制跨路径、硬件或 ABI 都能加载，底层拒绝后重新编译。模型权重、page cache、KV tensors、CUDA Graph 和 CUDA context 不属于可搬运资产。

## 冻结热点与合同

```bash
python tools/startup_assets.py publish-hotspots /path/to/offline_profile
python tools/startup_assets.py publish-contract /path/to/finalized_contract
```

热点来源需有原运行 manifest 和 `cuda_graph_hits.csv`；可附 `workload.json` 记录数据来源、RPS、请求数和长度分布。不同 workload 分开存放；使用发布命令返回的目录设置既有 `VLLM_CUDA_GRAPH_HOTSPOT_DIR`。未提供 workload 的表只代表人工选择的通用 profile，不声称自动识别在线 RPS。

预捕获规则不变：先取累计命中 99% 的前缀，再过滤 hits<10，补入 1、2，并限制到可捕获集合。线上新增统计位于 runs，只有显式发布才更新冻结资产。

合同仍需原 finalizer 审核和执行验证；统一仓库不自动批准 candidate。现有 D0/D1/D2 合同若只能依赖环境标签确定缓存状态，将被拒绝并在动作执行前回退，而不是假装已探测。

## 验证范围

`tests/compile/test_startup_assets.py` 覆盖路径优先级、校验、并发写入、受信任导入、迁移、步长/别名/符号边界与编译未命中回退。`tests/compile/startup_asset_smoke.py` 用两个短请求验证真实冷/热启动，不是性能基准。

真实验证记录在 `A100_0606/unified_assets_20261013/`。单卡短测不代替 TP/PP、persistent shell 或 mixed capture 的完整压力回归；这些优化没有因统一仓库而自动扩大支持范围。

2026-10-13 已修复 Dynamo frontend 与新输入校验的兼容性回归。前端包增加 `fake_evidence.graphpickle`：所有节点示例张量及后端入口张量先转为 FakeTensor，再在同一包中保存其元信息，保留共享符号、别名、步长和偏移，不保存权重数值。恢复后比较元信息契约，并在同一 FakeTensorMode/ShapeEnv 中调用后端，继续执行原有子图输入校验。数据相关常量、未知类型和旧的 direct-symbol-only 包不被猜测恢复，而是回退 tracing。

Qwen3-4B/14B 单卡短测均实际命中前端，未回退 tracing，输出 token IDs 与关闭前端的参考一致。资产损坏和配置变化测试记录在 `A100_0606/unified_assets_20261013/frontend_fix_v3/`。这些测试不将 exact-config 前端缓存扩大为通用 Dynamo guard cache，也未改变默认关闭状态。
