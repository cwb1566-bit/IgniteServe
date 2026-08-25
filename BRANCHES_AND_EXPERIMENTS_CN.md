# IgniteServe 分支、部署与实验使用说明

> 快照日期：2026-08-25
>
> vLLM 项目：`/home/wenbiao/vllm_my_A100`
>
> 配套 PyTorch 项目：`/home/wenbiao/IgniteServe-Pytorch`
> 主要验证环境：Python 3.10、PyTorch 2.7.0+cu126、CUDA 12.6、A100 PCIe 40 GB

本文档用于回答三个问题：当前有哪些分支、每个分支实现了什么、如何手动部署并复现实验。性能数据只代表文中给出的模型、硬件和参数，不应外推为所有模型或 GPU 上的固定收益。

## 1. 先看结论

- 单卡三项基础优化使用 `igniteServe/single-gpu-startup-optimizations`。该分支已经实现元数据感知的 canonical FX 多进程并行编译；它不是到 distributed 分支才新增的功能。
- TP=2 / PP=2 分布式启动实验使用 `igniteServe/distributed-startup-optimizations`。
- 需要分布式优化、通用 Engine Shell 复用、权重 I/O 流水和权重/编译重叠时，使用最新的 `igniteServe/persistent-engine-shell`。它继承前两个分支，而不是独立于它们。
- 真正的多进程完整 FX 图并行编译需要配套的修改版 `IgniteServe-Pytorch`。单卡分支对应 `IgniteServe-Pytorch@4dd0808`；只使用未经修改的官方 PyTorch 2.7 时，S4、DAG、元数据复用和自适应 CUDA Graph 仍可工作，但必须关闭 `VLLM_ENABLE_SUBPROCESS_COMPILE`。
- `main` 是上游 vLLM 参考分支，不包含 IgniteServe 冷启动优化，不应用作优化实验分支。

## 2. 分支继承关系

```text
main (上游 vLLM 参考)

igniteServe/single-gpu-startup-optimizations
  └─ igniteServe/distributed-startup-optimizations
       └─ igniteServe/persistent-engine-shell
```

当前服务器上的精确提交如下：

| 分支 | 提交 | 定位 |
|---|---|---|
| `main` | `ca5fb4bbd` | 上游 vLLM 参考，不含 IgniteServe 优化 |
| `igniteServe/single-gpu-startup-optimizations` | `a984d55b6` | 单卡三项基础优化、S4 资源复用与 canonical FX 多进程并行编译 |
| `igniteServe/distributed-startup-optimizations` | `af678326b` | 在基础分支上扩展 TP/PP 启动优化 |
| `igniteServe/persistent-engine-shell` | 见远程最新提交 | 最新超集：分布式优化 + 通用引擎复用 + I/O/编译重叠 |

配套 IgniteServe-Pytorch 分支：

| IgniteServe 分支 | IgniteServe-Pytorch 分支 | 提交 |
|---|---|---|
| `igniteServe/single-gpu-startup-optimizations` | `igniteServe/single-gpu-startup-optimizations` | `4dd0808` |
| `igniteServe/distributed-startup-optimizations` | `igniteServe/distributed-startup-optimizations` | `c9e68dc` |
| `igniteServe/persistent-engine-shell` | `igniteServe/persistent-engine-shell` | `6e905ed` |

2026-08-25 已通过仓库专用 SSH deploy key 将实验分支推送到相应的私有 GitHub 仓库。新机器配置读取权限后可直接获取：

```bash
git fetch --all --prune
git switch igniteServe/persistent-engine-shell
```

## 3. 各分支实现内容与收益

### 3.1 `igniteServe/single-gpu-startup-optimizations`：单卡三项基础优化

该分支包含论文主体的三项冷启动技术。

#### A. 状态感知启动 DAG

- 从 EngineCore 构造开始组织环境、模型初始化、权重预取/加载、编译、KV Cache、CUDA Graph 和运行时管理器。
- 当前完整单卡计划约 25 个节点。
- 支持权重 page-cache 异步预热，并在真实 `load_weights()` 前停止外部读取，减少 I/O 竞争。
- DAG 的主要价值是依赖正确性和可调度框架；单独启用 DAG 的稳定加速有限，不能把后续编译或 CUDA Graph 收益算给 DAG。

#### B. 元数据感知的编译优化、S4 与并行编译

- 将原生 `profile_run()` 的“编译 + 执行 + 峰值显存探测”拆成 compile-only 和 KV profile 两个可调度阶段。
- canonical 子图复用：Qwen/Llama 的重复 decoder 子图只编译少量规范图，其余子图别名复用。
- 单卡分支已经能够把 canonical GraphModule、FakeTensor/SymInt 输入和编译上下文提交给修改版 PyTorch 的 `FullFxCompilePool`，在隔离进程中并行执行完整 `compile_fx -> AOTAutograd -> Inductor` 路径，再由父进程加载产物并绑定 follower 子图。
- 并行编译通过 `VLLM_ENABLE_SUBPROCESS_COMPILE=1` 独立开启；它主要用于 cold compile miss，warm compile cache 时应关闭以避免无任务进程池开销。
- 持久化并恢复 piecewise 子图输入、fake-run 和数据流元数据。
- 以语义键复用 Triton `.best_config`，跳过兼容配置的在线 autotune benchmark。
- 复用 transformed TorchDynamo code artifact，减少反编译/产物生成开销。
- 复用 KV 峰值显存探测结果，兼容命中时跳过真实 `profile_run()`。
- 支持与模型绝对部署路径无关的 cache aliasing。

Qwen3-14B、FP16、TP=1、A100 的代表数据：

| 路径 | compile + KV profile | EngineCore Total Init | LLM 初始化 |
|---|---:|---:|---:|
| 原生 cold | 97.1154 s | 111.2342 s | 118.3785 s |
| S4 | 26.1260 s | 39.8928 s | 47.0618 s |

S4 相对原生 cold 将编译和探测主指标降低 70.9894 s（73.1%）。阶段分离 S0 本身基本中性，主要收益来自 canonical 复用、transformed-code 复用和 KV profile 复用。

#### C. 热点感知、自适应 CUDA Graph

- 离线统计 runtime shape 命中分布，只预捕获高频图。
- 运行时支持 exact replay、受成本和最大填充比例约束的向上填充、eager 回退。
- 缺失 decode shape 可在 idle 或满足 SLO slack 的机会中按需捕获。
- 向下拆分和完整 forward 多流并行没有进入正式 serving 路径，因为实测未获得稳定收益或无法满足安全条件。

Llama3-8B、ShareGPT、A100 的代表数据：

| 指标 | 原生 67 图 | 自适应 14 图 | 变化 |
|---|---:|---:|---:|
| CUDA Graph 预捕获 | 3.5481 s | 1.0421 s | -70.63% |
| EngineCore Total Init | 16.3070 s | 13.6799 s | -16.11% |
| Burst P99 TTFT | 17.7651 s | 15.1256 s | -14.86% |
| Steady P99 TPOT | 20.804 ms | 20.902 ms | +0.47% |

### 3.2 `igniteServe/distributed-startup-optimizations`：TP/PP 扩展

该分支继承单卡三项优化，并增加：

- TP/PP 可执行启动 DAG 和 rank/stage 感知依赖。
- 分布式自适应 CUDA Graph 预捕获；每个 rank/stage 仍捕获自己的本地图。
- 将单卡已有的进程隔离式完整 FX canonical 图并行编译适配到 TP/PP rank 与 pipeline stage。
- Full-FX worker import 与 CUDA 预热前移、异步 compiler identity/hash 预热、快速进程组关闭。
- topology-aware sharded checkpoint：TP rank 或 PP stage 只读取自己的权重分片。
- rank 内权重加载与结构 tracing/compile-only 重叠；若 best-config miss，autotune 在 `weights_ready` 屏障等待最终权重。

Llama3-8B、FP16、两张 A100 PCIe 40 GB、`max_model_len=2048` 的最佳累计结果：

| 拓扑与缓存 | 严格原生 | 分布式优化 | 减少 |
|---|---:|---:|---:|
| TP=2，模型/编译均冷 | 76.6489 s | 20.6014 s | 56.0475 s（73.1%） |
| TP=2，模型/编译均热 | 38.7347 s | 14.8314 s | 23.9033 s（61.7%） |
| PP=2，模型/编译均冷 | 51.2825 s | 17.7271 s | 33.5554 s（65.4%） |
| PP=2，模型/编译均热 | 30.4054 s | 8.8662 s | 21.5392 s（70.8%） |

上述最佳值包含 topology-sharded checkpoint、权重/编译重叠和两阶段 Full-FX 预热。若使用普通 Hugging Face checkpoint，权重加载收益会低于表中结果。

### 3.3 `igniteServe/persistent-engine-shell`：最新超集

该分支继承分布式分支，并新增两种必须严格区分的引擎复用语义。

#### 推荐：`general_shell_rebind`

保留拓扑级基础设施：

- EngineCore、worker PID、RPC/IPC 和共享内存；
- CUDA context；
- 固定 TP/PP 拓扑的 NCCL communicator 与 P2P 状态；
- persistent Full-FX compiler worker 进程。

每次模型切换重新构造：

- `VllmConfig`、模型模块和权重；
- 内存中的 Dynamo/Inductor callable 与 piecewise graph；
- KV profile/allocation 和 attention binding；
- CUDA Graph、scheduler、batch queue 与请求状态。

最新单卡 Llama3-8B、`max_model_len=2048` 实测：

| 缓存状态 | 完整新建实例 | 通用 shell 切换 | 单次收益 |
|---|---:|---:|---:|
| 模型与编译缓存充分预热 | 14.63 s | 12.69 s | 1.94 s（13.3%） |
| 模型与编译缓存均冷 | 19.03 s | 19.33 s | 无稳定收益，约 -1.6% |

第一次创建 shell 仍支付完整启动成本；收益从第二个模型实例开始。跨架构 Llama3-8B -> Qwen2.5-14B -> Llama3-8B，以及固定 TP=2/PP=2 的 A -> B -> A 已通过正确性测试，但尚无严格对齐的通用 TP/PP 多轮性能结论。

#### 历史专用实验：`inplace_weights`

该模式只适用于同结构模型，并保留编译运行时、KV Cache 与 CUDA Graph，只替换权重。它在单卡/TP2/PP2 曾测得约 49.8%～87.8% 的切换收益，但语义接近“热模型骨架”，不能用于回答任意模型实例切换的收益，也不是当前推荐的通用模式。

#### 最新 I/O 与依赖优化

- 下一 shard 异步 read-ahead；TP/PP 下支持 leader/shared 或 rank-local shard 策略。
- dense in-place loader 下，权重读取与值无关的 tracing/lowering/materialization 并行。
- best-config miss 时，只在真正需要运行 kernel 的 autotune 前等待权重；KV profile、CUDA Graph 和推理始终在权重完成后执行。
- maxlen=512 的旧 I/O 消融中，权重加载下降约 20%～22%，完整切换下降约 7%。

## 4. 从零部署

### 4.1 克隆私有仓库

新机器必须先配置对两个私有 GitHub 仓库的 SSH key 或 Personal Access Token。不要把 token 写进脚本或文档。

```bash
git clone git@github.com:cwb1566-bit/IgniteServe.git
git clone git@github.com:cwb1566-bit/IgniteServe-Pytorch.git
```

如果实验分支已经推送：

```bash
cd IgniteServe
git fetch --all --prune
git switch igniteServe/persistent-engine-shell

cd ../IgniteServe-Pytorch
git fetch --all --prune
git switch igniteServe/persistent-engine-shell
```

在当前服务器已有 checkout 时直接：

```bash
cd /home/wenbiao/vllm_my_A100
git status --short
git switch igniteServe/single-gpu-startup-optimizations
# 或：git switch igniteServe/distributed-startup-optimizations
# 或：git switch igniteServe/persistent-engine-shell
```

切换前先处理自己的未提交修改。不要用 `git reset --hard` 清理实验文件。

### 4.2 创建 Python 环境

```bash
conda create -n igniteserve python=3.10 -y
conda activate igniteserve

pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu126
```

### 4.3 安装 IgniteServe-Pytorch 修改

推荐在专用 Conda 环境对官方 PyTorch 2.7 wheel 安装 guarded Python overlay，速度远快于完整重编 PyTorch：

先切换到与 IgniteServe 相同名称的配套分支。单卡、distributed 和 persistent 三个版本分别对应同名分支；其中单卡分支固定在 `4dd0808`，已经包含真正的完整 FX 并行编译实现，并非未经修改的官方 PyTorch：

```bash
cd /path/to/IgniteServe-Pytorch
git switch igniteServe/single-gpu-startup-optimizations
# 或：git switch igniteServe/distributed-startup-optimizations
# 或：git switch igniteServe/persistent-engine-shell
python tools/install_igniteserve_overlay.py --python "$CONDA_PREFIX/bin/python"

python - <<'PY'
import torch
from torch._inductor.compile_fx_full_subproc import FullFxCompilePool
print(torch.__version__, FullFxCompilePool)
PY
```

恢复 overlay：

```bash
python tools/install_igniteserve_overlay.py \
  --python "$CONDA_PREFIX/bin/python" --restore
```

### 4.4 编译/安装 IgniteServe

```bash
cd /path/to/IgniteServe
python use_existing_torch.py
pip install -r requirements/build.txt
pip install -r requirements/cuda.txt
pip install -e . --no-build-isolation
```

验证实际加载的是当前源码：

```bash
PYTHONPATH=$PWD python - <<'PY'
import pathlib, torch, vllm
print("torch:", torch.__version__, pathlib.Path(torch.__file__).resolve())
print("vllm:", vllm.__version__, pathlib.Path(vllm.__file__).resolve())
print("CUDA:", torch.version.cuda, "GPUs:", torch.cuda.device_count())
PY
```

## 5. 两类缓存和统一资源目录

不要混淆以下两类数据：

```bash
export IGNITESERVE_ROOT=/path/to/IgniteServe
export RESOURCE_ROOT=$IGNITESERVE_ROOT/vllm_startup_resources
export CACHE_ROOT=$IGNITESERVE_ROOT/benchmark_results/runtime_cache
```

- `VLLM_STARTUP_RESOURCE_ROOT=$RESOURCE_ROOT`：IgniteServe 语义资源，包括 `parallel_compile_metadata/`、`best_config/`、`transformed_code/`、`kv_profile/`、`cuda_graph_hotspots/` 和启动 DAG 计划。
- `VLLM_CACHE_ROOT=$CACHE_ROOT`：原生 vLLM/TorchInductor 本地编译缓存。它不等价于 S4 语义资源根目录。
- `VLLM_CUDA_GRAPH_HOTSPOT_DIR=<exact-dir>`：单实例加载和导出同一个 workload profile 的精确目录。
- `VLLM_CUDA_GRAPH_HOTSPOT_ROOT=<parent-dir>`：按需多模型服务的父目录；客户端 hotspot key 与服务端参数共同生成子目录。

资源必须由 model architecture、dtype、TP/PP、GPU/compiler 配置等 manifest/hash 校验。不要仅按文件名强制跨环境复制。

## 6. 手动运行单卡实验

### 6.1 公共环境

```bash
cd /path/to/IgniteServe
conda activate igniteserve

export IGNITESERVE_ROOT=$PWD
export MODEL=/models/Meta-Llama-3-8B-Instruct
export RESOURCE_ROOT=$PWD/vllm_startup_resources
export CACHE_ROOT=$PWD/benchmark_results/manual_cache
mkdir -p "$RESOURCE_ROOT" "$CACHE_ROOT"

export PYTHONPATH=$PWD
export VLLM_USE_V1=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_STARTUP_RESOURCE_ROOT=$RESOURCE_ROOT
export VLLM_CACHE_ROOT=$CACHE_ROOT
```

### 6.2 原生/S0 对照

严格原生耦合路径：

```bash
CUDA_VISIBLE_DEVICES=0 \
VLLM_ENABLE_STARTUP_OPTIMIZATIONS=0 \
VLLM_STARTUP_COMPILE_MODE=native \
VLLM_STARTUP_DAG_MODE=off \
VLLM_ENABLE_ADAPTIVE_CUDAGRAPH=0 \
python cold_test.py --model "$MODEL" --dtype float16 --tp 1
```

只进行“compile-only + KV profile”阶段分离、禁用复用的 S0：

```bash
CUDA_VISIBLE_DEVICES=0 \
VLLM_ENABLE_STARTUP_OPTIMIZATIONS=0 \
VLLM_STARTUP_COMPILE_MODE=s0 \
VLLM_DISABLE_STRUCTURAL_REUSE=1 \
VLLM_STARTUP_DAG_MODE=off \
VLLM_ENABLE_ADAPTIVE_CUDAGRAPH=0 \
python cold_test.py --model "$MODEL" --dtype float16 --tp 1
```

### 6.3 单卡 S4 + DAG + 自适应 CUDA Graph

```bash
export HOTSPOT_DIR=$RESOURCE_ROOT/cuda_graph_hotspots/llama8b_manual_len2048_gpuutil0.9
mkdir -p "$HOTSPOT_DIR"

CUDA_VISIBLE_DEVICES=0 \
VLLM_ENABLE_STARTUP_OPTIMIZATIONS=1 \
VLLM_STARTUP_COMPILE_MODE=s4 \
VLLM_STARTUP_DAG_MODE=execute \
VLLM_ENABLE_WEIGHT_PAGECACHE_PREWARM=1 \
VLLM_ENABLE_ADAPTIVE_CUDAGRAPH=1 \
VLLM_CUDA_GRAPH_HOTSPOT_DIR=$HOTSPOT_DIR \
python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 --port 8000 \
  --model "$MODEL" --served-model-name Llama3-8B \
  --max-model-len 2048 --gpu-memory-utilization 0.9
```

第一次兼容 miss 会生成资源，不应当作 warm-resource 正式结果。先 seed，再用同一资源目录和独立 compile cache 做正式多轮测试。

### 6.4 真正的 canonical FX 并行编译

该能力已经在 `igniteServe/single-gpu-startup-optimizations` 中实现，并由后续两个分支继承。只在同名 IgniteServe-Pytorch overlay 已验证、且预计是 cold compile miss 时启用：

```bash
export VLLM_ENABLE_SUBPROCESS_COMPILE=1
export VLLM_ENABLE_PARALLEL_COMPILE_METADATA=1
export TORCHINDUCTOR_FULL_FX_COMPILE_WORKERS=4
export TORCHINDUCTOR_FX_COMPILE_INNER_THREADS=1
export TORCHINDUCTOR_FX_COMPILE_WORKER_IMPORTS=vllm._custom_ops
export TORCHINDUCTOR_FX_COMPILE_PREWARM_CUDA=1
```

已知 warm compile cache 时通常关闭 `VLLM_ENABLE_SUBPROCESS_COMPILE`，避免创建无任务的进程池。Llama3-8B 单卡四个 canonical jobs 的最终研究结果为：materialization 约 4.23 s、Model Compilation 14.5824 s、EngineCore Total Init 18.6148 s；这些数字依赖修改后的 PyTorch、模型结构和冷缓存状态。

## 7. 手动运行 TP=2 / PP=2

### 7.1 推荐环境

```bash
export VLLM_ENABLE_DISTRIBUTED_STARTUP_DAG=1
export VLLM_STARTUP_DAG_MODE=execute
export VLLM_ENABLE_STARTUP_OPTIMIZATIONS=1
export VLLM_STARTUP_COMPILE_MODE=s4
export VLLM_ENABLE_WEIGHT_PAGECACHE_PREWARM=1
export VLLM_ENABLE_WEIGHT_COMPILE_OVERLAP=1
export VLLM_ENABLE_TWO_PHASE_FULL_FX_PREWARM=1
export VLLM_ENABLE_ASYNC_COMPILER_HASH_PREWARM=1
export TORCHINDUCTOR_FULL_FX_FAST_SHUTDOWN=1
export VLLM_ENABLE_ADAPTIVE_CUDAGRAPH=1
export VLLM_ENABLE_DISTRIBUTED_ADAPTIVE_CUDAGRAPH=1
export VLLM_CUDA_GRAPH_HOTSPOT_TOP_K=2

export VLLM_ENABLE_SUBPROCESS_COMPILE=1
export TORCHINDUCTOR_FULL_FX_COMPILE_WORKERS=2
export TORCHINDUCTOR_FX_COMPILE_INNER_THREADS=1
export TORCHINDUCTOR_FX_COMPILE_WORKER_IMPORTS=vllm._custom_ops
export TORCHINDUCTOR_FX_COMPILE_PREWARM_CUDA=1
```

TP=2：

```bash
CUDA_VISIBLE_DEVICES=0,1 python \
  A100_0606/distributed_startup_exploration/distributed_smoke.py \
  --model "$MODEL" --tp 2 --pp 1 \
  --max-model-len 2048 --gpu-memory-utilization 0.8
```

PP=2：

```bash
CUDA_VISIBLE_DEVICES=0,1 python \
  A100_0606/distributed_startup_exploration/distributed_smoke.py \
  --model "$MODEL" --tp 1 --pp 2 \
  --max-model-len 2048 --gpu-memory-utilization 0.8
```

如果使用预先生成的 topology-sharded checkpoint，再增加：

```bash
export VLLM_ENABLE_V1_SHARDED_STATE=1

# 示例：
--load-format sharded_state \
--sharded-pattern 'model-global-rank-{global_rank}-part-{part}.safetensors'
```

`run_distributed_startup_ablation.py` 没有 argparse `--help`，直接运行会启动完整矩阵。它还写死了当前服务器的项目、Python、模型和 GPU 路径。手动复现前先修改文件顶部常量，或通过下列环境变量缩小实验：

```bash
IGNITESERVE_ABLATION_OUTPUT=$PWD/benchmark_results/tp2_d3 \
IGNITESERVE_ABLATION_REPEATS=3 \
IGNITESERVE_ABLATION_TOPOLOGIES=tp2 \
IGNITESERVE_ABLATION_VARIANTS=D3_adaptive_cudagraph \
IGNITESERVE_ABLATION_CACHE_MODE=paired \
python A100_0606/distributed_startup_exploration/run_distributed_startup_ablation.py
```

可选 variant 包括 `B0_native`、`D1_dag`、`D2_parallel_compile`、`D3_adaptive_cudagraph` 以及后续 D4-D8 探索项。

## 8. 手动运行通用 Engine Shell 模型切换

### 8.1 直接 A -> B -> A 正确性/计时

```bash
git switch igniteServe/persistent-engine-shell

export VLLM_ENABLE_PERSISTENT_ENGINE_SHELL=1
export VLLM_PERSISTENT_SHELL_REBIND_MODE=general
export VLLM_PERSISTENT_SHELL_INPLACE_WEIGHTS=0
export VLLM_PERSISTENT_SHELL_CLEAR_INMEM_COMPILE=1
export VLLM_ENABLE_WEIGHT_PAGECACHE_PREWARM=1
export VLLM_ENABLE_WEIGHT_SHARD_PIPELINE=1
export VLLM_ENABLE_WEIGHT_COMPILE_OVERLAP=1
export VLLM_PERSISTENT_SHELL_ALLOW_WEIGHT_COMPILE_OVERLAP=1

CUDA_VISIBLE_DEVICES=0 python \
  A100_0606/persistent_shell/benchmark_general_shell_switch.py \
  --model-a /models/Meta-Llama-3-8B-Instruct \
  --model-b /models/Qwen2.5-14B-Instruct \
  --cache-root "$CACHE_ROOT" \
  --hotspot-root "$RESOURCE_ROOT/cuda_graph_hotspots" \
  --cache-state warm \
  --tp 1 --pp 1 --max-model-len 2048 \
  --gpu-memory-utilization 0.8
```

`--cache-state cold` 会在每次 rebind 前尝试驱逐目标 safetensors page cache 并删除该测试 cache root 下的编译缓存。只对专用实验目录使用该选项。

### 8.2 按需多模型服务

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=$PWD \
VLLM_USE_V1=1 \
VLLM_ENABLE_STARTUP_OPTIMIZATIONS=1 \
VLLM_STARTUP_COMPILE_MODE=s4 \
VLLM_STARTUP_DAG_MODE=execute \
VLLM_ENABLE_PERSISTENT_ENGINE_SHELL=1 \
VLLM_PERSISTENT_SHELL_REBIND_MODE=general \
VLLM_PERSISTENT_SHELL_INPLACE_WEIGHTS=0 \
VLLM_ENABLE_WEIGHT_PAGECACHE_PREWARM=1 \
VLLM_ENABLE_WEIGHT_SHARD_PIPELINE=1 \
VLLM_ENABLE_WEIGHT_COMPILE_OVERLAP=1 \
VLLM_PERSISTENT_SHELL_ALLOW_WEIGHT_COMPILE_OVERLAP=1 \
VLLM_STARTUP_RESOURCE_ROOT=$RESOURCE_ROOT \
VLLM_CUDA_GRAPH_HOTSPOT_ROOT=$RESOURCE_ROOT/cuda_graph_hotspots \
python multi_model_server.py \
  --host 0.0.0.0 --port 8089 \
  --keepalive 30 --max-concurrent-models 1 \
  --model /models/Meta-Llama-3-8B-Instruct \
  --max-model-len 2048 --gpu-memory-utilization 0.9
```

客户端通过请求体中的 `model` 选择模型。请求从 Llama 切换到 Qwen 时，服务等待当前 shell 请求排空，然后原地 rebind，不新建 worker 进程：

```bash
curl -N http://127.0.0.1:8089/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "/models/Qwen2.5-14B-Instruct",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 32,
    "stream": true
  }'
```

当前通用 shell 只允许固定拓扑复用：单卡 shell 只能切换到单卡模型，TP=2 shell 只能切换到 TP=2。改变 TP/PP/world size 或设备映射必须重建 shell。

## 9. 手动生成和复用 CUDA Graph hotspot

服务端只设置统一父目录：

```bash
export VLLM_CUDA_GRAPH_HOTSPOT_ROOT=$RESOURCE_ROOT/cuda_graph_hotspots
```

客户端发送 `X-VLLM-HOTSPOT-KEY`，服务端再追加自己掌握的 `max_model_len` 和 `gpu_memory_utilization`。例如客户端 key：

```text
Llama3-8B__rps__hot__req2000__rps10
```

会对应：

```text
$RESOURCE_ROOT/cuda_graph_hotspots/
  Llama3-8B__rps__hot__req2000__rps10__len2048__gpuutil0.9/
    cuda_graph_hits.csv
    cuda_graph_hits_top99.csv
    cuda_graph_profile_manifest.json
    cuda_graph_cost_table.csv        # 启用成本采样时
```

手动 RPS 客户端：

```bash
python benchmarks_custom/vllm_benchmark_rps.py \
  --base-url http://127.0.0.1:8089 \
  --model "$MODEL" \
  --dataset benchmarks_custom/ShareGPT_validated_2000.json \
  --num-prompts 2000 --max-workers 50 --request-rate 10 \
  --seed 20260810 \
  --hotspot-key Llama3-8B__rps__hot__req2000__rps10 \
  --output-csv benchmark_results/llama8b-rps10.csv
```

离线 material 阶段关闭动态捕获，完成 workload 后优雅停止服务以导出 CSV；正式 online 阶段使用相同 key 和目录并启用自适应捕获：

```bash
# 离线分析
export VLLM_DISABLE_DYNAMIC_CUDA_GRAPH_CAPTURE=1

# 正式推理
unset VLLM_DISABLE_DYNAMIC_CUDA_GRAPH_CAPTURE
export VLLM_ENABLE_ADAPTIVE_CUDAGRAPH=1
```

自动化入口：

```bash
MODEL_ROOT=/models GPU=0 tools/run_igniteserve_e2e.sh --stage material
MODEL_ROOT=/models GPU=0 tools/run_igniteserve_e2e.sh --stage formal
```

## 10. 冷/热缓存的严格构造

### 冷模型 page cache

不要在共享服务器执行全局 `drop_caches`。只驱逐目标模型文件：

```bash
python A100_0606/persistent_shell/evict_model_pagecache.py "$MODEL"
```

用 `fincore` 检查驻留量：

```bash
fincore --bytes --output RES "$MODEL"/*.safetensors
```

### 热模型 page cache

```bash
vmtouch -t -q "$MODEL"/*.safetensors
```

没有 `vmtouch` 时可顺序读取文件，但不要与正式计时重叠，除非实验目标就是测试异步预热。

### 冷编译缓存

必须使用专用测试目录，并只删除该目录下的缓存：

```bash
test -n "$CACHE_ROOT" && test "$CACHE_ROOT" != / && \
  rm -rf "$CACHE_ROOT/torch_compile_cache" "$CACHE_ROOT/torch_extensions"
```

不要同时删除 `VLLM_STARTUP_RESOURCE_ROOT`，除非实验明确要求“无任何离线语义资源”。论文中的“cold compile cache + offline S4 resources”通常保留 best-config、metadata、transformed code 和 KV profile，只清理本地 Inductor/vLLM compile cache。

### 热编译缓存

先用完全相同的模型、dtype、TP/PP、长度预算、compiler/PyTorch/GPU 配置 seed 一次，随后保留 `VLLM_CACHE_ROOT`。必须通过日志中的 cache hit、canonical loaded/reused、best-config selected、transformed-code hit 和 KV-profile hit 确认，而不能只看时间。

## 11. 计时口径与验收

建议至少记录：

- `[TIMER] Init execution environment`
- `[TIMER] Init Model`
- `[TIMER] Loading Weights` 或 `load_weights`
- `[TIMER] Model Compilation`
- `[TIMER] KV Cache Init` / `KV Profile` / `KV Cache Allocation`
- `[TIMER] CUDA Graph Pre-capture`
- `[TIMER] Others`
- `[TIMER] Total Init`
- `init_seconds`
- 首次请求 TTFT、稳态 TPOT、P99 TPOT 和吞吐

实验规则：

1. 每个正式点至少三轮，交替执行基线与优化，报告中位数。
2. 每轮保存完整 stdout/stderr，不能只保存 CSV 汇总。
3. 检查 return code、生成输出、worker PID、cache hit 和 CUDA Graph shape，而不只比较计时。
4. 对 shell rebind，首次 shell 创建与后续切换必须分开报告。
5. 对 TP/PP，记录 GPU 拓扑、P2P/NVLink 状态和 checkpoint 是否 topology-sharded。

推荐正确性回归：

```bash
MODELS_ROOT=/models \
MODELS='Qwen3-8B Llama2-7B OPT-6.7B' \
GPU=0 \
tools/run_coldstart_correctness.sh

pytest -q \
  tests/v1/worker/test_adaptive_cudagraph.py \
  tests/v1/worker/test_cudagraph_batch_partition.py \
  tests/v1/worker/test_cudagraph_profile.py \
  tests/v1/worker/test_graph_matcher.py
```

## 12. 已知限制

- 项目是研究原型，不是所有 vLLM 模型、量化、LoRA、多模态和 speculative decoding 组合的通用生产版本。
- 通用 shell 切换前必须排空请求；当前不是多模型同时驻留方案。
- shell rebind 失败后的 backend rollback 尚未完全事务化。
- 通用 TP/PP shell 已有真实 GPU 正确性覆盖，但仍需正式多轮性能矩阵。
- 并行完整 FX 编译依赖 PyTorch 2.7 私有编译器 API；升级 PyTorch 必须重新适配 GraphPickler、FakeTensor 和 Inductor 生命周期。
- 下行 batch splitting、完整 forward 多流 replay 和持续负载下无约束 capture 未进入正式路径。
- 大型历史脚本中有硬编码服务器路径，运行前必须审查文件顶部常量；不要假设 `--help` 安全，`run_distributed_startup_ablation.py` 和 `run_llama8b_rebind_matrix.py` 当前会直接开始实验。

## 13. 结果与源码索引

- 总体架构：`docs/architecture.md`
- 参数说明：`docs/configuration.md`
- 部署说明：`docs/deployment.md`
- 基础性能：`docs/benchmarks.md`
- 编译分离完整研究记录：`compile_execution_separation.md`
- 分布式结果：`A100_0606/distributed_startup_exploration/DISTRIBUTED_FINE_GRAIN_RESULTS.md`
- 分布式三项消融：`A100_0606/distributed_startup_ablation_dagv4_final_20260822/summary.md`
- 通用 shell 设计与测试：`A100_0606/persistent_shell/general_shell_rebind/README.md`
- 历史 inplace shell 数据：`A100_0606/persistent_shell/llama8b_formal_20260823/summary.md`
- PyTorch 并行编译设计：`IgniteServe-Pytorch/docs/igniteserve_parallel_compile.md`
- PyTorch overlay 部署：`IgniteServe-Pytorch/docs/igniteserve_deployment.md`
