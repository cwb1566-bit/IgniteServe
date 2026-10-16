# Deployment and Startup

[Overview](../README.md) | [Architecture and source guide](architecture.md)

## Runtime and Build

The validated research environment uses Linux, Python 3.10, NVIDIA A100,
CUDA 12.6 and a modified Torch 2.7 runtime. Full-FX process compilation and
concurrent capture require that runtime; its sources/wheels are **not bundled
here**. A public release needs a corresponding pinned runtime distribution or
patch set. A version string alone is not a compatibility check.

After installing the compatible runtime into an isolated environment, build
this checkout against it:

```bash
python use_existing_torch.py
python -m pip install -r requirements/build.txt -r requirements/cuda.txt
python -m pip install -e . --no-build-isolation
```

`use_existing_torch.py` edits dependency files to remove Torch pins; it does not
install the patched runtime. Review that build-local diff rather than committing
it unintentionally. Check which source/runtime the environment actually imports:

```bash
PYTHONPATH="$PWD" python - <<'PY'
import inspect
import torch
import vllm
from torch._inductor.compile_fx_full_subproc import FullFxCompilePool
print('vLLM:', vllm.__file__)
print('Torch:', torch.__version__, torch.__file__)
for name in ('prewarm_workers', 'all_prewarm_futures', 'submit',
             'begin_shutdown_pool', 'wait_for_shutdown_pool', 'shutdown_pool'):
    assert hasattr(FullFxCompilePool, name), name
required = {'synchronize_before_capture', 'collect_before_capture',
            'empty_cache_before_capture', 'manage_default_generator'}
assert required <= set(inspect.signature(torch.cuda.graph).parameters)
PY
```

This is an API check, not a replacement for model tests. Stock Torch can disable
or fall back from unsupported paths; that is not evidence of process-parallel
compilation or concurrent capture. `PYTHONPATH` does not replace native extension
builds. For a second local checkout with identical native sources and a compatible
ABI, `python tools/prepare_local_runtime.py --from-checkout /path/to/built-checkout`
can attach existing artifacts. It is a development helper, not a release installer.

## Launch and Request

Run from the repository root on an available GPU:

```bash
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 VLLM_ENABLE_STARTUP_OPTIMIZATIONS=1 \
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/Qwen3-4B --served-model-name model \
  --max-model-len 2048 --host 127.0.0.1 --port 8000
```

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"model","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'
```

On the existing development server, the environment is `vllm_concurrent_capture`
and the checkout is `/home/wenbiao/data/igniteserve_declarative_startup`
(a view of `/data/wenbiao/igniteserve_declarative_startup`). These paths are not
requirements for other machines. Do not expose the unauthenticated test endpoint
to untrusted networks.

## Defaults and Overrides

The master switch fills **unset** settings at `VllmConfig` construction:

| Function | Setting supplied by the master switch |
|---|---|
| Packaged declarative DAG | `VLLM_STARTUP_DAG_MODE=execute` |
| Frontend reuse / process compilation | `VLLM_ENABLE_DYNAMO_FRONTEND_CACHE=1`, `VLLM_ENABLE_SUBPROCESS_COMPILE=1` |
| Adaptive graph management | `VLLM_ENABLE_ADAPTIVE_CUDAGRAPH=1` |
| Fast compiler shutdown | `TORCHINDUCTOR_FULL_FX_FAST_SHUTDOWN=1` |
| Weight prefetch / compile overlap | `VLLM_ENABLE_WEIGHT_SHARD_PIPELINE=1`, `VLLM_ENABLE_WEIGHT_COMPILE_OVERLAP=1` |
| Compiler prewarm | `VLLM_ENABLE_TWO_PHASE_FULL_FX_PREWARM=1`, `VLLM_ENABLE_ASYNC_COMPILER_HASH_PREWARM=1` |

S4 semantic caches, structural/global reuse, page-cache prewarm and cost-gated
splitting also consult the master policy at their consumers. Explicit overrides
win, including `=0`; a dirty shell can therefore change the minimal command's
behavior. Configuration is process-wide: use fresh processes for different
profiles. Turning the master off alone does not erase prior overrides, select
pristine upstream vLLM, or necessarily disable adaptive capture.

| Common override | Purpose |
|---|---|
| `VLLM_STARTUP_RESOURCE_ROOT=/path` | Read and write assets in that repository. |
| `VLLM_STARTUP_DAG_MODE=off` | Use this fork's sequential startup; not an upstream performance baseline. |
| `VLLM_STARTUP_CONTRACT_PATH`, `VLLM_STARTUP_ADAPTER_PATH` | Select a trusted declaration/adapter pair; both are validated. |
| `VLLM_STARTUP_DAG_MAX_WORKERS=1` | Serialize DAG scheduling for diagnosis, not internal compiler/capture work. |
| `TORCHINDUCTOR_FULL_FX_COMPILE_WORKERS` | Size of the modified Torch compiler pool. |
| `VLLM_ENABLE_ADAPTIVE_CUDAGRAPH=0` | Keep static precapture/replay, disable adaptive on-demand capture. |
| `VLLM_ENABLE_CONCURRENT_PIECEWISE_CUDAGRAPH_CAPTURE=0` | Disable background capture; safe-point maintenance may still run. |
| `VLLM_ENABLE_CUDAGRAPH_SPLIT=0` | Disable splitting without disabling ordinary graph replay. |
| `VLLM_CUDA_GRAPH_HOTSPOT_DIR=/path` | Select a compatible frozen workload profile. |
| `VLLM_ADAPTIVE_CUDAGRAPH_MIN_HITS` | Online trigger threshold, default `1`. |
| `VLLM_CUDA_GRAPH_PRECAPTURE_MIN_HITS` | Offline hit filter, default `10`. |
| `VLLM_FORCE_FULL_CUDAGRAPH_PRECAPTURE=1` | Select all registered capture shapes. |
| `VLLM_LOGGING_LEVEL=DEBUG` | Expand internal timings and progress. Default INFO keeps major summaries. |

V1 is already the default for supported configurations. `VLLM_WORKER_MULTIPROC_METHOD`
selects worker creation (`spawn` versus other methods), not compile parallelism.
The retired `VLLM_DISABLE_DYNAMIC_CUDA_GRAPH_CAPTURE`,
`VLLM_DYNAMIC_CUDA_GRAPH_CAPTURE_NO_REPLAY` and `VLLM_CUDAGRAPH_SPLIT_SHADOW`
flags no longer control the unified path; do not use them to disable capture.

## Assets and Hotspots

Root precedence is `VLLM_STARTUP_RESOURCE_ROOT`, then
`$XDG_CONFIG_HOME/igniteserve/config.json` (normally `~/.config/igniteserve/config.json`),
then `$XDG_CACHE_HOME/igniteserve/assets` (normally `~/.cache/igniteserve/assets`).
To choose `~/igniteserve_assets` when the environment variable is absent, set:

```json
{"startup_resource_root": "~/igniteserve_assets"}
```

That user setting already exists on the development server; it is not a hardcoded
default on every machine. Stage-specific explicit file/directory settings can
override the common root and are outside its export boundary. See the
[asset layout and validation table](architecture.md#asset-repository).

```bash
python tools/startup_assets.py inspect
python tools/startup_assets.py verify
python tools/startup_assets.py export /tmp/startup_assets.zip
VLLM_STARTUP_RESOURCE_ROOT=/new/assets \
  python tools/startup_assets.py import /tmp/startup_assets.zip --trusted
python tools/startup_assets.py publish-hotspots /path/to/offline_profile
```

Import only trusted executable assets. Export excludes `runs/`; checksums detect
corruption, not malicious code. Move the complete repository: global subgraph
entries may reference producer launch directories. Hardware/compiler/guard
validation still applies after migration. Weights, page cache, KV tensors and
live CUDA graphs are not serialized startup assets.

Offline hotspot selection takes the cumulative 99% prefix, filters hits below
10 and unsupported shapes, then adds registered shapes 1 and 2. It does not
refill the discarded tail. An explicit Top-K override changes that selection.
Missing/incompatible profiles use the full registered set, not a guaranteed
67 shapes on every configuration. Publish a compatible offline profile and use
the returned directory as `VLLM_CUDA_GRAPH_HOTSPOT_DIR`. Online observations go
to `runs/` without mutating the frozen profile; separate workloads/RPS explicitly.

## Validation and Debugging

### Distributed Startup and Capture

On this branch, the master startup switch selects a separate distributed
declaration when TP or PP exceeds one. Its worker-RPC capacity is one: each
action fans out to all ranks and waits for all replies before dependent actions.
Weight/compile overlap remains inside the fused worker action.

```bash
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0,1 \
VLLM_ENABLE_STARTUP_OPTIMIZATIONS=1 \
VLLM_ENABLE_DISTRIBUTED_BACKGROUND_CAPTURE=1 \
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/Qwen3-4B --tensor-parallel-size 2 \
  --disable-custom-all-reduce --max-model-len 512 \
  --max-num-seqs 8 --max-num-batched-tokens 64 \
  --enable-chunked-prefill --gpu-memory-utilization 0.45
```

For PP, replace `--tensor-parallel-size 2` with `--pipeline-parallel-size 2`.
The capture flag separates supported TP collectives from recorded pieces;
communication executes only on the foreground inference path. Both startup
and background graphs use stable private activation buffers. Completed graphs
become visible only after all ranks acknowledge them at a drained step boundary.
This publication path requires the default multiprocess EngineCore client.
With PP, continuous traffic may delay publication until the pipeline drains;
the implementation does not interrupt foreground work to force publication.
CUDA stream caching is thread-local. This requires the same modified Torch
runtime as single-GPU concurrent capture; no new Torch patch is required here.

Validation targets Qwen3-4B FP16, single-host TP=2/4 or PP=2/4 with NCCL, not DP,
multi-host, TP+PP combinations, arbitrary models, or a throughput improvement.
Custom all-reduce is not qualified by this acceptance. Rank-local cost-based
padding/splitting and whole-forward composite background capture are disabled
for distributed execution. Ranks use the same registered scheduler padding size
even while falling back to eager; local capture progress cannot change collective
sizes. LoRA, multimodal and hybrid-state restrictions remain.
Private activation buffers and per-piece pools consume additional memory. The
example deliberately reserves headroom and bounds batch sizes; large capture
sets and production memory-utilization settings require separate memory and
performance qualification. This branch does not claim throughput improvement.
Use `tools/validate_distributed_startup.py` for contract acceptance and
`tools/verify_distributed_capture.py --phase mixed` for real capture/replay checks.
The latter accepts `decode` and `prefill` for additional qualification. Shadow
validation is diagnostic only and must be disabled for performance measurements.
The packaged contract includes the two-rank acceptance matrix; four-rank checks
add cold DAG off/on equivalence and decode/prefill/mixed capture/replay checks.
`tools/validate_distributed_startup.py --model MODEL --gpus 0,1,2,3
--cold-timing --output PATH` compares the same optimized runtime with DAG off/on,
not native vLLM. Each case uses empty artifact/compiler caches, unchanged OS page
cache, and initial capture sizes `[1,2]`. It records LLM construction time and
stage timings; it does not measure API frontend startup or full 67-graph capture.

### Timing and Acceptance

INFO retains major completion times, reuse/fallback evidence and graph publication;
DEBUG retains fine-grained timing. JSON timing records and DAG cost measurements
are unaffected by verbosity. The fused weight/compile time includes both tasks;
a later readiness marker is not a second compilation. KV Profile includes any
reported `deferred_shutdown_wait_s`; do not call that entire interval cache I/O.
Overlapping and nested phase times must not be added as total latency.

Run focused CPU tests with the repository's test dependencies installed:

```bash
PYTHONPATH=. python -m pytest --noconftest -q tests/startup_runtime \
  tests/compile/test_startup_defaults.py tests/compile/test_startup_assets.py \
  tests/compile/test_compile_organization.py tests/compile/test_hierarchical_reuse.py \
  tests/compile/test_frontend_metadata.py tests/v1/worker/test_cuda_graph_manager.py \
  tests/v1/worker/test_distributed_capture.py \
  tests/v1/worker/test_cudagraph_adaptation.py tests/v1/worker/test_composite_cudagraph.py
```

For GPU acceptance, use `tools/validate_startup_contract.py` as described in the
[contract workflow](architecture.md#offline-contract-workflow).
`tools/verify_cudagraph_manager.py --model MODEL --gpu ID --phase mixed`
checks warmup, publication, actual replay and eager-reference outputs; repeat
with `prefill` and `decode`. `tools/profile_cudagraph_adaptation.py` calibrates
qualified cost rows. `tools/verify_online_composite.py --model MODEL --gpu ID
--batch 7 --publish` validates/publishes a measured composite profile; a fresh
`--policy-only --costs PATH` run verifies policy admission without synthetic
prices. Consult each tool's `--help`; choose an idle GPU and verify release after
testing. Forced-admission tests are correctness tests, not performance evidence.

Use actual token IDs, matched requests/arrival schedules and isolated cache
states for benchmarks. Separate cold composite construction from warm replay
and exclude startup when measuring steady-state latency. A successful request
or enabled switch does not prove background capture/reuse occurred. The packaged
contract's `validation` records its bounded acceptance, not universal model or
performance guarantees. Historical experiment tables are not current baselines.
