# IgniteServe

IgniteServe is a research inference system from CGCL, Huazhong University of
Science and Technology, based on vLLM 0.9.2. It targets three sources of LLM
startup and graph-execution overhead:

| Area | Implementation |
|---|---|
| Startup orchestration | Load a reviewed declaration and Python adapter; schedule actions by state readiness, resources and downstream cost. |
| Compilation | Restore validated subgraph inputs, compile in isolated processes, and reuse compatible artifacts within and across launches. |
| CUDA graphs | Select hotspot precapture shapes, capture missing decode/prefill/mixed shapes in the background, and apply cost-qualified padding or splitting. |

This branch is `IgniteServe/declarative-startup-runtime`. Its packaged startup
contract has 26 nodes covering EngineCore initialization through engine ready,
not the entire HTTP service. Single-GPU CUDA is the qualified contract scope;
other configurations retain the existing startup path. A switch enables a
policy, not a guarantee that every model or workload qualifies for it.

## Start Here

In an already built checkout with the compatible **modified PyTorch** runtime:

```bash
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 VLLM_ENABLE_STARTUP_OPTIMIZATIONS=1 \
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/Qwen3-4B --served-model-name model \
  --max-model-len 2048 --host 127.0.0.1 --port 8000
```

The repository does not include the modified PyTorch distribution, model
weights or runtime assets. Installing stock PyTorch is not sufficient to
reproduce all optimizations. Follow [deployment and startup](docs/deployment.md)
for prerequisites, common overrides, asset storage and validation.

Read [architecture and source guide](docs/architecture.md) for the three
optimization paths, safety boundaries and engine-porting interface. These two
guides are the maintained IgniteServe documentation; historical branch reports
are available in Git history, not deployment instructions for this branch.

## Upstream and License

The remaining [vLLM documentation](docs/README.md) describes upstream APIs,
models and development tooling; it does not define IgniteServe defaults.
Benchmark/test utilities retain their local usage information and `--help`.
See [LICENSE](LICENSE), [NOTICE](NOTICE), [SECURITY.md](SECURITY.md) and
[CONTRIBUTING.md](CONTRIBUTING.md). Upstream licensing and attribution are retained.
