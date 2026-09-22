# Mixed piecewise capture branch defaults

On igniteServe/mixed-piecewise-capture, adaptive CUDA graphs and concurrent
piecewise capture default to enabled. The runtime observation threshold defaults
to 1. Mixed and prefill triggers are enabled only after the existing single-GPU,
model and patched-PyTorch capability checks pass. Shadow validation stays off.

With a compatible hotspot CSV and no explicit TOP_K override, rank all records
by hit count (then capture priority), take the shortest prefix covering at least
99% of hits, then discard entries with fewer than 10 hits or unsupported shapes.
Do not refill the discarded tail. Add registered shapes 1 and 2 unconditionally.
Missing or invalid profiles retain the existing full registered-shape fallback.

Overrides remain available:

* VLLM_ENABLE_ADAPTIVE_CUDAGRAPH=0 disables the adaptive controller.
* VLLM_ENABLE_CONCURRENT_PIECEWISE_CUDAGRAPH_CAPTURE=0 disables background capture.
* VLLM_ADAPTIVE_CUDAGRAPH_MIN_HITS changes the online threshold (default 1).
* VLLM_CUDA_GRAPH_PRECAPTURE_MIN_HITS changes the offline filter (default 10).
* VLLM_FORCE_FULL_CUDAGRAPH_PRECAPTURE=1 selects all registered shapes.
* VLLM_CUDA_GRAPH_HOTSPOT_TOP_K explicitly overrides percentile selection.

To disable all dynamic capture, disable adaptive capture and set
VLLM_DISABLE_DYNAMIC_CUDA_GRAPH_CAPTURE=1 to disable the legacy path as well.
The legacy path is already disabled whenever the adaptive controller is enabled.
Hotspot/model paths and telemetry destinations are deployment-specific, not
hardcoded to the Llama experiment. The separate native v0.9.2 tree is unchanged.

The default-setting change is covered by CPU unit tests; the preceding GPU
performance experiment used the same values as explicit environment overrides.
