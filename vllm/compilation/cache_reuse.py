import contextlib
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Optional
from functools import lru_cache

from vllm.compilation.startup_resources import enabled_by_master


def _enabled(name: str) -> bool:
    return enabled_by_master(name)


def _hash_json(payload: Any, length: Optional[int] = None) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return digest if length is None else digest[:length]


def canonical_model_identity(model_path: str) -> str:
    """Return a deployment-path-independent identity for local HF models."""
    path = Path(model_path).expanduser()
    if not path.exists():
        return model_path
    resolved = path.resolve()
    config_path = resolved / "config.json"
    identity: dict[str, Any] = {}
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            config = {}
        for key in (
            "architectures", "model_type", "hidden_size", "intermediate_size",
            "num_attention_heads", "num_hidden_layers", "num_key_value_heads",
            "rope_theta", "vocab_size", "torch_dtype",
        ):
            if key in config:
                identity[key] = config[key]
        identity["config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    index_path = resolved / "model.safetensors.index.json"
    if index_path.exists():
        identity["safetensors_index_sha256"] = hashlib.sha256(index_path.read_bytes()).hexdigest()
    return "canonical-local-model://" + _hash_json(identity)


@contextlib.contextmanager
def normalized_model_paths(vllm_config):
    """Yield an isolated config with deployment paths normalized for hashing.

    Hashing can run concurrently with tokenizer and runtime-manager setup in
    the startup DAG. Mutating the shared ``vllm_config`` here would expose the
    temporary canonical URI to those readers, so normalization is applied to
    shallow copies of the config objects instead.
    """
    if not _enabled("VLLM_ENABLE_COMPILE_CACHE_ALIAS"):
        yield vllm_config
        return

    model_config = getattr(vllm_config, "model_config", None)
    if model_config is None:
        yield vllm_config
        return

    normalized_config = copy.copy(vllm_config)
    normalized_model_config = copy.copy(model_config)
    normalized_config.model_config = normalized_model_config
    hf_config = getattr(model_config, "hf_config", None)
    if hf_config is not None:
        normalized_model_config.hf_config = copy.copy(hf_config)

    attrs = ["model", "served_model_name", "tokenizer"]
    model = getattr(normalized_model_config, "model", None)
    canonical = (canonical_model_identity(model)
                 if isinstance(model, str) else model)
    for attr in attrs:
        if not hasattr(normalized_model_config, attr):
            continue
        value = getattr(normalized_model_config, attr)
        if isinstance(value, str) and value and Path(value).expanduser().exists():
            setattr(normalized_model_config, attr,
                    canonical_model_identity(value))
        elif attr == "model" and canonical is not None:
            setattr(normalized_model_config, attr, canonical)
    normalized_hf_config = getattr(normalized_model_config, "hf_config", None)
    if (normalized_hf_config is not None
            and hasattr(normalized_hf_config, "_name_or_path")):
        normalized_hf_config._name_or_path = canonical
    yield normalized_config


@lru_cache(maxsize=1)
def _native_code_identity():
    from . import startup_assets as assets
    package = Path(__file__).resolve().parents[1]
    return {p.name: assets.file_hash(p) for p in sorted(package.glob("*.so"))}


def compilation_context(vllm_config, compiler, options) -> dict:
    """Conservative backend identity, independent of model name/cache path."""
    import torch
    from torch._inductor import config
    from . import startup_assets as assets

    def stable(value):
        if value is None or type(value) in (str, int, float, bool):
            return value
        if isinstance(value, (tuple, list)):
            return [stable(v) for v in value]
        if isinstance(value, dict) and all(isinstance(k, str) for k in value):
            return {k: stable(v) for k, v in value.items()}
        # vLLM's post-grad pass manager exposes its compilation identity.
        from .inductor_pass import InductorPass
        from .pass_manager import PostGradPassManager
        if isinstance(value, (InductorPass, PostGradPassManager)):
            return {"pass": type(value).__qualname__, "uuid": str(value.uuid())}
        raise ValueError(f"unsupported compilation option: {type(value)}")

    parallel = vllm_config.parallel_config
    return {"backend": compiler.name, "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "compiler": compiler.compute_hash(vllm_config),
            "source": assets.source_hash("compilation", "model_executor"),
            "native_code": _native_code_identity(),
            "runtime": {"grad": torch.is_grad_enabled(),
                        "inference": torch.is_inference_mode_enabled(),
                        "deterministic": torch.are_deterministic_algorithms_enabled(),
                        "default_dtype": str(torch.get_default_dtype()),
                        "matmul_precision": torch.get_float32_matmul_precision(),
                        "fp16_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                        "bf16_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                        "cudnn": {name: getattr(torch.backends.cudnn, name) for name in
                                  ("enabled", "benchmark", "deterministic", "allow_tf32")},
                        "autocast": torch.is_autocast_enabled(),
                        "autocast_dtype": str(torch.get_autocast_dtype("cuda"))},
            "options": stable(options),
            "inductor": stable(config.save_config_portable()),
            "parallel": {"config": parallel.compute_hash(),
                         "rank": parallel.rank,
                         "dp_rank": parallel.data_parallel_rank,
                         "disable_custom_all_reduce": getattr(parallel, "disable_custom_all_reduce", None),
                         "enable_eplb": getattr(parallel, "enable_eplb", None)}}


def subgraph_identity(graph, example_inputs, context, runtime_shape=None):
    """Only graphs with current output evidence are globally reusable."""
    import torch
    from .input_contract import graph_key, input_contract
    if (context.get("options", {}).get("freezing")
            or context.get("inductor", {}).get("freezing")):
        raise ValueError("frozen tensor values require content-aware keys")
    from torch.utils._pytree import tree_leaves
    for node in graph.graph.nodes:
        for argument in tree_leaves((node.args, node.kwargs)):
            if (isinstance(argument, torch._ops.OpOverload)
                    and argument.namespace not in {"aten", "prims", "_C", "_C_cache_ops", "vllm"}):
                raise ValueError("unversioned nested custom operator")
        if node.op != "call_function":
            continue
        target = node.target
        if isinstance(target, torch._ops.OpOverload):
            # Unknown registered kernels may depend on unversioned native code.
            if target.namespace not in {"aten", "prims", "_C", "_C_cache_ops", "vllm"}:
                raise ValueError("unversioned custom operator")
        else:
            module = getattr(target, "__module__", "")
            if not (module.startswith("torch.") or module in {"torch", "operator", "_operator", "builtins"}):
                raise ValueError("unversioned Python code dependency")
    output = next(n for n in graph.graph.nodes if n.op == "output")

    def evidence(node):
        if "val" in node.meta:
            return node.meta["val"]
        return node.meta["example_value"]

    outputs = torch.fx.node.map_arg(output.args[0], evidence)
    placeholders = tuple(evidence(n) for n in graph.graph.nodes if n.op == "placeholder")
    if (input_contract(tuple(example_inputs), canonical=True)["value"]
            != input_contract(placeholders, canonical=True)["value"]):
        raise ValueError("current graph evidence does not match compilation inputs")
    return {"schema": "subgraph-v1", "ir": graph_key(graph),
            # Restored FakeTensors have fresh storages. Use the *current graph's*
            # matched placeholder/output evidence to retain output-view aliases.
            "signature": input_contract((placeholders, outputs),
                                        canonical=True)["value"],
            "context": context, "runtime_shape": runtime_shape}


class GlobalSubgraphCache:
    """Per-key descriptors; compiler-owned payloads remain at their source.

    The repository is trusted executable input, not a sandbox. Checksums detect
    corruption, not malicious replacement. Missing dependencies are cache misses.
    """

    def __init__(self, local_cache_dir: str, prefix: str = "") -> None:
        from .startup_resources import resource_root, resource_dir
        self.root = resource_root()
        self.path = resource_dir("compile") / "subgraphs"
        self.local = Path(local_cache_dir).resolve()
        self.base = self.local
        for _ in Path(prefix).parts:
            self.base = self.base.parent

    def lookup(self, identity) -> Optional[dict]:
        from . import startup_assets as assets
        if identity is None:
            return None
        data = assets.read_record(self.path / f"{assets.digest(identity)}.json",
                                  identity)
        return data if self.verify_entry(self.root, data) else None

    @staticmethod
    def verify_entry(root, data) -> bool:
        from . import startup_assets as assets
        if not isinstance(data, dict):
            return False
        try:
            files = data["files"]
            if not files or not data["candidates"]:
                return False
            if data["handle"][1] not in files:
                return False
            if not all(name in files for name in data["candidates"]):
                return False
            if not all(assets.file_hash(assets.member(root, name)) == sha
                       for name, sha in files.items()):
                return False
            return True
        except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError):
            return False

    def register(self, identity, handle, graph_index, runtime_shape) -> bool:
        from . import startup_assets as assets
        if identity is None:
            return False
        try:
            # Do not publish handles outside the configured repository, or
            # compiler variants without the supported FX-cache representation.
            code = Path(handle[1]).resolve()
            key = handle[0]
            if not isinstance(key, str) or not key.isalnum():
                return False
            fx_root = self.base / "inductor_cache" / "fxgraph" / key[1:3] / key
            candidates = sorted(p for p in fx_root.iterdir() if p.is_file())
            if not candidates:
                return False
            relative = lambda p: p.relative_to(self.root).as_posix()
            files = {relative(p): assets.file_hash(assets.member(self.root, relative(p)))
                     for p in [code, *candidates]}
            data = {"handle": [key, relative(code)], "files": files,
                    "candidates": [relative(p) for p in candidates],
                    "source_cache_dir": relative(self.local),
                    "graph_index": graph_index, "runtime_shape": runtime_shape}
            path = self.path / f"{assets.digest(identity)}.json"
            with assets.locked(path):
                assets.write_json(path, assets.record(identity, data))
            return True
        except (OSError, ValueError, TypeError, IndexError):
            return False

    def payloads(self, entry):
        """Verify the exact bytes used by the loader, not just earlier stats."""
        import hashlib
        from . import startup_assets as assets
        for name in entry["candidates"]:
            data = assets.member(self.root, name).read_bytes()
            if hashlib.sha256(data).hexdigest() != entry["files"][name]:
                raise ValueError("compiled subgraph changed during load")
            yield data
