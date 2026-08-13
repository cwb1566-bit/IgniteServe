import contextlib
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import torch.fx as fx

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
    identity: dict[str, Any] = {"basename": resolved.name}
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


def _normalize_fx_target(target: Any) -> str:
    import re
    return re.sub(r"\d+", "#", str(target))


def graph_semantic_key(graph: fx.GraphModule,
                       runtime_shape: Optional[int],
                       compiler_name: str) -> str:
    nodes = []
    for node in graph.graph.nodes:
        nodes.append({
            "op": node.op,
            "target": _normalize_fx_target(node.target),
            "args": _normalize_fx_target(node.args),
            "kwargs": _normalize_fx_target(node.kwargs),
        })
    payload = {
        "runtime_shape": runtime_shape,
        "compiler": compiler_name,
        "nodes": nodes,
    }
    return _hash_json(payload)


class GlobalSubgraphCache:
    """Small JSON index from semantic subgraph keys to compiler handles."""

    def __init__(self, local_cache_dir: str, prefix: str = "") -> None:
        local = Path(local_cache_dir).resolve()
        try:
            compile_root = local.parents[2]
        except IndexError:
            compile_root = local.parent
        self.path = compile_root / "global_subgraph_cache.json"
        self.prefix = prefix
        self.data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(self.data, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp_path, self.path)

    def lookup(self, key: str) -> Optional[Any]:
        entry = self.data.get(key)
        if isinstance(entry, dict):
            return entry.get("handle")
        return None

    def register(self, key: str, handle: Any, source_cache_dir: str,
                 graph_index: int, runtime_shape: Optional[int]) -> None:
        self.data[key] = {
            "handle": handle,
            "source_cache_dir": source_cache_dir,
            "graph_index": graph_index,
            "runtime_shape": runtime_shape,
            "prefix": self.prefix,
        }
        self.save()
