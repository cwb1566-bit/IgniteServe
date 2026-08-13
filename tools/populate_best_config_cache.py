#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import torch

from vllm.v1.worker import best_config_cache


def infer_model_name(root: str) -> str | None:
    name = Path(root).name
    prefix = "vllm_kv_profile_populate_"
    if name.startswith(prefix):
        return name[len(prefix):]
    return name or None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Populate the global Triton best_config semantic cache.")
    parser.add_argument("--store", default=str(best_config_cache.store_path()))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--roots", nargs="+", required=True,
                        help="Cache roots or directories containing *.best_config files.")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Optional source model labels, one per root.")
    args = parser.parse_args()

    os.environ["VLLM_BEST_CONFIG_CACHE_FILE"] = args.store
    os.environ["VLLM_ENABLE_BEST_CONFIG_CACHE"] = "1"
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = None

    summaries = []
    total_files = 0
    total_saved = 0
    for i, root in enumerate(args.roots):
        model = args.models[i] if args.models and i < len(args.models) else infer_model_name(root)
        stats = best_config_cache.ingest_best_config_tree(
            root, source_model=model, device=device)
        summaries.append(stats)
        total_files += int(stats.get("files", 0))
        total_saved += int(stats.get("saved", 0))
        print(json.dumps(stats, sort_keys=True))

    store = Path(args.store)
    entries = 0
    if store.exists():
        data = json.loads(store.read_text(encoding="utf-8"))
        entries = len(data.get("entries", {}))
    print(json.dumps({
        "store": str(store),
        "roots": len(args.roots),
        "files": total_files,
        "saved": total_saved,
        "entries": entries,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
