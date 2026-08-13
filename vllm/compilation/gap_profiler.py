import os
import pstats
import time
import cProfile

_profiler = None
_path = None


def start(label: str) -> None:
    global _profiler, _path
    if os.environ.get("VLLM_PROFILE_AFTER_BACKEND_GAP", "").lower() not in {"1", "true", "yes", "on"}:
        return
    out_dir = os.environ.get("VLLM_PROFILE_AFTER_BACKEND_GAP_DIR", "/tmp")
    os.makedirs(out_dir, exist_ok=True)
    ts = int(time.time() * 1000)
    _path = os.path.join(out_dir, f"after_backend_gap_{label}_{ts}.prof")
    _profiler = cProfile.Profile()
    _profiler.enable()


def stop() -> None:
    global _profiler, _path
    if _profiler is None:
        return
    profiler = _profiler
    path = _path
    _profiler = None
    _path = None
    profiler.disable()
    profiler.dump_stats(path)
    text_path = path + ".txt"
    with open(text_path, "w") as f:
        stats = pstats.Stats(path, stream=f).strip_dirs().sort_stats("cumtime")
        stats.print_stats(80)
        f.write("\n--- sort by tottime ---\n")
        stats.sort_stats("tottime").print_stats(80)
