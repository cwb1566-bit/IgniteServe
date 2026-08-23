import argparse
import asyncio
import copy
import gc
import inspect
import json
import os
import re
import time
from pathlib import Path
import torch

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.weight_pagecache import stop_async_weight_prewarm


HOTSPOT_KEY_HEADER = "x-vllm-hotspot-key"


def normalize_hotspot_key(hotspot_key: str | None) -> str | None:
    if not hotspot_key:
        return None
    normalized = re.sub(r"[^A-Za-z0-9._=-]+", "_", hotspot_key.strip())
    normalized = normalized.strip("._-/")
    return normalized or None


def format_hotspot_value(value) -> str:
    if isinstance(value, float):
        text = f"{value:.6g}"
    else:
        text = str(value)
    return re.sub(r"[^A-Za-z0-9._=-]+", "_", text)


def hotspot_root_dir() -> Path:
    root = os.environ.get("VLLM_CUDA_GRAPH_HOTSPOT_ROOT")
    if root:
        return Path(root)
    from vllm.compilation.startup_resources import resource_dir
    return resource_dir("cuda_graph_hotspots")


def reset_hotspot_analyzer_state(analyzer) -> None:
    analyzer.stats.clear()
    analyzer.recent_sizes.clear()
    analyzer.total_requests = 0
    analyzer.triggered_thresholds.clear()
    analyzer.pending_captures.clear()
    analyzer.has_exported_final = False


class TemplateManager:
    @staticmethod
    def get_prompt_and_stop(model_name: str, messages: list):
        model_name = model_name.lower()
        prompt = ""
        stop = []

        if "qwen" in model_name:
            for message in messages:
                prompt += (
                    f"<|im_start|>{message['role']}\n"
                    f"{message['content']}<|im_end|>\n"
                )
            prompt += "<|im_start|>assistant\n"
            stop = ["<|im_end|>"]

        elif "llama-2" in model_name:
            for message in messages:
                role = message.get("role", "user")
                content = message.get("content", "")
                if role == "user":
                    prompt += f"[INST] {content} [/INST]"
                else:
                    prompt += f" {content} "
            stop = []

        elif "opt" in model_name:
            for message in messages:
                prompt += f"\n\nUser: {message['content']}\n\nAssistant:"
            stop = ["User:"]

        else:
            for message in messages:
                role = message.get("role", "user")
                content = message.get("content", "")
                prompt += f"{role}: {content}\n"
            prompt += "assistant: "

        return prompt, stop


class EngineState:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.engine: AsyncLLMEngine | None = None
        self.hotspot_key: str | None = None
        self.lock = asyncio.Lock()
        self.last_active_time = time.time()
        self.is_loading = False


class MultiModelManager:
    def __init__(self, base_args: argparse.Namespace):
        self.base_args = base_args
        self.registry: dict[str, EngineState] = {}
        self.global_lock = asyncio.Lock()
        self.keepalive = base_args.keepalive
        self.max_concurrent_models = base_args.max_concurrent_models
        self.hotspot_lock = asyncio.Lock()
        self.current_hotspot_key: str | None = None
        self.current_hotspot_dir: Path | None = None
        self.persistent_shell_enabled = (
            os.environ.get(
                "VLLM_ENABLE_PERSISTENT_ENGINE_SHELL", "0").lower()
            in ("1", "true", "yes", "on")
            and self.max_concurrent_models == 1
        )
        self.shell_condition = asyncio.Condition()
        self.shell_engine: AsyncLLMEngine | None = None
        self.shell_identity: tuple[str, str | None] | None = None
        self.shell_active_requests = 0
        self.shell_switching = False
        self._idle_task = asyncio.create_task(self._idle_sweeper())

        if self.persistent_shell_enabled:
            print("[Persistent Shell] Enabled general model rebind; worker "
                  "processes and fixed-topology communication state will be "
                  "retained across model switches.")

    def _complete_hotspot_key(self, hotspot_key: str | None) -> str | None:
        normalized_key = normalize_hotspot_key(hotspot_key)
        if normalized_key is None:
            return None

        suffix_parts: list[str] = []
        if "__len" not in normalized_key:
            max_model_len = getattr(self.base_args, "max_model_len", None)
            if max_model_len is not None:
                suffix_parts.append(f"len{format_hotspot_value(max_model_len)}")
        if "__gpuutil" not in normalized_key:
            gpu_util = getattr(self.base_args, "gpu_memory_utilization", None)
            if gpu_util is not None:
                suffix_parts.append(f"gpuutil{format_hotspot_value(gpu_util)}")

        if suffix_parts:
            normalized_key = f"{normalized_key}__{'__'.join(suffix_parts)}"
        return normalized_key

    async def _apply_hotspot_key(self, hotspot_key: str | None) -> str | None:
        normalized_key = self._complete_hotspot_key(hotspot_key)
        if normalized_key is None:
            return None

        async with self.hotspot_lock:
            hotspot_dir = hotspot_root_dir() / normalized_key
            hotspot_dir.mkdir(parents=True, exist_ok=True)

            if normalized_key == self.current_hotspot_key:
                os.environ["VLLM_CUDA_GRAPH_HOTSPOT_DIR"] = str(hotspot_dir)
                return normalized_key

            try:
                from vllm.v1.worker.HotspotAnalyzer import HotspotAnalyzer
                if self.current_hotspot_key is not None:
                    HotspotAnalyzer.check_and_export(True)
                HotspotAnalyzer.output_dir = hotspot_dir
                reset_hotspot_analyzer_state(HotspotAnalyzer)
            except Exception as exc:
                print(f"[CUDA Graph Hotspot] Failed to rebind analyzer: {exc}")

            os.environ["VLLM_CUDA_GRAPH_HOTSPOT_DIR"] = str(hotspot_dir)
            self.current_hotspot_key = normalized_key
            self.current_hotspot_dir = hotspot_dir
            print("[CUDA Graph Hotspot] key="
                  f"{normalized_key} dir={hotspot_dir}")
            return normalized_key

    async def _shutdown_engine_instance(self, engine) -> None:
        """Best-effort explicit shutdown for V1 and legacy async engines."""
        shutdown_errors: list[tuple[str, Exception]] = []

        shutdown = getattr(engine, "shutdown", None)
        if callable(shutdown):
            try:
                result = shutdown()
                if inspect.isawaitable(result):
                    await result
                return
            except Exception as exc:  # pragma: no cover - defensive cleanup
                shutdown_errors.append(("shutdown", exc))

        shutdown_background_loop = getattr(engine, "shutdown_background_loop",
                                           None)
        if callable(shutdown_background_loop):
            try:
                result = shutdown_background_loop()
                if inspect.isawaitable(result):
                    await result
                return
            except Exception as exc:  # pragma: no cover - defensive cleanup
                shutdown_errors.append(("shutdown_background_loop", exc))

        if shutdown_errors:
            details = "; ".join(
                f"{name} failed: {exc!r}" for name, exc in shutdown_errors)
            raise RuntimeError(details)

    async def _unload_engine(self, model_name: str):
        state = self.registry.get(model_name)
        if not state or not state.engine:
            return

        async with state.lock:
            if state.engine is None:
                return

            print(f"\n[-] Unloading model: {model_name}")
            engine = state.engine
            state.engine = None
            state.is_loading = False
            try:
                await self._shutdown_engine_instance(engine)
                print(f"[-] Model {model_name} engine shutdown completed")
            except Exception as exc:
                print(f"[!] Model {model_name} shutdown error: {exc}")
            finally:
                del engine
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                print(f"[-] Model {model_name} resources released\n")

    async def shutdown_all(self):
        if not self._idle_task.done():
            self._idle_task.cancel()
            try:
                await self._idle_task
            except asyncio.CancelledError:
                pass
        async with self.global_lock:
            model_names = [
                name for name, state in self.registry.items()
                if state.engine is not None
            ]
        for model_name in model_names:
            await self._unload_engine(model_name)

    async def _evict_lru_model(self):
        loaded_models = [
            (name, state) for name, state in self.registry.items()
            if state.engine is not None and not state.is_loading
        ]
        if not loaded_models:
            return

        lru_model_name, _ = min(
            loaded_models,
            key=lambda item: item[1].last_active_time,
        )
        print(f"\n[!] GPU residency limit reached, evicting LRU model: "
              f"{lru_model_name}")
        await self._unload_engine(lru_model_name)

    async def _idle_sweeper(self):
        while True:
            await asyncio.sleep(10)
            now = time.time()
            models_to_unload = []

            async with self.global_lock:
                for name, state in self.registry.items():
                    if state.engine is not None and not state.is_loading:
                        if now - state.last_active_time > self.keepalive:
                            models_to_unload.append(name)

            for name in models_to_unload:
                if self.persistent_shell_enabled:
                    # Keep the shell/model ready until a different model takes
                    # ownership. Destroying the AsyncLLM here would also tear
                    # down the worker processes, CUDA context and NCCL state.
                    continue
                print(f"\n[Timeout] Model {name} idle for more than "
                      f"{self.keepalive}s, unloading")
                await self._unload_engine(name)

    async def _activate_persistent_shell(
            self, model_name: str,
            hotspot_key: str | None) -> AsyncLLMEngine:
        async with self.global_lock:
            if model_name not in self.registry:
                self.registry[model_name] = EngineState(model_name)
            state = self.registry[model_name]
            state.is_loading = True

        started = time.perf_counter()
        if hotspot_key is not None:
            await self._apply_hotspot_key(hotspot_key)
        req_args = copy.copy(self.base_args)
        req_args.model = model_name
        engine_args = AsyncEngineArgs.from_cli_args(req_args)
        weight_prewarm_proc = getattr(engine_args, "_weight_prewarm_proc",
                                      None)
        try:
            if self.shell_engine is None:
                print(f"[Persistent Shell] Creating shell for {model_name}")
                engine = AsyncLLMEngine.from_engine_args(engine_args)
                mode = "create"
                rebind_result = None
            else:
                rebind = getattr(self.shell_engine, "rebind_model", None)
                if not callable(rebind):
                    raise RuntimeError(
                        "Persistent shell rebind requires the vLLM V1 "
                        "AsyncLLM engine")
                old_identity = self.shell_identity
                print("[Persistent Shell] Rebinding "
                      f"{old_identity} -> {(model_name, hotspot_key)}")
                vllm_config = engine_args.create_engine_config()
                rebind_result = await rebind(vllm_config)
                engine = self.shell_engine
                mode = "general_rebind"

            for other_state in self.registry.values():
                if other_state is not state and other_state.engine is engine:
                    other_state.engine = None
                    other_state.is_loading = False
            state.engine = engine
            state.hotspot_key = hotspot_key
            state.last_active_time = time.time()
            state.is_loading = False
            print("[Persistent Shell] Activation finished: "
                  f"mode={mode} model={model_name} "
                  f"elapsed={time.perf_counter() - started:.3f}s")
            if rebind_result is not None:
                print("[Persistent Shell] Rebind metrics: "
                      f"{json.dumps(rebind_result, default=str)}")
            return engine
        except BaseException:
            state.is_loading = False
            raise
        finally:
            stop_async_weight_prewarm(
                weight_prewarm_proc, "persistent shell activation finished")

    async def _acquire_persistent_engine(
            self, model_name: str,
            hotspot_key: str | None) -> AsyncLLMEngine:
        identity = (model_name, hotspot_key)
        while True:
            async with self.shell_condition:
                if self.shell_switching:
                    await self.shell_condition.wait()
                    continue
                if (self.shell_engine is not None
                        and self.shell_identity == identity):
                    self.shell_active_requests += 1
                    self.registry[model_name].last_active_time = time.time()
                    return self.shell_engine
                if self.shell_active_requests == 0:
                    self.shell_switching = True
                    break
                await self.shell_condition.wait()

        try:
            engine = await self._activate_persistent_shell(
                model_name, hotspot_key)
        except BaseException:
            async with self.shell_condition:
                self.shell_switching = False
                self.shell_condition.notify_all()
            raise

        async with self.shell_condition:
            self.shell_engine = engine
            self.shell_identity = identity
            self.shell_active_requests += 1
            self.shell_switching = False
            self.shell_condition.notify_all()
            return engine

    async def _release_persistent_engine(self) -> None:
        async with self.shell_condition:
            self.shell_active_requests -= 1
            if self.shell_active_requests < 0:
                self.shell_active_requests = 0
                raise RuntimeError("Persistent shell request count underflow")
            self.shell_condition.notify_all()

    # async def get_or_load_engine(self, model_name: str) -> AsyncLLMEngine:
    #     async with self.global_lock:
    #         if model_name not in self.registry:
    #             self.registry[model_name] = EngineState(model_name)
    #         state = self.registry[model_name]
    #
    #     async with state.lock:
    #         state.last_active_time = time.time()
    #
    #         if state.engine is None:
    #             state.is_loading = True
    #             print("\n" + "=" * 50)
    #             print(f"[+] Cold start triggered for model: {model_name}")
    #
    #             async with self.global_lock:
    #                 active_count = sum(1 for s in self.registry.values() if s.engine is not None)
    #                 if active_count >= self.max_concurrent_models:
    #                     await self._evict_lru_model()
    #
    #             start_load = time.perf_counter()
    #             req_args = copy.copy(self.base_args)
    #             req_args.model = model_name
    #             engine_args = AsyncEngineArgs.from_cli_args(req_args)
    #
    #             try:
    #                 state.engine = AsyncLLMEngine.from_engine_args(engine_args)
    #                 load_time = time.perf_counter() - start_load
    #                 state.is_loading = False
    #                 print(f"[+] Model {model_name} initialized in "
    #                       f"{load_time:.2f}s")
    #                 print(f"{'=' * 50}\n")
    #             except Exception as exc:
    #                 state.is_loading = False
    #                 print("\nEngine initialization failed")
    #                 raise exc
    #
    #         return state.engine
    async def get_or_load_engine(self, model_name: str,
                                 hotspot_key: str | None = None) -> AsyncLLMEngine:
        async with self.global_lock:
            if model_name not in self.registry:
                self.registry[model_name] = EngineState(model_name)
            state = self.registry[model_name]

        async with state.lock:
            state.last_active_time = time.time()

            if (state.engine is not None and hotspot_key is not None
                    and state.hotspot_key != hotspot_key):
                print("[CUDA Graph Hotspot] Reloading model for hotspot key "
                      f"change: {state.hotspot_key} -> {hotspot_key}")
                old_engine = state.engine
                state.engine = None
                state.is_loading = False
                try:
                    await self._shutdown_engine_instance(old_engine)
                finally:
                    del old_engine
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

            if state.engine is None:
                state.is_loading = True
                print(f"\n{'=' * 50}")
                print(f"[+] Cold start triggered for model: {model_name}")

                async with self.global_lock:
                    active_count = sum(1 for s in self.registry.values() if s.engine is not None)
                    if active_count >= self.max_concurrent_models:
                        await self._evict_lru_model()

                start_load = time.perf_counter()
                weight_prewarm_proc = None

                # Page-cache prewarm is started by AsyncEngineArgs.from_cli_args()
                # and stopped in EngineCore before load_weights(). Keeping the old
                # vmtouch path here would duplicate I/O and contend with in-process
                # weight loading.
                print("[I/O Overlap] Page cache prewarm is delegated to the "
                      "engine startup path.")

                req_args = copy.copy(self.base_args)
                req_args.model = model_name
                engine_args = AsyncEngineArgs.from_cli_args(req_args)
                weight_prewarm_proc = getattr(engine_args,
                                              "_weight_prewarm_proc", None)

                try:
                    if self.current_hotspot_dir is not None:
                        os.environ["VLLM_CUDA_GRAPH_HOTSPOT_DIR"] = str(
                            self.current_hotspot_dir)
                    # Warm the tokenizer in the PARENT before the EngineCore is
                    # forked (VLLM_WORKER_MULTIPROC_METHOD defaults to "fork").
                    # Forking a threaded process can orphan the CPython import
                    # lock in the child; the child then deadlocks on the first
                    # dynamic import (e.g. Llama's SentencePiece tokenizer).
                    # Importing it here populates sys.modules so the forked
                    # child reuses the cached modules instead of importing.
                    try:
                        from vllm.transformers_utils.tokenizer import (
                            get_tokenizer)
                        get_tokenizer(
                            model_name,
                            tokenizer_mode=getattr(req_args, "tokenizer_mode",
                                                   "auto"),
                            trust_remote_code=getattr(
                                req_args, "trust_remote_code", False),
                            revision=getattr(req_args, "tokenizer_revision",
                                             None),
                        )
                    except Exception as exc:
                        print("[tokenizer-prewarm] skipped for "
                              f"{model_name}: {exc}")
                    state.engine = AsyncLLMEngine.from_engine_args(engine_args)
                    state.hotspot_key = hotspot_key
                    stop_async_weight_prewarm(weight_prewarm_proc,
                                              "multi-model engine ready")
                    weight_prewarm_proc = None

                    load_time = time.perf_counter() - start_load
                    state.is_loading = False

                    print(f"[+] Model {model_name} initialized, total: {load_time:.2f}s")
                    print(f"{'=' * 50}\n")

                except Exception as e:
                    state.is_loading = False
                    stop_async_weight_prewarm(weight_prewarm_proc,
                                              "multi-model engine init error")
                    import traceback
                    print("\nEngine initialization failed!!!")
                    traceback.print_exc()
                    raise e
            return state.engine

    async def generate_stream(self, model_name: str, prompt: str,
                              sampling_params: SamplingParams,
                              hotspot_key: str | None = None):
        if self.persistent_shell_enabled:
            # Defer the environment/profile switch until all requests using
            # the current model/profile have drained.
            normalized_hotspot_key = self._complete_hotspot_key(hotspot_key)
            engine = await self._acquire_persistent_engine(
                model_name, normalized_hotspot_key)
        else:
            normalized_hotspot_key = await self._apply_hotspot_key(hotspot_key)
            engine = await self.get_or_load_engine(
                model_name, normalized_hotspot_key)
            self.registry[model_name].last_active_time = time.time()

        request_id = f"req-{time.time()}"
        try:
            results_generator = engine.generate(
                prompt, sampling_params, request_id)
            async for request_output in results_generator:
                yield request_output
        finally:
            if self.persistent_shell_enabled:
                await self._release_persistent_engine()


global_cli_args = None
manager: MultiModelManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    del app
    global manager
    manager = MultiModelManager(global_cli_args)
    yield
    if manager is not None:
        await manager.shutdown_all()


app = FastAPI(lifespan=lifespan)


@app.get("/healthz")
async def healthz():
    loaded_models = []
    loading_models = []
    if manager is not None:
        for name, state in manager.registry.items():
            if state.engine is not None:
                loaded_models.append(name)
            if state.is_loading:
                loading_models.append(name)

    return {
        "status": "ok",
        "load_format": getattr(global_cli_args, "load_format", None),
        "keepalive": getattr(global_cli_args, "keepalive", None),
        "max_concurrent_models": getattr(global_cli_args,
                                         "max_concurrent_models", None),
        "loaded_models": loaded_models,
        "loading_models": loading_models,
        "storage_path": os.getenv("STORAGE_PATH"),
        "cuda_graph_hotspot_root": os.getenv("VLLM_CUDA_GRAPH_HOTSPOT_ROOT"),
        "cuda_graph_hotspot_dir": os.getenv("VLLM_CUDA_GRAPH_HOTSPOT_DIR"),
        "cuda_graph_hotspot_suffix": {
            "max_model_len": getattr(global_cli_args, "max_model_len", None),
            "gpu_memory_utilization": getattr(
                global_cli_args, "gpu_memory_utilization", None),
        },
    }


async def _set_model_profile(model_name: str, start: bool) -> dict:
    if manager is None:
        raise HTTPException(status_code=503, detail="Manager is not ready")
    state = manager.registry.get(model_name)
    if state is None or state.engine is None:
        raise HTTPException(status_code=404,
                            detail=f"Model is not loaded: {model_name}")
    if start:
        await state.engine.start_profile()
    else:
        await state.engine.stop_profile()
    return {"model": model_name, "profiling": start}


@app.post("/start_profile")
async def start_profile(request: Request):
    body = await request.json()
    return await _set_model_profile(body.get("model", ""), True)


@app.post("/stop_profile")
async def stop_profile(request: Request):
    body = await request.json()
    return await _set_model_profile(body.get("model", ""), False)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    hotspot_key = (
        request.headers.get(HOTSPOT_KEY_HEADER)
        or body.get("hotspot_key")
        or body.get("hotspotKey")
    )
    model_name = body.get("model")
    if not model_name:
        raise HTTPException(status_code=400, detail="Missing 'model' field")

    messages = body.get("messages", [])
    prompt, stop_tokens = TemplateManager.get_prompt_and_stop(
        model_name, messages)

    sampling_params = SamplingParams(
        temperature=body.get("temperature", 0.0),
        max_tokens=body.get("max_tokens", 256),
        stop=stop_tokens or None,
    )

    start_request = time.perf_counter()

    async def stream_results():
        first_token_sent = False
        last_output_text = ""

        try:
            assert manager is not None
            async for output in manager.generate_stream(model_name, prompt,
                                                        sampling_params,
                                                        hotspot_key):
                if not first_token_sent:
                    print(f">>> [{model_name}] TTFT: "
                          f"{time.perf_counter() - start_request:.4f}s")
                    first_token_sent = True

                current_text = output.outputs[0].text
                delta = current_text[len(last_output_text):]
                last_output_text = current_text

                if delta:
                    payload = json.dumps(
                        {"choices": [{"delta": {"content": delta}}]},
                        ensure_ascii=False,
                    )
                    yield f"data: {payload}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(stream_results(),
                             media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    from vllm.utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser(description="Multi-Model vLLM Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--keepalive",
                        type=int,
                        default=60,
                        help="Model idle lifetime in seconds.")
    parser.add_argument("--max-concurrent-models",
                        type=int,
                        default=1,
                        help="Max number of models allowed to stay resident.")
    parser.add_argument("--serverless-storage-path",
                        type=str,
                        default=None,
                        help="Set STORAGE_PATH for serverless_llm checkpoints.")

    parser = AsyncEngineArgs.add_cli_args(parser)
    global_cli_args = parser.parse_args()

    if global_cli_args.serverless_storage_path:
        os.environ["STORAGE_PATH"] = global_cli_args.serverless_storage_path

    print(f"Server starting at {global_cli_args.host}:{global_cli_args.port}")
    print("Scheduler Config: "
          f"keepalive={global_cli_args.keepalive}s, "
          f"max_concurrent={global_cli_args.max_concurrent_models}")
    print(f"vLLM load_format={getattr(global_cli_args, 'load_format', None)}")
    if os.getenv("STORAGE_PATH"):
        print(f"STORAGE_PATH={os.getenv('STORAGE_PATH')}")

    uvicorn.run(app,
                host=global_cli_args.host,
                port=global_cli_args.port,
                access_log=False)
