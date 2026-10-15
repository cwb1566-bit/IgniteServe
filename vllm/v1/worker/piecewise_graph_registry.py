# SPDX-License-Identifier: Apache-2.0
"""One publication index over engine-owned piecewise backend handles."""


class PiecewiseGraphRegistry:
    def __init__(self, compilation_config=None):
        self.compilation_config = compilation_config
        self.published = ()
        self.dynamic_shapes = frozenset()

    def backends(self):
        values = getattr(
            self.compilation_config, "cuda_piecewise_backends_list", {}
        )
        return sorted(values.values(), key=lambda b: b.piecewise_compile_index)

    def publish(self, shape, *, dynamic=False, gate=None):
        if gate is not None:
            backends = self.backends()
            if not backends or any(
                b.total_piecewise_compiles != len(backends)
                or b.concrete_size_entries.get(shape) is None
                or b.concrete_size_entries[shape].cudagraph is None
                for b in backends
            ):
                raise RuntimeError(f"incomplete piecewise graph chain: {shape}")
            for backend in backends:
                if not backend.finish_concurrent_capture(shape):
                    raise RuntimeError(f"cannot publish shape {shape}")
            gate.set()
        if dynamic:
            self.dynamic_shapes = self.dynamic_shapes | {shape}
        # Advertise only after every backend has finished publication.
        self.published = tuple(sorted(set(self.published) | {shape}))

    def ready_sizes(self, candidates=None):
        backends = self.backends()
        if not backends or any(
            b.total_piecewise_compiles != len(backends) for b in backends
        ):
            return ()
        ready = []
        for size in self.published:
            if candidates is not None and size not in candidates:
                continue
            entries = [b.concrete_size_entries.get(size) for b in backends]
            if all(
                e is not None
                and e.use_cudagraph
                and e.cudagraph is not None
                and not e.concurrent_capture_in_progress
                and (
                    e.concurrent_capture_gate is None
                    or e.concurrent_capture_gate.is_set()
                )
                for e in entries
            ):
                ready.append(size)
        return tuple(ready)

    def clear(self):
        self.published = ()
        self.dynamic_shapes = frozenset()
