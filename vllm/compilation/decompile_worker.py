# SPDX-License-Identifier: Apache-2.0
"""CPU-only diagnostic helper, invoked by path to avoid importing vLLM."""
import marshal
import os
import signal
import sys


def main() -> None:
    if sys.platform == "linux":
        import ctypes
        # Also stop on abrupt engine death, when Python atexit cannot run.
        if ctypes.CDLL(None).prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
            raise RuntimeError("Could not set decompiler parent-death signal")
        if os.getppid() != int(sys.argv[1]):
            return
    if hasattr(os, "nice"):
        os.nice(10)
    code = marshal.loads(sys.stdin.buffer.read())
    import depyf
    sys.stdout.buffer.write(depyf.decompile(code).encode("utf-8"))


if __name__ == "__main__":
    main()
