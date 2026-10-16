import threading

from vllm.compilation import transformed_code_cache as cache


def example(x):
    return x + 1


def test_background_export_is_nonblocking_and_atomic(tmp_path, monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    saved = []

    class Process:
        returncode = 0

        def __init__(self, *args, **kwargs):
            assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""

        def communicate(self, data, timeout):
            entered.set()
            assert release.wait(5)
            return b"def example(x): return x + 1\n", b""

        def poll(self):
            return 0

    monkeypatch.setattr(cache.subprocess, "Popen", Process)
    monkeypatch.setattr(cache, "save", lambda *args: saved.append(args))
    target = tmp_path / "transformed_code.py"
    thread = cache.save_in_background(example.__code__, example.__code__,
                                     str(target), type)
    assert entered.wait(5)
    assert not target.exists()
    release.set()
    thread.join(5)
    assert not thread.is_alive()
    assert target.read_text().startswith("def example")
    assert len(saved) == 1
    assert not cache._jobs


def test_failed_export_does_not_publish(tmp_path, monkeypatch):
    class Process:
        returncode = 1

        def __init__(self, *args, **kwargs):
            pass

        def communicate(self, *args, **kwargs):
            return b"partial", b"decompile failed"

        def poll(self):
            return 1

    monkeypatch.setattr(cache.subprocess, "Popen", Process)
    target = tmp_path / "transformed_code.py"
    thread = cache.save_in_background(example.__code__, example.__code__,
                                     str(target), type)
    thread.join(5)
    assert not thread.is_alive()
    assert not target.exists()
    assert not cache._jobs


def test_shutdown_rejects_new_exports(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "_closing", True)
    def unexpected(*args, **kwargs):
        raise AssertionError("must not start a child after shutdown")
    monkeypatch.setattr(cache.subprocess, "Popen", unexpected)
    target = tmp_path / "transformed_code.py"
    thread = cache.save_in_background(example.__code__, example.__code__,
                                     str(target), type)
    thread.join(5)
    assert not thread.is_alive()
    assert not target.exists()
    assert not cache._jobs


def test_timeout_reaps_child(tmp_path, monkeypatch):
    killed = []
    waited = []

    class Process:
        def __init__(self, *args, **kwargs):
            pass

        def communicate(self, *args, **kwargs):
            raise cache.subprocess.TimeoutExpired("decompiler", 120)

        def poll(self):
            return None

        def kill(self):
            killed.append(True)

        def wait(self):
            waited.append(True)

    monkeypatch.setattr(cache.subprocess, "Popen", Process)
    target = tmp_path / "transformed_code.py"
    thread = cache.save_in_background(example.__code__, example.__code__,
                                     str(target), type)
    thread.join(5)
    assert not thread.is_alive()
    assert killed == waited == [True]
    assert not target.exists()
    assert not cache._jobs
