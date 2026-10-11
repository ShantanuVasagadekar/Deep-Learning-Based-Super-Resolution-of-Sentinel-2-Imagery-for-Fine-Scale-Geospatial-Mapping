import importlib.util
import sys
import time
from pathlib import Path
from PyQt5.QtTest import QTest

ROOT = Path(__file__).resolve().parents[1]


def load_gui(name, filename):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT/filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def wait_jobs(window, timeout=30):
    deadline = time.monotonic()+timeout
    while window._workers or window._is_busy:
        if time.monotonic() >= deadline:
            raise AssertionError("GUI operation failed to finish before timeout")
        QTest.qWait(10)
    QTest.qWait(10)  # Flush deferred QObject deletion before memory checks or the next job.
