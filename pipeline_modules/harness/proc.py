"""Process helpers for attached / external harness jobs."""

from __future__ import annotations

import os
import sys


def pid_is_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        number = int(pid)
    except (TypeError, ValueError):
        return False
    if number <= 0:
        return False
    if sys.platform == "win32":
        return _pid_is_alive_windows(number)
    try:
        os.kill(number, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def pid_created_at(pid: int | None) -> float | None:
    """Process creation time (seconds since epoch), or None when unknown."""
    if pid is None:
        return None
    try:
        import psutil

        return float(psutil.Process(int(pid)).create_time())
    except Exception:
        return None


def pid_matches(pid: int | None, created_at: float | None) -> bool:
    """True when ``pid`` is alive and (when a creation time was recorded at
    attach time) still is the same process instance. Guards against Windows
    PID reuse, where a dead job's pid gets handed to an unrelated process and
    plain liveness checks report the job as running forever."""
    if not pid_is_alive(pid):
        return False
    if created_at is None:
        return True
    actual = pid_created_at(pid)
    if actual is None:
        return True
    return abs(actual - float(created_at)) <= 5.0


def tree_memory_mb(pid: int | None) -> float | None:
    """Sum of private commit bytes (vms) over ``pid`` and its descendants, MB.

    vms/PagefileUsage is the committed amount — the resource that actually
    runs out when parallel big-array steps OOM — so this is the planning
    metric. Returns None when the process is gone or psutil is unavailable.
    """
    if pid is None:
        return None
    try:
        number = int(pid)
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    try:
        import psutil
    except ImportError:  # pragma: no cover - psutil ships with the runtime env
        return None
    try:
        root = psutil.Process(number)
        procs = [root, *root.children(recursive=True)]
    except Exception:
        return None
    total = 0
    for proc in procs:
        try:
            total += int(proc.memory_info().vms)
        except Exception:
            continue
    return total / (1024 * 1024)


def _pid_is_alive_windows(pid: int) -> bool:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return ctypes.get_last_error() in {5, 0x5}  # access denied => likely alive
    try:
        code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return True
        return int(code.value) == STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)
