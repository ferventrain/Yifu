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
