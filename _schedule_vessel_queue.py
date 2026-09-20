"""Register and start a vessel queue scheduled task with optional sample args."""
import subprocess
import sys

PYTHON = r"C:\Users\admin\micromamba\envs\yifu\python.exe"
SCRIPT = r"S:\Yifu\_queue_YF2025063002_vessel.py"


def run(args: list[str]) -> int:
    proc = subprocess.run(args, capture_output=True, text=True, encoding="gbk", errors="replace")
    out = ((proc.stdout or "") + (proc.stderr or "")).strip()
    if out:
        print(out)
    return proc.returncode


task = sys.argv[1]
samples = sys.argv[2:]
tr = f'"{PYTHON}" "{SCRIPT}"' + (" " + " ".join(f'"{s}"' for s in samples) if samples else "")
code = run(["schtasks", "/Create", "/TN", task, "/F", "/TR", tr, "/SC", "ONCE", "/ST", "23:59"])
if code != 0:
    sys.exit("schtasks /Create failed")
code = run(["schtasks", "/Run", "/TN", task])
if code != 0:
    sys.exit("schtasks /Run failed")
print(f"task {task} started, samples={samples or '(default)'}")
