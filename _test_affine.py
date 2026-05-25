from pathlib import Path
import numpy as np

p = "H:/arivis-analysis/YF2026041701shanda/nao_1/transforms/fwd_1_tmpy81k6ki60GenericAffine.mat"
text = Path(p).read_text(encoding="utf-8", errors="ignore")
transform_line = ""
parameters_line = ""
for line in text.splitlines():
    stripped = line.strip()
    if stripped.startswith("Transform:"):
        transform_line = stripped
    elif stripped.startswith("Parameters:"):
        parameters_line = stripped

print("transform_line found:", bool(transform_line))
print("parameters_line found:", bool(parameters_line))
if parameters_line:
    values = [float(v) for v in parameters_line.split(":", 1)[1].split()]
    print("values count:", len(values))
    if len(values) >= 12:
        A = np.asarray(values[:9], dtype=np.float64).reshape(3, 3)
        print("A:\n", A)
        print("det:", np.linalg.det(A))
