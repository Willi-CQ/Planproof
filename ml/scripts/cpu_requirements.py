"""Print the portable lock without CUDA/torch wheels; install official CPU torch first."""

from pathlib import Path
import re

lock = Path(__file__).resolve().parents[1] / "requirements.lock"
for line in lock.read_text().splitlines():
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        continue
    name = re.split(r"[=;<>\s]", stripped)[0].lower().replace("_", "-")
    if name not in {"torch", "torchvision", "triton"} and not name.startswith("nvidia-"):
        print(stripped)
