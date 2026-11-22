"""Read Tesseract word TSV through the system CLI or the bundled native binding."""
from io import BytesIO
import os
from pathlib import Path
import shutil
import subprocess


TSV_HEADER = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext"
ASSETS = Path(__file__).resolve().parents[2] / "planproof_runtime"


def bundled():
    return os.environ.get("PLANPROOF_COMPACT_CPU") == "1"


def available():
    return bundled() or bool(shutil.which("tesseract"))


def read_tsv(image, psm, timeout, variables=None):
    variables = variables or {}
    if bundled():
        from tesserocr import PyTessBaseAPI
        with PyTessBaseAPI(path=str(ASSETS), lang="eng", psm=psm, variables=variables) as api:
            api.SetImage(image)
            if not api.Recognize(timeout=timeout * 1000):
                raise subprocess.TimeoutExpired("tesseract", timeout)
            body = api.GetTSVText(0)
        return body if body.startswith("level\t") else TSV_HEADER + "\n" + body
    blob = BytesIO()
    image.save(blob, format="PNG")
    command = ["tesseract", "stdin", "stdout", "-l", "eng", "--psm", str(psm)]
    for name, value in variables.items():
        command.extend(["-c", f"{name}={value}"])
    proc = subprocess.run([*command, "tsv"], input=blob.getvalue(), capture_output=True, timeout=timeout)
    if proc.returncode:
        raise OSError("Tesseract failed")
    return proc.stdout.decode("utf-8", "replace")
