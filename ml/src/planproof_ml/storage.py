"""Keep generated training artifacts out of source-controlled directories."""

from pathlib import Path


def ensure_artifact_directory(path: Path) -> Path:
    resolved = path.resolve()
    roots = set()
    for base in (Path.cwd().resolve(), Path(__file__).resolve().parent):
        for parent in (base, *base.parents):
            if (parent / ".git").exists() or (
                (parent / "ml/pyproject.toml").is_file()
                and (parent / "data/gpu-training-datasets.csv").is_file()
            ):
                roots.add(parent)
                break
    allowed = (
        ".artifacts",
        ".cache",
        "runs",
        "exports",
        "artifacts",
        "data/raw",
        "data/downloads",
        "data/extracted",
        "data/processed",
        "data/evaluation",
    )
    for root in roots:
        if resolved.is_relative_to(root):
            relative = resolved.relative_to(root).as_posix()
            if not any(relative == prefix or relative.startswith(prefix + "/") for prefix in allowed):
                raise ValueError(
                    "Write generated data/models outside the repository or into its ignored artifact directories"
                )
    return resolved
