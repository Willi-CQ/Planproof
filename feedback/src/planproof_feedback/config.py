from __future__ import annotations

import os
import tomllib
from pathlib import Path


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config.toml"


def read_env(path: Path) -> dict[str, str]:
    """Read literal KEY=VALUE assignments; never execute or expand shell content."""
    values = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, sep, value = line.partition("=")
            if not sep or not key.strip().replace("_", "a").isalnum():
                raise ValueError("Invalid assignment in the local .env file")
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            values[key.strip()] = value
    return values


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    path = path.resolve()
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    config["config_path"] = str(path)
    config["env_file"] = str((path.parent / config["env_file"]).resolve())
    for key in ("audit_dir", "report_dir"):
        if key in config:
            config[key] = str((path.parent / config[key]).resolve())
    for key in ("question_file", "inventory_file", "pack_dir"):
        config["benchmark"][key] = str((path.parent / config["benchmark"][key]).resolve())
    return config


def get_key(config: dict, provider: dict) -> str:
    name = provider["api_key_env"]
    key = os.environ.get(name) or read_env(Path(config["env_file"])).get(name, "")
    if not key or key.startswith("<") or key == "PASTE_YOUR_FULL_VERDA_KEY_HERE":
        raise ValueError(f"Paste your full key after {name}= in {config['env_file']}")
    if any(c.isspace() for c in key):
        raise ValueError(f"{name} must not contain whitespace")
    return key
