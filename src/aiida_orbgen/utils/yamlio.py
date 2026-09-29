"""Low-level JSON / YAML readers used by the ``parameters`` runner.

Kept tiny and AiiDA-free (same idea as ``aiida_uranium_workflow.utils.yamlio``)
so the config loader can be imported and unit-tested without a profile.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import json


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file, requiring the root to be a mapping."""
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(
            f"YAML root of {path} must be a mapping, got {type(data).__name__}"
        )
    return data


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON file."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, data: Any) -> Path:
    """Write ``data`` as pretty JSON, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=4, ensure_ascii=False)
        handle.write("\n")
    return path


__all__ = ["read_yaml", "read_json", "write_json"]
