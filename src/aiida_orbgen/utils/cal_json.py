"""Helpers for the ``output.json`` identifier map written by ``run``.

Layout (same convention as ``aiida_uranium_workflow.utils.cal_json``)::

    {
        "workflow": "orbgen.gridsearch",
        "input_json": "/abs/path/input.json",
        "abacus": {
            "orbgen": {
                "<preset>": "<WorkChain UUID>"
            }
        }
    }

The leaf identifier is the WorkChain **UUID** (stable across profile
re-imports); legacy integer pks are still accepted by every reader. The
top-level ``workflow`` field lets ``report`` re-derive the method without
being told again.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from aiida_orbgen.utils.yamlio import read_json

__all__ = [
    "SubmittedJob",
    "WORKFLOW_TO_KEY",
    "build_cal_json",
    "write_cal_json",
    "default_result_path",
    "read_output_json",
    "collect_job_entries",
]

#: ``workflow -> inner JSON key`` used for the second level of the map.
WORKFLOW_TO_KEY: dict[str, str] = {
    "orbgen.calc": "orbgen",
    "orbgen.gridsearch": "orbgen",
}


@dataclass
class SubmittedJob:
    """One submitted WorkChain, as recorded in ``output.json``."""

    backend: str                     # "abacus"
    key: str                         # "orbgen"
    preset_name: str                 # leaf name in output.json
    uuid: str | None = None
    pk: int | None = None
    workflow: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def identifier(self) -> str:
        """Canonical identifier written to ``output.json``."""
        if self.uuid:
            return self.uuid
        if self.pk is not None:
            return str(self.pk)
        raise ValueError("SubmittedJob has neither uuid nor pk")


def build_cal_json(
    jobs: Iterable[SubmittedJob],
    *,
    workflow: str,
    input_json: str | Path | None = None,
) -> "OrderedDict[str, Any]":
    """Group ``jobs`` into the nested ``output.json`` layout."""
    jobs = list(jobs)
    duplicates: dict[tuple[str, str, str], int] = {}
    for job in jobs:
        key = (job.backend, job.key, job.preset_name)
        duplicates[key] = duplicates.get(key, 0) + 1

    out: "OrderedDict[str, Any]" = OrderedDict()
    out["workflow"] = workflow
    if input_json is not None:
        out["input_json"] = str(Path(input_json).resolve())

    for index, job in enumerate(jobs):
        backend_dict = out.setdefault(job.backend, OrderedDict())
        key_dict = backend_dict.setdefault(job.key or WORKFLOW_TO_KEY.get(workflow, "orbgen"), OrderedDict())
        name = job.preset_name
        if duplicates[(job.backend, job.key, job.preset_name)] > 1:
            name = f"{name}#{index}"
        key_dict[name] = job.identifier
    return out


def write_cal_json(
    jobs: Iterable[SubmittedJob],
    *,
    output_path: str | Path,
    workflow: str,
    input_json: str | Path | None = None,
) -> Path:
    """Write the ``output.json`` identifier map."""
    from aiida_orbgen.utils.yamlio import write_json

    data = build_cal_json(jobs, workflow=workflow, input_json=input_json)
    return write_json(output_path, data)


def default_result_path(input_json: str | Path) -> Path:
    """``<input_dir>/output.json`` — used when ``--output`` is omitted."""
    return Path(input_json).resolve().parent / "output.json"


def read_output_json(path: str | Path) -> dict:
    """Read an ``output.json``, raising a readable error when malformed."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"output.json not found: {path}")
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object, got {type(data).__name__}")
    return data


def collect_job_entries(data: dict) -> list[tuple[str, str]]:
    """Flatten an ``output.json`` into ``[(label, identifier), ...]``.

    ``label`` is ``"<backend>/<key>/<preset>"`` so log lines stay readable.
    Non-scalar leaves and pure metadata keys (``workflow``, ``input_json``)
    are ignored.
    """
    entries: list[tuple[str, str]] = []
    for backend, by_key in data.items():
        if backend in ("workflow", "input_json"):
            continue
        if not isinstance(by_key, dict):
            continue
        for key, by_preset in by_key.items():
            if not isinstance(by_preset, dict):
                continue
            for preset, identifier in by_preset.items():
                if isinstance(identifier, bool):
                    continue
                if isinstance(identifier, (int, str)):
                    entries.append((f"{backend}/{key}/{preset}", str(identifier)))
    return entries
