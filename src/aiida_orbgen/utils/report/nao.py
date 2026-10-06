"""``aiida-orbgen nao`` — generate the contracted NAOs of one NSW reference.

The expensive half of this project is the NSW step: SIAB builds the primitive
spherical-wave basis and ABACUS runs the reference DFT with it.  The contraction
(SIAB's spillage fit) only *reads* the results of that run, so it can be repeated as
often as wanted — and SIAB does several schemes in one go, because ``orbitals`` is a
**list**, one entry per scheme (see ``project/test/pbe/pbe_orbgen.json``)::

    "orbitals": [
      {"nzeta": [3, 2, 2, 1], "geoms": [0], "nbands": "occ",  "checkpoint": null},
      {"nzeta": [4, 3, 3, 2], "geoms": [0], "nbands": "occ*2", "checkpoint": null},
      {"nzeta": [4, 3, 3, 2, 1], "geoms": [0], "nbands": "occ",
       "checkpoint": null,
       "model_kwargs": {"lloc_min": 4, "vloc_aux": "/abs/path/U.pbe-n-nc.UPF"}}
    ]

This module turns such a config (or the one the run itself used, written out as
``<out>/orbgen.json`` when none is given) into the fit, and summarises what came out:
one row per entry with its spillage, its radial functions per ``l`` and whether the
file validates.

Two things that are easy to get wrong, both verified on 2026-10-06:

``checkpoint`` chains only grow
    Entry *n* may start from entry *n-1* only if it asks for at least as many radial
    functions per ``l``; a shrinking step dies inside ``SIAB/orb/cascade.py`` with
    ``ValueError: axes don't match array`` — after the previous entry has already been
    optimised.  And even a legal chain is not the same fit: ``2s2p1d1f -> 3s2p2d1f``
    chained ended at spillage 1.259e-3 where two independent entries reached 9.720e-4.
    So chain only for the "cascade" hierarchy, never for comparing schemes.

a ``g`` channel needs ``model_kwargs``
    SIAB reads ``lloc_min`` / ``vloc_aux`` only from an orbital's ``model_kwargs``
    block; written flat (as in ``pbe_orbgen.json``) they are silently ignored, the
    ``g`` channel comes out empty and the file fails validation
    (``per l are [4, 3, 3, 2, 0] but the requested scheme 4s3p3d2f1g needs […1]``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "NaoOutcome",
    "config_for_node",
    "default_config",
    "describe_entries",
    "summarise_outcome",
    "write_config",
]


def default_config(siab_json: dict, *, l_max: int | None = None,
                   r_cut: float | None = None) -> dict:
    """The config to fit with when the caller supplied none: the run's own.

    Written out rather than used in memory, so the ``orbitals`` list is on disk and can
    be edited before the next run — that is the whole point of the file being explicit.
    """
    from aiida_orbgen.interfaces.nsw import apply_grid_point

    if l_max is None and r_cut is None:
        return json.loads(json.dumps(siab_json))
    return apply_grid_point(siab_json, int(l_max), float(r_cut))


def config_for_node(node, *, siab_config: str | Path | None = None) -> tuple[dict, str]:
    """``(config, source)`` for one workchain: the file given, else the run's own.

    The stored ``siab_json`` is the config the NSW run was *built* with, so its
    ``bessel_nao_rcut`` / ``ecutjy`` / ``geoms`` match the reference tree by
    construction — which is exactly what the fit needs.
    """
    if siab_config is not None:
        path = Path(siab_config).expanduser().resolve()
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle), f"--config {path}"
    siab_info = getattr(getattr(node, "outputs", None), "siab_info", None)
    info = siab_info.get_dict() if siab_info is not None else {}
    content = node.inputs.siab_json.get_content()
    if isinstance(content, bytes):
        content = content.decode("utf-8")
    config = default_config(
        json.loads(content),
        l_max=info.get("lmax"),
        r_cut=info.get("rcut"),
    )
    return config, "the run's own siab_json (grid point applied)"


def write_config(config: dict, path: str | Path) -> Path:
    """Write the config the fit will use, indented, next to its products."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


def describe_entries(config: dict) -> list[str]:
    """One line per ``orbitals`` entry: what will be fitted, and from where."""
    from aiida_orbgen.utils.report.validate import nzeta_string

    lines: list[str] = []
    for index, orbital in enumerate(config.get("orbitals") or []):
        nzeta = list(orbital.get("nzeta") or [])
        scheme = nzeta_string(nzeta) if nzeta else "?"
        model_kwargs = orbital.get("model_kwargs") or {}
        bits = [
            f"[{index}] {scheme}",
            f"nbands={orbital.get('nbands', '?')}",
            f"checkpoint={orbital.get('checkpoint')}",
        ]
        if model_kwargs.get("vloc_aux"):
            bits.append("vloc_aux (l>=%s from the auxiliary potential)"
                        % model_kwargs.get("lloc_min", 4))
        elif len(nzeta) > 4 and nzeta[4]:
            bits.append("⚠ asks for a g channel without model_kwargs.vloc_aux — "
                        "it will come out empty")
        lines.append("  ".join(bits))
    return lines


def config_problems(config: dict, *, l_max: int, r_cut: float) -> list[str]:
    """Why this config cannot be fitted against that tree (empty when it can).

    The fit reads the tree's folders and primitive orbital, so anything that would make
    SIAB look for a *different* primitive (``bessel_nao_rcut`` / ``ecutjy`` /
    ``lmaxmax``) turns a minute-long fit into a fresh reference DFT.  Better to say so
    than to let a run quietly recompute DFT.
    """
    problems: list[str] = []
    from aiida_orbgen.interfaces.nsw import folder_rcut

    configured = [float(value) for value in config.get("bessel_nao_rcut") or []]
    if not configured:
        problems.append("the config has no bessel_nao_rcut")
    elif folder_rcut(float(r_cut)) not in {folder_rcut(value) for value in configured}:
        problems.append(
            f"the config's bessel_nao_rcut {configured} does not cover r_cut={r_cut:g} au "
            f"of the reference tree"
        )
    geoms = config.get("geoms") or []
    configured_l_max = geoms[0].get("lmaxmax") if geoms else None
    if configured_l_max is not None and int(configured_l_max) != int(l_max):
        problems.append(
            f"the config's lmaxmax={configured_l_max} does not match l_max={l_max} of "
            f"the reference tree"
        )
    if not (config.get("orbitals") or []):
        problems.append("the config has no orbitals entry to fit")
    return problems


@dataclass
class NaoOutcome:
    """What one ``orbigen nao`` invocation produced for one workchain."""

    node_pk: int
    config_path: Path | None = None
    config_source: str = ""
    out_dir: Path | None = None
    dft_root: str = ""
    status: str = ""
    entries: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.entries) and all(entry.get("ok") for entry in self.entries)

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_pk": self.node_pk,
            "config": str(self.config_path) if self.config_path else None,
            "config_source": self.config_source,
            "out_dir": str(self.out_dir) if self.out_dir else None,
            "dft_root": self.dft_root,
            "status": self.status,
            "entries": self.entries,
            "warnings": self.warnings,
        }


def summarise_outcome(config: dict, final_orbital: dict, out_dir: str | Path) -> NaoOutcome:
    """Turn the report's final-orbital outcome into one row per ``orbitals`` entry."""
    from aiida_orbgen.utils.report.validate import nzeta_string

    out = NaoOutcome(
        node_pk=int(final_orbital.get("node_pk") or 0),
        out_dir=Path(out_dir),
        dft_root=str(final_orbital.get("dft_root") or ""),
        status=str(final_orbital.get("status") or ""),
    )
    produced = [Path(path) for path in final_orbital.get("files") or []]
    spillages = [float(value) for value in final_orbital.get("spillage") or []]
    validated = {Path(item["file"]).name: item
                 for item in final_orbital.get("validated") or []}
    failure = str(final_orbital.get("validation_failed") or "")
    for index, orbital in enumerate(config.get("orbitals") or []):
        nzeta = list(orbital.get("nzeta") or [])
        name = f"_{nzeta_string(nzeta)}.orb" if nzeta else ""
        match = next((path for path in produced if path.name.endswith(name)), None)
        check = validated.get(match.name) if match else None
        entry: dict[str, Any] = {
            "index": index,
            "scheme": nzeta_string(nzeta) if nzeta else "?",
            "nzeta": nzeta,
            "nbands": orbital.get("nbands"),
            "checkpoint": orbital.get("checkpoint"),
            "file": str(match) if match else None,
            "bytes": match.stat().st_size if match and match.is_file() else None,
            "spillage": spillages[index] if index < len(spillages) else None,
            "per_l": list(check.get("per_l")) if check else None,
            "ok": bool(check.get("ok")) if check else False,
            "reason": (check.get("reason") if check else None)
                      or (failure or None)
                      or (None if match else "no orbital file was produced"),
        }
        out.entries.append(entry)
    if final_orbital.get("message"):
        out.warnings.append(str(final_orbital["message"]))
    if failure:
        out.warnings.append(f"a produced orbital failed validation: {failure}")
    return out


def render_summary(rows: list[NaoOutcome]) -> str:
    """The Markdown table ``aiida-orbgen nao`` prints (and writes to ``nao.md``)."""
    lines = [
        "# NAO from one NSW reference",
        "",
        "Δ-fitted schemes of one reference run; the DFT was read, not recomputed.",
        "",
        "| node | # | scheme | nbands | checkpoint | spillage | radial per l | bytes | ok | file |",
        "| ---: | ---: | --- | --- | ---: | ---: | --- | ---: | :--: | --- |",
    ]
    for outcome in rows:
        for entry in outcome.entries:
            spillage = ("—" if entry["spillage"] is None
                        else f"{entry['spillage']:.4e}")
            counts = " ".join(str(value) for value in entry["per_l"] or []) or "—"
            size = "—" if entry["bytes"] is None else str(entry["bytes"])
            verdict = "✅" if entry["ok"] else f"❌ {entry['reason'] or ''}".strip()
            lines.append(
                f"| {outcome.node_pk} | {entry['index']} | `{entry['scheme']}` "
                f"| {entry['nbands']} | {entry['checkpoint']} | {spillage} "
                f"| {counts} | {size} | {verdict} "
                f"| `{Path(entry['file']).name if entry['file'] else '—'}` |"
            )
    return "\n".join(lines) + "\n"
