"""Markdown report generation for orbgen runs (``aiida-orbgen report``).

The report is built from AiiDA provenance only — no filesystem assumptions —
so it works for a single ``OrbgenCalcWorkChain`` (one ``(l_max, r_cut)``) as
well as for an ``OrbgenGridSearchWorkChain`` (a whole candidate grid).

Public API
----------

* :func:`collect_summary` — walk a WorkChain node and return an
  :class:`OrbgenRunSummary` (grid points, energies, statuses).
* :func:`render_report` — turn that summary (+ orbital-export results) into
  the Markdown text written to ``report.md``.
* :func:`generate_report` — collect + render + write, in one call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Union

__all__ = [
    "GridPoint",
    "OrbgenRunSummary",
    "collect_summary",
    "render_report",
    "generate_report",
    "CALC_LABEL",
    "GRID_LABEL",
]

CALC_LABEL = "OrbgenCalcWorkChain"
GRID_LABEL = "OrbgenGridSearchWorkChain"


# ---------------------------------------------------------------------------
#  Summary collection
# ---------------------------------------------------------------------------


@dataclass
class GridPoint:
    """One ``(l_max, r_cut)`` point, i.e. one ``OrbgenCalcWorkChain``."""

    l_max: int | None
    r_cut: float | None
    pk: int
    exit_status: int | None
    finished_ok: bool
    process_state: str | None
    tolerance_meV: float | None = None
    delta_max_per_atom_meV: float | None = None
    delta_max_meV: float | None = None
    delta_min_per_atom_meV: float | None = None
    per_struct: list[dict] = field(default_factory=list)
    children: list[dict] = field(default_factory=list)
    siab_info: dict = field(default_factory=dict)
    output_dir: str | None = None
    error: str | None = None

    @property
    def n_data(self) -> int:
        """Number of dimers with both PW and LCAO energies."""
        return sum(
            1 for entry in self.per_struct
            if isinstance(entry.get("dE_per_atom"), (int, float))
        )

    @property
    def key(self) -> tuple:
        return (
            self.l_max if self.l_max is not None else -1,
            self.r_cut if self.r_cut is not None else -1.0,
        )


@dataclass
class OrbgenRunSummary:
    """Everything the report needs about one submitted WorkChain."""

    node_pk: int
    node_uuid: str
    label: str
    kind: str                       # "calc" | "gridsearch"
    status: str
    exit_status: int | None
    process_state: str | None
    tolerance_meV: float | None
    search_strategy: str | None = None
    grid: list[GridPoint] = field(default_factory=list)
    best: dict | None = None
    grid_summary: dict | None = None
    candidates: list[tuple[int, float]] = field(default_factory=list)

    @property
    def is_grid(self) -> bool:
        return self.kind == "gridsearch" or len(self.grid) > 1


def _status_label(node) -> str:
    if getattr(node, "is_finished_ok", False):
        return f"Finished OK [{node.exit_status}]"
    state = getattr(node, "process_state", None)
    label = state.value.capitalize() if state else "—"
    exit_status = getattr(node, "exit_status", None)
    if exit_status:
        return f"{label} [{exit_status}]"
    return label


def _status_emoji(node) -> str:
    if getattr(node, "is_finished_ok", False):
        return "🟢"
    state = getattr(node, "process_state", None)
    if state is not None and state.value in ("excepted", "killed"):
        return "🔴"
    return "🟡"


def _get_dict(node, name: str) -> dict:
    try:
        output = node.outputs[name]
    except (AttributeError, KeyError):
        return {}
    try:
        return dict(output.get_dict())
    except Exception:  # noqa: BLE001 — a non-Dict output is simply skipped
        return {}


def _calc_node_label(node) -> str:
    return getattr(node, "process_label", "") or type(node).__name__


def _extra(node, key: str):
    """Read one AiiDA extra without raising when it is missing."""
    try:
        return node.base.extras.all.get(key)
    except Exception:  # noqa: BLE001
        return None


def _collect_grid_point(calc_node) -> GridPoint:
    """Read one ``OrbgenCalcWorkChain`` node into a :class:`GridPoint`."""
    l_max: int | None = None
    r_cut: float | None = None
    try:
        l_max = int(calc_node.inputs.l_max.value)
        r_cut = float(calc_node.inputs.r_cut.value)
    except Exception:  # noqa: BLE001 — fall back to the outputs below
        pass

    energies = _get_dict(calc_node, "energies")
    siab_info = _get_dict(calc_node, "siab_info")
    if l_max is None:
        l_max = siab_info.get("lmax")
    if r_cut is None:
        r_cut = siab_info.get("rcut")

    per_struct = energies.get("delta_E_per_struct") or []
    min_per_atom = None
    values = [
        float(entry["dE_per_atom"]) * 1000.0
        for entry in per_struct
        if isinstance(entry.get("dE_per_atom"), (int, float))
    ]
    if values:
        min_per_atom = min(values)

    output_dir = None
    try:
        output_dir = calc_node.inputs.output_dir.value
    except Exception:  # noqa: BLE001
        pass

    children = []
    for child in getattr(calc_node, "called", []):
        # CalcFunctionNodes (energy aggregation …) are not processes the user
        # can open with `verdi process show`; keep the table to real children.
        if type(child).__name__ == "CalcFunctionNode":
            continue
        children.append({
            "pk": child.pk,
            "label": _calc_node_label(child),
            "status": _status_label(child),
            "emoji": _status_emoji(child),
            "task": _extra(child, "task") or "—",
            "basis": _extra(child, "basis") or "—",
            "exit_status": child.exit_status,
        })

    tolerance = None
    try:
        tolerance = float(calc_node.inputs.abacus_config.get_dict().get("tolerance_meV"))
    except Exception:  # noqa: BLE001
        tolerance = None

    return GridPoint(
        l_max=int(l_max) if l_max is not None else None,
        r_cut=float(r_cut) if r_cut is not None else None,
        pk=calc_node.pk,
        exit_status=calc_node.exit_status,
        finished_ok=bool(calc_node.is_finished_ok or calc_node.exit_status == 304),
        process_state=calc_node.process_state.value if calc_node.process_state else None,
        tolerance_meV=tolerance,
        delta_max_per_atom_meV=_as_float(energies.get("delta_E_max_per_atom_meV")),
        delta_max_meV=_as_float(energies.get("delta_E_max_meV")),
        delta_min_per_atom_meV=min_per_atom,
        per_struct=list(per_struct),
        children=children,
        siab_info=siab_info,
        output_dir=output_dir,
    )


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def collect_summary(node) -> OrbgenRunSummary:
    """Walk a WorkChain node (single point or grid search) into a summary."""
    label = _calc_node_label(node)

    if label == GRID_LABEL:
        kind = "gridsearch"
    elif label == CALC_LABEL:
        kind = "calc"
    else:
        # Not an orbgen WorkChain: still report status, but no grid data.
        return OrbgenRunSummary(
            node_pk=node.pk,
            node_uuid=str(node.uuid),
            label=label,
            kind="unknown",
            status=_status_label(node),
            exit_status=node.exit_status,
            process_state=node.process_state.value if node.process_state else None,
            tolerance_meV=None,
        )

    grid_summary = _get_dict(node, "grid_summary")
    tolerance = _as_float(grid_summary.get("tolerance_meV"))

    if kind == "calc":
        point = _collect_grid_point(node)
        if tolerance is None:
            tolerance = point.tolerance_meV
        grid = [point]
        best = None
        strategy = None
        candidates = [(point.l_max, point.r_cut)] if point.l_max is not None else []
    else:
        grid = [
            _collect_grid_point(child)
            for child in getattr(node, "called", [])
            if _calc_node_label(child) == CALC_LABEL
        ]
        grid.sort(key=lambda point: point.key)
        if tolerance is None:
            for point in grid:
                if point.tolerance_meV is not None:
                    tolerance = point.tolerance_meV
                    break
        strategy = grid_summary.get("search_strategy")
        candidates = [(point.l_max, point.r_cut) for point in grid
                      if point.l_max is not None]
        best = _best_from_summary(grid_summary, grid)

    return OrbgenRunSummary(
        node_pk=node.pk,
        node_uuid=str(node.uuid),
        label=label,
        kind=kind,
        status=_status_label(node),
        exit_status=node.exit_status,
        process_state=node.process_state.value if node.process_state else None,
        tolerance_meV=tolerance,
        search_strategy=strategy,
        grid=grid,
        best=best,
        grid_summary=grid_summary or None,
        candidates=candidates,
    )


def _best_from_summary(grid_summary: dict, grid: list[GridPoint]) -> dict | None:
    """Best point: as recorded by the grid search, else the lowest Max ΔE."""
    if grid_summary.get("best_calc_pk"):
        return {
            "l_max": grid_summary.get("best_l_max"),
            "r_cut": grid_summary.get("best_r_cut"),
            "delta_max_per_atom_meV": _as_float(
                grid_summary.get("best_delta_per_atom_meV")
            ),
            "calc_pk": grid_summary.get("best_calc_pk"),
            "source": "grid_summary",
        }
    ranked = [
        point for point in grid
        if point.delta_max_per_atom_meV is not None
        and point.delta_max_per_atom_meV > 0
    ]
    if not ranked:
        return None
    best_point = min(ranked, key=lambda point: point.delta_max_per_atom_meV)
    return {
        "l_max": best_point.l_max,
        "r_cut": best_point.r_cut,
        "delta_max_per_atom_meV": best_point.delta_max_per_atom_meV,
        "calc_pk": best_point.pk,
        "source": "lowest-deltaE",
    }


def select_best_point(summary: OrbgenRunSummary) -> GridPoint | None:
    """The :class:`GridPoint` a follow-up (final orbital) step should use."""
    if summary.best and summary.best.get("calc_pk"):
        for point in summary.grid:
            if point.pk == summary.best["calc_pk"]:
                return point
    if summary.kind == "calc" and summary.grid:
        return summary.grid[0]
    ranked = [
        point for point in summary.grid
        if point.delta_max_per_atom_meV is not None
        and point.delta_max_per_atom_meV > 0
    ]
    if ranked:
        return min(ranked, key=lambda point: point.delta_max_per_atom_meV)
    return summary.grid[0] if summary.grid else None


# ---------------------------------------------------------------------------
#  Rendering
# ---------------------------------------------------------------------------


def _format_cell(value: float | None, tolerance: float | None, fmt: str = "{:.1f}") -> str:
    """Format one ΔE cell; bold when it exceeds the tolerance."""
    if value is None:
        return "—"
    text = fmt.format(float(value))
    if tolerance is not None and float(value) > float(tolerance):
        return f"**{text}**"
    return text


def _matrix(summary: OrbgenRunSummary, value_of, *, title: str, note: str = "") -> list[str]:
    """Render one ``l_max × r_cut`` matrix."""
    l_max_values = sorted({point.l_max for point in summary.grid if point.l_max is not None})
    r_cut_values = sorted({point.r_cut for point in summary.grid if point.r_cut is not None})
    lines = [f"## {title}", ""]
    if note:
        lines.extend([note, ""])
    header = ["l_max \\ r_cut"] + [f"{rcut:.1f}" for rcut in r_cut_values]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| ---: | " + " | ".join([":---:"] * len(r_cut_values)) + " |")
    by_key = {point.key: point for point in summary.grid}
    for l_max in l_max_values:
        row = [f"**{l_max}**"]
        for r_cut in r_cut_values:
            point = by_key.get((l_max, r_cut))
            row.append("—" if point is None else value_of(point))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def render_report(
    summary: OrbgenRunSummary,
    *,
    orbital_files: Iterable[Any] = (),
    final_orbitals: Iterable[dict] | None = None,
    orbital_dirs: Iterable[Any] = (),
    include_process_logs: bool = True,
) -> str:
    """Render the Markdown report for ``summary``."""
    lines: list[str] = []
    lines.append("# Orbgen Report")
    lines.append("")
    lines.append(f"**Workflow**: {summary.label}<{summary.node_pk}>")
    lines.append(f"**UUID**: `{summary.node_uuid}`")
    lines.append(f"**Status**: {summary.status}")
    if summary.tolerance_meV is not None:
        lines.append(
            f"**Tolerance**: {summary.tolerance_meV} meV/atom "
            f"(0.1 kcal/mol/atom)"
        )
    if summary.search_strategy:
        lines.append(f"**Search strategy**: {summary.search_strategy}")
    if summary.candidates:
        lines.append(
            "**Candidates**: "
            + ", ".join(f"(l_max={lm}, r_cut={rcut:g})" for lm, rcut in summary.candidates)
        )
    lines.append(f"**Report generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("---")
    lines.append("")

    if not summary.grid:
        lines.append("_(no OrbgenCalcWorkChain information available for this node)_")
        lines.append("")
        return _render_tail(lines, orbital_files, final_orbitals, orbital_dirs)

    tolerance = summary.tolerance_meV

    # ── ΔE tables ────────────────────────────────────────────────────────
    if summary.is_grid:
        lines.extend(_matrix(
            summary,
            lambda point: _format_cell(point.delta_max_per_atom_meV, tolerance),
            title="1a. ΔE Max (per atom, meV) — l_max × r_cut",
            note="Cells in **bold** exceed the convergence tolerance.",
        ))
        lines.extend(_matrix(
            summary,
            lambda point: str(point.n_data) if point.n_data else "—",
            title="1a2. Per-dimer ΔE data count — l_max × r_cut",
            note=("Number of dimers that produced a per-atom ΔE value "
                  "(both PW and LCAO:nsw energies available)."),
        ))
        lines.extend(_matrix(
            summary,
            lambda point: _format_cell(point.delta_min_per_atom_meV, tolerance),
            title="1b. ΔE Min (per atom, meV) — l_max × r_cut",
        ))
        lines.extend(_matrix(
            summary,
            lambda point: _exit_cell(point),
            title="Exit Code matrix",
            note=("OrbgenCalcWorkChain exit_status per grid point. "
                  "`0`=OK · `304`=WARNING_TOLERANCE_EXCEEDED · other=real failure."),
        ))

    best = summary.best
    if best and best.get("l_max") is not None:
        verdict = ""
        value = best.get("delta_max_per_atom_meV")
        if value is not None and tolerance is not None:
            verdict = " ✅ converged" if value <= tolerance else " ❌ NOT converged"
        lines.append(
            f"**Best**: (l_max={best['l_max']}, r_cut={_rcut_str(best.get('r_cut'))}) → "
            f"{value if value is None else f'{value:.2f}'} meV/atom{verdict} "
            f"(OrbgenCalcWorkChain<{best.get('calc_pk')}>, picked by "
            f"{best.get('source')})"
        )
        lines.append("")

    # ── single point summary ─────────────────────────────────────────────
    if not summary.is_grid:
        point = summary.grid[0]
        lines.append("## 1. Single (l_max, r_cut) point")
        lines.append("")
        lines.append("| Property | Value |")
        lines.append("| --- | --- |")
        lines.append(f"| l_max | {point.l_max} |")
        lines.append(f"| r_cut (au) | {point.r_cut:g} |")
        lines.append(f"| OrbgenCalcWorkChain | <{point.pk}> |")
        lines.append(f"| Status | {_point_status(point)} |")
        lines.append(f"| ΔE_max (per atom) | {_fmt(point.delta_max_per_atom_meV)} meV |")
        lines.append(f"| ΔE_max (per system) | {_fmt(point.delta_max_meV)} meV |")
        lines.append(f"| ΔE_min (per atom) | {_fmt(point.delta_min_per_atom_meV)} meV |")
        if point.tolerance_meV is not None:
            lines.append(f"| Tolerance | {point.tolerance_meV} meV/atom |")
        lines.append("")

    # ── orbital inventory ────────────────────────────────────────────────
    lines.append("## 2. Basis (primitive NSW orbital) inventory")
    lines.append("")
    lines.append("| l_max | r_cut | family label | primitive .orb | output dir |")
    lines.append("| ---: | ---: | --- | --- | --- |")
    for point in summary.grid:
        info = point.siab_info or {}
        lines.append(
            f"| {point.l_max} | {_rcut_str(point.r_cut)} "
            f"| `{info.get('family_label', '—')}` "
            f"| `{info.get('nsw_filename', '—')}` "
            f"| `{point.output_dir or '—'}` |"
        )
    lines.append("")

    # ── per-grid details ─────────────────────────────────────────────────
    lines.append("## 3. Per-point details")
    lines.append("")
    for point in summary.grid:
        lines.append(
            f"### grid (l_max={point.l_max}, r_cut={_rcut_str(point.r_cut)})"
            f" — OrbgenCalcWorkChain<{point.pk}> <a id=\"grid-{point.pk}\"></a>"
        )
        lines.append("")
        lines.append(f"- Status: {_point_status(point)}")
        lines.append(f"- ΔE_max/atom: {_fmt(point.delta_max_per_atom_meV)} meV")
        lines.append(f"- ΔE_max/system: {_fmt(point.delta_max_meV)} meV")
        pertmags = (point.siab_info or {}).get("pertmags")
        if pertmags:
            lines.append(f"- Perturbations (Å): {pertmags}")
        lines.append("")
        if point.per_struct:
            lines.append("**Per-dimer energies:**")
            lines.append("")
            lines.append("| Dimer | N atoms | E_pw (Ha) | E_lcao_nsw (Ha) | ΔE/atom (meV) |")
            lines.append("| --- | ---: | ---: | ---: | ---: |")
            for entry in point.per_struct:
                d_per_atom = entry.get("dE_per_atom")
                lines.append(
                    f"| {entry.get('folder', '—')} | {entry.get('n_atoms', '—')} "
                    f"| {_fmt(entry.get('E_pw'), 6)} | {_fmt(entry.get('E_lcao_nsw'), 6)} "
                    f"| {_fmt(None if d_per_atom is None else float(d_per_atom) * 1000.0, 3)} |"
                )
            lines.append("")
        else:
            lines.append("_(no per-dimer energies — all calcs failed or were skipped)_")
            lines.append("")
        if point.children:
            lines.append("**abacus.base children:**")
            lines.append("")
            lines.append("| Dimer | basis | pk | Status | exit |")
            lines.append("| --- | --- | ---: | --- | ---: |")
            for child in point.children:
                lines.append(
                    f"| {child['task']} | {child['basis']} | <{child['pk']}> "
                    f"| {child['emoji']} {child['status']} | {child['exit_status']} |"
                )
            lines.append("")

    # ── process logs ─────────────────────────────────────────────────────
    if include_process_logs:
        from aiida.cmdline.utils.common import get_workchain_report

        lines.append("## 4. Process reports")
        lines.append("")
        lines.append("Process logs (same as `verdi process report`).")
        lines.append("")
        for point in summary.grid:
            try:
                from aiida.orm import load_node

                wc_node = load_node(point.pk)
                report_text = get_workchain_report(wc_node, "REPORT")
            except Exception as exc:  # noqa: BLE001
                report_text = f"<failed to read process report: {exc}>"
            lines.append(f"### OrbgenCalcWorkChain<{point.pk}> (l_max={point.l_max}, r_cut={_rcut_str(point.r_cut)})")
            lines.append("")
            lines.append("```")
            lines.append(report_text or "_(no report logs)_")
            lines.append("```")
            lines.append("")

    return _render_tail(lines, orbital_files, final_orbitals, orbital_dirs)


def _render_tail(
    lines: list[str],
    orbital_files: Iterable[Any],
    final_orbitals: Iterable[dict] | None = None,
    orbital_dirs: Iterable[Any] = (),
) -> str:
    """Append the orbital-file sections and join everything."""
    orbital_files = list(orbital_files)
    lines.append("## 5. Orbital files")
    lines.append("")
    if orbital_files:
        lines.append("| File | Directory | Source | Family | l_max | r_cut |")
        lines.append("| --- | --- | --- | --- | ---: | ---: |")
        for item in orbital_files:
            lines.append(
                f"| `{Path(item.path).name}` | `{Path(item.path).parent.name or '.'}` "
                f"| {item.source} | `{item.family_label or '—'}` "
                f"| {item.l_max if item.l_max is not None else '—'} "
                f"| {item.r_cut if item.r_cut is not None else '—'} |"
            )
    else:
        lines.append("_(no orbital file could be exported — see the log for details)_")
    lines.append("")

    lines.append("## 6. Final CSW-NAO orbitals")
    lines.append("")
    final_orbitals = list(final_orbitals or [])
    if not final_orbitals:
        lines.append("_(final orbital generation was not requested)_")
    for entry in final_orbitals:
        point = ""
        if entry.get("l_max") is not None:
            r_cut = entry.get("r_cut")
            r_cut = int(r_cut) if isinstance(r_cut, (int, float)) and float(r_cut).is_integer() \
                else r_cut
            point = f"l_max={entry['l_max']}, r_cut={r_cut}"
        lines.append(f"### {point or 'grid point'} — {entry.get('status')}")
        lines.append("")
        for key, label in (
            ("dir", "Orbital directory"),
            ("dft_root", "DFT root"),
            ("dft_root_source", "DFT root from"),
            ("config", "SIAB config"),
            ("config_source", "Config source"),
            ("command", "Command"),
            ("log", "Log"),
        ):
            if entry.get(key) is not None:
                lines.append(f"- {label}: `{entry[key]}`")
        assembled = entry.get("assemble") or {}
        if assembled:
            lines.append("- Reference assembly: " + str(
                assembled.get("summary") or assembled.get("message") or "failed"
            ))
            for folder, detail_entry in (assembled.get("folders") or {}).items():
                detail = detail_entry.get("status")
                if detail_entry.get("dev_eV") is not None:
                    detail += (f" (bands={detail_entry.get('bands')}, "
                               f"ΔE={detail_entry['dev_eV']:.1e} eV)")
                if detail_entry.get("message"):
                    detail += f" — {detail_entry['message']}"
                lines.append(f"    - `{folder}`: {detail}")
        if entry.get("quarantined"):
            moved = ", ".join(f"`{folder}`→{name}" for folder, name in
                              entry["quarantined"].items())
            lines.append(f"- Stale reference data moved aside: {moved}")
        if entry.get("config_notice"):
            lines.append(f"- ⚠ {entry['config_notice']}")
        if entry.get("message"):
            lines.append(f"- Note: {entry['message']}")
        produced = entry.get("files") or []
        if produced:
            lines.append("")
            lines.append("| Orbital file | Bytes |")
            lines.append("| --- | ---: |")
            for path in produced:
                path = Path(path)
                size = path.stat().st_size if path.exists() else 0
                lines.append(f"| `{path.name}` | {size} |")
        lines.append("")

    # An index of everything the report directory holds, so a report generated
    # for one grid point still documents the orbitals of the others.
    orbital_dirs = [Path(path) for path in orbital_dirs]
    if orbital_dirs:
        lines.append("")
        lines.append("### Orbital directories in this report")
        lines.append("")
        for directory in sorted(orbital_dirs):
            files = sorted(directory.glob("*.orb"))
            listed = ", ".join(f"`{f.name}`" for f in files) or "_no .orb yet_"
            lines.append(f"- `{directory.name}/`: {listed}")
    lines.append("")
    return "\n".join(lines)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _point_status(point: GridPoint) -> str:
    if point.finished_ok and point.exit_status in (0, 304):
        return f"Finished [{point.exit_status}]"
    return f"{point.process_state or '—'} [{point.exit_status}]"


def _exit_cell(point: GridPoint) -> str:
    if point.exit_status is None:
        return "—"
    if point.exit_status == 0:
        return f"[0](#grid-{point.pk})"
    if point.exit_status == 304:
        return f"[304 ⚠](#grid-{point.pk})"
    return f"[**{point.exit_status}** ✗](#grid-{point.pk})"


def _rcut_str(r_cut: float | None) -> str:
    return "—" if r_cut is None else format(float(r_cut), "g")


def generate_report(
    node_identifier: Union[int, str],
    output_path: Union[str, Path],
    *,
    profile: str | None = None,
    orbital_files: Iterable[Any] = (),
    final_orbitals: Iterable[dict] | None = None,
    orbital_dirs: Iterable[Any] = (),
    include_process_logs: bool = True,
    node=None,
) -> str:
    """Collect + render + write a Markdown report; returns its path."""
    from aiida import load_profile

    load_profile(profile)
    if node is None:
        from aiida.orm import load_node

        node = load_node(node_identifier)

    summary = collect_summary(node)
    text = render_report(
        summary,
        orbital_files=orbital_files,
        final_orbitals=final_orbitals,
        orbital_dirs=orbital_dirs,
        include_process_logs=include_process_logs,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return str(output_path)
