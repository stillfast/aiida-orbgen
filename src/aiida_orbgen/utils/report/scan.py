"""``report.md`` for the two value-selection scans.

A scan chooses parameters; it produces no orbital flat, so
:func:`~aiida_orbgen.utils.report.generate_one_report` — which fits the contracted
CSW-NAO orbital against a reference DFT tree — has nothing to do for it.  What a scan
*does* have is the thing such a run exists for: the total energies of the reference
geometries in two bases and the difference between them

.. math::  \\Delta E = E_{\\mathrm{lcao}} - E_{\\mathrm{pw}},

so this module renders exactly that, from the provenance:

* every child's ``misc.total_energy`` (the ``AbacusBaseWorkChain`` nodes the scan
  submitted, which is where the raw energies live — the ``basis_decision`` Dict only
  keeps the differences);
* paired per geometry through
  :func:`~aiida_orbgen.workflows._children.geometries_from_entries`, i.e. through the
  same normalised geometry names the scan used (``dimer-2.8``, never the folder name
  ``U-dimer-2.80-12au``, which changes with ``r_cut``);
* judged through the same rule the scan applied,
  :func:`~aiida_orbgen.workflows.ladder.basis_table`.

The products are ``report.md``, ``energies.csv`` (the same numbers, one row per
candidate × geometry) and ``decision.json`` (the decision Dict verbatim), so the table
can be re-made offline.

For ``orbgen.ecutwfc`` the table is the other way round — one row per plane-wave cutoff,
with the per-geometry energy and the neighbouring-step difference the convergence rule
uses — and, when ``with_lcao`` ran, the LCAO-vs-PW block at the reference cutoff.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

__all__ = [
    "ScanReport",
    "build_scan_report",
    "render_scan_report",
    "write_scan_report",
]

MEV_PER_EV = 1000.0


@dataclass
class ScanReport:
    """Everything ``report.md`` needs, read off the provenance."""

    workflow: str
    pk: int
    uuid: str
    label: str
    status: str
    exit_status: int | None
    decision: dict = field(default_factory=dict)
    #: ``"label::geometry" -> {"e_pw": eV, "e_nsw": eV, "n_atoms": N}``
    cells: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: candidate label -> row of ``ladder.basis_table`` (empty for ecutwfc)
    table: list[dict] = field(default_factory=list)
    geometries: list[str] = field(default_factory=list)
    #: the winner, recomputed from the table
    chosen: dict = field(default_factory=dict)
    #: ecutwfc only: ``cutoff -> {geometry: energy}``
    curve: dict[float, dict[str, float]] = field(default_factory=dict)
    lcao_cutoff: float | None = None
    #: the AbacusBaseWorkChain children, for the per-child table at the end
    children: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
#  reading the provenance
# ---------------------------------------------------------------------------
def _scan_children(node) -> list[dict]:
    """The scan's children as its own ``ctx.children_info`` saw them.

    ``AbacusBaseWorkChain`` children carry the synthetic task name (``label::geometry``)
    and the basis in their extras — that is what the workchain keyed its own bookkeeping
    by, and it is the only place the raw energies survive.
    """
    children: list[dict] = []
    for child in getattr(node, "called", []) or []:
        if child.process_label != "AbacusBaseWorkChain":
            continue
        extras = child.base.extras.all
        geometry, label = extras.get("task"), extras.get("label")
        if geometry is None or label is None:
            continue
        children.append({
            "task": f"{label}::{geometry}",
            "basis": str(extras.get("basis") or ""),
            "node": child,
            "n_atoms": int(extras.get("n_atoms") or 1),
        })
    return children


def build_scan_report(node, *, report=None) -> ScanReport:
    """Collect a scan's energies from the database into a :class:`ScanReport`."""
    from aiida_orbgen.workflows._children import seconds_of, split_task
    from aiida_orbgen.workflows.extract import collect_child_energies

    outputs = getattr(node, "outputs", None)
    decision: dict = {}
    workflow = node.process_label
    if outputs is not None and "basis_decision" in outputs:
        decision = outputs.basis_decision.get_dict()
        workflow = "orbgen.basis"
    elif outputs is not None and "ecutwfc_decision" in outputs:
        decision = outputs.ecutwfc_decision.get_dict()
        workflow = "orbgen.ecutwfc"
    else:
        raise ValueError(
            f"node {getattr(node, 'pk', node)} carries neither a basis_decision nor an "
            f"ecutwfc_decision output; is it an `orbgen.basis` / `orbgen.ecutwfc` run?"
        )

    report_data = ScanReport(
        workflow=workflow,
        pk=int(node.pk),
        uuid=str(node.uuid),
        label=str(node.label or ""),
        status=f"{node.process_state.value} [{node.exit_status}]",
        exit_status=node.exit_status,
        decision=decision,
    )

    collected = collect_child_energies(_scan_children(node), report=report)
    report_data.warnings.extend(collected.notes)

    cells: dict[str, dict[str, Any]] = {}
    for entry in collected.entries:
        label, geometry = split_task(entry.folder)
        slot = cells.setdefault(f"{label}::{geometry}", {"n_atoms": int(entry.n_atoms)})
        slot["e_nsw" if entry.basis != "pw" else "e_pw"] = float(entry.energy)

    if workflow == "orbgen.ecutwfc":
        # the cutoff of a PW child is part of its label ("pw@150"), and the LCAO child
        # ran at the reference cutoff
        curve: dict[float, dict[str, float]] = {}
        for key, slot in cells.items():
            label, geometry = split_task(key)
            if label.startswith("pw@") and slot.get("e_pw") is not None:
                cutoff = float(label.split("@", 1)[1])
                curve.setdefault(cutoff, {})[geometry] = float(slot["e_pw"])
        report_data.curve = curve
        report_data.lcao_cutoff = float(decision.get("reference_ecutwfc") or 0.0)
        report_data.cells = cells
        report_data.geometries = sorted({g for entry in curve.values() for g in entry})
        return report_data

    # `orbgen.basis`: the same table the scan itself built, plus the raw energies.
    from aiida_orbgen.workflows._children import geometries_from_entries
    from aiida_orbgen.workflows.ladder import basis_table

    pw_reference = {
        key.split("::", 1)[1]: {"energy": slot["e_pw"], "n_atoms": slot["n_atoms"]}
        for key, slot in cells.items()
        if slot.get("e_pw") is not None and key.split("::", 1)[0].startswith("pw")
    }
    rows_by_label = geometries_from_entries(
        # the one-off PW reference is not a candidate -- passing its children through as
        # entries produced a spurious "pw" row that could never be judged
        [entry for entry in collected.entries if entry.basis != "pw"],
        pw_reference=pw_reference,
    )
    table = basis_table(
        {label: {"candidate": _candidate_of(decision, label), "geometries": geometries}
         for label, geometries in rows_by_label.items()},
        float(decision.get("tolerance_meV") or 100.0),
        reference=decision.get("reference"),
    )
    from aiida_orbgen.workflows._children import n_primitive_functions
    from aiida_orbgen.workflows.ladder import candidate_cost, pick_cheapest

    seconds: dict[str, float] = {}
    for child in _scan_children(node):
        label, _geometry = child["task"].split("::", 1)
        value = seconds_of(child["node"])
        if value is not None:
            seconds[label] = seconds.get(label, 0.0) + value
    gate = decision.get("atomization_tolerance_meV")
    for row in table["rows"]:
        row["seconds"] = seconds.get(row["label"])
        candidate = row.get("candidate") or {}
        if row.get("nchi") is None and candidate:
            row["nchi"] = n_primitive_functions(
                candidate.get("r_cut"), candidate.get("ecutjy"), candidate.get("l_max")
            )
        if row.get("cost") is None:
            row["cost"] = candidate_cost(row.get("nchi"), candidate.get("r_cut"))
        drift = row.get("atomization_vs_reference_meV")
        row["gate_ok"] = (gate is None) or (
            drift is not None and abs(float(drift)) <= float(gate)
        )
        row["passed"] = bool(row["tolerance_ok"]) and bool(row["gate_ok"])
    report_data.table = table["rows"]
    # Recompute the winner from the table instead of trusting the stored one: a scan run
    # by an older checkout recorded a wrong `best` (the reference point, picked by a few
    # cached seconds), and the report has to show what the numbers say now.
    report_data.chosen = pick_cheapest(
        {"rows": table["rows"]},
        by=str(decision.get("by") or "seconds"),
        require_atomization=gate is not None,
        atomization_tolerance_meV=gate,
        reference=decision.get("reference"),
    )
    if report_data.chosen and decision.get("best") and \
            report_data.chosen["label"] != decision.get("best"):
        report_data.warnings.append(
            f"the stored decision says best = {decision['best']}, but this table says "
            f"{report_data.chosen['label']} — the run was made before the cost/verdict "
            f"fixes, so trust the table (and rerun if you need the decision itself "
            f"corrected)"
        )
    report_data.cells = cells
    order = decision.get("evaluation_order") or decision.get("ran") or []
    geometries = sorted(
        {key.split("::", 1)[1] for key in cells},
        key=lambda name: ("monomer" in name, name),
    )
    report_data.geometries = geometries
    report_data.decision["_row_order"] = order
    return report_data


def _candidate_of(decision: dict, label: str) -> dict:
    for row in decision.get("table") or []:
        if row.get("label") == label:
            return dict(row.get("candidate") or {})
    point = decision.get("reference_point") or {}
    if label == decision.get("reference"):
        return dict(point)
    parts = label.split("_")
    try:
        return {"r_cut": float(parts[0][1:]), "l_max": int(parts[1][1:]),
                "ecutjy": float(parts[2][1:])}
    except (IndexError, ValueError):
        return {}


# ---------------------------------------------------------------------------
#  rendering (pure: a ScanReport in, Markdown out)
# ---------------------------------------------------------------------------
def _fmt(value, spec: str = ".4f", dash: str = "—") -> str:
    if value is None:
        return dash
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def _fmt_atom(name: str) -> str:
    """``dimer-2.1`` -> ``U2 2.10 A``, ``monomer`` -> ``U1`` in the header row."""
    if "monomer" in name:
        return "monomer"
    parts = name.split("-")
    try:
        return f"dimer {float(parts[-1]):.2f} A"
    except ValueError:
        return name


def _header(data: ScanReport) -> list[str]:
    decision = data.decision
    lines = [
        f"# Orbgen {'Basis' if data.workflow == 'orbgen.basis' else 'Ecutwfc'} Scan Report",
        "",
        f"**Workflow**: `{data.workflow}` (PK {data.pk})",
        f"**UUID**: `{data.uuid}`",
        f"**Status**: {data.status}",
        f"**Report generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    if data.workflow == "orbgen.basis":
        point = decision.get("reference_point") or {}
        pw = decision.get("pw_reference") or {}
        lines += [
            f"**Reference point**: `{decision.get('reference')}` "
            f"(l_max={point.get('l_max')}, r_cut={point.get('r_cut')} au, "
            f"ecutjy={point.get('ecutjy')} Ry)",
            f"**Criterion**: max |ΔE|/atom over the dimers ≤ "
            f"{_fmt(decision.get('tolerance_meV'), '.3g')} meV"
            + (f", atomization gate ≤ "
               f"{_fmt(decision.get('atomization_tolerance_meV'), '.3g')} meV"
               if decision.get("atomization_tolerance_meV") is not None else ""),
            f"**PW reference**: {pw.get('source') or 'n/a'}, "
            f"ecutwfc = {_fmt(pw.get('ecutwfc'), '.3g')} Ry, "
            f"{pw.get('n_geometries')} geometries",
            f"**Candidates**: ran {len(decision.get('ran') or [])} of "
            f"{len(decision.get('planned') or [])}"
            + (" (stopped at the first passing one)" if decision.get("stopped_early") else ""),
            "",
            "ΔE = `E_lcao − E_pw` of the same geometry, from the children's "
            "`misc.total_energy`.",
            "",
        ]
    else:
        lines += [
            f"**Ladder**: ecutwfc = "
            + ", ".join(_fmt(v, '.6g') for v in decision.get("values") or []) + " Ry",
            f"**Criterion**: the total energy stops moving between neighbouring cutoffs "
            f"(≤ {_fmt(decision.get('tolerance_meV'), '.3g')} meV/atom)",
            f"**Chosen**: **{_fmt(decision.get('chosen'), '.6g')} Ry**"
            + ("" if decision.get("converged") else " (NOT converged — run a wider ladder)"),
            "",
        ]
    for warning in data.warnings:
        lines.append(f"> {warning.strip()}")
    if data.warnings:
        lines.append("")
    return lines


def _pw_by_geometry(data: ScanReport) -> dict[str, float]:
    """``{geometry: E_pw}`` from the PW children.

    The PW energy of a geometry is shared by every candidate — that is what makes the
    comparison a comparison — so it is *not* stored in the LCAO cells and has to be
    looked up per geometry.  (Reading it from the candidate's own cell is how the first
    version of this table printed ``E_pw`` and ``ΔE`` as "—" for every candidate.)
    """
    out: dict[str, float] = {}
    for key, slot in data.cells.items():
        label, geometry = key.split("::", 1)
        if label.startswith("pw") and slot.get("e_pw") is not None:
            out[geometry] = float(slot["e_pw"])
    return out


def _render_basis_tables(data: ScanReport) -> list[str]:
    geometries = data.geometries
    decision = data.decision
    order = decision.get("_row_order") or []
    rows = sorted(
        data.table,
        key=lambda row: (order.index(row["label"]) if row["label"] in order else len(order),
                         row["label"]),
    )
    header = ["candidate", *(_fmt_atom(g) for g in geometries), "max dimer", "dA", "cost",
              "verdict"]

    def cell(value, spec="+.2f"):
        return "—" if value is None else format(float(value), spec)

    out = [
        "## 1. Energy difference per candidate (meV/atom)",
        "",
        "| " + " | ".join(header) + " |",
        "| --- | " + " | ".join("---:" for _ in header[1:]) + " |",
    ]
    for row in rows:
        dE = row.get("dE_per_atom_meV") or {}
        cells = [cell(dE.get(g)) for g in geometries]
        label = row["label"] + (" *(reference)*" if row["label"] == decision.get("reference") else "")
        verdict = "✓" if row.get("passed") else "✗"
        if not row.get("gate_ok", True):
            verdict += " *gate*"
        if data.chosen and row["label"] == data.chosen.get("label"):
            verdict += " **chosen**"
        out.append(
            f"| {label} | " + " | ".join(cells)
            + f" | {_fmt(row.get('dE_max_abs_meV'), '.2f')}"
            + f" | {cell(row.get('atomization_vs_reference_meV'), '+.1f')}"
            + " | " + (f"{_fmt(row.get('seconds'), '.0f')} s" if row.get("seconds") is not None
                       else (f"{_fmt(row.get('cost'), '.0f')} (proxy)" if row.get("cost")
                             else "—"))
            + f" | {verdict} |"
        )
    for label in decision.get("planned_but_not_run") or []:
        out.append(f"| {label} | " + " | ".join("—" for _ in geometries)
                   + " | — | — | — | not run |")
    pw_energies = _pw_by_geometry(data)
    out += [
        "",
        "`max dimer` is the criterion (the monomer is the atomization reference, so its "
        "error is listed but not judged); `dA` is the atomization-energy difference "
        "against the reference candidate.",
        "",
        "## 2. Energies behind the differences (eV)",
        "",
        "### 2a. PW reference (shared by every candidate)",
        "",
        "| geometry | N | E_pw (eV) |",
        "| --- | ---: | ---: |",
    ]
    for geometry in geometries:
        n_atoms = 1
        for key, slot in data.cells.items():
            if key.endswith(f"::{geometry}") and key.startswith("pw"):
                n_atoms = int(slot.get("n_atoms") or 1)
                break
        out.append(f"| {geometry} | {n_atoms} | {_fmt(pw_energies.get(geometry))} |")
    out += [
        "",
        "### 2b. LCAO (NSW) against the PW reference",
        "",
        "| candidate | geometry | N | E_pw (eV) | E_nsw (eV) | ΔE (meV) | ΔE/atom (meV) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        label = row["label"]
        for geometry in geometries:
            slot = data.cells.get(f"{label}::{geometry}") or {}
            e_nsw = slot.get("e_nsw")
            e_pw = pw_energies.get(geometry)
            n_atoms = int(slot.get("n_atoms") or 1)
            d_e = None if e_pw is None or e_nsw is None else (e_nsw - e_pw) * MEV_PER_EV
            out.append(
                f"| {label} | {geometry} | {n_atoms} | {_fmt(e_pw)} | {_fmt(e_nsw)} "
                f"| {cell(d_e)} | {cell(None if d_e is None else d_e / n_atoms)} |"
            )

    atomization = [
        "## 3. Atomization energy `A = E(dimer) − 2·E(monomer)` (eV)",
        "",
        "| candidate | A_lcao | A_pw | A_lcao − A_pw (meV) | vs reference (meV) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    any_atomization = False
    for row in rows:
        label = row["label"]
        nsw = {
            geometry: float(slot["e_nsw"])
            for geometry in geometries
            for slot in [data.cells.get(f"{label}::{geometry}") or {}]
            if slot.get("e_nsw") is not None
        }
        pw = {geometry: energy for geometry, energy in pw_energies.items()}
        monomer = next((g for g in geometries if "monomer" in g), None)
        dimers = [g for g in geometries if "monomer" not in g]
        if monomer is None or not dimers:
            continue
        worst_nsw = max((nsw[g] - 2 * nsw[monomer] for g in dimers if g in nsw), default=None)
        worst_pw = max((pw[g] - 2 * pw[monomer] for g in dimers if g in pw), default=None)
        if worst_nsw is None or worst_pw is None:
            continue
        any_atomization = True
        atomization.append(
            f"| {label} | {_fmt(worst_nsw)} | {_fmt(worst_pw)} "
            f"| {cell((worst_nsw - worst_pw) * MEV_PER_EV, '+.1f')} "
            f"| {cell(row.get('atomization_vs_reference_meV'), '+.1f')} |"
        )
    out += atomization if any_atomization else []
    return out


def _render_ecutwfc_tables(data: ScanReport) -> list[str]:
    decision = data.decision
    values = [float(v) for v in decision.get("values") or sorted(data.curve)]
    geometries = data.geometries
    out = [
        "## 1. PW total energy vs ecutwfc",
        "",
        "| ecutwfc (Ry) | " + " | ".join(_fmt_atom(g) for g in geometries)
        + " | step: max |ΔE|/atom | within |",
        "| ---: | " + " | ".join("---:" for _ in geometries) + " | ---: | :--: |",
    ]
    steps = {float(step["from"]): step for step in decision.get("steps") or []}
    for cutoff in values:
        energies = data.curve.get(cutoff, {})
        cells = [_fmt(energies.get(g)) for g in geometries]
        step = steps.get(cutoff)
        marker = "—"
        within = "—"
        if step:
            marker = _fmt(step.get("max_meV"), ".2f") + " meV/atom"
            within = "✓" if step.get("within_tolerance") else "✗"
        out.append(f"| {_fmt(cutoff, '.6g')} | " + " | ".join(cells)
                   + f" | {marker} | {within} |")
    out += [
        "",
        "`step` is the difference against the *next* cutoff of the ladder, worst geometry "
        "first; the decision is the smallest cutoff from which every further step is "
        "inside the tolerance.",
        "",
    ]

    lcao = [
        "## 2. LCAO vs PW at the reference cutoff (eV, meV/atom)",
        "",
    ]
    cutoff = data.lcao_cutoff
    rows = []
    for key, slot in data.cells.items():
        label, geometry = key.split("::", 1)
        if label.startswith("pw@"):
            continue
        pw = (data.curve.get(cutoff) or {}).get(geometry)
        if pw is None or slot.get("e_nsw") is None:
            continue
        n_atoms = int(slot.get("n_atoms") or 1)
        d_e = (slot["e_nsw"] - pw) * MEV_PER_EV
        rows.append((geometry, n_atoms, pw, slot["e_nsw"], d_e, d_e / n_atoms))
    if rows:
        lcao += [
            f"Reference cutoff: **{_fmt(cutoff, '.6g')} Ry**",
            "",
            "| geometry | N | E_pw (eV) | E_nsw (eV) | ΔE (meV) | ΔE/atom (meV) |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for geometry, n_atoms, pw, nsw, d_e, per_atom in sorted(rows):
            lcao.append(f"| {geometry} | {n_atoms} | {_fmt(pw)} | {_fmt(nsw)} "
                        f"| {_fmt(d_e, '+.2f')} | {_fmt(per_atom, '+.2f')} |")
        worst = max(rows, key=lambda row: abs(row[5]))
        info = decision.get("lcao_vs_pw") or {}
        lcao += [
            "",
            f"Worst geometry: **{worst[0]}** at {_fmt(worst[5], '+.2f')} meV/atom "
            + ("✓" if info.get("tolerance_ok") else "✗")
            + f" — the basis of this run against the converged PW limit.",
            "",
        ]
    return out + (lcao if rows else [])


def render_scan_report(data: ScanReport) -> str:
    """The Markdown body of a scan's ``report.md`` (pure)."""
    lines = _header(data)
    lines += _render_basis_tables(data) if data.workflow == "orbgen.basis" \
        else _render_ecutwfc_tables(data)

    decision = data.decision
    lines += ["## Final", ""]
    if data.workflow == "orbgen.basis":
        best = (data.chosen or {}).get("label") or decision.get("best")
        chosen = data.chosen or (decision.get("best_row") or {})
        if best:
            lines += [
                f"- **chosen**: `{best}`"
                + (" — the reference point itself (no cheaper candidate passed)"
                   if best == decision.get("reference") else ""),
                f"- candidate: r_cut **{chosen.get('candidate', {}).get('r_cut')} au**, "
                f"l_max **{chosen.get('candidate', {}).get('l_max')}**, "
                f"ecutjy **{chosen.get('candidate', {}).get('ecutjy')} Ry**",
                f"- max |ΔE|/atom (dimers): "
                f"{_fmt(chosen.get('dE_max_abs_meV'), '.2f')} meV"
                + (f", dA = {_fmt(chosen.get('atomization_vs_reference_meV'), '+.1f')} meV"
                   if chosen.get("atomization_vs_reference_meV") is not None else ""),
                f"- cost: {_fmt(chosen.get('seconds'), '.0f', '—')} s "
                f"of children, nchi = {chosen.get('nchi')}",
                "",
                "Feed these values into the `input.json` of an `orbgen.calc` run "
                "(`parameters.orbgen`), then `aiida-orbgen report` on *that* run to get "
                "the contracted orbital — see the project's `REPRODUCE.md`.",
            ]
        else:
            lines.append("- no candidate met the criterion: widen the ladder or relax "
                         "`tolerance_meV` / `atomization_tolerance_meV`")
    else:
        lines += [
            f"- **chosen ecutwfc**: {_fmt(decision.get('chosen'), '.6g')} Ry"
            + ("" if decision.get("converged") else " (**not converged**)"),
            f"- steps: "
            + ", ".join(f"{_fmt(s.get('max_meV'), '.2f')}"
                        for s in decision.get("steps") or []) + " meV/atom",
            "",
            "The cutoff affects the *reference* DFT only: an LCAO basis is not a "
            "plane-wave basis, so the orbitals themselves do not depend on it. The "
            "`pw_reference` block of this run can be reused by `orbgen.basis` "
            "(`input.json[\"scan\"][\"pw_reference_pk\"]`).",
        ]

    children = data.children
    if children:
        lines += ["", "## Children", "",
                  "| # | task | basis | PK | exit | seconds |", "| ---: | --- | --- | ---: | ---: | ---: |"]
        for index, child in enumerate(children, start=1):
            lines.append(
                f"| {index} | {child['task']} | {child['basis']} | {child['node'].pk} "
                f"| {child['node'].exit_status} | "
                f"{_fmt(child.get('seconds'), '.0f', '—')} |"
            )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
#  writing the products
# ---------------------------------------------------------------------------
def _write_energies_csv(data: ScanReport, path: Path) -> Path:
    """One row per (candidate, geometry): ``E_pw``, ``E_nsw`` and their difference.

    The PW energy of a geometry is shared by every candidate (that is the whole point of
    the scan), so it is looked up by geometry rather than stored per candidate — and the
    PW children appear once, as the reference block.
    """
    def row(candidate, geometry, n_atoms, e_pw, e_nsw, note=""):
        d_e = None if e_pw is None or e_nsw is None else (e_nsw - e_pw) * MEV_PER_EV
        return [
            candidate, geometry, n_atoms,
            "" if e_pw is None else f"{e_pw:.8f}",
            "" if e_nsw is None else f"{e_nsw:.8f}",
            "" if d_e is None else f"{d_e:.4f}",
            "" if d_e is None else f"{d_e / int(n_atoms or 1):.4f}",
            note,
        ]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["candidate", "geometry", "n_atoms", "e_pw_eV", "e_nsw_eV",
                         "delta_e_meV", "delta_e_per_atom_meV", "note"])
        if data.workflow == "orbgen.basis":
            for key, slot in sorted(data.cells.items()):
                label, geometry = key.split("::", 1)
                if label.startswith("pw"):
                    writer.writerow(row("PW (reference)", geometry, slot.get("n_atoms"),
                                        slot.get("e_pw"), None, "PW reference"))
            pw_by_geometry = {
                key.split("::", 1)[1]: slot["e_pw"]
                for key, slot in data.cells.items()
                if key.split("::", 1)[0].startswith("pw") and slot.get("e_pw") is not None
            }
            for key, slot in sorted(data.cells.items()):
                label, geometry = key.split("::", 1)
                if label.startswith("pw"):
                    continue
                e_pw = pw_by_geometry.get(geometry)
                writer.writerow(row(label, geometry, slot.get("n_atoms"), e_pw,
                                    slot.get("e_nsw"),
                                    "" if e_pw is not None
                                    else "no PW energy for this geometry"))
        else:
            for cutoff in sorted(data.curve):
                for geometry, energy in sorted(data.curve[cutoff].items()):
                    writer.writerow(row(f"ecutwfc@{cutoff:g}", geometry, "", energy, None,
                                        "PW ladder"))
            for key, slot in sorted(data.cells.items()):
                label, geometry = key.split("::", 1)
                if label.startswith("pw@"):
                    continue
                e_pw = (data.curve.get(data.lcao_cutoff) or {}).get(geometry)
                writer.writerow(row(f"LCAO@{data.lcao_cutoff:g}Ry", geometry,
                                    slot.get("n_atoms"), e_pw, slot.get("e_nsw"),
                                    "LCAO vs PW at the reference cutoff"
                                    if e_pw is not None else "no PW energy"))
    return path


def write_scan_report(
    node,
    output_dir: str | Path,
    *,
    report_name: str = "report.md",
    profile: str | None = None,
) -> dict[str, Path]:
    """Write ``report.md``, ``energies.csv`` and ``decision.json`` for one scan."""
    from aiida import load_profile
    from aiida.orm import load_node

    load_profile(profile)
    if not hasattr(node, "pk"):
        node = load_node(node)

    data = build_scan_report(node)
    data.children = _scan_children(node)
    for child in data.children:
        try:
            child["seconds"] = float(
                (child["node"].mtime - child["node"].ctime).total_seconds()
            )
        except Exception:  # noqa: BLE001 - timestamps are cosmetic
            child["seconds"] = None

    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "report": out_dir / report_name,
        "csv": out_dir / "energies.csv",
        "json": out_dir / "decision.json",
    }
    paths["report"].write_text(render_scan_report(data), encoding="utf-8")
    _write_energies_csv(data, paths["csv"])
    paths["json"].write_text(
        json.dumps(data.decision, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return paths
