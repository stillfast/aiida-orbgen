"""Utility for generating markdown reports from an OrbgenGridSearchWorkChain.

Public API
----------
- ``generate_report(pk_or_uuid, l_max_list, r_cut_list, output_path)``
    Generate the full report immediately (workflow must have finished).
- ``monitor_and_report(pk_or_uuid, l_max_list, r_cut_list, output_path, poll_interval=60)``
    Poll until the workflow reaches a terminal state, then generate the report.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from aiida import orm
from aiida.cmdline.utils.common import get_workchain_report
from aiida.engine import ProcessState
from aiida.orm import CalcJobNode, WorkChainNode

# ── helpers ──────────────────────────────────────────────────────────────────


def _status_emoji(node):
    if hasattr(node, "is_finished_ok") and node.is_finished_ok:
        return "🟢"
    if hasattr(node, "process_state") and node.process_state and node.process_state.value in ("excepted", "killed"):
        return "🔴"
    return "🟡"


def _slug(text: str) -> str:
    """Generate a GitHub-flavoured markdown anchor slug for a heading text.

    Rules (matching GitHub's slugger):
      - lowercase
      - remove ``<`` ``>`` (HTML tag brackets)
      - replace every run of non-``[a-z0-9]`` with a single ``-``
        (note: ``_`` is also replaced, not preserved)
      - strip leading/trailing ``-``
    """
    import re
    s = text.lower()
    s = s.replace("<", "").replace(">", "")
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-")


def _status_label(node):
    if hasattr(node, "is_finished_ok") and node.is_finished_ok:
        return f"Finished [{node.exit_status}]"
    if hasattr(node, "process_state") and node.process_state:
        s = node.process_state.value.capitalize()
        es = f" [{node.exit_status}]" if node.exit_status is not None else ""
        return f"{s}{es}"
    return "—"


def _collect_grid_data(node, l_max_list, r_cut_list):
    """Walk children of an OrbgenGridSearchWorkChain and collect ΔE data.

    Returns a dict mapping ``(l_max, r_cut) -> {
        'wc_pk':       OrbgenCalcWorkChain PK,
        'max_dE_atom': ΔE_max (per atom, meV)  (or None if missing),
        'max_dE_sys':  ΔE_max (per system, meV) (or None if missing),
        'tolerance':   tolerance_meV from inputs,
        'per_struct':  [{'folder': str, 'dE_per_atom': float (meV), ...}],
        'exit_status': int,
    }``.
    """
    grid_data: dict[tuple, dict] = {}
    for c in node.called:
        if not isinstance(c, WorkChainNode):
            continue
        if getattr(c, "process_label", "") != "OrbgenCalcWorkChain":
            continue
        try:
            l_max = int(c.inputs.l_max.value)
            r_cut = float(c.inputs.r_cut.value)
        except (AttributeError, KeyError, ValueError):
            continue
        info: dict = {
            "wc_pk": c.pk,
            "max_dE_atom": None,
            "max_dE_sys": None,
            "tolerance": None,
            "per_struct": [],
            "exit_status": c.exit_status,
        }
        # tolerance: 来自 OrbgenGridSearchWC 的 inputs.abacus.tolerance_meV
        try:
            info["tolerance"] = float(
                node.inputs.abacus.parameters.get_dict()["input"].get("scf_thr", None)
            )  # placeholder
        except Exception:
            pass
        # 找 create_energies_dict 子 calcfunction
        for cc in c.called:
            if type(cc).__name__ == "CalcFunctionNode" and "energies" in (cc.label or "").lower():
                for olink in cc.base.links.get_outgoing().all():
                    node_dict = olink.node
                    if not hasattr(node_dict, "get_dict"):
                        continue
                    try:
                        d = node_dict.get_dict()
                    except Exception:
                        continue
                    info["max_dE_atom"] = d.get("delta_E_max_per_atom_meV")
                    info["max_dE_sys"] = d.get("delta_E_max_meV")
                    per_struct = d.get("delta_E_per_struct", []) or []
                    info["per_struct"] = per_struct
                    # Compute min ΔE from per_struct
                    min_vals = [
                        e["dE_per_atom"] * 1000
                        for e in per_struct
                        if isinstance(e.get("dE_per_atom"), (int, float))
                    ]
                    info["min_dE_atom"] = min(min_vals) if min_vals else None
                break
        # tolerance from abacus.json of OrbgenCalcWC inputs
        try:
            abacus_params = c.inputs.parameters.get_dict()
            # 这一行只是占位; OrbgenCalcWC 的 parameters 是 lcao/pw 的 INPUT
        except Exception:
            pass
        grid_data[(l_max, r_cut)] = info
    # tolerance 也尝试从 gridsearch wc 本身拿
    try:
        tol = float(node.inputs.tolerance_meV.value)
        for k in grid_data:
            grid_data[k]["tolerance"] = tol
    except Exception:
        pass
    return grid_data


# ── tree formatting ──────────────────────────────────────────────────────────


def _collect_calc_nodes(node):
    """Recursively collect CalcJobNode instances from *node* and its called children."""
    results = []
    for child in node.called:
        if isinstance(child, CalcJobNode):
            results.append(child)
        elif isinstance(child, WorkChainNode):
            results.extend(_collect_calc_nodes(child))
    return results


def _format_child_tree(wc_node, depth=0):
    """Build a code-fence tree of the sub-process structure."""
    indent = "    " * depth
    wc_emoji = _status_emoji(wc_node)
    wc_type = getattr(wc_node, "process_label", type(wc_node).__name__)
    lines = [f"{indent}{wc_emoji} {wc_type}<{wc_node.pk}> {_status_label(wc_node)}"]

    child_display_lines = []
    calc_nodes_out = []
    for node in wc_node.called:
        emoji = _status_emoji(node)
        ntype = getattr(node, "process_label", type(node).__name__)
        if isinstance(node, CalcJobNode):
            link = f"[{ntype}<{node.pk}>](#calc-{node.pk})"
            child_display_lines.append(f"{emoji} {link} {_status_label(node)}")
            calc_nodes_out.append((node, wc_node))
        elif isinstance(node, WorkChainNode):
            child_display_lines.append(f"{emoji} {ntype}<{node.pk}> {_status_label(node)}")
            for calc_child in _collect_calc_nodes(node):
                calc_nodes_out.append((calc_child, wc_node))
        else:
            child_display_lines.append(f"{emoji} {ntype}<{node.pk}> {_status_label(node)}")
    for idx, line in enumerate(child_display_lines):
        prefix = "├── " if idx < len(child_display_lines) - 1 else "└── "
        lines.append(indent + prefix + line)
    return "\n".join(lines), calc_nodes_out


def _format_child_tree_table(wc_node, l_max=None, r_cut=None):
    """Build a clickable markdown table of the sub-process structure.

    Each row: ``[emoji] [Type<pk>](anchor) [Status]`` with anchor links to
    the per-calc detail sections further down in the report.
    """
    wc_pk = wc_node.pk
    rows = []
    # Top row: the WC itself (link to the grid heading)
    rows.append({
        "depth": 0,
        "branch": "─",
        "type": f"**OrbgenCalcWorkChain<{wc_pk}>**",
        "status": _status_label(wc_node),
        "link": f"#grid-{wc_pk}",
        "emoji": _status_emoji(wc_node),
        "pk": wc_pk,
    })
    for node in wc_node.called:
        emoji = _status_emoji(node)
        ntype = getattr(node, "process_label", type(node).__name__)
        link = f"#calc-{node.pk}"
        rows.append({
            "depth": 1,
            "branch": "├─" if node is not list(wc_node.called)[-1] else "└─",
            "type": f"[{ntype}<{node.pk}>]({link})",
            "status": _status_label(node),
            "link": link,
            "emoji": emoji,
            "pk": node.pk,
        })
    return rows


# ── core API ─────────────────────────────────────────────────────────────────


def _format_cell(value, tol, formatter="{:.1f}"):
    """Format a single ΔE cell with a color hint based on tolerance."""
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    if v == 0.0:
        return f"_{formatter.format(v)}_"  # 0 often means no data
    s = formatter.format(v)
    if tol is not None and v > float(tol):
        s = f"**{s}**"  # bold if exceeds tolerance
    return s


def generate_report(
    pk_or_uuid: Union[int, str],
    l_max_list: list,
    r_cut_list: list,
    output_path: Union[str, Path],
    tolerance_mev: Optional[float] = None,
) -> str:
    """Generate a full markdown report for an OrbgenGridSearchWorkChain.

    Parameters
    ----------
    pk_or_uuid:
        PK or UUID of the workflow node.
    l_max_list:
        List of ``l_max`` candidates (e.g. ``[3, 4, 5]``).
    r_cut_list:
        List of ``r_cut`` candidates (e.g. ``[8.0, 9.0, 10.0, 11.0, 12.0]``).
    output_path:
        Where to write the report.
    tolerance_mev:
        Optional override for the convergence tolerance (meV/atom).
        If None, read from ``OrbgenGridSearchWorkChain.inputs.tolerance_meV``.

    Returns
    -------
    The absolute path of the written report file.
    """
    node = orm.load_node(pk_or_uuid)
    wc_label = getattr(node, "process_label", type(node).__name__)

    grid_data = _collect_grid_data(node, l_max_list, r_cut_list)
    if not grid_data:
        raise ValueError(f"no OrbgenCalcWorkChain children found in node {pk_or_uuid}")

    # Decide tolerance for bold marking
    if tolerance_mev is None:
        for v in grid_data.values():
            if v["tolerance"] is not None:
                tolerance_mev = v["tolerance"]
                break
    if tolerance_mev is None:
        try:
            tolerance_mev = float(node.inputs.tolerance_meV.value)
        except Exception:
            tolerance_mev = None

    # ── compose report ──
    lines: list[str] = []
    lines.append("# Orbgen Grid Search Report")
    lines.append("")
    lines.append(f"**Workflow**: {wc_label}<{node.pk}>")
    lines.append(f"**Status**: {_status_label(node)}")
    lines.append(f"**Report generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if tolerance_mev is not None:
        lines.append(f"**Tolerance**: {tolerance_mev} meV/atom")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── 1a. ΔE Max matrix (l_max × r_cut) ──
    lines.append("## 1a. ΔE Max (per atom, meV) — l_max × r_cut")
    lines.append("")
    lines.append("Cells in **bold** exceed the convergence tolerance.")
    lines.append("")
    rc_strs = [f"{r:.1f}" for r in r_cut_list]
    header = ["l_max \\ r_cut"] + rc_strs
    separator = ["---:"] + [":---:"] * len(rc_strs)
    rows = [header, separator]
    best = (None, None, None)  # (l_max, r_cut, value) lowest max ΔE
    for l_max in sorted(set(l_max_list)):
        row = [str(l_max)]
        for r_cut in r_cut_list:
            info = grid_data.get((l_max, r_cut))
            v = info["max_dE_atom"] if info else None
            row.append(_format_cell(v, tolerance_mev))
            if v is not None and v > 0:
                if best[2] is None or v < best[2]:
                    best = (l_max, r_cut, v)
        rows.append(row)
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    lines.append("")

    # ── 1a2. Per-dimer data count matrix (l_max × r_cut) ──
    lines.append("## 1a2. Per-dimer ΔE data count — l_max × r_cut")
    lines.append("")
    lines.append("Number of dimers that successfully produced a per-atom ΔE value (i.e. with both `lcao` and `pw` energies available) for each grid point.")
    lines.append("")
    rc_strs = [f"{r:.1f}" for r in r_cut_list]
    header = ["l_max \\ r_cut"] + rc_strs
    separator = ["---:"] + [":---:"] * len(rc_strs)
    rows = [header, separator]
    for l_max in sorted(set(l_max_list)):
        row = [str(l_max)]
        for r_cut in r_cut_list:
            info = grid_data.get((l_max, r_cut))
            if info is None:
                row.append("—")
            else:
                # count valid per-dimer entries (those with numeric dE_per_atom)
                cnt = sum(
                    1 for e in (info.get("per_struct") or [])
                    if isinstance(e.get("dE_per_atom"), (int, float))
                )
                row.append(str(cnt) if cnt else "—")
        rows.append(row)
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    lines.append("")

    # ── 1b. ΔE Min matrix (l_max × r_cut) ──
    lines.append("## 1b. ΔE Min (per atom, meV) — l_max × r_cut")
    lines.append("")
    rows = [header, separator]
    best_min = (None, None, None)  # (l_max, r_cut, value) lowest min ΔE
    for l_max in sorted(set(l_max_list)):
        row = [str(l_max)]
        for r_cut in r_cut_list:
            info = grid_data.get((l_max, r_cut))
            v = info["min_dE_atom"] if info else None
            row.append(_format_cell(v, tolerance_mev))
            if v is not None and v > 0:
                if best_min[2] is None or v < best_min[2]:
                    best_min = (l_max, r_cut, v)
        rows.append(row)
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    lines.append("")

    # ── summary ──
    if best[0] is not None:
        verdict = "✅ converged" if (tolerance_mev is not None and best[2] <= tolerance_mev) else "❌ NOT converged"
        lines.append(
            f"**Best (lowest Max ΔE)**: (l_max={best[0]}, r_cut={best[1]:.1f}) → {best[2]:.2f} meV/atom {verdict}"
        )
    if best_min[0] is not None:
        lines.append(
            f"**Best (lowest Min ΔE)**: (l_max={best_min[0]}, r_cut={best_min[1]:.1f}) → {best_min[2]:.2f} meV/atom"
        )
    lines.append("")
    lines.append("[⬇ Jump to Exit Code matrix](#exit-code-matrix) or [per-grid details](#2-per-grid-details)")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── 1b. Exit Code matrix (OrbgenCalcWC exit_status) ──
    lines.append("## Exit Code matrix")
    lines.append("")
    lines.append("OrbgenCalcWorkChain exit_status for each grid point. Click any cell to jump to that grid's details.")
    lines.append("")
    rc_strs = [f"{r:.1f}" for r in r_cut_list]
    header = ["l_max \\ r_cut"] + rc_strs
    separator = ["---:"] + [":---:"] * len(rc_strs)
    rows = [header, separator]
    for l_max in sorted(set(l_max_list)):
        row = [f"**{l_max}**"]
        for r_cut in r_cut_list:
            info = grid_data.get((l_max, r_cut))
            if info is None or info.get("exit_status") is None:
                row.append("—")
                continue
            es = info["exit_status"]
            wc_pk = info["wc_pk"]
            anchor = f"grid-{wc_pk}"
            if es == 0:
                cell = f"[0](#{anchor})"
            elif es == 304:
                cell = f"[304 ⚠](#{anchor})"
            else:
                cell = f"[**{es}** ✗](#{anchor})"
            row.append(cell)
        rows.append(row)
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    lines.append("")
    lines.append("Legend: `0`=OK · `304`=WARNING_TOLERANCE_EXCEEDED (lcao above tolerance) · other=real failure")
    lines.append("")
    lines.append("[⬆ Back to ΔE matrix](#1a-de-max-per-atom-mev--l_max--r_cut)")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── 2. Per-grid details (lcao vs pw for each dimer) ──
    lines.append("## 2. Per-grid details")
    lines.append("")

    grid_summaries = []  # (l_max, r_cut, info, wc_node)
    for (l_max, r_cut) in sorted(grid_data.keys()):
        info = grid_data[(l_max, r_cut)]
        wc_pk = info["wc_pk"]
        try:
            wc_node = orm.load_node(wc_pk)
        except Exception:
            continue
        lines.append(
            f"### grid (l_max={l_max}, r_cut={r_cut:.1f}) — OrbgenCalcWorkChain<{wc_pk}>"
            f" <a id=\"grid-{wc_pk}\"></a>"
        )
        lines.append("")
        lines.append(f"[⬆ Back to ΔE matrix](#1a-de-max-per-atom-mev--l_max--r_cut)")
        lines.append("")

        lines.append("| Property | Value |")
        lines.append("| --- | --- |")
        lines.append(f"| WorkChain | OrbgenCalcWorkChain<{wc_pk}> |")
        lines.append(f"| Status | {_status_label(wc_node)} |")
        lines.append(f"| Exit | `{wc_node.exit_status}` |")
        lines.append(f"| ΔE_max (per atom) | {info['max_dE_atom']} meV |" if info["max_dE_atom"] is not None else "| ΔE_max (per atom) | — |")
        lines.append(f"| ΔE_max (per system) | {info['max_dE_sys']} meV |" if info["max_dE_sys"] is not None else "| ΔE_max (per system) | — |")
        lines.append("")

        # Per-dimer table
        per = info["per_struct"]
        if per:
            lines.append("**Per-dimer ΔE/atom (meV):**")
            lines.append("")
            lines.append("| Dimer | E_pw (Ha) | E_lcao_nsw (Ha) | ΔE/atom (meV) |")
            lines.append("| --- | ---: | ---: | ---: |")
            for entry in per:
                lines.append(
                    f"| {entry['folder']} | {entry['E_pw']:.6f} | "
                    f"{entry['E_lcao_nsw']:.6f} | {entry['dE_per_atom']*1000:.3f} |"
                )
            lines.append("")
        else:
            lines.append("_(no per-dimer energies — all calcs failed or were skipped)_")
            lines.append("")

        # ── Full AbacusCalc table: (dimer, basis) → (E, exit, cached) ──
        abacus_rows = []  # list of dicts: {'dimer', 'basis', 'E', 'exit', 'cached', 'pk', 'wc_pk'}
        for ab_base_wc in wc_node.called:
            if type(ab_base_wc).__name__ != "WorkChainNode":
                continue
            if getattr(ab_base_wc, "process_label", "") != "AbacusBaseWorkChain":
                continue
            ab_base_wc_pk = ab_base_wc.pk
            ab_base_wc_exit = ab_base_wc.exit_status
            # extract dimer / basis from inputs
            dimer = None
            basis = None
            try:
                # OrbgenCalcWC 提交时设的 extras task
                pass
            except Exception:
                pass
            # 从 AbacusCalc 的 inputs.parameters.basis_type 拿
            for ab_calc in ab_base_wc.called:
                if type(ab_calc).__name__ != "CalcJobNode":
                    continue
                if getattr(ab_calc, "process_label", "") != "AbacusCalculation":
                    continue
                try:
                    b = ab_calc.inputs.parameters.get_dict()["input"].get("basis_type", "?")
                except Exception:
                    b = "?"
                # task/dimer 来自 extras (用 .all 避免 missing key 异常)
                task = "?"
                try:
                    task = ab_base_wc.base.extras.all.get("task", "?")
                except Exception:
                    pass
                cached = "—"
                try:
                    cf = ab_calc.base.extras.all.get("_aiida_cached_from", None)
                    cached = cf[:8] if cf else "—"
                except Exception:
                    pass
                try:
                    E = ab_calc.outputs.misc.get_dict().get("total_energy", None) if ab_calc.process_state.value == "finished" else None
                except Exception:
                    E = None
                E_str = f"{E:.6f}" if isinstance(E, (int, float)) else "—"
                abacus_rows.append({
                    "dimer": task,
                    "basis": b,
                    "E": E_str,
                    "exit_calc": ab_calc.exit_status,
                    "exit_basewc": ab_base_wc_exit,
                    "cached": cached,
                    "calc_pk": ab_calc.pk,
                    "base_wc_pk": ab_base_wc_pk,
                })
        if abacus_rows:
            lines.append("**Per-dimer energies + exit codes:**")
            lines.append("")
            lines.append("| Dimer | basis | E (Ha) | AbacusCalc exit | AbacusBaseWC exit | cached |")
            lines.append("| --- | --- | ---: | :---: | :---: | :---: |")
            # 按 dimer 排序再按 basis
            def _dimer_key(r):
                # 提取 dimer 里的数字 part (e.g. U-dimer-1.89-12au → (1.89, 12.0))
                import re
                m = re.findall(r"(\d+\.?\d*)", r["dimer"])
                if len(m) >= 2:
                    try:
                        return (float(m[0]), float(m[1]))
                    except ValueError:
                        return (0, 0)
                return (0, 0)
            for row in sorted(abacus_rows, key=lambda r: (_dimer_key(r), r["basis"])):
                # mark failures
                es_calc = row["exit_calc"]
                es_base = row["exit_basewc"]
                es_calc_str = f"**{es_calc}**" if es_calc and es_calc != 0 else str(es_calc)
                es_base_str = f"**{es_base}**" if es_base and es_base != 0 else str(es_base)
                lines.append(
                    f"| {row['dimer']} | {row['basis']} | {row['E']} | "
                    f"{es_calc_str} | {es_base_str} | {row['cached']} |"
                )
            lines.append("")
            # exit code summary
            exit_codes = sorted({r["exit_calc"] for r in abacus_rows if r["exit_calc"] is not None})
            cached_count = sum(1 for r in abacus_rows if r["cached"] != "—")
            lines.append(f"  - Total AbacusCalc in this grid: {len(abacus_rows)}")
            lines.append(f"  - Distinct exit codes: {exit_codes}")
            lines.append(f"  - Cached results: {cached_count}/{len(abacus_rows)}")
            lines.append("")

        # Sub-process tree (as clickable table)
        tree_rows = _format_child_tree_table(wc_node, l_max=l_max, r_cut=r_cut)
        _, calc_nodes = _format_child_tree(wc_node)  # collect calc nodes for later
        lines.append("**Sub-process tree (clickable):**")
        lines.append("")
        lines.append("| | Type | Status |")
        lines.append("| --- | --- | --- |")
        for r in tree_rows:
            prefix = r["branch"]
            lines.append(
                f"| {r['emoji']} {prefix} | {r['type']} | {r['status']} |"
            )
        lines.append("")

        # Links
        lines.append(
            f"- [OrbgenCalcWorkChain<{wc_pk}> process report](#grid-details-{wc_pk})"
        )
        lines.append("")
        lines.append("---")
        lines.append("")

        grid_summaries.append((l_max, r_cut, info, wc_node))

    # ── 3. Workchain Reports (process logs) ──
    lines.append("## 3. OrbgenCalcWorkChain process reports")
    lines.append("")
    lines.append("Process logs (same as `verdi process report`) for each grid point.")
    lines.append("")
    for l_max, r_cut, info, wc_node in grid_summaries:
        wc_pk = wc_node.pk
        lines.append(f"### grid-details-{wc_pk}  <a id=\"grid-details-{wc_pk}\"></a>")
        lines.append("")
        lines.append(f"**Grid**: (l_max={l_max}, r_cut={r_cut:.1f})")
        lines.append("")
        lines.append(f"[⬆ Back to grid details](#grid-{wc_pk})")
        lines.append("")
        wc_report = get_workchain_report(wc_node, "REPORT")
        if wc_report:
            lines.append("```")
            lines.append(wc_report)
            lines.append("```")
        else:
            lines.append("_(no report logs)_")
        lines.append("")
        lines.append("---")
        lines.append("")

    # ── write ──
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return str(output_path)


def monitor_and_report(
    pk_or_uuid: Union[int, str],
    l_max_list: list,
    r_cut_list: list,
    output_path: Union[str, Path],
    tolerance_mev: Optional[float] = None,
    poll_interval: int = 60,
    max_poll_time: Optional[int] = None,
    on_progress: Optional[callable] = None,
) -> str:
    """Poll the workflow until it reaches a terminal state, then generate the report."""
    _TERMINAL_STATES = {ProcessState.FINISHED, ProcessState.EXCEPTED, ProcessState.KILLED}
    node = orm.load_node(pk_or_uuid)
    start_time = time.time()
    while node.process_state not in _TERMINAL_STATES:
        state_label = node.process_state.value if node.process_state else "UNKNOWN"
        if max_poll_time is not None:
            elapsed = time.time() - start_time
            if elapsed >= max_poll_time:
                print(f"[monitor] node<{node.pk}> max_poll_time ({max_poll_time}s) reached, "
                      f"generating report with current state={state_label}...")
                break
        if on_progress:
            on_progress(state_label)
        else:
            print(f"[monitor] node<{node.pk}> state={state_label} — waiting {poll_interval}s...")
        time.sleep(poll_interval)
        node = orm.load_node(node.pk)
    state_label = node.process_state.value if node.process_state else "UNKNOWN"
    if on_progress:
        on_progress(state_label)
    else:
        print(f"[monitor] node<{node.pk}> state={state_label} — terminated, generating report...")
    return generate_report(pk_or_uuid, l_max_list, r_cut_list, output_path, tolerance_mev)
