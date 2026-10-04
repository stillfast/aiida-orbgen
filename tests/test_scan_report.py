"""``report.md`` for a scan, and the two bugs the first real rerun exposed.

A scan's report is the table the scan exists for — the total energies of the reference
geometries in both bases and the difference between them — so these tests pin the
arithmetic, the products (``report.md`` / ``energies.csv`` / ``decision.json``) and the
cost rule that decides *which* candidate the table calls the winner.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aiida_orbgen.utils.report.scan import (
    ScanReport,
    _write_energies_csv,
    render_scan_report,
)
from aiida_orbgen.workflows._children import seconds_of
from aiida_orbgen.workflows.ladder import basis_table, pick_cheapest

REPO = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
#  cost and the reference fallback (the 2026-10-04 rerun)
# ---------------------------------------------------------------------------
#: The rerun's own rows: the reference's children came out of AiiDA's cache (18 s of
#: bookkeeping), the two accepted candidates really ran (5684 s / 8846 s).
RERUN = {
    "r12_l4_j150": {"dE": 23.22, "dA": 0.0, "seconds": 18.2, "nchi": 1095, "r_cut": 12.0},
    "r9_l4_j150": {"dE": 170.26, "dA": -172.2, "seconds": 17.7, "nchi": 810, "r_cut": 9.0},
    "r10_l4_j150": {"dE": 77.77, "dA": -76.2, "seconds": 15.7, "nchi": 904, "r_cut": 10.0},
    "r12_l3_j150": {"dE": 56.21, "dA": 66.0, "seconds": 5684.0, "nchi": 708, "r_cut": 12.0},
    "r11_l4_j150": {"dE": 39.03, "dA": -37.7, "seconds": 8846.0, "nchi": 995, "r_cut": 11.0},
}


def _rows(seconds: str = "measured") -> dict:
    """The rerun's rows, with honest costs (or the cached ones it really had)."""
    from aiida_orbgen.workflows.ladder import candidate_cost

    rows = []
    for label, spec in RERUN.items():
        l_max = 3 if "l3" in label else 4
        cost = candidate_cost(spec["nchi"], spec["r_cut"])
        rows.append({
            "label": label,
            "candidate": {"r_cut": spec["r_cut"], "l_max": l_max, "ecutjy": 150.0},
            "dE_per_atom_meV": {"dimer-2.4": spec["dE"]},
            "dE_max_abs_meV": spec["dE"],
            "atomization_vs_reference_meV": spec["dA"],
            # a real run costs ~10x the static proxy; a cached one reports seconds of
            # bookkeeping (the rerun's own numbers, in `RERUN`)
            "seconds": spec["seconds"] if seconds == "cached" else cost * 10.0,
            "nchi": spec["nchi"],
            "cost": cost,
            "tolerance_ok": spec["dE"] <= 100.0,
        })
    return {"rows": rows}


def test_the_reference_is_a_fallback_not_the_cheapest_candidate():
    """The reference is the ladder's starting point (its most expensive basis).

    The scan had already accepted `r11_l4_j150` in `evaluate_candidate_step`;
    `finalize` then re-picked from the table and could still return the reference.
    """
    table = _rows()
    best = pick_cheapest(table, by="seconds", require_atomization=True,
                         atomization_tolerance_meV=43.36, reference="r12_l4_j150")
    assert best["label"] == "r11_l4_j150"          # the only one inside the gate
    # with the gate relaxed to 100 meV, the cheaper accepted candidate wins
    loose = pick_cheapest(table, by="seconds", require_atomization=True,
                          atomization_tolerance_meV=100.0, reference="r12_l4_j150")
    assert loose["label"] == "r10_l4_j150"


def test_cached_seconds_must_not_decide_the_winner():
    """A re-run's cached children report bookkeeping seconds, not cost.

    On 2026-10-04 a rerun reused every child of the reference point, which then looked
    like the *cheapest* candidate (18 s against 8846 s) and was returned as the winner
    next to a warning that no cheaper candidate had passed -- while one had.
    """
    # the gate that was in force: only the reference and r11 are left in the running
    table = _rows(seconds="cached")
    naive = pick_cheapest(table, by="seconds", require_atomization=True,
                          atomization_tolerance_meV=43.36)
    assert naive["label"] == "r12_l4_j150"          # 18 s of cache bookkeeping < 8846 s

    fixed = pick_cheapest(table, by="seconds", require_atomization=True,
                          atomization_tolerance_meV=43.36, reference="r12_l4_j150")
    assert fixed["label"] == "r11_l4_j150"


def test_unknown_seconds_rank_by_the_cost_proxy():
    """`by="seconds"` with everything cached falls back to the static proxy."""
    table = _rows(seconds="cached")
    for row in table["rows"]:
        row["seconds"] = None
    best = pick_cheapest(table, by="seconds", require_atomization=True,
                         atomization_tolerance_meV=100.0, reference="r12_l4_j150")
    assert best["label"] == "r10_l4_j150"          # proxy 817 < 1318 < 1711


def test_seconds_of_returns_nothing_for_a_cached_calculation():
    """A cached child did not run, so its bookkeeping seconds are not a cost."""

    def node(cached: bool, seconds: float = 5.0):
        inner = SimpleNamespace(
            process_label="AbacusCalculation",
            base=SimpleNamespace(caching=SimpleNamespace(is_created_from_cache=cached)),
        )
        from datetime import datetime, timedelta

        start = datetime(2026, 10, 4, 8, 0, 0)
        return SimpleNamespace(
            process_label="AbacusBaseWorkChain",
            called=[inner],
            ctime=start,
            mtime=start + timedelta(seconds=seconds),
            base=SimpleNamespace(
                caching=SimpleNamespace(is_created_from_cache=False)
            ),
        )

    assert seconds_of(node(cached=True)) is None
    assert seconds_of(node(cached=False, seconds=42.0)) == pytest.approx(42.0)


# ---------------------------------------------------------------------------
#  the report itself
# ---------------------------------------------------------------------------
def _scan_report() -> ScanReport:
    cells = {
        "pw::dimer-2.4": {"n_atoms": 2, "e_pw": -100.0},
        "pw::monomer": {"n_atoms": 1, "e_pw": -50.0},
        "r11_l4_j150::dimer-2.4": {"n_atoms": 2, "e_nsw": -99.98},
        "r11_l4_j150::monomer": {"n_atoms": 1, "e_nsw": -49.989},
        "r12_l4_j150::dimer-2.4": {"n_atoms": 2, "e_nsw": -99.90},
        "r12_l4_j150::monomer": {"n_atoms": 1, "e_nsw": -49.95},
    }
    return ScanReport(
        workflow="orbgen.basis",
        pk=1,
        uuid="uuid",
        label="",
        status="finished [0]",
        exit_status=0,
        decision={
            "reference": "r12_l4_j150",
            "tolerance_meV": 100.0,
            "atomization_tolerance_meV": 43.36,
            "pw_reference": {"source": "computed", "ecutwfc": 180.0, "n_geometries": 2},
            "reference_point": {"r_cut": 12.0, "l_max": 4, "ecutjy": 150.0},
            "ran": ["r12_l4_j150", "r11_l4_j150"],
            "planned": ["r12_l4_j150", "r11_l4_j150", "r10_l4_j150"],
            "planned_but_not_run": ["r10_l4_j150"],
            "stopped_early": True,
        },
        cells=cells,
        geometries=["dimer-2.4", "monomer"],
        table=[
            {"label": "r12_l4_j150", "candidate": {"r_cut": 12.0, "l_max": 4, "ecutjy": 150.0},
             "dE_per_atom_meV": {"dimer-2.4": 50.0, "monomer": 0.0},
             "dE_max_abs_meV": 50.0, "atomization_vs_reference_meV": 0.0,
             "seconds": None, "nchi": 1095, "cost": 2071.9, "tolerance_ok": True,
             "gate_ok": True, "passed": True},
            {"label": "r11_l4_j150", "candidate": {"r_cut": 11.0, "l_max": 4, "ecutjy": 150.0},
             "dE_per_atom_meV": {"dimer-2.4": 10.0, "monomer": 1.0},
             "dE_max_abs_meV": 10.0, "atomization_vs_reference_meV": -12.0,
             "seconds": 100.5, "nchi": 995, "cost": 1317.7, "tolerance_ok": True,
             "gate_ok": True, "passed": True},
        ],
        chosen={"label": "r11_l4_j150",
                "candidate": {"r_cut": 11.0, "l_max": 4, "ecutjy": 150.0},
                "dE_max_abs_meV": 10.0, "atomization_vs_reference_meV": -12.0,
                "seconds": 100.5, "nchi": 995},
    )


def test_the_report_prints_the_energies_and_their_differences():
    text = render_scan_report(_scan_report())
    assert "## 1. Energy difference per candidate (meV/atom)" in text
    assert "## 2. Energies behind the differences (eV)" in text
    assert "### 2a. PW reference" in text and "### 2b. LCAO (NSW)" in text
    assert "## 3. Atomization energy" in text
    # the raw energies of both bases, and the difference between them
    assert "-100.0000" in text and "-99.9800" in text
    assert "+20.00" in text            # (E_nsw - E_pw) of the dimer, in meV
    assert "+10.00" in text            # ... per atom
    # the criterion column and the winner
    assert "10.00" in text and "**chosen**" in text
    assert "r11_l4_j150" in text
    # what was not run stays visible
    assert "not run" in text


def test_the_report_marks_the_gate_and_the_stored_decision_disagreement():
    data = _scan_report()
    data.table[0]["dA"] = 200.0
    data.table[0]["atomization_vs_reference_meV"] = 200.0
    data.table[0]["gate_ok"] = False
    data.table[0]["passed"] = False
    text = render_scan_report(data)
    assert "*gate*" in text
    assert "**chosen**" in text


def test_the_csv_carries_one_row_per_candidate_and_geometry(tmp_path):
    data = _scan_report()
    path = _write_energies_csv(data, tmp_path / "energies.csv")
    lines = path.read_text().splitlines()
    assert lines[0].startswith("candidate,geometry,n_atoms,e_pw_eV,e_nsw_eV")
    rows = [line.split(",") for line in lines[1:]]
    assert len(rows) == len(data.cells)
    pw_rows = [r for r in rows if r[0] == "PW (reference)"]
    assert len(pw_rows) == 2 and all(r[4] == "" for r in pw_rows)
    lcao = [r for r in rows if r[0] == "r11_l4_j150"]
    assert len(lcao) == 2 and all(r[3] != "" and r[4] != "" for r in lcao)
    assert float(lcao[0][5]) == pytest.approx(20.0, abs=0.01)


def test_report_routes_scan_workflows_to_the_scan_renderer(tmp_path, monkeypatch, capsys):
    """`aiida-orbgen report -i output.json -o ./` has to work for a scan."""
    from aiida_orbgen.cli import run as cli_run
    from aiida_orbgen.utils.config import WORKFLOW_BASIS

    calls = {}

    def fake_write(node, out_dir, *, report_name="report.md", profile=None):
        calls["node"] = node
        calls["out_dir"] = Path(out_dir)
        calls["report_name"] = report_name
        out = Path(out_dir) / report_name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("# scan\n")
        return {"report": out, "csv": out.parent / "energies.csv",
                "json": out.parent / "decision.json"}

    monkeypatch.setattr("aiida_orbgen.utils.report.scan.write_scan_report", fake_write)
    monkeypatch.setattr("aiida.load_profile", lambda *a, **k: None)
    monkeypatch.setattr("aiida.orm.load_node", lambda identifier: f"node:{identifier}")

    output_json = tmp_path / "output.json"
    output_json.write_text(json.dumps({
        "workflow": WORKFLOW_BASIS,
        "abacus": {"orbgen": {"u_14ve": "uuid:1234"}},
    }))
    out_dir = tmp_path / "out"
    assert cli_run.main(["report", "-i", str(output_json), "-o", str(out_dir)]) == 0
    assert calls["node"] == "node:uuid:1234"
    assert calls["out_dir"] == out_dir.resolve()
    assert (out_dir / "report.md").is_file()
    assert "ok ->" in capsys.readouterr().out
