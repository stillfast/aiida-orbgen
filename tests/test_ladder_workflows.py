"""Tests for the two value-selection WorkChains.

``orbgen.ecutwfc`` and ``orbgen.basis`` submit through a scheduler, so what is pinned
down here is what the decisions rest on:

* the decision rules of ``workflows/ladder.py`` (``pw_convergence``,
  ``plan_candidates``, ``basis_table``, ``pick_cheapest``, ``describe_row``);
* the shared helpers of ``workflows/_children.py`` (geometry naming, the
  ``parameters.input`` merge, the PW-reference bookkeeping);
* the shape of the two WorkChains -- in particular that ``orbgen.basis`` runs **one**
  PW reference for the whole scan instead of one per candidate, which is the whole
  reason the scan exists next to ``orbgen.gridsearch``.

The numbers used below are the ones the U reference of
``project/u_14ve`` produced, so a change of the rules shows up against real data and
not against invented ones.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from aiida_orbgen.interfaces.nsw import apply_grid_point, compute_nbes_per_l
from aiida_orbgen.workflows import OrbgenBasisScanWorkChain, OrbgenCalcWorkChain, \
    OrbgenEcutwfcWorkChain
from aiida_orbgen.workflows._children import (
    cell_edges_bohr,
    geometries_from_entries,
    geometry_key,
    n_primitive_functions,
    rcut_fits_cell,
    siab_tree_problem,
    split_task,
    synthetic_task,
    with_input_overrides,
)
from aiida_orbgen.workflows.ecutwfc import DEFAULT_ECUTWFC_SCALE
from aiida_orbgen.workflows.energies import ChildEnergy
from aiida_orbgen.workflows.ladder import (
    basis_table,
    candidate_cost,
    candidate_label,
    describe_row,
    no_basis_message,
    pick_cheapest,
    plan_candidates,
    plan_evaluation_order,
    pw_convergence,
)

REPO = Path(__file__).resolve().parent.parent
WORKFLOWS = REPO / "src" / "aiida_orbgen" / "workflows"
BASIS = WORKFLOWS / "basis.py"
ECUTWFC = WORKFLOWS / "ecutwfc.py"


# ---------------------------------------------------------------------------
#  ecutwfc: the convergence rule
# ---------------------------------------------------------------------------
#: The measured per-atom move of the U reference between neighbouring cutoffs
#: (Ry): 100->120, 120->150, 150->180, 180->200, in meV/atom.
MEASURED_STEPS = (436.85, 63.38, 5.94, 1.34)
MEASURED_VALUES = (100.0, 120.0, 150.0, 180.0, 200.0)


def _measured_curve() -> dict:
    """One geometry whose per-atom energy moves exactly like the real A-scan.

    ``n_atoms = 1`` so that the energy of the entry *is* the per-atom energy the
    measured steps are quoted in; a real dimer would put a factor 2 in between.
    """
    energy = 0.0
    curve = {}
    for index, value in enumerate(MEASURED_VALUES):
        if index:
            energy += MEASURED_STEPS[index - 1] / 1000.0
        curve[value] = {"dimer-2.4": {"energy": energy, "n_atoms": 1}}
    return curve


def test_pw_convergence_picks_the_first_cutoff_that_settles():
    """180 Ry is where 5.94 meV/atom per step drops below 5 meV... it does not.

    With a 6 meV/atom criterion the 150->180 step is inside it and 120->150 (63 meV) is
    not, so 150 Ry is the answer; with 5 meV/atom the same step is *outside* it and the
    answer moves to 180 Ry.  That sensitivity is the point of reporting the steps.
    """
    decision = pw_convergence(_measured_curve(), 6.0)
    assert decision["chosen"] == 150.0
    assert decision["converged"] is True
    assert [round(step["max_meV"], 2) for step in decision["steps"]] == list(
        MEASURED_STEPS
    )
    assert [step["within_tolerance"] for step in decision["steps"]] == [
        False, False, True, True
    ]

    tighter = pw_convergence(_measured_curve(), 5.0)
    assert tighter["chosen"] == 180.0
    assert [step["within_tolerance"] for step in tighter["steps"]] == [
        False, False, False, True
    ]


def test_pw_convergence_says_so_when_the_ladder_is_too_short():
    """An unconverged ladder must not silently return its last value as the answer."""
    decision = pw_convergence(_measured_curve(), 1.0)
    assert decision["converged"] is False
    assert decision["chosen"] == MEASURED_VALUES[-1]

    single = pw_convergence(
        {100.0: {"dimer-2.4": {"energy": -1.0, "n_atoms": 2}}}, 5.0
    )
    assert single["converged"] is False
    assert single["steps"] == []


def test_pw_convergence_max_is_taken_over_the_geometries():
    """The worst geometry decides, not the average."""
    curve = {
        100.0: {"a": {"energy": 0.0, "n_atoms": 1}, "b": {"energy": 0.0, "n_atoms": 1}},
        150.0: {"a": {"energy": 0.001, "n_atoms": 1},  # 1 meV/atom: fine
                "b": {"energy": 0.030, "n_atoms": 1}},  # 30 meV/atom: not fine
    }
    decision = pw_convergence(curve, 5.0)
    assert decision["steps"][0]["max_meV"] == pytest.approx(30.0)
    assert decision["steps"][0]["within_tolerance"] is False
    assert decision["converged"] is False


def test_the_default_scale_is_increasing():
    assert list(DEFAULT_ECUTWFC_SCALE) == sorted(DEFAULT_ECUTWFC_SCALE)
    assert 1.0 in DEFAULT_ECUTWFC_SCALE and max(DEFAULT_ECUTWFC_SCALE) > 2.0


# ---------------------------------------------------------------------------
#  basis: the ladder
# ---------------------------------------------------------------------------
REFERENCE = {"r_cut": 12.0, "l_max": 4, "ecutjy": 150.0}


def test_ladder_reduces_one_parameter_at_a_time_from_the_reference():
    """Cost order, not cartesian order: the reference is always the first candidate."""
    candidates = plan_candidates(
        r_cut_values=[11.0, 10.0], l_max_values=[3], ecutjy_values=[125.0, 100.0],
        reference=REFERENCE,
    )
    assert [candidate_label(c) for c in candidates] == [
        "r12_l4_j150", "r12_l4_j125", "r12_l4_j100", "r12_l3_j150",
        "r11_l4_j150", "r10_l4_j150",
    ]
    # every candidate but the reference is cheaper in exactly one parameter
    reference = candidates[0]
    for candidate in candidates[1:]:
        cheaper = [
            key for key in ("r_cut", "l_max", "ecutjy")
            if candidate[key] < reference[key]
        ]
        assert len(cheaper) == 1, candidate
        assert all(candidate[key] <= reference[key] for key in reference)


def test_ladder_drops_duplicates_and_keeps_the_reference():
    candidates = plan_candidates(
        ecutjy_values=[150.0, 125.0], reference=REFERENCE,
    )
    assert [candidate_label(c) for c in candidates] == ["r12_l4_j150", "r12_l4_j125"]


def test_the_ladder_is_evaluated_cheapest_first_and_the_reference_last():
    """The reference is the most expensive candidate, so it may not be the answer.

    A scan that stopped at the first passing candidate of the *plan* order would stop at
    the reference and never find the reduction it exists for.  (Whether the reference
    itself is accurate enough is a property of the reference *point*, not of the order:
    a ladder whose reference fails the tolerance ends with no winner at all.)
    """
    plan = plan_candidates(
        r_cut_values=[11.0, 10.0], l_max_values=[3], ecutjy_values=[125.0, 100.0],
        reference=REFERENCE,
    )
    order = plan_evaluation_order(plan, reference_label="r12_l4_j150")
    assert [candidate_label(c) for c in order][-1] == "r12_l4_j150"
    assert [candidate_label(c) for c in order][0] == "r10_l4_j150"   # cheapest
    # a baseline-first order is what the atomization gate needs
    order = plan_evaluation_order(plan, reference_label="r12_l4_j150", baseline_first=True)
    assert [candidate_label(c) for c in order][0] == "r12_l4_j150"
    assert len(order) == len(plan)


def test_the_evaluation_order_uses_the_cost_proxy_when_it_has_one():
    plan = plan_candidates(ecutjy_values=[125.0], reference=REFERENCE)
    # deliberately inverted costs: the order has to follow them, not the parameters
    order = plan_evaluation_order(
        plan, costs={"r12_l4_j150": 1.0, "r12_l4_j125": 2.0},
        reference_label="r12_l4_j150",
    )
    assert [candidate_label(c) for c in order] == ["r12_l4_j125", "r12_l4_j150"]
    assert plan_evaluation_order([], reference_label=None) == []


def test_the_no_winner_diagnosis_names_the_real_cause():
    """Three ways to end with no winner, three different things to change.

    The Si smoke run hit the first one: nothing passed *and* the reference point itself
    was 257 meV/atom off, because r_cut 7 au is simply too small there -- the answer is a
    larger reference point, not a wider ladder.
    """
    bad_reference = basis_table(
        {"r7_l3_j100": {
            "candidate": {"r_cut": 7.0, "l_max": 3, "ecutjy": 100.0},
            "geometries": {"dimer-2.4": {"e_nsw": -99.5, "e_pw": -100.0, "n_atoms": 2}},
            "seconds": 1124.0, "nchi": 314}},
        100.0, reference="r7_l3_j100",
    )
    message = no_basis_message(bad_reference, 100.0, reference="r7_l3_j100")
    assert "reference point" in message and "larger reference point" in message

    # the bare table of the earlier fixtures: something did pass, but the gate rejected it
    table = basis_table(_rows(), 20.0, reference="r12_l4_j150")
    assert table["within_tolerance"] == ["r12_l4_j150", "r11_l4_j125"]
    message = no_basis_message(
        table, 0.01, reference="r12_l4_j150", atomization_tolerance_meV=50.0
    )
    assert "widen the ladder" in message or "reference point" in message
    assert "atomization gate" in message

    assert "could be evaluated" in no_basis_message({"rows": []}, 100.0)


def test_the_cost_proxy_grows_with_the_basis():
    assert candidate_cost(192, 11.0) < candidate_cost(192, 12.0)      # same nchi, bigger cell
    assert candidate_cost(192, 12.0) < candidate_cost(221, 12.0)      # more functions
    assert candidate_cost(None, 12.0) is None
    assert candidate_cost(192, None) is None


def test_exhaustive_is_the_cartesian_product():
    candidates = plan_candidates(
        r_cut_values=[11.0, 10.0], l_max_values=[4, 3], ecutjy_values=[150.0, 125.0],
        reference=REFERENCE, strategy="exhaustive",
    )
    assert len(candidates) == 8
    assert len({candidate_label(c) for c in candidates}) == 8
    assert {c["l_max"] for c in candidates} == {4, 3}


def test_unknown_strategy_is_an_error_not_a_silent_ladder():
    with pytest.raises(ValueError):
        plan_candidates(reference=REFERENCE, strategy="quickest")
    with pytest.raises(ValueError):
        # the ladder needs a reference point to reduce
        plan_candidates(ecutjy_values=[125.0], reference=None)


# ---------------------------------------------------------------------------
#  basis: the table and the choice
# ---------------------------------------------------------------------------
def _rows() -> dict:
    """Three candidates of the measured U scan: reference, chosen one, one below it.

    ``r11_l4_j125`` passed with dA = -21.6 meV; ``r10_l4_j150`` failed the tolerance
    (dA of the atomization energy was fine, the per-atom dE was not); the reference is
    the slowest but always valid.
    """
    def geometry(dimer: float, nsw: float, pw: float, monomer: bool = False):
        return {
            "e_nsw": nsw, "e_pw": pw, "n_atoms": 1 if monomer else 2,
        }

    return {
        "r12_l4_j150": {
            "candidate": {"r_cut": 12.0, "l_max": 4, "ecutjy": 150.0},
            "geometries": {
                "dimer-2.4": geometry(2.4, -100.0000, -100.0000),
                "monomer": geometry(0.0, -50.0000, -50.0000, monomer=True),
            },
            "seconds": 75.0, "nchi": 221,
        },
        "r11_l4_j125": {
            "candidate": {"r_cut": 11.0, "l_max": 4, "ecutjy": 125.0},
            "geometries": {
                # +0.022 meV/atom on the dimer (dE/atom), and an atomization energy
                # 21.6 meV below the reference one -- the defect-like criterion
                "dimer-2.4": geometry(2.4, -99.999956, -100.000000),
                "monomer": geometry(0.0, -49.989178, -50.000000, monomer=True),
            },
            "seconds": 39.0, "nchi": 192,
        },
        "r10_l4_j150": {
            "candidate": {"r_cut": 10.0, "l_max": 4, "ecutjy": 150.0},
            "geometries": {
                "dimer-2.4": geometry(2.4, -99.9, -100.000000),  # 50 meV/atom
                "monomer": geometry(0.0, -49.95, -50.000000, monomer=True),
            },
            "seconds": 30.0, "nchi": 173,
        },
    }


def test_basis_table_measures_the_per_atom_and_the_atomization_error():
    table = basis_table(_rows(), 10.0, reference="r12_l4_j150")
    by_label = {row["label"]: row for row in table["rows"]}

    reference = by_label["r12_l4_j150"]
    assert reference["dE_max_abs_meV"] == pytest.approx(0.0)
    assert reference["atomization_meV"] == pytest.approx(0.0)
    assert reference["tolerance_ok"] is True

    chosen = by_label["r11_l4_j125"]
    assert chosen["dE_max_abs_meV"] == pytest.approx(0.022, abs=0.002)
    assert chosen["tolerance_ok"] is True
    # d(atomization) = (E_dimer - 2 E_monomer) difference, in meV: -21.6
    assert chosen["atomization_vs_reference_meV"] == pytest.approx(-21.6, abs=0.5)

    failing = by_label["r10_l4_j150"]
    assert failing["dE_max_abs_meV"] == pytest.approx(50.0, abs=0.5)
    assert failing["tolerance_ok"] is False
    assert table["within_tolerance"] == ["r12_l4_j150", "r11_l4_j125"]


def test_cheapest_means_cheapest_measured_seconds():
    table = basis_table(_rows(), 10.0, reference="r12_l4_j150")
    best = pick_cheapest(table, by="seconds")
    assert best["label"] == "r11_l4_j125"
    # the failing candidate is cheaper than everything and must still never win
    assert pick_cheapest(table, by="nchi")["label"] == "r11_l4_j125"
    assert pick_cheapest(table, by="cost")["label"] == "r11_l4_j125"


def test_the_atomization_gate_can_reject_the_cheapest_candidate():
    """A tighter gate rejects the 21.6 meV atomization drift and falls back."""
    table = basis_table(_rows(), 10.0, reference="r12_l4_j150")
    strict = pick_cheapest(
        table, by="seconds", require_atomization=True, atomization_tolerance_meV=10.0
    )
    assert strict["label"] == "r12_l4_j150"
    loose = pick_cheapest(
        table, by="seconds", require_atomization=True, atomization_tolerance_meV=30.0
    )
    assert loose["label"] == "r11_l4_j125"


def test_pick_cheapest_returns_nothing_instead_of_raising():
    table = basis_table(_rows(), 0.01, reference="r12_l4_j150")
    assert table["within_tolerance"] == ["r12_l4_j150"] or \
        table["within_tolerance"] == []
    empty = basis_table({}, 100.0)
    assert pick_cheapest(empty) is None


def test_pick_cheapest_rejects_an_unknown_cost_key():
    table = basis_table(_rows(), 10.0, reference="r12_l4_j150")
    with pytest.raises(ValueError):
        pick_cheapest(table, by="wallclock")


def test_describe_row_prints_what_the_decision_used():
    table = basis_table(_rows(), 10.0, reference="r12_l4_j150")
    row = next(r for r in table["rows"] if r["label"] == "r11_l4_j125")
    line = describe_row(row, 10.0)
    assert "r11_l4_j125" in line
    assert "0.02" in line            # the per-atom error
    assert "✓" in line               # inside the tolerance
    assert "39s" in line             # the measured cost
    assert "dA=" in line             # the atomization drift


# ---------------------------------------------------------------------------
#  child bookkeeping
# ---------------------------------------------------------------------------
def test_every_child_is_submitted_with_the_preset_parameters():
    """`parameters` is not optional: a child without it silently loses `abacus.json`.

    `build_abacus_child_inputs` merges the preset's `parameters.input` last, so passing
    `{}` does not mean "no overrides" -- it means the preset is dropped and
    `apply_input_overrides`' `ks_solver: scalapack_gvx` wins.  `orbgen.basis` did that
    for every LCAO child (found by reading the INPUT of a real child in the database:
    it said scalapack_gvx while the workchain input said genelpa).
    """
    import inspect

    from aiida_orbgen.workflows._children import submit_child

    signature = inspect.signature(submit_child)
    assert signature.parameters["parameters"].default is inspect.Parameter.empty

    module = __import__("aiida_orbgen.workflows.basis", fromlist=["basis"])
    for step in ("submit_candidate_step", "submit_pw_step"):
        source = inspect.getsource(getattr(module.OrbgenBasisScanWorkChain, step))
        assert "submit_child(" in source
        assert "with_input_overrides(options" in source, step


def test_with_input_overrides_alone_keeps_the_preset(tmp_path):
    """The "no override of my own" case still has to carry the preset."""
    from aiida_orbgen.workflows._children import with_input_overrides

    options = {"parameters": {"input": {"ks_solver": "genelpa", "scf_thr": 1e-4},
                              "max_iterations": 4}}
    assert with_input_overrides(options, None) == options["parameters"]
    merged = with_input_overrides(options, {"ecutwfc": 150})
    assert merged["input"] == {"ks_solver": "genelpa", "scf_thr": 1e-4, "ecutwfc": 150}
    assert merged["max_iterations"] == 4


def test_geometry_key_does_not_depend_on_the_basis_parameters():
    """SIAB folds r_cut into the folder name; pairing must not."""
    entry = {"proto": "dimer", "pert": 2.8, "folder": "U-dimer-2.80-11au"}
    assert geometry_key(entry) == "dimer-2.8"
    other = {"proto": "dimer", "pert": 2.8, "folder": "U-dimer-2.80-12au"}
    assert geometry_key(other) == geometry_key(entry)
    monomer = {"proto": "monomer", "pert": 0.0, "folder": "U-monomer-11au"}
    assert "monomer" in geometry_key(monomer)
    # and the name is what makes the atomization difference findable
    assert geometry_key(monomer) == "monomer"


def test_synthetic_task_round_trip():
    task = synthetic_task("r11_l4_j125", "dimer-2.8")
    assert task == "r11_l4_j125::dimer-2.8"
    assert split_task(task) == ("r11_l4_j125", "dimer-2.8")
    assert split_task("dimer-2.8") == ("dimer-2.8", "dimer-2.8")


def test_with_input_overrides_merges_into_input_and_keeps_siblings():
    """``abacus.json`` may carry more than ``input`` (``pw.yml`` has max_iterations)."""
    options = {"parameters": {"input": {"ks_solver": "dav", "ecutwfc": 100},
                              "max_iterations": 4}}
    merged = with_input_overrides(options, {"ecutwfc": 180})
    assert merged["input"] == {"ks_solver": "dav", "ecutwfc": 180}
    assert merged["max_iterations"] == 4
    # the caller's options are untouched (the ladder overrides per candidate)
    assert options["parameters"]["input"]["ecutwfc"] == 100


def test_geometries_from_entries_pairs_nsw_with_the_shared_pw_reference():
    entries = [
        ChildEnergy(folder="r11_l4_j125::dimer-2.8", basis="lcao",
                    energy=-99.99, n_atoms=2),
    ]
    reference = {"dimer-2.8": {"energy": -100.0, "n_atoms": 2}}
    rows = geometries_from_entries(entries, pw_reference=reference)
    assert rows["r11_l4_j125"]["dimer-2.8"]["e_nsw"] == pytest.approx(-99.99)
    assert rows["r11_l4_j125"]["dimer-2.8"]["e_pw"] == pytest.approx(-100.0)
    assert rows["r11_l4_j125"]["dimer-2.8"]["pw_from_reference"] is True

    # a PW child of the scan itself wins over the shared reference
    with_pw = geometries_from_entries(
        entries + [ChildEnergy(folder="r11_l4_j125::dimer-2.8", basis="pw",
                               energy=-101.0, n_atoms=2)],
        pw_reference=reference,
    )
    assert with_pw["r11_l4_j125"]["dimer-2.8"]["e_pw"] == pytest.approx(-101.0)
    assert "pw_from_reference" not in with_pw["r11_l4_j125"]["dimer-2.8"]


def test_geometries_from_entries_skips_geometries_without_a_pw_energy():
    entries = [ChildEnergy(folder="r12_l4_j150::dimer-9.9", basis="lcao",
                           energy=-1.0, n_atoms=2)]
    rows = geometries_from_entries(entries, pw_reference={"dimer-2.8": {"energy": -2.0}})
    assert "e_pw" not in rows["r12_l4_j150"]["dimer-9.9"]


def test_a_cached_siab_tree_without_files_is_detected(tmp_path):
    """The failure mode of AiiDA's cache: a stored Dict whose files are gone.

    Without the check, a scan submits every child of that candidate and each one dies
    on a missing INPUT -- with nothing in the report pointing at the cache.
    """
    stru = tmp_path / "STRU"
    inp = tmp_path / "INPUT"
    orb = tmp_path / "Si_gga_7au_100Ry_21s20p20d.orb"
    info = {"orb_path": str(orb), "dft": [{"stru": str(stru), "input": str(inp)}]}
    assert siab_tree_problem(info) is not None            # files do not exist yet
    assert "orbital" in siab_tree_problem(info)

    orb.write_text("x")
    assert "STRU" in siab_tree_problem(info)
    stru.write_text("x")
    assert "INPUT" in siab_tree_problem(info)
    inp.write_text("x")
    assert siab_tree_problem(info) is None                # everything is there
    assert siab_tree_problem(None) is not None
    assert siab_tree_problem({"orb_path": None, "dft": []}) is not None


def test_the_rcut_cell_check_is_in_bohr_on_both_sides():
    """A 35 Bohr cubic cell allows r_cut up to 17.5 au.

    ``interfaces/stru.py`` names its constant ``BOHR_TO_ANG`` while holding the number of
    *Bohr per Angstrom*, so converting the cell through it puts a factor 3.6 between the
    check and ``r_cut`` -- which is exactly the bug that made a valid r_cut 7 au fail on
    a 35 Bohr cell.  The edges therefore come straight from ``LATTICE_CONSTANT`` (Bohr)
    times the dimensionless vectors.
    """
    parsed = {"lattice_constant": 35.0,
              "lattice_vectors": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}
    edges = cell_edges_bohr(parsed)
    assert edges == [35.0, 35.0, 35.0]
    assert rcut_fits_cell(7.0, edges) is None
    assert rcut_fits_cell(17.5, edges) is None
    assert "17.500" in rcut_fits_cell(18.0, edges)
    # the *smallest* edge decides, and a cell we cannot read must not block a run
    assert rcut_fits_cell(9.0, [20.0, 8.0, 30.0]) is not None
    assert rcut_fits_cell(50.0, []) is None
    assert rcut_fits_cell(50.0, cell_edges_bohr({"lattice_constant": None})) is None
    assert rcut_fits_cell(50.0, cell_edges_bohr(None)) is None


def test_n_primitive_functions_grows_with_every_parameter():
    base = n_primitive_functions(11.0, 125.0, 4)
    assert base == sum(
        int(n) * (2 * l + 1)
        for l, n in enumerate(compute_nbes_per_l(11.0, 125.0, 4))
    )
    assert n_primitive_functions(12.0, 125.0, 4) > base
    assert n_primitive_functions(11.0, 150.0, 4) > base
    assert n_primitive_functions(11.0, 125.0, 5) > base


def test_the_solver_follows_the_basis_the_child_runs():
    """`genelpa` + PW and `dav` + LCAO are refused by ABACUS, not merely slow.

    `orbgen.ecutwfc` submits PW children *and* (with `with_lcao`) LCAO children from the
    same `abacus.json`, so one preset has to survive both.  Found by running the CLI
    route with the `lcao_only` preset: 12 of 18 children died.
    """
    from aiida_orbgen.workflows.siab import _solver_for_basis

    assert _solver_for_basis("genelpa", "pw") == "dav"
    assert _solver_for_basis("scalapack_gvx", "pw") == "dav"
    assert _solver_for_basis("dav", "lcao") == "genelpa"
    assert _solver_for_basis("cg", "lcao") == "genelpa"
    # what matches stays, and an unknown keyword is left to ABACUS
    assert _solver_for_basis("genelpa", "lcao") == "genelpa"
    assert _solver_for_basis("dav", "pw") == "dav"
    assert _solver_for_basis("cg", "pw") == "cg"
    assert _solver_for_basis("mystery", "pw") == "mystery"
    # ... and a preset that names no solver gets the basis default
    assert _solver_for_basis(None, "pw") == "dav"
    assert _solver_for_basis(None, "lcao") == "genelpa"


def test_apply_grid_point_can_override_the_jy_cutoff():
    config = {"ecutjy": 150, "bessel_nao_rcut": [12], "geoms": [{"lmaxmax": 4}]}
    assert apply_grid_point(config, 4, 11.0)["ecutjy"] == 150      # unchanged
    assert apply_grid_point(config, 4, 11.0, 125.0)["ecutjy"] == 125.0
    assert config["ecutjy"] == 150                                 # input untouched


# ---------------------------------------------------------------------------
#  the two WorkChains: shape and reuse
# ---------------------------------------------------------------------------
def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize("path", [BASIS, ECUTWFC])
def test_the_scans_are_orchestration_only(path):
    """Like ``batch.py``: no shelling out, no SIAB file parsing of their own."""
    source = _source(path)
    for pattern in (r"subprocess\.", r"generate_all_from_json\(", r"read_stru\("):
        assert not __import__("re").search(pattern, source, __import__("re").MULTILINE), \
            f"{pattern!r} does not belong in {path.name}"
    # ... while still reusing the SIAB layer.  The basis scan builds one tree per
    # candidate, so it calls the calcfunction; the ecutwfc scan inherits the single-tree
    # step of OrbgenCalcWorkChain (pinned in the reuse test below).
    if path is BASIS:
        assert "run_siab_pipeline(" in source
    assert "build_abacus_child_inputs" not in source  # it goes through _children


def test_the_pw_reference_is_run_once_not_per_candidate():
    """The optimisation that justifies ``orbgen.basis`` over a grid search."""
    module = __import__("aiida_orbgen.workflows.basis", fromlist=["basis"])
    submit_pw = inspect.getsource(module.OrbgenBasisScanWorkChain.submit_pw_step)
    assert 'basis="pw"' in submit_pw
    submit_candidate = inspect.getsource(
        module.OrbgenBasisScanWorkChain.submit_candidate_step
    )
    assert 'basis="lcao_nsw"' in submit_candidate
    assert '"pw"' not in submit_candidate.replace("lcao_nsw", "")

    # the PW step must sit *outside* the loop over candidates
    tree = ast.parse(_source(BASIS))
    loops = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Call)
        and getattr(node.func.func, "id", "") == "while_"
    ]
    assert len(loops) == 1, "exactly one loop over the candidates"
    looped = {getattr(arg, "attr", "") for arg in loops[0].args}
    assert "submit_candidate_step" in looped
    assert "submit_pw_step" not in looped


def test_the_ecutwfc_run_replaces_the_child_submission_and_the_analysis():
    for name in ("submit_children", "extract_energies_step", "finalize"):
        assert getattr(OrbgenEcutwfcWorkChain, name) is not getattr(
            OrbgenCalcWorkChain, name
        ), name
    # ... and reuses the expensive, SIAB-facing steps instead of copying them
    for name in ("run_siab_pipeline_step", "ensure_pseudo_family", "inspect_children"):
        assert getattr(OrbgenEcutwfcWorkChain, name) is getattr(
            OrbgenCalcWorkChain, name
        ), name


def test_the_ecutwfc_scan_runs_one_lcao_child_at_the_reference_cutoff():
    module = __import__("aiida_orbgen.workflows.ecutwfc", fromlist=["ecutwfc"])
    source = inspect.getsource(module.OrbgenEcutwfcWorkChain.submit_children)
    assert 'jobs: list[tuple[str, float]] = [("pw", value) for value in values]' in source
    assert 'jobs.append(("lcao_nsw", reference))' in source


@pytest.fixture(scope="module")
def profile():
    """The default AiiDA profile, or a skip if this machine has none configured."""
    from aiida import load_profile
    from aiida.common import ProfileConfigurationError

    try:
        load_profile()
    except ProfileConfigurationError:  # pragma: no cover - no AiiDA here
        pytest.skip("no AiiDA profile configured")


def test_exit_codes_of_the_scans_are_distinct(profile):
    """Two scans sharing a process label may not share a number with a different meaning."""
    for cls in (OrbgenEcutwfcWorkChain, OrbgenBasisScanWorkChain):
        codes = [code.status for code in cls.spec().exit_codes.values()]
        assert len(codes) == len(set(codes)), cls.__name__
    assert "WARNING_ECUTWFC_NOT_CONVERGED" in OrbgenEcutwfcWorkChain.spec().exit_codes
    assert "ERROR_ECUTWFC_LADDER" in OrbgenEcutwfcWorkChain.spec().exit_codes
    assert "WARNING_NO_BASIS_WITHIN_TOLERANCE" in OrbgenBasisScanWorkChain.spec().exit_codes
    assert "ERROR_RCUT_TOO_LARGE" in OrbgenBasisScanWorkChain.spec().exit_codes


def test_the_registered_entry_points_are_these_classes(profile):
    from aiida.plugins import WorkflowFactory

    assert WorkflowFactory("orbgen.ecutwfc") is OrbgenEcutwfcWorkChain
    assert WorkflowFactory("orbgen.basis") is OrbgenBasisScanWorkChain
