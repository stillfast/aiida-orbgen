"""Tests for the workflow layer.

``OrbgenCalcWorkChain`` / ``OrbgenGridSearchWorkChain`` themselves need a daemon,
so what is pinned down here is the pure logic they call into, plus source-level
regression tests for the two bugs that motivated extracting it:

* ``workflows/_grid.py``                  -- grid construction / stopping rule
* ``interfaces/nsw.py``                   -- the single ``(l_max, r_cut)`` override
* ``utils/report/assemble.py``            -- "SIAB will not run a DFT" guard
* ``utils/report/validate.py``            -- produced ``.orb`` vs requested nzeta
* ``calculations/pseudo_family.py``       -- relative ``pseudo_dir`` resolution
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
from pathlib import Path

import pytest

from aiida_orbgen.interfaces.nsw import apply_grid_point, folder_rcut
from aiida_orbgen.utils.config import WORKFLOW_CALC, WORKFLOW_GRIDSEARCH
from aiida_orbgen.utils.report.validate import (
    nzeta_string,
    read_orbital,
    scheme_of_name,
    spillage_values,
    validate_orbital,
)
from aiida_orbgen.workflows import OrbgenCalcWorkChain, OrbgenGridSearchWorkChain
from aiida_orbgen.workflows.energies import (
    ChildEnergy,
    best_of,
    evaluate_energies,
    is_soft_success,
    pair_energies,
    tolerance_verdict,
)
from aiida_orbgen.workflows._grid import (
    GridEntry,
    build_cartesian_grid,
    build_multi_json_grid,
    has_pending_iterative,
    work_dir_name,
)

REPO = Path(__file__).resolve().parent.parent
WORKFLOWS = REPO / "src" / "aiida_orbgen" / "workflows"
BATCH = WORKFLOWS / "batch.py"
SIAB = WORKFLOWS / "siab.py"

# --- which SIAB? -----------------------------------------------------------
# There are two ABACUS-CSW-NAO checkouts on this machine and they are not
# equivalent: the one under ``/home/liguozhou/install`` still does
# ``from scipy.integrate import simps``, which scipy removed in 1.12, while the
# workspace checkout (``.../orbgen/ABACUS-CSW-NAO``) uses
# ``from scipy.integrate import simpson as simps``.  A bare ``import SIAB``
# succeeds for both -- only the deep import fails -- and in a pytest session the
# interpreter happens to resolve SIAB to the *broken* one, so the tests below pin
# the working checkout explicitly and drop the broken path.  Without this they
# would either error deep inside SIAB or silently skip the guard tests.
_WORKING_SIAB = "/home/liguozhou/abacus/calculations/orbgen/ABACUS-CSW-NAO"
_BROKEN_SIAB = "/home/liguozhou/install/ABACUS-CSW-NAO"
if os.path.isdir(_WORKING_SIAB):
    if _WORKING_SIAB not in sys.path:
        sys.path.insert(0, _WORKING_SIAB)
    sys.path[:] = [p for p in sys.path if p != _BROKEN_SIAB]
    for _name in [m for m in list(sys.modules) if m == "SIAB" or m.startswith("SIAB.")]:
        del sys.modules[_name]           # drop a partial/other-checkout import

try:  # pragma: no cover - environment dependent
    from SIAB.abacus.api import build_abacus_jobs  # noqa: F401
    from SIAB.driver.main import init  # noqa: F401

    HAS_SIAB = True
    SIAB_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    HAS_SIAB = False
    SIAB_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
requires_siab = pytest.mark.skipif(
    not HAS_SIAB, reason=f"usable SIAB (ABACUS-CSW-NAO) not importable ({SIAB_IMPORT_ERROR})"
)


# ---------------------------------------------------------------------------
#  Grid bookkeeping
# ---------------------------------------------------------------------------
def test_cartesian_grid_is_sorted_and_deduplicated():
    grid = build_cartesian_grid([4, 2, 2], [10.0, 8.0, 10.0], siab_json="node")
    assert [(e.l_max, e.r_cut) for e in grid] == [
        (2, 8.0), (2, 10.0), (4, 8.0), (4, 10.0)
    ]
    assert [e.index for e in grid] == [0, 1, 2, 3]
    assert all(e.siab_json == "node" for e in grid)


def test_multi_json_grid_keeps_file_order():
    grid = build_multi_json_grid([(4, 10.0, "a"), (3, 9.0, "b")])
    assert [e.siab_json for e in grid] == ["a", "b"]
    assert [e.index for e in grid] == [0, 1]
    assert grid[1].point == (3, 9.0)


def test_gridentry_label_and_dir_name():
    entry = GridEntry(l_max=4, r_cut=10.0, index=0)
    assert entry.label == "l_max=4, r_cut=10"
    assert entry.point == (4, 10.0)

    # one spelling everywhere (workflow *and* report): lmax4_rcut10
    from aiida_orbgen.interfaces.nsw import legacy_point_dir_name, point_dir_name

    assert work_dir_name(4, 10.0) == point_dir_name(4, 10.0) == "lmax4_rcut10"
    assert work_dir_name(3, 9) == "lmax3_rcut9"
    assert work_dir_name(4, 9.4) == "lmax4_rcut9.4"

    class _Point:
        l_max, r_cut = 4, 10.0

    from aiida_orbgen.utils.report.orbitals import (
        point_dir_aliases, point_dir_name as report_point_dir_name,
    )

    assert report_point_dir_name(_Point()) == "lmax4_rcut10"
    # trees written before 2026-09-29 used `lmax4_rcut10p0` and stay readable
    assert point_dir_aliases(_Point()) == ["lmax4_rcut10", "lmax4_rcut10p0"]
    assert legacy_point_dir_name(4, 10.0) == "lmax4_rcut10p0"
    assert legacy_point_dir_name(3, 9) == "lmax3_rcut9"   # no `.` -> no `p0`


def test_has_pending_iterative_stopping_rule():
    grid = build_cartesian_grid([3, 4], [9.0], siab_json=None)
    assert has_pending_iterative(grid, 0) is True
    assert has_pending_iterative(grid, 1) is True
    assert has_pending_iterative(grid, 2) is False                      # exhausted
    assert has_pending_iterative(grid, 0, best={"l_max": 3}) is False    # found
    assert has_pending_iterative(grid, 0, dry_run=True) is False


def test_grid_is_never_unpacked_into_two_names():
    """Regression: ``l_max, r_cut = self.ctx.grid[0]`` used to raise.

    ``ctx.grid`` holds :class:`GridEntry`, and unpacking it into two names blew
    up on the *first* submission -- so the ``iterative`` strategy (the
    WorkChain's own default) was unusable, while every CLI run passed
    ``exhaustive`` and never noticed.  Guard the shape of the code, not just the
    data, because the data alone cannot catch a re-introduced unpack.
    """
    tree = ast.parse(BATCH.read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        target = node.targets[0]
        if not isinstance(target, (ast.Tuple, ast.List)) or len(target.elts) != 2:
            continue
        if "ctx.grid" in ast.unparse(node.value):
            offenders.append(f"line {node.lineno}: {ast.unparse(node)}")
    assert offenders == [], (
        "ctx.grid entries are GridEntry objects; read .l_max / .r_cut instead "
        "of unpacking: " + "; ".join(offenders)
    )


# ---------------------------------------------------------------------------
#  The single (l_max, r_cut) override
# ---------------------------------------------------------------------------
def base_config() -> dict:
    return {
        "element": "U",
        "bessel_nao_rcut": [9],
        "ecutjy": 100,
        "geoms": [{"proto": "dimer", "pertkind": "stretch", "pertmags": "auto",
                   "lmaxmax": 4}],
        "orbitals": [{"nzeta": [2, 2, 2, 2, 1], "geoms": [0], "nbands": "occ",
                      "checkpoint": None}],
    }


def test_apply_grid_point_writes_the_integer_rcut_spelling():
    """Both layers must produce the same config -- that is the whole point."""
    config = apply_grid_point(base_config(), 3, 9.0)
    assert config["bessel_nao_rcut"] == [9]           # 9.0 -> 9, not [9.0]
    assert config["geoms"][0]["lmaxmax"] == 3
    # the report layer reaches the same implementation (it used to wrap its own)
    from aiida_orbgen.utils.report import orbitals as report_orbitals

    assert report_orbitals.apply_grid_point is apply_grid_point
    # 9.4 is not integral: keep the fraction, SIAB names it `9.4au`
    assert apply_grid_point(base_config(), 4, 9.4)["bessel_nao_rcut"] == [9.4]


def test_apply_grid_point_does_not_mutate_its_argument():
    original = base_config()
    apply_grid_point(original, 5, 12.0)
    assert original["bessel_nao_rcut"] == [9]
    assert original["geoms"][0]["lmaxmax"] == 4


def test_folder_rcut_convention():
    assert folder_rcut(9.0) == 9
    assert isinstance(folder_rcut(9.0), int)
    assert folder_rcut(9.4) == 9.4
    assert folder_rcut(10) == 10


def test_stop_on_first_valid_keeps_iterating():
    """``stop_on_first_valid=False`` = run the whole grid, then choose."""
    grid = build_cartesian_grid([3, 4], [9.0], siab_json=None)
    found = {"l_max": 3}
    assert has_pending_iterative(grid, 1, best=found) is False
    assert has_pending_iterative(grid, 1, best=found, stop_on_first_valid=False) is True
    # still stops once everything has been tried
    assert has_pending_iterative(grid, 2, best=found, stop_on_first_valid=False) is False


def test_cap_grid_applies_the_abacus_limits():
    from aiida_orbgen.workflows._grid import cap_grid

    grid = build_cartesian_grid([3, 4, 5], [9.0, 11.0], siab_json="node")
    capped = cap_grid(grid, max_l_max=4, max_r_cut=10.0)
    assert [(e.l_max, e.r_cut) for e in capped] == [(3, 9.0), (4, 9.0)]
    assert [e.index for e in capped] == [0, 1]          # indices re-numbered
    assert cap_grid(grid) == grid


def test_build_explicit_grid_accepts_pairs_only():
    from aiida_orbgen.workflows._grid import build_explicit_grid

    grid = build_explicit_grid([[4, 11.0], (4, 12)], siab_json="node")
    assert [(e.l_max, e.r_cut) for e in grid] == [(4, 11.0), (4, 12.0)]
    assert all(e.siab_json == "node" for e in grid)
    with pytest.raises(ValueError):
        build_explicit_grid([[4, 11.0, 12.0]], siab_json="node")


def test_grid_search_spec_exposes_the_ported_options():
    """The three capabilities taken from the deleted advanced.py."""
    spec = OrbgenGridSearchWorkChain.spec()
    for name in ("candidates", "stop_on_first_valid", "search_strategy"):
        assert name in spec.inputs, f"{name} missing from the grid search spec"


def test_workflow_classes_are_exported_from_one_place():
    """``WorkflowFactory('orbgen.gridsearch')`` and this import must agree."""
    from aiida_orbgen.workflows import batch

    assert OrbgenGridSearchWorkChain is batch.OrbgenGridSearchWorkChain
    assert OrbgenCalcWorkChain is batch.OrbgenCalcWorkChain
    assert WORKFLOW_CALC == "orbgen.calc"
    assert WORKFLOW_GRIDSEARCH == "orbgen.gridsearch"


def test_dry_run_is_not_implemented_with_exitcode_zero():
    """A step cannot stop an outline by returning ``ExitCode(0)``.

    AiiDA maps a zero-status ExitCode to ``None`` and keeps stepping, so such a
    guard is inert and every dry-run decision has to go through ``_dry_run``.
    """
    tree = ast.parse(BATCH.read_text(encoding="utf-8"))
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "dry_run" in ast.unparse(node.test):
            body = "\n".join(ast.unparse(stmt) for stmt in node.body)
            if "ExitCode(0)" in body:
                bad.append(f"line {node.lineno}")
    assert bad == [], f"inert dry-run guards (ExitCode(0)) at {bad}"
    for cls in (OrbgenCalcWorkChain, OrbgenGridSearchWorkChain):
        assert "_dry_run" in vars(cls), f"{cls.__name__} must expose _dry_run"


def test_step_1_and_1_5_are_dry_run_guarded():
    """They used to run SIAB and register a pseudo family during a dry run."""
    source = BATCH.read_text(encoding="utf-8")
    for step in ("def run_siab_pipeline_step", "def ensure_pseudo_family"):
        body = source[source.index(step):]
        body = body[:body.index("\n    def ", 1)] if "\n    def " in body[1:] else body
        assert "self._dry_run" in body, f"{step} lacks a dry-run guard"


# ---------------------------------------------------------------------------
#  Produced-orbital validation
# ---------------------------------------------------------------------------
def test_nzeta_string_and_scheme_lookup():
    assert nzeta_string([4, 3, 2, 2, 1]) == "4s3p2d2f1g"
    assert nzeta_string([3, 2, 2, 1, 0]) == "3s2p2d1f"
    schemes = [[3, 2, 2, 1, 0], [4, 3, 2, 2, 1]]
    assert scheme_of_name("U_gga_10au_100Ry_4s3p2d2f1g.orb", schemes) == [4, 3, 2, 2, 1]
    assert scheme_of_name("U_gga_10au_100Ry_unrelated.orb", schemes) is None


def test_spillage_values_are_scraped_from_the_log(tmp_path):
    log = tmp_path / "orbgen.log"
    log.write_text(
        "2026-09-19 - INFO - OrbgenCascade: orbital optimization ends with "
        "spillage = 1.20492740e-03\n"
        "2026-09-19 - INFO - OrbgenCascade: orbital optimization ends with "
        "spillage = 1.74717634e-03\n"
    )
    assert spillage_values(log) == pytest.approx([1.20492740e-03, 1.74717634e-03])
    assert spillage_values(tmp_path / "missing.log") == []


def test_validate_orbital_reports_a_missing_g_channel(tmp_path, monkeypatch):
    """The 2026-09-29 bug: the file name says ``1g``, the header says ``0``."""
    import aiida_orbgen.utils.report.validate as validate

    def fake_read(path):
        return {"elem": "U", "rcut": 10.0, "ecut": 100.0, "nr": 1001,
                "per_l": [4, 3, 2, 2, 0]}              # no g channel

    monkeypatch.setattr(validate, "read_orbital", fake_read)
    path = tmp_path / "U_gga_10au_100Ry_4s3p2d2f1g.orb"
    path.write_text("placeholder")

    bad = validate_orbital(path, [[4, 3, 2, 2, 1]])
    assert bad["ok"] is False
    assert "radial functions per l" in bad["reason"]

    good = validate_orbital(path, [[4, 3, 2, 2, 0]])
    assert good["ok"] is True
    assert good["expected"] == [4, 3, 2, 2, 0]


@requires_siab
def test_validate_orbital_on_a_real_file(tmp_path):
    """Round-trip a tiny real orbital generated by SIAB."""
    from aiida_orbgen.interfaces.nsw import generate_nsw

    config = {"element": "Si", "bessel_nao_rcut": [6], "ecutjy": 40,
              "primitive_type": "reduced", "geoms": [{"lmaxmax": 1}]}
    path = generate_nsw(config, output_dir=str(tmp_path), lmaxmax=1)
    assert validate_orbital(path, [[4, 3, 2, 2, 1]])["ok"] is False   # name mismatch
    per_l = read_orbital(path)["per_l"]
    assert validate_orbital(path, [per_l])["ok"] is True


# ---------------------------------------------------------------------------
#  pseudo_dir resolution
# ---------------------------------------------------------------------------
def test_resolve_upf_path_absolute_passthrough(tmp_path):
    from aiida_orbgen.calculations.pseudo_family import resolve_upf_path

    upf = tmp_path / "U.upf"
    upf.write_text("dummy")
    assert resolve_upf_path(str(upf), tmp_path) == str(upf)


def test_resolve_upf_path_relative_uses_the_first_existing_base(tmp_path):
    from aiida_orbgen.calculations.pseudo_family import resolve_upf_path

    config_dir, run_dir = tmp_path / "cfg", tmp_path / "run"
    config_dir.mkdir()
    run_dir.mkdir()
    (run_dir / "U.upf").write_text("dummy")
    assert resolve_upf_path("./U.upf", config_dir, run_dir) == str(run_dir / "U.upf")


def test_resolve_upf_path_error_lists_where_it_looked(tmp_path):
    from aiida_orbgen.calculations.pseudo_family import resolve_upf_path

    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_upf_path("./missing.upf", tmp_path)
    assert "missing.upf" in str(excinfo.value)
    assert str(tmp_path) in str(excinfo.value)


def test_run_siab_pipeline_keeps_the_config_next_to_the_run():
    """The grid-point config must not be a /tmp tempfile any more."""
    source = SIAB.read_text(encoding="utf-8")
    assert "import tempfile" not in source
    assert "NamedTemporaryFile" not in source
    assert "siab_config.json" in source


def test_the_workchains_return_no_unstored_data_nodes():
    """`self.out(..., Str(...))` aborts the whole workchain — it did, on 2026-09-29.

    AiiDA refuses "tried returning an unstored `Data` node" *after* the steps have
    run, so every output written later (``energies``) is lost and the report's ΔE
    tables come out empty.  Data nodes must come from a ``@calcfunction`` (see
    ``workflows/results.py``) or from the context, never from a constructor call in
    the outline.
    """
    data_constructors = {
        "Str", "Dict", "Int", "Float", "Bool", "List", "SinglefileData",
        "StructureData", "KpointsData", "FolderData", "UpfData", "RemoteData",
    }
    offenders = []
    for node in ast.walk(ast.parse(BATCH.read_text(encoding="utf-8"))):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "out" and len(node.args) >= 2):
            continue
        value = node.args[1]
        if not isinstance(value, ast.Call):
            continue                     # a Name/Attribute/Subscript: ctx or inputs
        func = value.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name in data_constructors:
            offenders.append(f"line {node.lineno}: self.out(..., {name}(...))")
    assert not offenders, (
        "these would make the workchain EXCEPT instead of returning its outputs:\n"
        + "\n".join(offenders)
    )


def test_the_workchain_hands_the_children_the_family_it_registered():
    """Registering a family and using another label is how "wrong pseudo" happens.

    ``ensure_pseudo_family`` returns the label to use, which is *not* always the one
    it was handed: when the derived name already belongs to a family built from
    other files, it registers a content-suffixed label (see
    ``calculations/pseudo_family.label_for_pair``).  The step has to keep that
    answer, and ``submit_children`` has to take the label from one place.
    """
    tree = ast.parse(BATCH.read_text(encoding="utf-8"))
    calc_wc = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "OrbgenCalcWorkChain"
    )
    methods = {
        node.name: node for node in calc_wc.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    stored = [
        node for node in ast.walk(methods["ensure_pseudo_family"])
        if isinstance(node, ast.Attribute) and node.attr == "pseudo_family_label"
        and isinstance(node.ctx, ast.Store)
    ]
    assert stored, "ensure_pseudo_family must keep the label it got back"

    assigned = [
        node for node in ast.walk(methods["submit_children"])
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "family_label"
                for target in node.targets)
    ]
    assert len(assigned) == 1, "family_label must be decided in exactly one place"
    value = assigned[0].value
    assert isinstance(value, ast.Call) and getattr(value.func, "attr", "") == \
        "_effective_family_label", ast.dump(value)


def test_siab_facing_code_lives_in_one_module():
    """The WorkChain module must only orchestrate.

    Everything that shells out to SIAB, builds its INPUT/STRU or names its
    folders belongs to ``workflows/siab.py`` (plus ``workflows/_grid.py`` for the
    naming/stopping rules); this is what keeps ``batch.py`` readable.
    """
    batch = BATCH.read_text(encoding="utf-8")
    # What is forbidden is *code* that shells out, not the word: batch.py's
    # docstrings legitimately explain the dry-run bug ("the run still launched the
    # SIAB subprocess"), so match call sites instead of a bare substring.
    forbidden = (
        r"^\s*import subprocess\b",
        r"subprocess\.(run|Popen|call|check_call|check_output)\b",
        r"generate_all_from_json\(",
        r"read_stru\(",
    )
    for pattern in forbidden:
        assert not re.search(pattern, batch, re.MULTILINE), (
            f"{pattern!r} belongs in workflows/siab.py"
        )
    siab = SIAB.read_text(encoding="utf-8")
    for expected in ("generate_all_from_json", "build_abacus_child_inputs",
                     "run_siab_pipeline"):
        assert expected in siab


# ---------------------------------------------------------------------------
#  "SIAB will not run a DFT" guard
# ---------------------------------------------------------------------------
def _tiny_siab_config(root: Path) -> Path:
    """A minimal SIAB-valid config (dummy UPF, r_cut 6, lmaxmax 0: fast)."""
    upf = root / "Si.upf"
    upf.write_text("dummy UPF, only needed because SIAB checks it exists\n")
    config = {
        "element": "Si",
        "pseudo_dir": str(upf),
        "abacus_command": "abacus",
        "bessel_nao_rcut": [6],
        "ecutjy": 40,
        "ecutwfc": 40,
        "primitive_type": "reduced",
        "spill_guess": "atomic",   # SIAB then also prepares the monomer job
        "geoms": [{"proto": "dimer", "pertkind": "stretch",
                   "pertmags": [2.2], "nbands": 8, "nspin": 1,
                   "lmaxmax": 0}],
        "orbitals": [{"nzeta": [1], "geoms": [0], "nbands": "occ",
                      "checkpoint": None}],
    }
    path = root / "orbgen.json"
    path.write_text(json.dumps(config, indent=2))
    return path


# ---------------------------------------------------------------------------
#  reference geometries: every `geoms` entry becomes a job, including the monomer
# ---------------------------------------------------------------------------
def _reference_config(**geom_overrides) -> dict:
    """A SIAB config with one dimer reference (what `geoms` may contain)."""
    dimer = {"proto": "dimer", "pertkind": "stretch", "pertmags": [2.4, 2.75],
             "nbands": 40, "nspin": 1, "lmaxmax": 4, "celldm": 35}
    dimer.update(geom_overrides)
    return {
        "element": "U",
        "pseudo_dir": "/tmp/U.pbe-n-nc.UPF",
        "ecutwfc": 150, "ecutjy": 100,
        "bessel_nao_rcut": [10], "primitive_type": "reduced", "fit_basis": "jy",
        "spill_guess": "atomic",
        "geoms": [dimer],
    }


@requires_siab
def test_generate_all_mirrors_the_monomer_job_siab_adds(tmp_path):
    """SIAB's spillage step needs a monomer reference, and SIAB builds that job itself.

    `SIAB/abacus/api.py:build_abacus_jobs` appends `proto='monomer'` with `nbands=69`
    when `spill_guess == 'atomic'`, and `SIAB/driver/main.py` then initialises the
    orbitals from `model_kwargs['jobdir'] = U-monomer-<rcut>au`.  `geoms` cannot name it
    (SIAB's `GeomAssert` rejects `monomer`), so `generate_all` appends it — otherwise the
    final-orbital step dies with `FileNotFoundError: 'U-monomer-10au/OUT.ABACUS/INPUT'`
    after the whole grid has been paid for (2026-09-30).
    """
    from aiida_orbgen.interfaces.pipeline import generate_all

    result = generate_all(_reference_config(), output_root=str(tmp_path), dr=0.05)

    folders = [entry["folder"] for entry in result["dft"]]
    assert folders == ["U-dimer-2.40-10au", "U-dimer-2.75-10au", "U-monomer-10au"]
    # `pertmags` still describes the reference *dimer* set (what the report prints)
    assert result["pertmags"] == [2.4, 2.75]

    monomer = result["dft"][-1]
    assert monomer["proto"] == "monomer"
    assert os.path.isfile(monomer["input"]) and os.path.isfile(monomer["stru"])
    # nbands=69 is what the atomic guess indexes its bands against — not the dimers' 40
    text = Path(monomer["input"]).read_text()
    assert [line for line in text.splitlines() if line.split()[:1] == ["nbands"]][0].split()[1] == "69"


@requires_siab
def test_no_monomer_job_without_the_atomic_guess(tmp_path):
    from aiida_orbgen.interfaces.pipeline import generate_all

    config = _reference_config()
    config["spill_guess"] = "random"
    result = generate_all(config, output_root=str(tmp_path), dr=0.05)
    assert [entry["proto"] for entry in result["dft"]] == ["dimer", "dimer"]


@requires_siab
def test_a_second_reference_geometry_is_expanded_too(tmp_path):
    """`geoms` may list a dimer *and* a trimer; both are reference states."""
    from aiida_orbgen.interfaces.pipeline import generate_all

    config = _reference_config()
    config["geoms"].append({"proto": "trimer", "pertkind": "stretch", "pertmags": [0.0],
                            "nbands": 40, "nspin": 1, "lmaxmax": 4, "celldm": 35})
    result = generate_all(config, output_root=str(tmp_path), dr=0.05)
    protos = [entry["proto"] for entry in result["dft"]]
    assert protos.count("dimer") == 2 and protos.count("trimer") == 1
    assert len({entry["folder"] for entry in result["dft"]}) == len(protos)


@requires_siab
def test_duplicate_geometry_entries_do_not_duplicate_jobs(tmp_path):
    from aiida_orbgen.interfaces.pipeline import generate_all

    config = _reference_config()
    config["geoms"].append(dict(config["geoms"][0]))          # same dimer twice
    result = generate_all(config, output_root=str(tmp_path), dr=0.05)
    folders = [entry["folder"] for entry in result["dft"]]
    assert len(folders) == len(set(folders))


@requires_siab
def test_normalise_renames_out_dir_so_siab_reuses_the_data(tmp_path):
    """The trap that cost a surprise DFT run on 2026-09-29.

    A tree assembled from AiiDA carries the plugin's INPUT (``suffix aiida``)
    next to ``OUT.aiida``.  SIAB's ``DuplicateCheck`` compares the whole INPUT,
    so it classifies the folder as *new work* (``rundft`` would recompute it),
    and even once its INPUT has been rewritten it still looks for
    ``OUT.ABACUS`` -- i.e. at data that is not there.  Normalisation fixes both:
    INPUT by SIAB's own autoset, output directory renamed to the declared
    suffix.
    """
    from aiida_orbgen.utils.report.assemble import (
        normalise_reference_for_siab,
        siab_jobs_pending,
    )

    root = tmp_path / "dft"
    folder = root / "Si-dimer-2.20-6au"
    out = folder / "OUT.aiida"
    out.mkdir(parents=True)
    (folder / "INPUT").write_text("INPUT_PARAMETERS\nsuffix aiida\n")
    (out / "running_scf.log").write_text("Finish Time\n")
    for name in ("data-0-S", "data-0-T", "WFC_NAO_GAMMA1.txt"):
        (out / name).write_text("placeholder\n")
    config_path = _tiny_siab_config(tmp_path)

    # SIAB sees the folder as new work (the INPUT does not match its autoset)
    assert siab_jobs_pending(root, config_path) == [
        "Si-dimer-2.20-6au", "Si-monomer-6au"
    ]

    result = normalise_reference_for_siab(
        root, config_path, optional=["Si-monomer-6au"]
    )
    assert result["renamed"] == {"Si-dimer-2.20-6au": "OUT.aiida -> OUT.ABACUS"}
    assert result["blocking_jobs"] == []
    assert result["missing_data"] == []
    assert result["ok"] is True
    assert (folder / "OUT.ABACUS" / "running_scf.log").is_file()
    assert not out.exists()

    again = normalise_reference_for_siab(
        root, config_path, optional=["Si-monomer-6au"]
    )
    assert again["ok"] is True and again["renamed"] == {}


@requires_siab
def test_normalise_leaves_quarantined_folders_alone(tmp_path):
    """``<name>.stale`` is the report layer's quarantine -- hands off.

    Renaming the output directory of quarantined data would make SIAB see it as
    usable reference data again, which is the opposite of what the quarantine is
    for.
    """
    from aiida_orbgen.utils.report.assemble import normalise_reference_for_siab

    root = tmp_path / "dft"
    stale = root / "Si-dimer-2.20-6au.stale"
    out = stale / "OUT.aiida"
    out.mkdir(parents=True)
    (stale / "INPUT").write_text("INPUT_PARAMETERS\nsuffix aiida\n")
    (out / "running_scf.log").write_text("Finish Time\n")
    config_path = _tiny_siab_config(tmp_path)

    result = normalise_reference_for_siab(root, config_path)
    assert result["renamed"] == {}
    assert (out / "running_scf.log").is_file()
    assert not (stale / "OUT.ABACUS").exists()


@requires_siab
def test_normalise_stands_its_ground_on_details(tmp_path):
    """The required/optional split must not be sloppy.

    A *dimer* without reference data blocks the spillage (refuse unless the user
    allowed a DFT); the *monomer* is the atomic initial guess SIAB computes by
    itself in minutes, so its absence must not block anything.
    """
    from aiida_orbgen.utils.report.assemble import normalise_reference_for_siab

    root = tmp_path / "dft"
    (root / "Si-dimer-2.20-6au").mkdir(parents=True)     # empty: no INPUT, no OUT
    config_path = _tiny_siab_config(tmp_path)

    refused = normalise_reference_for_siab(
        root, config_path, required=["Si-dimer-2.20-6au"],
        optional=["Si-monomer-6au"],
    )
    assert refused["ok"] is False
    assert refused["missing_data"] == ["Si-dimer-2.20-6au"]
    assert "--force-final-orbital" in refused["message"]

    allowed = normalise_reference_for_siab(
        root, config_path, required=["Si-dimer-2.20-6au"], allow_run=True,
    )
    assert allowed["ok"] is True

    # a required folder that *does* carry data is fine even without the monomer:
    # the monomer is the initial guess SIAB computes itself, so it never blocks
    root2 = tmp_path / "dft2"
    folder2 = root2 / "Si-dimer-2.20-6au"
    out2 = folder2 / "OUT.ABACUS"
    out2.mkdir(parents=True)
    (folder2 / "INPUT").write_text("INPUT_PARAMETERS\nsuffix ABACUS\n")
    for name in ("running_scf.log", "data-0-S", "data-0-T", "WFC_NAO_GAMMA1.txt"):
        (out2 / name).write_text("placeholder\n")

    monomer_result = normalise_reference_for_siab(
        root2, config_path, required=["Si-dimer-2.20-6au"],
        optional=["Si-monomer-6au"],
    )
    assert monomer_result["ok"] is True
    assert monomer_result["missing_data"] == []
    assert monomer_result["blocking_jobs"] == []


# ---------------------------------------------------------------------------
#  ΔE extraction / tolerance verdict (workflows/energies.py)
# ---------------------------------------------------------------------------
def test_304_is_a_soft_success_only_with_energies():
    assert is_soft_success(True, 0) is True
    assert is_soft_success(False, 304, has_energies=True) is True
    # 304 without an energies node means the extraction really failed
    assert is_soft_success(False, 304, has_energies=False) is False
    assert is_soft_success(False, 302) is False
    assert is_soft_success(False, None) is False


def test_pair_energies_normalises_per_atom():
    result = pair_energies([
        ChildEnergy("U-dimer-1.89", "pw", -100.000),
        ChildEnergy("U-dimer-1.89", "lcao", -99.998, n_atoms=2),
        ChildEnergy("U-dimer-2.09", "pw", -100.000),
        ChildEnergy("U-dimer-2.09", "lcao", -99.990, n_atoms=2),
    ])
    assert result["n_pw"] == 2 and result["n_lcao"] == 2
    assert [e["folder"] for e in result["delta_E_per_struct"]] == [
        "U-dimer-1.89", "U-dimer-2.09"
    ]
    # 2 meV total over 2 atoms = 1 meV/atom
    assert result["delta_E_per_struct"][0]["dE"] == pytest.approx(2e-3)
    assert result["delta_E_per_struct"][0]["dE_per_atom"] == pytest.approx(1e-3)
    assert result["delta_E_max_meV"] == pytest.approx(10.0)
    assert result["delta_E_max_per_atom_meV"] == pytest.approx(5.0)
    assert result["delta_E_max_eV"] == pytest.approx(0.010)


def test_pair_energies_skips_geometry_without_a_pw_partner():
    result = pair_energies([
        ChildEnergy("only-lcao", "lcao", -1.0),
        ChildEnergy("pair", "pw", -2.0),
        ChildEnergy("pair", "lcao", -1.999),
    ])
    assert [e["folder"] for e in result["delta_E_per_struct"]] == ["pair"]
    assert result["n_lcao"] == 2 and result["n_pw"] == 1


def test_pair_energies_with_nothing_to_pair_is_zero_not_perfect():
    """An unpaired run must not look like a 0 meV convergence."""
    result = pair_energies([ChildEnergy("dimer", "lcao", -1.0)])
    assert result["delta_E_per_struct"] == []
    assert result["delta_E_max_per_atom_meV"] == 0.0
    # ... which is why the grid side uses evaluate_energies(), where a missing
    # key becomes inf and the point is rejected rather than accepted
    assert evaluate_energies({}, 4.2)["tolerance_ok"] is False
    assert evaluate_energies({}, 4.2)["delta_per_atom_meV"] == float("inf")


def test_tolerance_verdict_uses_the_per_atom_standard():
    ok, message = tolerance_verdict(4.2, 4.2)
    assert ok is True and "tolerance OK" in message
    bad, message = tolerance_verdict(4.21, 4.2)
    assert bad is False and "EXCEEDED" in message
    assert "per atom" not in message.replace("/atom", "")  # wording sanity


def test_best_of_takes_the_first_acceptable_point():
    entries = [
        {"l_max": 2, "tolerance_ok": False, "delta_per_atom_meV": 100.0},
        {"l_max": 3, "tolerance_ok": True, "delta_per_atom_meV": 3.0},
        {"l_max": 4, "tolerance_ok": True, "delta_per_atom_meV": 1.0},
    ]
    assert best_of(entries)["l_max"] == 3          # cheapest acceptable, not lowest
    assert best_of(entries[:1]) is None
    assert best_of([], tolerance_meV=4.2) is None
    # the explicit tolerance is a fallback for entries without the flag
    assert best_of([{"l_max": 5, "delta_per_atom_meV": 1.0}], tolerance_meV=4.2) == \
        {"l_max": 5, "delta_per_atom_meV": 1.0}
