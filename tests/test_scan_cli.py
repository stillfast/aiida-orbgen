"""``aiida-orbgen run -i input.json`` for the two value-selection scans.

The point of these tests is the *separation*: ``input.json`` carries every parameter of
the scan (the ladder, the criterion, the reference point) and the plugin carries the
code, so the CLI route must be able to express anything the WorkChains can do — and a
mistake in the section has to fail in ``check``/``--dry-run``, not after the queue has
been fed.

Everything here is offline except the two tests that must instantiate AiiDA ``Data``
nodes (``scan_workchain_inputs``), which load the default profile and skip without one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aiida_orbgen.cli._common import (
    METHOD_SPECS,
    get_method_spec,
    plan_runs,
    scan_workchain_inputs,
)
from aiida_orbgen.cli.run import main
from aiida_orbgen.utils.config import (
    SCAN_KEYS,
    WORKFLOW_BASIS,
    WORKFLOW_CALC,
    WORKFLOW_ECUTWFC,
    ConfigLoader,
    canonical_scan_config,
    scan_families,
)

REPO = Path(__file__).resolve().parent.parent

ECUTWFC_SCAN = {"ecutwfc_values": [100, 120, 150, 180, 200], "with_lcao": True}
BASIS_SCAN = {
    "reference_ecutjy": 150.0,
    "ecutjy_values": [125, 100],
    "l_max_values": [3],
    "r_cut_values": [11, 10],
    "ecutwfc": 180,
    "atomization_tolerance_meV": 50,
}


def _scan_presets() -> dict[str, dict]:
    """``{"<file>:<preset>": config}`` of the shipped ``parameters/scan/`` tree."""
    import yaml

    base = REPO / "src" / "aiida_orbgen" / "parameters" / "scan"
    out: dict[str, dict] = {}
    for path in sorted(base.glob("*.yml")):
        for name, config in (yaml.safe_load(path.read_text()) or {}).items():
            out[f"{path.stem}:{name}"] = config or {}
    return out


def _first_preset_with(key: str) -> dict:
    """The ``{"file": "preset"}`` slot entry of a shipped preset that sets ``key``.

    Reading it from the files keeps the fixtures working when the project renames or
    re-splits its ladders (it did once already: `basis_ladder` -> `*_step1`/`*_step2`).
    """
    for where, config in _scan_presets().items():
        if key in config:
            file_name, preset = where.split(":", 1)
            return {file_name: preset}
    raise AssertionError(f"no shipped scan preset sets {key!r}")


@pytest.fixture(scope="module")
def profile():
    """The default AiiDA profile, or a skip when this machine has none."""
    from aiida import load_profile
    from aiida.common import ProfileConfigurationError

    try:
        load_profile()
    except ProfileConfigurationError:  # pragma: no cover - no AiiDA here
        pytest.skip("no AiiDA profile configured")


# ---------------------------------------------------------------------------
#  the scan section itself
# ---------------------------------------------------------------------------
def test_the_scan_section_is_passed_through_but_typed():
    scan = canonical_scan_config(dict(BASIS_SCAN))
    assert scan["ecutjy_values"] == [125, 100]
    assert scan["l_max_values"] == [3]
    assert scan["r_cut_values"] == [11, 10]
    assert scan["atomization_tolerance_meV"] == 50
    # an empty scan is simply "no scan", not an error
    assert canonical_scan_config(None) == {}
    assert canonical_scan_config({}) == {}


def test_the_scan_families_are_what_decides_the_workflow():
    assert scan_families(ECUTWFC_SCAN) == {"ecutwfc"}
    assert scan_families(BASIS_SCAN) == {"basis"}
    assert scan_families(ECUTWFC_SCAN) != scan_families(BASIS_SCAN)
    # `ecutwfc` (the PW cutoff of the reference children) belongs to the basis scan, so
    # a basis input cannot be mistaken for the cutoff ladder
    assert "ecutwfc" in SCAN_KEYS and SCAN_KEYS["ecutwfc"][0] == "basis"
    assert "ecutwfc_values" in SCAN_KEYS and SCAN_KEYS["ecutwfc_values"][0] == "ecutwfc"


@pytest.mark.parametrize("scan, match", [
    ({"nope": 1}, "unknown key"),
    ({"strategy": "quickest"}, "not one of"),
    ({"by": "wallclock"}, "not one of"),
    ({"ecutwfc_values": []}, "non-empty list"),
    ({"ecutwfc_values": ["100"]}, "only contain numbers"),
    ({"with_lcao": "yes"}, "true or false"),
    ({"ecutwfc": "180"}, "must be a number"),
    ({"pw_reference": [1, 2]}, "must be a dict"),
])
def test_a_bad_scan_section_is_rejected_by_name(scan, match):
    """A mistyped ladder must not become a valid but different scan."""
    with pytest.raises((KeyError, TypeError, ValueError), match=match):
        canonical_scan_config(scan)


def test_the_allowed_keys_are_documented_in_the_error():
    with pytest.raises(KeyError) as excinfo:
        canonical_scan_config({"r_cut": 11})
    assert "r_cut_values" in str(excinfo.value)


# ---------------------------------------------------------------------------
#  the scan preset slot: one level of the parameters/ tree
# ---------------------------------------------------------------------------
def test_a_scan_can_be_named_instead_of_spelled_out(tmp_path):
    """`parameters.scan` holds the whole ladder, so input.json only names it."""
    basis_ladder = _first_preset_with("r_cut_values")
    path = _write_input(tmp_path, presets={
        "abacus": {"lcao_only": "lcao"},
        "orbgen": {"u_14ve": "ref_r12_l4_j150"},
        "scan": basis_ladder,
    })
    bundle = ConfigLoader(path).load_all()
    assert bundle.workflow == WORKFLOW_BASIS          # from the preset's keys
    assert bundle.scan["ecutjy_values"]                # whatever the project's ladder is
    assert bundle.scan["r_cut_values"]
    assert "atomization_tolerance_meV" in bundle.scan
    assert bundle.candidates() == [(4, 12.0)]        # the reference point


def test_the_two_presets_of_one_scan_pair_with_the_same_reference(tmp_path):
    """`orbgen/u_14ve.yml#ref_r12_l4_j150` serves both scans."""
    for slot, workflow in ((_first_preset_with("ecutwfc_values"), WORKFLOW_ECUTWFC),
                           (_first_preset_with("r_cut_values"), WORKFLOW_BASIS)):
        path = _write_input(tmp_path, presets={
            "abacus": {"lcao_only": "lcao"},
            "orbgen": {"u_14ve": "ref_r12_l4_j150"},
            "scan": slot,
        })
        bundle = ConfigLoader(path).load_all()
        assert bundle.workflow == workflow, slot
        # one candidate, so `plan_runs` accepts it as the reference point
        assert bundle.candidates() == [(4, 12.0)]


def test_the_production_preset_is_the_basis_the_ladders_chose(tmp_path):
    path = _write_input(tmp_path, presets={
        "abacus": {"lcao_only": "lcao"},
        "orbgen": {"u_14ve": "prod_r11_l4_j125"},
    })
    bundle = ConfigLoader(path).load_all()
    assert bundle.workflow == WORKFLOW_CALC
    assert bundle.candidates() == [(4, 11.0)]
    assert bundle.orbgen_presets[0].config["ecutjy"] == 125


@pytest.mark.parametrize("preset, l_max, r_cut", [
    ("prod_r10_l4_j150", 4, 10.0),
    ("prod_r11_l4_j150", 4, 11.0),
    ("prod_r11_l4_j125", 4, 11.0),
    ("prod_g_r11_l4_j125", 4, 11.0),
])
def test_every_point_the_ladder_can_pick_has_a_production_preset(
    tmp_path, preset, l_max, r_cut
):
    """Whichever point `orbgen.basis` chooses is one line away from its report run."""
    path = _write_input(tmp_path, presets={
        "abacus": {"abacus": "production"},
        "orbgen": {"u_14ve": preset},
    })
    bundle = ConfigLoader(path).load_all()
    assert bundle.workflow == WORKFLOW_CALC
    assert bundle.candidates() == [(l_max, r_cut)]


def test_the_requests_in_the_production_presets_can_actually_be_fitted():
    """A g channel needs an auxiliary potential, or `report` rejects the result.

    SIAB fills `l >= lloc_min` from `model_kwargs.vloc_aux`; without it the produced
    orbital comes back with an empty g channel (`[4, 3, 2, 2, 0]` for a requested
    `4s3p2d2f1g`), which is exactly the validation failure in the project's
    `ecutjy_150` report.  So: no g without the auxiliary potential.
    """
    import yaml

    path = (REPO / "src" / "aiida_orbgen" / "parameters" / "orbgen" / "u_14ve.yml")
    presets = yaml.safe_load(path.read_text())
    for name, preset in presets.items():
        if not name.startswith("prod_"):
            continue
        orbital = preset["orbitals"][0]
        nzeta = orbital["nzeta"]
        model_kwargs = orbital.get("model_kwargs") or {}
        if len(nzeta) > 4 and nzeta[4]:
            assert model_kwargs.get("vloc_aux"), (
                f"{name} asks for a g channel without vloc_aux: the g channel would "
                f"come back empty and report would flag it"
            )
        else:
            assert not model_kwargs.get("vloc_aux"), name


def test_the_scan_slot_accepts_the_same_three_spellings(tmp_path):
    for spec in ({"scan": "pw_cutoff_quick"},
                 {"scan": ["pw_cutoff_quick"]},
                 {"scan": {"scan": "pw_cutoff_quick"}}):
        path = _write_input(tmp_path, presets={"abacus": {"lcao_only": "lcao"},
                                               "orbgen": "u_14ve", **spec},
                            single_point=False)
        bundle = ConfigLoader(path).load_all()
        assert bundle.workflow == WORKFLOW_ECUTWFC, spec
        assert bundle.scan["ecutwfc_values"] == [100.0, 150.0, 200.0]


def test_the_inline_section_overrides_the_preset_key_by_key(tmp_path):
    """A project keeps its ladders in the preset tree and varies one key per run."""
    basis_ladder = _first_preset_with("r_cut_values")
    path = _write_input(tmp_path, presets={
        "abacus": {"lcao_only": "lcao"},
        "orbgen": {"u_14ve": "ref_r12_l4_j150"},
        "scan": basis_ladder,
    })
    payload = json.loads(path.read_text())
    payload["scan"] = {"by": "nchi", "ecutjy_values": [125]}
    path.write_text(json.dumps(payload))
    bundle = ConfigLoader(path).load_all()
    assert bundle.scan["by"] == "nchi"                # inline wins
    assert bundle.scan["ecutjy_values"] == [125.0]    # inline wins
    assert bundle.scan["r_cut_values"]                # still from the preset
    assert any("overrides" in w for w in bundle.warnings)


def test_an_unknown_scan_preset_lists_what_exists(tmp_path):
    path = _write_input(tmp_path, presets={"abacus": {"lcao_only": "lcao"},
                                           "orbgen": "u_14ve",
                                           "scan": {"u_14ve": "nope"}})
    with pytest.raises(KeyError, match="available"):
        ConfigLoader(path).load_all()


def test_two_scan_presets_may_not_set_the_same_key(tmp_path):
    path = _write_input(tmp_path, presets={
        "abacus": {"lcao_only": "lcao"}, "orbgen": "u_14ve",
        "scan": {"scan": ["pw_cutoff_standard", "pw_cutoff_with_basis_check"]},
    }, single_point=False)
    with pytest.raises(KeyError, match="overlap"):
        ConfigLoader(path).load_all()


def test_a_scan_preset_is_validated_like_the_inline_section(tmp_path):
    """The scan slot gets the same schema check, so a typo fails in `check`."""
    assert _scan_presets(), "the scan preset tree is empty"
    assert any("ecutwfc_values" in config for config in _scan_presets().values())
    assert any("r_cut_values" in config for config in _scan_presets().values())
    with pytest.raises(KeyError, match="unknown key"):
        canonical_scan_config({"ecutwfc_value": [100]})     # singular typo


# ---------------------------------------------------------------------------
#  the loader turns the section into a workflow
# ---------------------------------------------------------------------------
def _single_point_siab(tmp_path: Path) -> dict:
    """A SIAB config with exactly one ``(l_max, r_cut)`` candidate.

    A scan reduces from one *reference point*, so the orbgen preset has to be a single
    point — the same shape `aiida-orbgen select` writes into ``static.siab_config``.
    """
    upf = tmp_path / "Si.upf"
    if not upf.exists():
        upf.write_text("dummy UPF, only needed because SIAB checks it exists\n")
    return {
        "element": "Si",
        "pseudo_dir": str(upf),
        "ecutjy": 100,
        "ecutwfc": 100,
        "primitive_type": "reduced",
        "spill_guess": "atomic",
        "bessel_nao_rcut": [11],
        "geoms": [{"proto": "dimer", "pertkind": "stretch", "pertmags": [2.4],
                   "nbands": 8, "nspin": 1, "lmaxmax": 4}],
        "orbitals": [{"nzeta": [2, 2, 2, 1, 1], "geoms": [0], "nbands": "occ",
                      "checkpoint": None}],
    }


def _write_input(tmp_path: Path, scan=None, workflow=None, presets=None,
                 single_point: bool = True) -> Path:
    pseudo = tmp_path / "U.pbe-n-nc.14ve.UPF"
    pseudo.write_text("<UPF version='1.0.0'>z_valence=\"14\"</UPF>\n")
    static = {"pseudo_path": str(pseudo), "metadata": "yeesuan"}
    parameters = presets or {"abacus": {"lcao_only": "lcao"}, "orbgen": "test"}
    if single_point and presets is None:
        static["siab_config"] = _single_point_siab(tmp_path)
        static["siab_config_name"] = "u_ref"
    payload = {
        "parameters": parameters,
        "static": static,
        "profile": "aiida_profile",
        "code": {"abacus": "abacus_lts@yeesuan"},
    }
    if workflow:
        payload["workflow"] = workflow
    if scan is not None:
        payload["scan"] = scan
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload, indent=4))
    return path


def test_abacus_input_in_static_overrides_the_preset(tmp_path):
    """Per-run INPUT keys belong in `input.json`, not in a new preset file.

    `static.abacus_input` is merged into the ABACUS INPUT of every child of the run,
    after the preset — which is where `scf_thr`/`mixing_beta` for a difficult system go
    now that the scans run on the CLI.
    """
    path = _write_input(tmp_path)
    payload = json.loads(path.read_text())
    payload["static"]["abacus_input"] = {"scf_thr": 1e-4, "mixing_beta": 0.1,
                                         "ks_solver": "dav"}
    path.write_text(json.dumps(payload))
    bundle = ConfigLoader(path).load_all()
    resolved = bundle.abacus_presets[0].config["abacus"]["parameters"]["input"]
    assert resolved["scf_thr"] == 1e-4
    assert resolved["mixing_beta"] == 0.1
    assert resolved["ks_solver"] == "dav"

    # ... and it goes through the same validation as a preset
    payload["static"]["abacus_input"] = {"ecutjy": 150}       # SIAB-only key
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="ecutjy"):
        ConfigLoader(path).load_all()

    payload["static"]["abacus_input"] = "scf_thr=1e-4"
    path.write_text(json.dumps(payload))
    with pytest.raises(TypeError, match="abacus_input"):
        ConfigLoader(path).load_all()


def test_the_cutoff_ladder_selects_the_ecutwfc_workflow(tmp_path):
    bundle = ConfigLoader(_write_input(tmp_path, ECUTWFC_SCAN)).load_all()
    assert bundle.workflow == WORKFLOW_ECUTWFC
    assert bundle.workflow_explicit is False
    assert bundle.scan["ecutwfc_values"] == [100.0, 120.0, 150.0, 180.0, 200.0]


def test_the_basis_ladder_selects_the_basis_scan(tmp_path):
    bundle = ConfigLoader(_write_input(tmp_path, BASIS_SCAN)).load_all()
    assert bundle.workflow == WORKFLOW_BASIS
    assert bundle.scan["ecutjy_values"] == [125.0, 100.0]


def test_without_a_scan_the_candidate_grid_still_decides(tmp_path):
    """An input.json that never mentions `scan` behaves exactly as before."""
    path = _write_input(tmp_path)
    bundle = ConfigLoader(path).load_all()
    assert bundle.workflow == WORKFLOW_CALC          # one candidate -> plain calc
    assert bundle.scan == {}
    # ... and a multi-point preset still auto-detects the grid search
    path = _write_input(tmp_path, presets={"abacus": "test", "orbgen": "test"},
                        single_point=False)
    bundle = ConfigLoader(path).load_all()
    assert bundle.workflow == "orbgen.gridsearch"


def test_an_explicit_workflow_wins_over_the_scan_keys(tmp_path):
    """`workflow` is the escape hatch, e.g. to run a scan with odd keys."""
    path = _write_input(tmp_path, ECUTWFC_SCAN, workflow=WORKFLOW_BASIS)
    bundle = ConfigLoader(path).load_all()
    assert bundle.workflow == WORKFLOW_BASIS
    assert bundle.workflow_explicit is True


def test_mixing_the_two_ladders_in_one_input_is_refused(tmp_path):
    mixed = dict(BASIS_SCAN, ecutwfc_values=[100, 150])
    with pytest.raises(ValueError, match="mixes the keys of two scans"):
        ConfigLoader(_write_input(tmp_path, mixed)).load_all()
    # ... unless the workflow is stated explicitly
    path = _write_input(tmp_path, mixed, workflow=WORKFLOW_BASIS)
    assert ConfigLoader(path).load_all().workflow == WORKFLOW_BASIS


def test_the_scan_workflows_are_registered_methods():
    for name, class_name in ((WORKFLOW_ECUTWFC, "OrbgenEcutwfcWorkChain"),
                             (WORKFLOW_BASIS, "OrbgenBasisScanWorkChain")):
        spec = get_method_spec(name)
        assert spec.class_name == class_name
        assert spec.entry_point == name
        assert spec.key == "orbgen"      # second-level key of output.json


# ---------------------------------------------------------------------------
#  the plan
# ---------------------------------------------------------------------------
def test_the_plan_of_a_scan_needs_exactly_one_reference_point(tmp_path):
    """`l_max`/`r_cut` of the plan are the *reference point* of the ladder.

    An orbgen preset with several r_cut values is a grid, and which of its points is
    the reference cannot be guessed — so that has to be an error, not a silent pick.
    """
    path = _write_input(tmp_path, BASIS_SCAN, presets={"abacus": "test", "orbgen": "test"},
                        single_point=False)
    bundle = ConfigLoader(path).load_all()
    assert len(bundle.candidates()) > 1
    with pytest.raises(ValueError, match="exactly one reference point"):
        plan_runs(bundle, output_root=tmp_path / "run")


def test_the_scan_plan_carries_the_ladder_and_its_own_run_directory(tmp_path):
    input_json = _write_input(tmp_path, BASIS_SCAN)
    bundle = ConfigLoader(input_json).load_all()
    plans = plan_runs(bundle, output_root=tmp_path / "run")
    assert len(plans) == 1
    plan = plans[0]
    assert plan.workflow == WORKFLOW_BASIS
    assert plan.scan == bundle.scan
    assert plan.output_dir == (tmp_path / "run" / "u_ref" / "scan_basis").resolve()
    assert plan.siab_json_path.is_file()
    line = plan.describe_scan()
    assert "ecutjy 125/100" in line and "l_max 3" in line and "r_cut 11/10" in line
    assert "atomization gate 50 meV" in line and "strategy=ladder" in line


def test_the_cutoff_plan_prints_the_ladder_it_will_run(tmp_path):
    input_json = _write_input(tmp_path, ECUTWFC_SCAN)
    bundle = ConfigLoader(input_json).load_all()
    plan = plan_runs(bundle, output_root=tmp_path / "run")[0]
    assert plan.output_dir == (tmp_path / "run" / "u_ref" / "scan_ecutwfc").resolve()
    line = plan.describe_scan()
    assert line.startswith("ecutwfc ladder [100, 120, 150, 180, 200] Ry")
    assert "one LCAO child" in line
    # without values: the scale spelling, which the workflow resolves from the baseline
    bundle = ConfigLoader(
        _write_input(tmp_path, {"ecutwfc_scale": [1.0, 1.5, 2.0]})
    ).load_all()
    assert "baseline x 1, 1.5, 2" in plan_runs(bundle, output_root=tmp_path / "run")[0].describe_scan()


# ---------------------------------------------------------------------------
#  the workchain inputs
# ---------------------------------------------------------------------------
def test_the_scan_inputs_keep_the_types_the_workchain_declares(tmp_path, profile):
    from aiida import orm

    input_json = _write_input(tmp_path, BASIS_SCAN)
    bundle = ConfigLoader(input_json).load_all()
    plan = plan_runs(bundle, output_root=tmp_path / "run")[0]
    inputs = scan_workchain_inputs(plan)

    assert isinstance(inputs["ecutjy_values"], orm.List)
    assert inputs["ecutjy_values"].get_list() == [125.0, 100.0]
    assert isinstance(inputs["l_max_values"], orm.List)
    assert isinstance(inputs["reference_ecutjy"], orm.Float)
    assert isinstance(inputs["atomization_tolerance_meV"], orm.Float)
    assert isinstance(inputs["ecutwfc"], orm.Float)
    assert "stop_on_first_pass" not in inputs         # absent = the workflow default
    # nothing invented: the workflow inputs are exactly the scan keys
    assert set(inputs) == set(BASIS_SCAN)


def test_the_cutoff_inputs_keep_the_boolean(tmp_path, profile):
    from aiida import orm

    input_json = _write_input(tmp_path, ECUTWFC_SCAN)
    bundle = ConfigLoader(input_json).load_all()
    plan = plan_runs(bundle, output_root=tmp_path / "run")[0]
    inputs = scan_workchain_inputs(plan)
    assert isinstance(inputs["with_lcao"], orm.Bool)
    assert inputs["with_lcao"].value is True
    assert isinstance(inputs["ecutwfc_values"], orm.List)


def test_the_built_inputs_validate_against_the_workchain_spec(tmp_path, profile):
    """`aiida-orbgen run` must submit something the WorkChain accepts.

    This is the check that would have caught the orm.Int-for-a-Float bug without a
    cluster: the spec validates the exact mapping ``build_workchain_inputs`` produces.
    """
    from aiida.plugins import WorkflowFactory

    from aiida_orbgen.cli._common import build_workchain_inputs

    for scan, entry in ((ECUTWFC_SCAN, "orbgen.ecutwfc"), (BASIS_SCAN, "orbgen.basis")):
        bundle = ConfigLoader(_write_input(tmp_path, scan)).load_all()
        plan = plan_runs(bundle, output_root=tmp_path / "run")[0]
        assert plan.entry_point == entry
        inputs = build_workchain_inputs(plan)
        WorkflowFactory(entry).spec().inputs.validate(inputs)
        # the four inputs every workchain of the family gets
        for key in ("siab_json", "abacus_config", "output_dir", "l_max", "r_cut"):
            assert key in inputs, key


def test_an_inline_pw_reference_is_taken_verbatim(tmp_path, profile):
    from aiida import orm

    reference = {"ecutwfc": 180.0,
                 "geometries": {"dimer-2.8": {"energy": -100.0, "n_atoms": 2}}}
    scan = dict(BASIS_SCAN, pw_reference=reference)
    bundle = ConfigLoader(_write_input(tmp_path, scan)).load_all()
    plan = plan_runs(bundle, output_root=tmp_path / "run")[0]
    inputs = scan_workchain_inputs(plan)
    assert isinstance(inputs["pw_reference"], orm.Dict)
    assert inputs["pw_reference"].get_dict() == reference
    assert "pw_reference_pk" not in inputs


def test_the_pw_reference_is_wrapped_because_aiida_forbids_dotted_top_level_keys(profile):
    """Why `scan.pw_reference` is `{"geometries": {...}}` and not the bare mapping.

    Geometry names of a perturbed dimer are `dimer-2.8`, and an AiiDA ``Dict`` refuses
    a *top-level* key with a dot — the failure only shows up when the node is created,
    i.e. after the queue has been fed.  Nested keys are fine, so one level of structure
    buys storability; the WorkChain accepts either shape.
    """
    from aiida import orm
    from aiida.common.exceptions import ValidationError

    with pytest.raises(ValidationError, match="cannot contain"):
        orm.Dict(dict={"dimer-2.8": {"energy": -100.0}})
    assert orm.Dict(dict={"geometries": {"dimer-2.8": {"energy": -100.0}}}) is not None


def test_a_json_integer_gets_the_float_port_the_workchain_declares(tmp_path, profile):
    """`"atomization_tolerance_meV": 50` is an int in JSON but a Float in the spec."""
    from aiida import orm

    scan = dict(BASIS_SCAN)
    scan["atomization_tolerance_meV"] = 50          # int, not 50.0
    scan["reference_ecutjy"] = 150                  # int, not 150.0
    scan["ecutwfc"] = 180                           # int, not 180.0
    bundle = ConfigLoader(_write_input(tmp_path, scan)).load_all()
    plan = plan_runs(bundle, output_root=tmp_path / "run")[0]
    inputs = scan_workchain_inputs(plan)
    for key in ("atomization_tolerance_meV", "reference_ecutjy", "ecutwfc"):
        assert isinstance(inputs[key], orm.Float), key
        assert not isinstance(inputs[key], orm.Int), key


def test_the_two_ways_of_giving_a_pw_reference_are_exclusive(tmp_path, profile):
    scan = dict(BASIS_SCAN, pw_reference={"geometries": {"monomer": {"energy": -1.0}}},
                pw_reference_pk=469224)
    bundle = ConfigLoader(_write_input(tmp_path, scan)).load_all()
    plan = plan_runs(bundle, output_root=tmp_path / "run")[0]
    with pytest.raises(ValueError, match="mutually exclusive"):
        scan_workchain_inputs(plan)


def test_a_wrong_pw_reference_pk_says_what_is_wrong(tmp_path, profile):
    """Reusing a PW reference is the whole reason the cutoff scan runs first."""
    from aiida_orbgen.cli._common import reuse_pw_reference

    with pytest.raises(ValueError, match="cannot load that node"):
        reuse_pw_reference(999_999_999)


# ---------------------------------------------------------------------------
#  the CLI itself
# ---------------------------------------------------------------------------
def test_check_validates_a_scan_input_offline(tmp_path, capsys):
    path = _write_input(tmp_path, BASIS_SCAN)
    assert main(["check", "-i", str(path)]) == 0
    out = capsys.readouterr().out
    assert "workflow=orbgen.basis" in out
    assert "scan_basis" in out
    assert "ecutjy 125/100" in out


def test_check_rejects_a_bad_scan_section(tmp_path, capsys):
    path = _write_input(tmp_path, {"ecutjy_values": [125, "100"]})
    assert main(["check", "-i", str(path)]) == 1
    assert "only contain numbers" in capsys.readouterr().err


def test_run_dry_run_plans_a_scan_without_submitting(tmp_path, capsys):
    path = _write_input(tmp_path, ECUTWFC_SCAN)
    assert main(["run", "-i", str(path), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "[plan] workflow=orbgen.ecutwfc" in out
    assert "ecutwfc ladder [100, 120, 150, 180, 200] Ry" in out
    assert "nothing submitted" in out
    assert not (tmp_path / "output.json").exists()


def test_report_accepts_a_scan_output_json(tmp_path, monkeypatch, capsys):
    """A scan's report is its energy table, not an orbital flat (see test_scan_report).

    This used to be a refusal; it now writes `report.md` / `energies.csv` /
    `decision.json` through the scan renderer (patched here so the test stays offline).
    """
    from aiida_orbgen.cli import run as cli_run

    monkeypatch.setattr(
        "aiida_orbgen.utils.report.scan.write_scan_report",
        lambda node, out_dir, **kwargs: {"report": Path(out_dir) / kwargs.get(
            "report_name", "report.md"), "csv": Path(out_dir) / "energies.csv",
            "json": Path(out_dir) / "decision.json"},
    )
    monkeypatch.setattr("aiida.load_profile", lambda *a, **k: None)
    monkeypatch.setattr("aiida.orm.load_node", lambda identifier: identifier)
    output = tmp_path / "output.json"
    output.write_text(json.dumps({
        "workflow": WORKFLOW_BASIS,
        "abacus": {"orbgen": {"test": "uuid:1234"}},
    }))
    assert cli_run.main(["report", "-i", str(output), "-o", str(tmp_path / "out")]) == 0
    out = capsys.readouterr().out
    assert "energies" in out and "decision" in out


def test_the_method_registry_lists_all_four_workflows():
    assert set(METHOD_SPECS) == {
        "orbgen.calc", "orbgen.gridsearch", "orbgen.ecutwfc", "orbgen.basis"
    }
