"""Offline tests for the ``parameters``-driven CLI (``run`` / ``check``).

Everything here runs **without** an AiiDA profile: the config loader, the
submission planner, the ``output.json`` helpers and the Markdown renderer are
all AiiDA-free by design.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from types import SimpleNamespace

from aiida_orbgen.cli._common import plan_runs
from aiida_orbgen.cli.run import main
from aiida_orbgen.utils.cal_json import (
    SubmittedJob,
    build_cal_json,
    collect_job_entries,
    default_result_path,
    write_cal_json,
)
from aiida_orbgen.spec import AbacusSpec, OrbgenSpec
from aiida_orbgen.utils.select import (
    choose_point,
    selection_payload,
    write_selection,
)
from aiida_orbgen.utils.config import (
    DEFAULT_TOLERANCE_MEV,
    ConfigLoader,
    WORKFLOW_CALC,
    WORKFLOW_GRIDSEARCH,
    candidates_from_orbgen,
    canonical_abacus_config,
    SIAB_ONLY_INPUT_KEYS,
    validate_abacus_input,
    validate_siab_config,
)
from aiida_orbgen.utils.report.orbgen import GridPoint, OrbgenRunSummary, render_report

# ---------------------------------------------------------------------------
#  fixtures
# ---------------------------------------------------------------------------


def _write_input(tmp_path, parameters, extra=None):
    """Write a minimal input.json (+ a dummy UPF) and return its path."""
    pseudo = tmp_path / "U.pbe-n-nc.14ve.UPF"
    if not pseudo.exists():
        pseudo.write_text("<UPF version='1.0.0'>z_valence=\"14\"</UPF>\n")

    payload = {
        "parameters": parameters,
        "static": {"pseudo_path": str(pseudo), "metadata": "yeesuan"},
        "profile": "aiida_profile",
        "code": {"abacus": "abacus_lts@yeesuan"},
    }
    if extra:
        payload.update(extra)
    path = tmp_path / "input.json"
    path.write_text(json.dumps(payload, indent=4))
    return path


# ---------------------------------------------------------------------------
#  ConfigLoader
# ---------------------------------------------------------------------------


def test_dict_form_selects_the_named_file_and_preset(tmp_path):
    """The reference spelling: {"abacus": {"test": "test"}}."""
    path = _write_input(
        tmp_path,
        {"abacus": {"test": "test"}, "orbgen": {"test": "test"}},
    )
    bundle = ConfigLoader(path).load_all()

    assert [preset.name for preset in bundle.abacus_presets] == ["test"]
    assert [preset.name for preset in bundle.orbgen_presets] == ["test"]
    assert bundle.abacus_presets[0].source.endswith("parameters/abacus/test.yml#test")
    assert bundle.orbgen_presets[0].source.endswith("parameters/orbgen/test.yml#test")

    # canonical abacus.json shape
    config = bundle.abacus_presets[0].config
    assert config["basis"] == ["pw", "lcao_nsw"]
    assert config["abacus"]["parameters"]["input"]["ks_solver"] == "scalapack_gvx"
    assert config["abacus"]["code"] == "abacus_lts@yeesuan"
    assert config["abacus"]["max_iterations"] == 4
    assert config["tolerance_meV"] == DEFAULT_TOLERANCE_MEV == 100.0
    # scheduler options come from parameters/metadata.yml
    assert config["abacus"]["metadata"]["options"]["queue_name"] == "q_ysuan"

    # static.pseudo_path is injected as SIAB's pseudo_dir
    assert bundle.orbgen_presets[0].config["pseudo_dir"] == bundle.pseudo_path


def test_static_tolerance_overrides_the_preset(tmp_path):
    """`static.tolerance_meV` is the run-level knob: it wins over the preset.

    The workflow's standard is 100 meV/atom (the paper's strict 0.1 kcal/mol/atom =
    4.2 meV/atom was out of reach for a tractable U basis), and a run may relax or
    tighten it without editing `parameters/abacus/*.yml`.
    """
    path = _write_input(
        tmp_path,
        {"abacus": {"test": "test"}, "orbgen": {"test": "test"}},
    )
    assert ConfigLoader(path).load_all().abacus_presets[0].config["tolerance_meV"] == 100.0

    # rewrite with an explicit tolerance
    payload = json.loads(path.read_text())
    payload["static"]["tolerance_meV"] = 250
    path.write_text(json.dumps(payload, indent=4))
    config = ConfigLoader(path).load_all().abacus_presets[0].config
    assert config["tolerance_meV"] == 250.0


def _fake_upf(path, *, projectors=((0, 984), (1, 984)), dij=None, declare=None, nwfc=None):
    """A minimal UPF with the blocks the checks look at."""
    blocks = "\n".join(
        f'<PP_BETA.{index} angular_momentum="{l}" cutoff_radius_index="{ci}" '
        f'label="X{index}">0.0 1.0 2.0</PP_BETA.{index}>'
        for index, (l, ci) in enumerate(projectors, start=1)
    )
    count = len(projectors) if declare is None else declare
    wavefunctions = len(projectors) if nwfc is None else nwfc
    values = " ".join("0.0" for _ in range(len(projectors) ** 2)) if dij is None else dij
    path.write_text(
        f'<UPF version="2.0.1">\n<PP_HEADER number_of_proj="{count}" '
        f'mesh_size="3" number_of_wfc="{wavefunctions}" z_valence="14"/>\n{blocks}\n'
        f"<PP_DIJ>{values}</PP_DIJ>\n</UPF>\n"
    )
    return path


def test_upf_warning_for_a_projector_without_a_reference_state(tmp_path):
    """The file that broke ABACUS's LCAO path lists its PP_BETA blocks as
    ``l = 0,0,1,2,3,1``: that order is what mispaired the projectors with the radial
    functions (see ``UPF-INVESTIGATION.md``), and the extra ``l=1`` channel without a
    reference state is how such an order arises.  Both are reported.

    The working ``U.pbe-n-nc.14ve.UPF`` has 5 projectors for 5 wavefunctions and is
    grouped, so it is clean even though two of its *s* projectors share a cutoff radius.
    """
    from aiida_orbgen.utils.upf import upf_warnings

    # exactly the working file's shape: two s projectors, one per other l, 5 chi
    good = _fake_upf(tmp_path / "good.UPF", projectors=((0, 984), (0, 984), (1, 985)),
                     nwfc=3)
    assert upf_warnings(good) == []

    # the new file's shape: an extra l=1 at the same cutoff, 5 chi, not grouped by l
    bad = _fake_upf(tmp_path / "bad.UPF",
                    projectors=((0, 990), (1, 990), (2, 990), (1, 990)), nwfc=3)
    warnings = upf_warnings(bad)
    assert len(warnings) == 2
    assert "not grouped by angular momentum (order [0, 1, 2, 1])" in warnings[0]
    assert "--sort-by-l" in warnings[0]
    assert "4 projectors but only 3 atomic wavefunctions" in warnings[1]
    assert "l=1 twice" in warnings[1]

    # grouping them by l (the same pseudopotential, renumbered) is clean
    grouped = _fake_upf(tmp_path / "grouped.UPF",
                        projectors=((0, 990), (1, 990), (1, 991)), nwfc=2)
    assert not any("not grouped" in w for w in upf_warnings(grouped))


def test_upf_warning_for_blocks_that_disagree_with_the_header(tmp_path):
    from aiida_orbgen.utils.upf import upf_warnings

    text = _fake_upf(tmp_path / "mismatch.UPF", projectors=((0, 1), (1, 2)),
                     declare=3).read_text()
    assert any("number_of_proj=3" in w for w in upf_warnings(tmp_path / "mismatch.UPF"))

    _fake_upf(tmp_path / "short_dij.UPF", projectors=((0, 1), (1, 2)), dij="0.0 0.0 0.0")
    assert any("PP_DIJ holds 3 values" in w
               for w in upf_warnings(tmp_path / "short_dij.UPF"))
    assert text


def test_the_loader_reports_upf_problems_on_check(tmp_path):
    """`check` must say it before anything is submitted, not after the DFT."""
    _fake_upf(tmp_path / "U.test.UPF", projectors=((0, 990), (1, 990), (1, 990)),
              nwfc=2)
    path = _write_input(
        tmp_path, {"abacus": {"test": "test"}, "orbgen": {"test": "test"}},
        extra={"static": {"pseudo_path": str(tmp_path / "U.test.UPF"),
                          "metadata": "yeesuan"}},
    )
    bundle = ConfigLoader(path).load_all()
    assert any("l=1 twice" in w for w in bundle.warnings)


def test_inline_siab_config_replaces_the_orbgen_preset(tmp_path):
    """``static.siab_config`` is how `select` re-runs one chosen grid point."""
    inline = {
        "element": "U",
        "ecutjy": 150,
        "ecutwfc": 150,
        "bessel_nao_rcut": [10.0],
        "geoms": [{"proto": "dimer", "pertkind": "stretch", "pertmags": [2.2],
                   "lmaxmax": 4}],
        "orbitals": [_orbital(nzeta=[3, 2, 2, 1], geoms=[0])],
    }
    path = _write_input(
        tmp_path,
        {"abacus": {"test": "test"}, "orbgen": {"test": "test"}},
    )
    payload = json.loads(path.read_text())
    payload["static"]["siab_config"] = inline
    path.write_text(json.dumps(payload))

    bundle = ConfigLoader(path).load_all()

    assert [preset.name for preset in bundle.orbgen_presets] == ["inline"]
    assert bundle.orbgen_presets[0].config["ecutjy"] == 150
    # static.pseudo_path is still injected into the inline config
    assert bundle.orbgen_presets[0].config["pseudo_dir"] == bundle.pseudo_path
    # one (l_max, r_cut) -> the loader picks orbgen.calc with no explicit workflow
    assert bundle.workflow == WORKFLOW_CALC
    assert bundle.workflow_explicit is False
    assert bundle.candidates() == [(4, 10.0)]
    # parameters['orbgen'] is still there, so the override has to say so
    assert any("static.siab_config is set" in warning for warning in bundle.warnings)


def test_inline_siab_config_without_the_orbgen_slot(tmp_path):
    """An inline config makes ``parameters['orbgen']`` unnecessary."""
    path = _write_input(tmp_path, {"abacus": {"test": "test"}})
    payload = json.loads(path.read_text())
    payload["static"]["siab_config"] = {
        "element": "U", "ecutjy": 150, "bessel_nao_rcut": [10.0],
        "geoms": [{"proto": "dimer", "pertkind": "stretch", "pertmags": [2.2],
                   "lmaxmax": 4}],
        "orbitals": [_orbital(nzeta=[3, 2, 2, 1], geoms=[0])],
    }
    path.write_text(json.dumps(payload))

    bundle = ConfigLoader(path).load_all()
    assert [preset.name for preset in bundle.orbgen_presets] == ["inline"]
    assert not any("is ignored" in warning for warning in bundle.warnings)


def test_inline_siab_config_is_validated_like_a_preset(tmp_path):
    """An inline config is not a way to skip the checks a preset gets."""
    path = _write_input(tmp_path, {"abacus": {"test": "test"}})
    payload = json.loads(path.read_text())
    payload["static"]["siab_config"] = {"element": "U"}     # no ecutjy / geoms
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="ecutjy"):
        ConfigLoader(path).load_all()


def test_inline_siab_config_name_names_the_run_directory(tmp_path):
    """The name is used for `run/<name>/lmax4_rcut10`, so it must be editable."""
    path = _write_input(tmp_path, {"abacus": {"test": "test"}})
    payload = json.loads(path.read_text())
    payload["static"]["siab_config"] = {
        "element": "U", "ecutjy": 150, "bessel_nao_rcut": [10.0],
        "geoms": [{"proto": "dimer", "pertkind": "stretch", "pertmags": [2.2],
                   "lmaxmax": 4}],
        "orbitals": [_orbital(nzeta=[3, 2, 2, 1], geoms=[0])],
    }
    path.write_text(json.dumps(payload))

    # default name
    assert [p.name for p in ConfigLoader(path).load_all().orbgen_presets] == ["inline"]

    payload["static"]["siab_config_name"] = "pbe_nc_150Ry"
    path.write_text(json.dumps(payload))
    bundle = ConfigLoader(path).load_all()
    assert [p.name for p in bundle.orbgen_presets] == ["pbe_nc_150Ry"]
    from aiida_orbgen.cli._common import plan_runs

    plans = plan_runs(bundle, output_root=tmp_path / "run")
    assert plans[0].output_dir.parent.name == "pbe_nc_150Ry"

    payload["static"]["siab_config_name"] = "   "
    path.write_text(json.dumps(payload))
    with pytest.raises(TypeError, match="siab_config_name"):
        ConfigLoader(path).load_all()


def test_inline_siab_config_must_be_an_object(tmp_path):
    path = _write_input(tmp_path, {"abacus": {"test": "test"}})
    payload = json.loads(path.read_text())
    payload["static"]["siab_config"] = "parameters/orbgen/test.yml"
    path.write_text(json.dumps(payload))
    with pytest.raises(TypeError, match="static.*siab_config"):
        ConfigLoader(path).load_all()


def test_flat_form_and_auto_workflow(tmp_path):
    """A single (l_max, r_cut) candidate needs no grid search."""
    path = _write_input(tmp_path, {"abacus": "test", "orbgen": "u_14ve"})
    bundle = ConfigLoader(path).load_all()

    assert [preset.name for preset in bundle.abacus_presets] == ["test"]
    assert [preset.name for preset in bundle.orbgen_presets] == ["u_14ve"]
    # u_14ve carries bessel_nao_rcut [9, 10] → two candidates → grid search
    assert bundle.workflow == WORKFLOW_GRIDSEARCH
    assert bundle.workflow_explicit is False
    assert bundle.candidates() == [(4, 9.0), (4, 10.0)]


def test_workflow_can_be_forced(tmp_path):
    path = _write_input(
        tmp_path,
        {"abacus": "test", "orbgen": "test"},
        extra={"workflow": WORKFLOW_CALC},
    )
    bundle = ConfigLoader(path).load_all()
    assert bundle.workflow == WORKFLOW_CALC
    assert bundle.workflow_explicit is True


def test_list_form_gives_several_presets(tmp_path):
    path = _write_input(tmp_path, {"abacus": ["test", "u_14ve"], "orbgen": "test"})
    bundle = ConfigLoader(path).load_all()
    assert [preset.name for preset in bundle.abacus_presets] == ["test", "u_14ve"]
    assert len(bundle.pairs()) == 2


def test_unknown_preset_reports_the_available_ones(tmp_path):
    path = _write_input(tmp_path, {"abacus": "nope", "orbgen": "test"})
    with pytest.raises(KeyError, match="nope"):
        ConfigLoader(path).load_all()


def test_unknown_slot_is_rejected(tmp_path):
    path = _write_input(tmp_path, {"abacus": "test", "vasp": "test"})
    with pytest.raises(KeyError, match="unknown slot"):
        ConfigLoader(path).load_all()


def test_missing_required_key_is_rejected(tmp_path):
    path = tmp_path / "input.json"
    path.write_text(json.dumps({"parameters": {"abacus": "test"}}))
    with pytest.raises(KeyError, match="static"):
        ConfigLoader(path).load_all()


def test_missing_pseudo_path_is_rejected(tmp_path):
    path = _write_input(tmp_path, {"abacus": "test", "orbgen": "test"})
    payload = json.loads(path.read_text())
    payload["static"]["pseudo_path"] = str(tmp_path / "does-not-exist.UPF")
    path.write_text(json.dumps(payload))
    with pytest.raises(FileNotFoundError, match="pseudo_path"):
        ConfigLoader(path).load_all()


def test_canonical_abacus_config_accepts_the_nested_spelling():
    preset = {
        "basis": ["lcao_nsw"],
        "abacus": {
            "code": "abacus@host",
            "parameters": {"input": {"ecutwfc": 120}},
            "metadata": {"options": {"queue_name": "q2"}},
        },
        "tolerance_meV": 3.0,
    }
    config = canonical_abacus_config(preset, code=None, options={"queue_name": "q1"})
    assert config["basis"] == ["lcao_nsw"]
    assert config["abacus"]["code"] == "abacus@host"
    assert config["abacus"]["parameters"]["input"] == {"ecutwfc": 120}
    assert config["abacus"]["metadata"]["options"]["queue_name"] == "q2"
    assert config["tolerance_meV"] == 3.0


def test_candidates_respect_the_max_caps():
    orbgen = {"bessel_nao_rcut": [8, 10, 12], "geoms": [{"lmaxmax": 4}]}
    assert candidates_from_orbgen(orbgen) == [(4, 8.0), (4, 10.0), (4, 12.0)]
    assert candidates_from_orbgen(orbgen, {"max_r_cut": 10}) == [(4, 8.0), (4, 10.0)]
    assert candidates_from_orbgen(orbgen, {"max_l_max": 3}) == []


# ---------------------------------------------------------------------------
#  SIAB preset validation (the `KeyError: 'ecutjy'` failure mode)
# ---------------------------------------------------------------------------

def _orbital(**overrides) -> dict:
    """A single orbital entry with SIAB's four compulsory keys (ORBITAL_COMPULSORY_)."""
    orbital = {"nzeta": [3, 2, 2, 1, 0], "geoms": [0], "nbands": "occ",
               "checkpoint": None}
    orbital.update(overrides)
    return orbital


_COMPLETE_SIAB = {
    "element": "U",
    "ecutwfc": 150,
    "ecutjy": 100,
    "bessel_nao_rcut": [9, 10],
    "geoms": [{"proto": "dimer", "pertkind": "stretch", "pertmags": "auto",
               "lmaxmax": 4}],
    "orbitals": [_orbital()],
    "abacus_command": "abacus",
}


def test_validate_siab_config_rejects_a_missing_ecutjy():
    """`generate_nsw` indexes `ecutjy` — it must fail at load time, not in the daemon."""
    config = dict(_COMPLETE_SIAB)
    config.pop("ecutjy")
    with pytest.raises(ValueError, match="ecutjy"):
        validate_siab_config(config)


def test_validate_siab_config_rejects_incomplete_geoms():
    config = dict(_COMPLETE_SIAB)
    config["geoms"] = [{"proto": "dimer"}]
    with pytest.raises(ValueError, match="pertkind"):
        validate_siab_config(config)


def test_validate_siab_config_warns_about_final_orbital_keys():
    config = {key: value for key, value in _COMPLETE_SIAB.items()
              if key != "abacus_command"}
    warnings = validate_siab_config(config)
    assert any("abacus_command" in warning for warning in warnings)
    assert all("missing required" not in warning for warning in warnings)


def test_loader_rejects_an_incomplete_orbgen_preset(tmp_path, monkeypatch):
    """A preset missing `ecutjy` aborts `run`/`check` with a readable error."""
    from aiida_orbgen.utils import config as config_module

    tree = tmp_path / "parameters"
    (tree / "abacus").mkdir(parents=True)
    (tree / "orbgen").mkdir(parents=True)
    (tree / "metadata.yml").write_text("yeesuan:\n  options:\n    queue_name: q\n")
    (tree / "abacus" / "abacus.yml").write_text("test:\n  basis: [pw, lcao_nsw]\n")
    # deliberately incomplete: no ecutjy / geoms / orbitals
    (tree / "orbgen" / "orbgen.yml").write_text(
        "broken:\n  element: U\n  bessel_nao_rcut: [9]\n"
    )
    monkeypatch.setattr(config_module, "PARAMETERS_DIR", tree)
    monkeypatch.setattr(config_module, "METADATA_FILE", tree / "metadata.yml")

    path = _write_input(tmp_path, {"abacus": "test", "orbgen": "broken"})
    with pytest.raises(ValueError, match="ecutjy"):
        ConfigLoader(path).load_all()


# ---------------------------------------------------------------------------
#  ABACUS preset validation (the `ecutjy` → "Bad parameter" failure mode)
# ---------------------------------------------------------------------------


def test_validate_siab_config_rejects_flat_vloc_aux():
    """SIAB only reads vloc_aux/lloc_min from `model_kwargs`; flat = ignored."""
    config = dict(_COMPLETE_SIAB)
    config["orbitals"] = [_orbital(nzeta=[4, 3, 2, 2, 1], geoms=[0],
                                   lloc_min=4, vloc_aux="/tmp/U.UPF")]
    with pytest.raises(ValueError, match="model_kwargs"):
        validate_siab_config(config)

    config["orbitals"] = [_orbital(nzeta=[4, 3, 2, 2, 1], geoms=[0],
                                   model_kwargs={"lloc_min": 4,
                                                 "vloc_aux": "/tmp/U.UPF"})]
    warnings = validate_siab_config(config)  # must not raise
    assert all("model_kwargs" not in warning for warning in warnings)


def test_validate_abacus_input_rejects_siab_only_keys():
    """`ecutjy` inside parameters.input makes ABACUS abort the whole INPUT."""
    config = {"abacus": {"parameters": {"input": {"ks_solver": "scalapack_gvx",
                                                  "ecutjy": 100}}}}
    with pytest.raises(ValueError, match="ecutjy"):
        validate_abacus_input(config)


def test_validate_abacus_input_warns_about_managed_keys():
    config = {"abacus": {"parameters": {"input": {"basis_type": "lcao"}}}}
    warnings = validate_abacus_input(config)
    assert len(warnings) == 1 and "basis_type" in warnings[0]


def test_shipped_abacus_presets_carry_no_siab_only_keys():
    """Regression guard for the 2026-09-19 run: no preset may inject `ecutjy`."""
    from aiida_orbgen.utils.config import PARAMETERS_DIR, read_yaml

    for table_path in sorted((PARAMETERS_DIR / "abacus").glob("*.yml")):
        table = read_yaml(table_path)
        for name, preset in table.items():
            overrides = (preset.get("parameters") or {}).get("input") or {}
            bad = sorted(set(overrides) & set(SIAB_ONLY_INPUT_KEYS))
            assert not bad, f"{table_path.name}#{name} sets {bad} in parameters.input"


def test_loader_forwards_no_siab_only_key_to_the_abacus_input(tmp_path):
    path = _write_input(tmp_path, {"abacus": "test", "orbgen": "test"})
    bundle = ConfigLoader(path).load_all()
    overrides = bundle.abacus_presets[0].config["abacus"]["parameters"]["input"]
    assert "ecutjy" not in overrides
    assert overrides["ks_solver"] == "scalapack_gvx"
    assert bundle.warnings == []


# ---------------------------------------------------------------------------
#  final-orbital plumbing (point selection, preset re-read, DFT root discovery)
# ---------------------------------------------------------------------------


def _make_dft_tree(root, folders):
    for name in folders:
        (root / name / "OUT.ABACUS").mkdir(parents=True)
        (root / name / "OUT.ABACUS" / "running_scf.log").write_text("Finish Time\n")


def _two_point_summary():
    def point(l_max, r_cut, pk, element="U", pertmags=(1.89, 2.09)):
        return GridPoint(
            l_max=l_max, r_cut=r_cut, pk=pk, exit_status=304, finished_ok=True,
            process_state="finished", tolerance_meV=4.2,
            delta_max_per_atom_meV=10.0 * pk, delta_max_meV=20.0,
            delta_min_per_atom_meV=8.0, per_struct=[],
            siab_info={
                "family_label": f"siab-u-{int(r_cut)}au",
                "nsw_filename": f"U_gga_{int(r_cut)}au_100Ry.orb",
                "dft": [{"folder": f"{element}-dimer-{pert}-{int(r_cut)}au"}
                        for pert in pertmags],
            },
            output_dir=f"/tmp/run/lmax{l_max}_rcut{int(r_cut)}p0",
        )

    return OrbgenRunSummary(
        node_pk=1, node_uuid="uuid", label="OrbgenGridSearchWorkChain",
        kind="gridsearch", status="Finished [0]", exit_status=0,
        process_state="finished", tolerance_meV=4.2,
        grid=[point(4, 9.0, 101), point(4, 10.0, 102)],
        best={"l_max": 4, "r_cut": 9.0, "delta_max_per_atom_meV": 1010.0,
              "calc_pk": 101, "source": "lowest-deltaE"},
    )


def test_choose_point_prefers_the_point_the_dft_tree_covers(tmp_path):
    from aiida_orbgen.utils.report.orbitals import choose_point_for_dft_root

    summary = _two_point_summary()
    # a tree prepared for r_cut = 10 only (the ΔE-best point is r_cut = 9!)
    _make_dft_tree(tmp_path, [f"U-dimer-{p}-10au" for p in (1.89, 2.09)]
                   + ["U-monomer-10au"])

    point, status = choose_point_for_dft_root(summary, tmp_path)
    assert point is not None and point.pk == 102 and point.r_cut == 10.0
    assert status == {"l_max=4, r_cut=9au": "incomplete",
                      "l_max=4, r_cut=10au": "complete"}


def test_choose_point_returns_nothing_for_an_unrelated_tree(tmp_path):
    from aiida_orbgen.utils.report.orbitals import choose_point_for_dft_root

    _make_dft_tree(tmp_path, ["U-dimer-1.89-7au"])
    point, status = choose_point_for_dft_root(_two_point_summary(), tmp_path)
    assert point is None
    assert set(status.values()) == {"incomplete"}


def test_dft_root_is_read_from_input_json(tmp_path):
    from aiida_orbgen.utils.report.orbitals import (
        dft_root_from_input_json,
        load_preset_config,
    )

    input_json = tmp_path / "input.json"
    input_json.write_text(json.dumps({
        "parameters": {"abacus": "test", "orbgen": "test"},
        "static": {"dft_root": "/tmp/reference_runs"},
    }))
    assert dft_root_from_input_json(input_json) == Path("/tmp/reference_runs")
    assert dft_root_from_input_json(tmp_path / "nope.json") is None

    # loading presets needs a full input.json; a preset-free one must not raise
    config, reason = load_preset_config(input_json, 4, 10.0)
    assert config is None and reason


def test_load_preset_config_applies_the_grid_point(tmp_path):
    from aiida_orbgen.utils.report.orbitals import load_preset_config

    path = _write_input(tmp_path, {"abacus": "test", "orbgen": "test"})
    config, reason = load_preset_config(path, 4, 10.0)
    assert reason is None
    assert config["bessel_nao_rcut"] == [10]
    assert config["geoms"][0]["lmaxmax"] == 4
    # the fixed preset keeps vloc_aux nested where SIAB actually reads it
    assert "vloc_aux" not in config["orbitals"][1]
    assert config["orbitals"][1]["model_kwargs"]["vloc_aux"].endswith(".UPF")


def test_configs_differ_is_order_insensitive():
    from aiida_orbgen.utils.report.orbitals import configs_differ

    assert not configs_differ({"a": 1, "b": [1, 2]}, {"b": [1, 2], "a": 1})
    assert configs_differ({"a": 1}, {"a": 2})


def test_claim_path_overwrites_previous_runs_but_dedups_within_one(tmp_path):
    """Re-exporting must not pile up `lmax…_rcut…_` duplicates."""
    from aiida_orbgen.utils.report.orbitals import _claim_path

    class _Point:
        def __init__(self, pk, l_max, r_cut):
            self.pk, self.l_max, self.r_cut = pk, l_max, r_cut

    point_a, point_b = _Point(1, 4, 9.0), _Point(2, 4, 10.0)

    # a file left behind by an earlier report run: same node -> overwrite
    stale = tmp_path / "U_gga_9au_100Ry.orb"
    stale.write_text("old")

    claimed: dict = {}
    assert _claim_path(tmp_path, stale.name, point_a, claimed) == stale
    assert _claim_path(tmp_path, stale.name, point_a, claimed) == stale

    # a *different* point wanting the same name in the same run -> prefixed
    prefixed = _claim_path(tmp_path, stale.name, point_b, claimed)
    assert prefixed.name == "lmax4_rcut10_U_gga_9au_100Ry.orb"


def test_final_orbital_rerun_and_idempotency(tmp_path, monkeypatch):
    """A re-run reports the files it overwrote, and skips work that is up to date."""
    import os

    import aiida_orbgen.utils.report.orbitals as orbitals
    from aiida_orbgen.utils.report.orbitals import generate_final_orbital

    summary = _two_point_summary()
    point = summary.grid[1]  # (4, 10)
    _make_dft_tree(tmp_path / "dft", ["U-dimer-1.89-10au", "U-dimer-2.09-10au",
                                      "U-monomer-10au"])

    stub = tmp_path / "fake_orbgen.sh"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        "out='.'\n"
        "while [[ $# -gt 0 ]]; do case \"$1\" in -o) out=\"$2\"; shift 2;; *) shift;; esac; done\n"
        "printf 'fresh\\n' > \"$out/U_gga_10au_100Ry_3s2p2d1f.orb\"\n"
    )
    stub.chmod(0o755)

    monkeypatch.setattr(
        "aiida.orm.load_node", lambda pk: type("N", (), {"pk": pk})()
    )
    monkeypatch.setattr(
        orbitals, "_build_siab_config", lambda *a, **k: {"element": "U", "geoms": [{}]}
    )

    out_dir = tmp_path / "out"
    work_dir = out_dir / "lmax4_rcut10"
    work_dir.mkdir(parents=True)
    stale = work_dir / "U_gga_10au_100Ry_3s2p2d1f.orb"
    stale.write_text("stale\n")
    old = os.path.getmtime(stale) - 3600
    os.utime(stale, (old, old))          # older than the reference DFT

    result = generate_final_orbital(
        summary, out_dir, dft_root=tmp_path / "dft", point=point,
        orbgen_command=str(stub),
    )
    assert result["status"] == "ok", result
    assert result["dir"] == str(work_dir)
    assert Path(result["config"]).parent == work_dir
    # the overwritten file is still reported as produced (mtime, not name diff)
    assert [Path(p).name for p in result["files"]] == ["U_gga_10au_100Ry_3s2p2d1f.orb"]
    assert stale.read_text() == "fresh\n"

    # second run: the products are newer than the reference DFT -> no recompute
    again = generate_final_orbital(
        summary, out_dir, dft_root=tmp_path / "dft", point=point,
        orbgen_command=str(stub),
    )
    assert again["status"] == "already-present", again
    assert "already newer than the reference DFT" in again["message"]

    # ... unless a redo is requested
    redone = generate_final_orbital(
        summary, out_dir, dft_root=tmp_path / "dft", point=point,
        orbgen_command=str(stub), redo=True,
    )
    assert redone["status"] == "ok", redone


# ---------------------------------------------------------------------------
#  basis consistency of a reference tree (nao vs the config's primitive basis)
# ---------------------------------------------------------------------------


def _fake_reference(jobdir, nao, natom=2):
    """Minimal reference geometry: STRU + OUT.ABACUS/{data-0-H,running_scf.log}."""
    out = jobdir / "OUT.ABACUS"
    out.mkdir(parents=True)
    (out / "data-0-H").write_text(f"{nao} 1.0\n")
    (out / "running_scf.log").write_text("...\nFinish Time : now\n")
    atoms = "\n".join(
        f"{9.0 + i * 1.89:.8f} 9.26060186 9.26060186 0 0 0" for i in range(natom)
    )
    (jobdir / "STRU").write_text(
        "ATOMIC_SPECIES\nU 1.000000 U.pbe-n-nc.14ve.UPF\n\n"
        "NUMERICAL_ORBITAL\nU_gga_9au_100Ry_27s27p26d26f25g.orb\n\n"
        "LATTICE_CONSTANT\n35.000000  // add lattice constant(a.u.)\n"
        "LATTICE_VECTORS\n1.00000000 0.00000000 0.00000000\n"
        "0.00000000 1.00000000 0.00000000\n0.00000000 0.00000000 1.00000000\n"
        "ATOMIC_POSITIONS\nCartesian_angstrom_center_xyz\nU\n0.00\n"
        f"{natom}\n{atoms}\n"
    )
    return jobdir


def test_reference_basis_mismatch_is_detected(tmp_path):
    """A tree computed with another ecutjy must count as stale, not as done."""
    from aiida_orbgen.utils.report.orbitals import (
        _running_scf_done,
        expected_nao,
        expected_nao_by_folder,
    )

    # dimer of U with the 9au primitive basis: 100 Ry -> 645/atom, 150 Ry -> 810
    jobdir = _fake_reference(tmp_path / "U-dimer-1.89-9au", nao=1290, natom=2)
    cfg100 = {"ecutjy": 100, "ecutwfc": 150}
    cfg150 = {"ecutjy": 150, "ecutwfc": 150}

    assert expected_nao(cfg100, 4, 9.0, 2) == 1290
    assert expected_nao(cfg150, 4, 9.0, 2) == 1620

    assert _running_scf_done(jobdir, 1290)[0] is True
    done, why = _running_scf_done(jobdir, 1620)
    assert done is False and "different primitive basis" in why

    # ecutjy absent -> SIAB's own fallback to ecutwfc
    assert expected_nao({"ecutwfc": 150}, 4, 9.0, 2) == 1620
    by_folder = expected_nao_by_folder(tmp_path, ["U-dimer-1.89-9au"], cfg150, 4, 9.0)
    assert by_folder == {"U-dimer-1.89-9au": 1620}


# ---------------------------------------------------------------------------
#  DFT-tree resolution (one static.dft_root may serve every grid point)
# ---------------------------------------------------------------------------


def test_resolve_dft_root_searches_one_level_down(tmp_path):
    """`static.dft_root` may point at a directory of per-point runs."""
    from aiida_orbgen.utils.report.orbitals import resolve_dft_root

    summary = _two_point_summary()
    parent = tmp_path / "u_14ve"
    _make_dft_tree(parent / "run_lmax4_rcut9",
                   [f"U-dimer-{p}-9au" for p in (1.89, 2.09)] + ["U-monomer-9au"])
    _make_dft_tree(parent / "run_lmax4_rcut10",
                   [f"U-dimer-{p}-10au" for p in (1.89, 2.09)] + ["U-monomer-10au"])

    root9, origin9 = resolve_dft_root(summary.grid[0], None, dft_root=parent)
    root10, origin10 = resolve_dft_root(summary.grid[1], None, dft_root=parent)
    assert root9 == parent / "run_lmax4_rcut9"
    assert root10 == parent / "run_lmax4_rcut10"
    assert "run_lmax4_rcut9" in origin9 and "run_lmax4_rcut10" in origin10


def test_primitive_export_falls_back_to_the_reference_tree(tmp_path, monkeypatch):
    """The recorded `orb_path` points at the submitting machine's scratch dir.

    `/tmp/test_orbgen_output/...` is usually gone by the time a report runs, so
    the primitive has to be found where SIAB actually wrote it: inside the
    reference-DFT tree that the spillage step uses.
    """
    from aiida_orbgen.utils.report.orbitals import export_primitive_orbitals

    name = "U_gga_10au_150Ry_37s37p36d36f36g.orb"
    root = tmp_path / "run_lmax4_rcut10"
    (root / "primitive_jy").mkdir(parents=True)
    (root / "primitive_jy" / name).write_text("primitive orbital\n")

    point = GridPoint(
        l_max=4, r_cut=10.0, pk=410517, exit_status=304, finished_ok=False,
        process_state="finished",
        siab_info={
            "nsw_filename": name,
            "orb_path": "/tmp/test_orbgen_output/lmax4_rcut10p0/primitive_jy/" + name,
            "family_label": "siab-u-nr-pbe-z14-nsw-10au-150Ry-g",
        },
    )
    summary = SimpleNamespace(grid=[point])

    # This test is about the *filesystem* fallback, not about the database one: the
    # family lookup is switched off so the result cannot depend on which
    # AtomicOrbitalData nodes happen to be in the profile being used (it does —
    # `siab-u-nr-pbe-z14-nsw-10au-150Ry-g` exists for real, and then the "nothing to
    # fall back to" half of this test would find it).
    from aiida_orbgen.utils.report import orbitals as report_orbitals

    monkeypatch.setattr(report_orbitals, "_atomic_orbital_data_nodes",
                        lambda *args, **kwargs: [])

    # without a search root there is nothing to fall back to ...
    files, warnings = export_primitive_orbitals(summary, tmp_path / "out")
    assert files == []
    assert any("no .orb found in AiiDA" in w for w in warnings)

    # ... with one, the reference tree supplies the file
    files, warnings = export_primitive_orbitals(
        summary, tmp_path / "out2", search_roots=[root]
    )
    assert [f.path.name for f in files] == [name]
    assert files[0].kind == "primitive"
    assert files[0].source.startswith("filesystem:")
    assert (tmp_path / "out2" / name).read_text().startswith("primitive orbital")


def test_primitive_export_prefers_the_archived_orbital(tmp_path, monkeypatch):
    """Runs submitted after 2026-09-29 archive the primitive in provenance.

    That copy survives the scratch directory (`siab_info['orb_path']` points at
    `/tmp/...`), so it must be used before the family lookup or the filesystem
    fallbacks.
    """
    import aiida.orm
    from aiida_orbgen.utils.report.orbitals import export_primitive_orbitals

    name = "U_gga_10au_100Ry_30s30p29d29f28g.orb"

    class _Repo:
        def get_object_content(self, name, mode="rb"):
            return b"archived primitive\n"

    class _Node:
        pk = 4242
        base = SimpleNamespace(repository=_Repo())

    monkeypatch.setattr(aiida.orm, "load_node", lambda pk: _Node())

    point = GridPoint(
        l_max=4, r_cut=10.0, pk=410517, exit_status=304, finished_ok=False,
        process_state="finished",
        siab_info={"nsw_filename": name,
                   "orb_path": "/tmp/test_orbgen_output/primitive_jy/" + name,
                   "family_label": "siab-u-nr-pbe-z14-nsw-10au-100Ry-g"},
        primitive_orbital_pk=_Node.pk,
    )
    files, warnings = export_primitive_orbitals(
        SimpleNamespace(grid=[point]), tmp_path
    )
    assert warnings == []
    assert [f.path.name for f in files] == [name]
    assert files[0].source == f"aiida:primitive_orbital<{_Node.pk}>"
    assert (tmp_path / name).read_text() == "archived primitive\n"


def test_resolve_dft_root_prefers_the_per_point_mapping(tmp_path):
    from aiida_orbgen.utils.report.orbitals import resolve_dft_root

    summary = _two_point_summary()
    tree9 = tmp_path / "nine"
    _make_dft_tree(tree9, [f"U-dimer-{p}-9au" for p in (1.89, 2.09)]
                   + ["U-monomer-9au"])
    input_json = tmp_path / "input.json"
    input_json.write_text(json.dumps({
        "static": {"dft_root": str(tmp_path / "fallback"),
                   "dft_roots": {"lmax4_rcut9": str(tree9)}},
    }))

    root, origin = resolve_dft_root(summary.grid[0], None, input_json=input_json)
    assert root == tree9
    assert "dft_roots[lmax4_rcut9]" in origin

    # a point without a mapping entry falls back to static.dft_root
    root10, origin10 = resolve_dft_root(summary.grid[1], None,
                                        input_json=input_json)
    assert root10 == tmp_path / "fallback"
    assert origin10 == "input.json:static.dft_root"


def test_stale_reference_data_is_quarantined_before_a_rerun(tmp_path, monkeypatch):
    """A stale monomer must be moved aside, or SIAB would skip recomputing it."""
    import aiida_orbgen.utils.report.orbitals as orbitals
    from aiida_orbgen.utils.report.orbitals import generate_final_orbital

    summary = _two_point_summary()
    point = summary.grid[1]                       # (4, 10), ecutjy = 100 in _COMPLETE_SIAB
    dft = tmp_path / "dft"
    _make_dft_tree(dft, ["U-dimer-1.89-10au", "U-dimer-2.09-10au"])
    # monomer with a STRU + 100 Ry data (720 fns), config below asks 150 Ry (904)
    _fake_reference(dft / "U-monomer-10au", nao=720, natom=1)

    stub = tmp_path / "fake_orbgen.sh"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        "out='.'\n"
        "while [[ $# -gt 0 ]]; do case \"$1\" in -o) out=\"$2\"; shift 2;; *) shift;; esac; done\n"
        "printf 'orb\\n' > \"$out/U_gga_10au_100Ry_3s2p2d1f.orb\"\n"
    )
    stub.chmod(0o755)
    monkeypatch.setattr("aiida.orm.load_node", lambda pk: type("N", (), {"pk": pk})())
    monkeypatch.setattr(
        orbitals, "_build_siab_config",
        lambda *a, **k: {"element": "U", "ecutjy": 150, "ecutwfc": 150,
                         "geoms": [{"lmaxmax": 4}]},
    )

    result = generate_final_orbital(
        summary, tmp_path / "out", dft_root=dft, point=point,
        orbgen_command=str(stub), assemble=False,
    )
    assert result["status"] == "ok", result
    assert result["quarantined"] == {"U-monomer-10au": "U-monomer-10au.stale"}
    assert (dft / "U-monomer-10au.stale" / "OUT.ABACUS").is_dir()
    assert not (dft / "U-monomer-10au").exists()
    # dimers (expected_nao None without a STRU) stay untouched
    assert (dft / "U-dimer-1.89-10au" / "OUT.ABACUS").is_dir()


def test_monomer_only_gap_is_computed_automatically(tmp_path, monkeypatch):
    """A missing monomer is filled in; a missing dimer needs --force."""
    import aiida_orbgen.utils.report.orbitals as orbitals
    from aiida_orbgen.utils.report.orbitals import generate_final_orbital

    summary = _two_point_summary()
    point = summary.grid[1]  # (4, 10)
    dimers = ["U-dimer-1.89-10au", "U-dimer-2.09-10au"]
    _make_dft_tree(tmp_path / "dft", dimers)          # no monomer on purpose

    stub = tmp_path / "fake_orbgen.sh"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        "out='.'\n"
        "while [[ $# -gt 0 ]]; do case \"$1\" in -o) out=\"$2\"; shift 2;; *) shift;; esac; done\n"
        "printf 'orb\\n' > \"$out/U_gga_10au_100Ry_3s2p2d1f.orb\"\n"
    )
    stub.chmod(0o755)
    monkeypatch.setattr("aiida.orm.load_node", lambda pk: type("N", (), {"pk": pk})())
    monkeypatch.setattr(
        orbitals, "_build_siab_config", lambda *a, **k: {"element": "U", "geoms": [{}]}
    )

    run = generate_final_orbital(
        summary, tmp_path / "out", dft_root=tmp_path / "dft", point=point,
        orbgen_command=str(stub), assemble=False,
    )
    assert run["status"] == "ok", run            # monomer-only gap -> allowed

    skipped = generate_final_orbital(
        summary, tmp_path / "out2", dft_root=tmp_path / "dft", point=point,
        orbgen_command=str(stub), assemble=False, run_missing="none",
    )
    assert skipped["status"] == "skipped"
    assert "reference DFT incomplete (1/3 geometries)" in skipped["message"]

    # a missing *dimer* is never run implicitly
    _make_dft_tree(tmp_path / "dft2", ["U-monomer-10au"])
    blocked = generate_final_orbital(
        summary, tmp_path / "out3", dft_root=tmp_path / "dft2", point=point,
        orbgen_command=str(stub), assemble=False,
    )
    assert blocked["status"] == "skipped"
    assert "--force-final-orbital" in blocked["message"]


# ---------------------------------------------------------------------------
#  INPUT filtering (out_wfc_lcao must reach ABACUS for LCAO jobs)
# ---------------------------------------------------------------------------


def test_apply_input_overrides_keeps_out_wfc_lcao():
    """Regression: filtering it out left the AiiDA LCAO jobs without WFC_NAO."""
    from aiida_orbgen.static.defaults import AIIDA_MANAGED_KEYS, apply_input_overrides

    assert "out_wfc_lcao" not in AIIDA_MANAGED_KEYS
    merged = apply_input_overrides({
        "pseudo_dir": "./pseudo/", "orbital_dir": "./orbital/",
        "suffix": "ABACUS", "basis_type": "lcao", "bessel_nao_rcut": 9.0,
        "ecutwfc": 150, "out_wfc_lcao": 1, "out_mat_hs": "1 12",
    })
    assert merged["out_wfc_lcao"] == 1
    assert merged["out_mat_hs"] == "1 12"
    assert not {"pseudo_dir", "orbital_dir", "suffix", "basis_type",
                "bessel_nao_rcut"} & set(merged)


def test_loader_does_not_warn_about_out_wfc_lcao():
    from aiida_orbgen.utils.config import validate_abacus_input

    assert validate_abacus_input(
        {"abacus": {"parameters": {"input": {"out_wfc_lcao": 1}}}}
    ) == []
    warnings = validate_abacus_input(
        {"abacus": {"parameters": {"input": {"basis_type": "lcao"}}}}
    )
    assert len(warnings) == 1 and "basis_type" in warnings[0]


# ---------------------------------------------------------------------------
#  reference-tree assembly (H C = S C eps → WFC_NAO_GAMMA1.txt)
# ---------------------------------------------------------------------------


def _write_triu(path, matrix, fmt="{:.10e}"):
    """ABACUS data-{ik}-{H,S} layout: dim, then the upper triangle row-major."""
    import numpy as np

    n = matrix.shape[0]
    values = [str(n)]
    for i in range(n):
        values.extend(fmt.format(v) for v in matrix[i, i:])
    Path(path).write_text("\n".join(values) + "\n")
    return np.asarray([float(v) for v in values[1:]])


def test_reconstruct_wfc_round_trip(tmp_path):
    """Rebuilding C from H/S reproduces the eigenvalues listed in istate.info."""
    import numpy as np

    from aiida_orbgen.utils.report.assemble import (
        RY2EV,
        read_triu_abacus,
        reconstruct_wfc,
    )

    norb, nband = 12, 5
    ham = np.diag(np.linspace(-1.0, 1.0, norb))
    _write_triu(tmp_path / "data-0-H", ham)
    _write_triu(tmp_path / "data-0-S", np.eye(norb))

    energies = np.diag(ham)[:nband] * RY2EV
    occ = [1.0] * 3 + [0.0] * (nband - 3)
    (tmp_path / "istate.info").write_text(
        "\n".join(f"{i + 1} {e:.6f} {o:.6f}"
                  for i, (e, o) in enumerate(zip(energies, occ))) + "\n"
    )

    info = reconstruct_wfc(tmp_path)
    assert info["ok"] and info["bands"] == nband and info["occupied"] == 3
    assert info["dev_eV"] < 1e-6

    text = (tmp_path / "WFC_NAO_GAMMA1.txt").read_text().splitlines()
    assert text[0].startswith(str(nband))
    assert text[1].startswith(str(norb))
    assert read_triu_abacus(tmp_path / "data-0-S").shape == (norb, norb)


def test_reconstruct_wfc_refuses_a_bad_match(tmp_path):
    """A mismatching istate.info must not be turned into a bogus WFC."""
    import numpy as np
    import pytest

    from aiida_orbgen.utils.report.assemble import reconstruct_wfc

    _write_triu(tmp_path / "data-0-H", np.diag([-1.0, -0.5, 0.5, 1.0]))
    _write_triu(tmp_path / "data-0-S", np.eye(4))
    (tmp_path / "istate.info").write_text("1 -99.0 1.0\n2 -98.0 1.0\n")

    with pytest.raises(ValueError, match="deviate"):
        reconstruct_wfc(tmp_path)
    assert not (tmp_path / "WFC_NAO_GAMMA1.txt").exists()


def test_folder_is_complete_detects_a_finished_tree(tmp_path):
    from aiida_orbgen.utils.report.assemble import _folder_is_complete

    jobdir = tmp_path / "U-dimer-1.89-9au"
    out = jobdir / "OUT.ABACUS"
    out.mkdir(parents=True)
    for name in ("data-0-S", "data-0-T", "WFC_NAO_GAMMA1.txt"):
        (out / name).write_text("x")
    assert not _folder_is_complete(jobdir)          # running_scf.log missing
    (out / "running_scf.log").write_text("...\nFinish Time : now\n")
    assert _folder_is_complete(jobdir)


# ---------------------------------------------------------------------------
#  plan / CLI
# ---------------------------------------------------------------------------


def test_plan_runs_materialises_the_siab_json(tmp_path):
    path = _write_input(tmp_path, {"abacus": "test", "orbgen": "test"})
    bundle = ConfigLoader(path).load_all()
    plans = plan_runs(bundle, output_root=tmp_path / "run")

    assert len(plans) == 1
    plan = plans[0]
    assert plan.workflow == WORKFLOW_GRIDSEARCH
    assert plan.siab_json_path.is_file()
    assert plan.siab_json_path.parent.name == "configs"
    assert plan.candidates == [(4, 9.0), (4, 10.0)]

    written = json.loads(plan.siab_json_path.read_text())
    assert written["element"] == "U"
    assert written["pseudo_dir"] == bundle.pseudo_path


def test_plan_runs_calc_uses_the_grid_point_directory(tmp_path):
    path = _write_input(tmp_path, {"abacus": "test", "orbgen": "test"})
    bundle = ConfigLoader(path).load_all()
    plans = plan_runs(bundle, output_root=tmp_path / "run", workflow=WORKFLOW_CALC)

    plan = plans[0]
    assert plan.candidates == [(4, 9.0)]  # --only 0 → first candidate
    # the same spelling the workflow and the report use (point_dir_name), not the
    # pre-2026-09-29 ``lmax4_rcut9p0`` the plan builder used to write
    from aiida_orbgen.interfaces.nsw import point_dir_name

    assert plan.output_dir.name == point_dir_name(4, 9.0) == "lmax4_rcut9"


def test_cli_check_and_dry_run_write_no_output_json(tmp_path, capsys):
    path = _write_input(tmp_path, {"abacus": {"test": "test"}, "orbgen": {"test": "test"}})
    run_root = tmp_path / "run"

    assert main(["check", "-i", str(path), "--output-root", str(run_root)]) == 0
    out = capsys.readouterr().out
    assert "workflow=orbgen.gridsearch" in out
    assert "(l_max=4, r_cut=9)" in out
    assert not (tmp_path / "output.json").exists()

    assert main(["run", "-i", str(path), "--dry-run", "--output-root", str(run_root)]) == 0
    out = capsys.readouterr().out
    assert "--dry-run: nothing submitted" in out
    assert not (tmp_path / "output.json").exists()


def test_cli_check_rejects_a_broken_input(tmp_path, capsys):
    path = _write_input(tmp_path, {"abacus": "no-such-preset", "orbgen": "test"})
    assert main(["check", "-i", str(path)]) == 1
    assert "invalid input" in capsys.readouterr().err


def test_cli_run_reports_a_missing_input(tmp_path, capsys):
    assert main(["run", "-i", str(tmp_path / "nope.json"), "--dry-run"]) == 1
    assert "input.json not found" in capsys.readouterr().err


def test_report_validates_paths_before_touching_aiida(tmp_path, capsys):
    """`--siab-json` / `--dft-root` are checked before any profile is loaded."""
    output_json = tmp_path / "output.json"
    output_json.write_text(json.dumps({
        "workflow": WORKFLOW_GRIDSEARCH,
        "abacus": {"orbgen": {"test": "not-a-uuid"}},
    }))

    assert main([
        "report", "-i", str(output_json), "-o", str(tmp_path),
        "--siab-json", str(tmp_path / "missing.json"),
    ]) == 1
    assert "--siab-json not found" in capsys.readouterr().err

    assert main([
        "report", "-i", str(output_json), "-o", str(tmp_path),
        "--dft-root", str(tmp_path / "missing-dir"),
    ]) == 1
    assert "--dft-root not a directory" in capsys.readouterr().err


# ---------------------------------------------------------------------------
#  output.json
# ---------------------------------------------------------------------------


def test_output_json_round_trip(tmp_path):
    jobs = [
        SubmittedJob(backend="abacus", key="orbgen", preset_name="test",
                     uuid="11111111-2222-3333-4444-555555555555", pk=1,
                     workflow=WORKFLOW_GRIDSEARCH),
        SubmittedJob(backend="abacus", key="orbgen", preset_name="u_14ve",
                     uuid="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", pk=2,
                     workflow=WORKFLOW_GRIDSEARCH),
    ]
    out_path = tmp_path / "output.json"
    write_cal_json(jobs, output_path=out_path, workflow=WORKFLOW_GRIDSEARCH,
                   input_json=tmp_path / "input.json")

    data = json.loads(out_path.read_text())
    assert data["workflow"] == WORKFLOW_GRIDSEARCH
    assert data["abacus"]["orbgen"]["test"].startswith("11111111")
    assert collect_job_entries(data) == [
        ("abacus/orbgen/test", "11111111-2222-3333-4444-555555555555"),
        ("abacus/orbgen/u_14ve", "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"),
    ]


def test_output_json_default_path_and_legacy_pks(tmp_path):
    input_json = tmp_path / "input.json"
    input_json.write_text("{}")
    assert default_result_path(input_json) == tmp_path / "output.json"

    data = build_cal_json(
        [SubmittedJob(backend="abacus", key="orbgen", preset_name="test", pk=410274)],
        workflow=WORKFLOW_GRIDSEARCH,
    )
    assert data["abacus"]["orbgen"]["test"] == "410274"
    assert collect_job_entries(data) == [("abacus/orbgen/test", "410274")]


# ---------------------------------------------------------------------------
#  report rendering
# ---------------------------------------------------------------------------


def _summary() -> OrbgenRunSummary:
    points = [
        GridPoint(
            l_max=3, r_cut=9.0, pk=101, exit_status=304, finished_ok=True,
            process_state="finished", tolerance_meV=4.2,
            delta_max_per_atom_meV=10.0, delta_max_meV=20.0,
            delta_min_per_atom_meV=8.0,
            per_struct=[{"folder": "U-dimer-1.89-9au", "n_atoms": 2,
                         "E_pw": -100.0, "E_lcao_nsw": -99.99, "dE_per_atom": 0.005}],
            siab_info={"family_label": "siab-u-9au", "nsw_filename": "U_gga_9au_100Ry.orb"},
            output_dir="/tmp/run/lmax3_rcut9p0",
        ),
        GridPoint(
            l_max=4, r_cut=10.0, pk=102, exit_status=0, finished_ok=True,
            process_state="finished", tolerance_meV=4.2,
            delta_max_per_atom_meV=2.0, delta_max_meV=4.0,
            delta_min_per_atom_meV=1.0,
            per_struct=[],
            siab_info={"family_label": "siab-u-10au", "nsw_filename": "U_gga_10au_100Ry.orb"},
            output_dir="/tmp/run/lmax4_rcut10p0",
        ),
    ]
    return OrbgenRunSummary(
        node_pk=42, node_uuid="uuid-42", label="OrbgenGridSearchWorkChain",
        kind="gridsearch", status="Finished [0]", exit_status=0,
        process_state="finished", tolerance_meV=4.2, search_strategy="exhaustive",
        grid=points,
        best={"l_max": 4, "r_cut": 10.0, "delta_max_per_atom_meV": 2.0,
              "calc_pk": 102, "source": "grid_summary"},
    )


def test_render_report_contains_matrices_and_best():
    text = render_report(_summary(), include_process_logs=False)

    assert "# Orbgen Report" in text
    assert "## 1a. ΔE Max (per atom, meV) — l_max × r_cut" in text
    assert "| l_max \\ r_cut | 9.0 | 10.0 |" in text
    # 10.0 meV exceeds the 4.2 meV tolerance → bold, 2.0 does not
    assert "**10.0**" in text and "| 2.0 |" in text
    assert "**Best**: (l_max=4, r_cut=10)" in text
    assert "siab-u-10au" in text
    assert "_(no orbital file could be exported — see the log for details)_" in text


def test_render_report_shows_spillage_and_orbital_validation():
    """Quality numbers must reach the report, not just the result dict."""
    from aiida_orbgen.utils.report.orbgen import render_report
    from aiida_orbgen.utils.report.orbgen import OrbgenRunSummary

    summary = OrbgenRunSummary(
        node_pk=1, node_uuid="u", label="OrbgenGridSearchWorkChain",
        kind="gridsearch", status="Finished [304]", exit_status=304,
        process_state="finished", tolerance_meV=4.2,
        search_strategy="exhaustive", grid=[],
    )
    text = render_report(
        summary,
        final_orbitals=[{
            "l_max": 4, "r_cut": 10.0,
            "status": "ok",
            "spillage": [1.20492740e-03, 1.74717634e-03],
            "normalise": {"ok": True, "message": "SIAB would recompute nothing"},
            "validated": [{
                "file": "/tmp/U_gga_10au_150Ry_4s3p2d2f1g.orb", "ok": True,
                "per_l": [4, 3, 2, 2, 1], "expected": [4, 3, 2, 2, 1],
            }],
            "files": ["/tmp/U_gga_10au_150Ry_4s3p2d2f1g.orb"],
        }],
    )
    assert "Spillage (converged)" in text
    assert "1.204927e-03" in text and "1.747176e-03" in text
    assert "SIAB consistency: SIAB would recompute nothing" in text
    assert "4 3 2 2 1" in text and "✅" in text


def test_render_report_single_point_and_final_orbital():
    summary = _summary()
    summary.kind = "calc"
    summary.grid = summary.grid[1:]
    text = render_report(
        summary,
        final_orbitals=[
            {"status": "ok", "l_max": 4, "r_cut": 10.0, "dft_root": "/tmp/dft",
             "files": []},
            {"status": "failed", "l_max": 4, "r_cut": 9.0,
             "message": "reference DFT incomplete", "files": []},
        ],
        include_process_logs=False,
    )
    assert "## 1. Single (l_max, r_cut) point" in text
    assert "## 6. Final CSW-NAO orbitals" in text
    # one subsection per grid point, each with its own status
    assert "### l_max=4, r_cut=10 — ok" in text
    assert "### l_max=4, r_cut=9 — failed" in text
    assert "Note: reference DFT incomplete" in text


# ---------------------------------------------------------------------------
#  spec.py — the typed, validated view of a run
# ---------------------------------------------------------------------------
def test_spec_rejects_an_unreachable_nzeta_scheme():
    """The check that used to cost a full reference DFT before failing."""
    config = dict(_COMPLETE_SIAB)
    # r_cut=9, ecutjy=100, reduced gives 27 radial functions for l=0
    config["orbitals"] = [_orbital(nzeta=[28, 2, 2, 1, 0], geoms=[0])]
    with pytest.raises(ValueError, match="not reachable"):
        validate_siab_config(config)

    config["orbitals"] = [_orbital(nzeta=[27, 2, 2, 1, 0], geoms=[0])]
    assert validate_siab_config(config) is not None      # fits, only warnings


def test_spec_rejects_nzeta_beyond_lmaxmax():
    config = dict(_COMPLETE_SIAB)
    # trailing zeros do not raise the requested l_max: [3,2,2,1,0,0] is still 3
    config["orbitals"] = [_orbital(nzeta=[3, 2, 2, 1, 0, 0], geoms=[0])]
    validate_siab_config(config)
    # a non-zero term at l=5 does (geoms[0].lmaxmax is 4)
    config["orbitals"] = [_orbital(nzeta=[3, 2, 2, 1, 0, 1], geoms=[0])]
    with pytest.raises(ValueError, match="lmaxmax"):
        validate_siab_config(config)


def test_spec_rejects_a_geometry_index_that_does_not_exist():
    config = dict(_COMPLETE_SIAB)
    config["orbitals"] = [_orbital(nzeta=[3, 2, 2, 1, 0], geoms=[3])]
    with pytest.raises(ValueError, match="geoms\\[3\\]"):
        validate_siab_config(config)


def test_spec_rejects_unknown_orbital_keys():
    """Unknown keys would be ignored by SIAB; saying so beats dropping them."""
    config = dict(_COMPLETE_SIAB)
    config["orbitals"] = [_orbital(nzeta=[3, 2, 2, 1, 0], geoms=[0],
                           llocmin=4)]                 # typo
    with pytest.raises(ValueError, match="llocmin"):
        validate_siab_config(config)


def test_spec_accepts_every_shipped_preset():
    """Guard against the validator rejecting configurations SIAB accepts."""
    from aiida_orbgen.utils.yamlio import read_yaml
    from aiida_orbgen.utils.config import PARAMETERS_DIR

    seen = 0
    for table in sorted((PARAMETERS_DIR / "orbgen").glob("*.yml")):
        for name, preset in (read_yaml(table) or {}).items():
            validate_siab_config(preset, source=f"{table.name}#{name}")
            seen += 1
    assert seen >= 3


def test_spec_rejects_a_monomer_reference_geometry():
    """SIAB only accepts dimer/trimer/square/tetrahedron/octahedron/cube in `geoms`.

    The monomer of the `atomic` initial guess is a *job* SIAB appends itself, mirrored
    by `interfaces.pipeline.generate_all`; writing it into `geoms` makes SIAB abort
    ("proto should be a file or one of the following: ...") at the very end of a report
    run, after the whole grid has been paid for (2026-09-30).
    """
    config = dict(_COMPLETE_SIAB)
    config["geoms"] = [dict(config["geoms"][0]),
                       {**config["geoms"][0], "proto": "monomer", "pertmags": [0.0]}]
    with pytest.raises(ValueError, match="geoms\\[1\\].proto='monomer'"):
        validate_siab_config(config)


def test_siab_compulsory_orbital_keys_are_required_here_too():
    """SIAB rejects an orbital without `checkpoint`; catch it before the DFT runs.

    2026-09-30: a hand-written scan config omitted `checkpoint` (and `nbands`), passed
    `aiida-orbgen check`, ran a whole grid — and then `report` failed at the very last
    step with "orbital 0 does not have all the compulsory keys".
    """
    config = dict(_COMPLETE_SIAB)
    config["orbitals"] = [{"nzeta": [3, 2, 2, 1, 0], "geoms": [0]}]   # not SIAB-complete
    with pytest.raises(ValueError, match="missing .*checkpoint"):
        validate_siab_config(config)

    # the same orbital with SIAB's four keys is accepted
    config["orbitals"] = [_orbital()]
    validate_siab_config(config)


def test_presets_shared_by_name_agree():
    """A preset name must mean the same thing in every shipped file.

    ``test`` exists both in ``orbgen.yml`` (the file the flat spelling
    ``{"orbgen": "test"}`` reads) and in ``test.yml`` (``{"orbgen": {"test": "test"}}``).
    The two differed only in ``ecutjy`` (100 vs 150), i.e. the same preset name named
    two different orbitals, and only the 150 Ry one could reuse the reference DFT tree
    in ``project/u_14ve``. They agree now; this keeps a one-sided edit from
    re-opening the trap.
    """
    from aiida_orbgen.utils.config import PARAMETERS_DIR
    from aiida_orbgen.utils.yamlio import read_yaml

    seen: dict[str, tuple[str, dict]] = {}
    for table in sorted((PARAMETERS_DIR / "orbgen").glob("*.yml")):
        for name, preset in (read_yaml(table) or {}).items():
            if name in seen:
                other_table, other_preset = seen[name]
                assert preset == other_preset, (
                    f"preset {name!r} differs between {other_table} and {table.name}: "
                    f"{ {k: (other_preset.get(k), preset.get(k)) for k in set(other_preset) | set(preset) if other_preset.get(k) != preset.get(k)} }"
                )
            else:
                seen[name] = (table.name, preset)
    assert len(seen) >= 2                      # u_14ve + test (in two files)


def test_spec_reports_achievable_nzeta_and_grid():
    spec = OrbgenSpec.model_validate(_COMPLETE_SIAB)
    assert spec.lmaxmax == 4
    assert spec.grid_points() == [(3, 9.0), (3, 10.0)]
    # r_cut=9, ecutjy=100, reduced
    assert spec.achievable_nzeta[0] == 27
    assert spec.to_siab_config()["bessel_nao_rcut"] == [9.0, 10.0]


def test_abacus_spec_warns_when_lcao_outputs_are_switched_off():
    base = {"basis": ["pw", "lcao_nsw"], "tolerance_meV": 4.2}
    assert AbacusSpec(**base).warnings() == []

    spec = AbacusSpec(**base, parameters_input={"out_wfc_lcao": 0})
    warnings = spec.warnings()
    assert len(warnings) == 1 and "out_wfc_lcao" in warnings[0]

    # a pure-PW run does not need them
    assert AbacusSpec(basis=["pw"], parameters_input={"out_wfc_lcao": 0}).warnings() == []


# ---------------------------------------------------------------------------
#  the report reads the family the children actually used
# ---------------------------------------------------------------------------
class _FakeOutput:
    def __init__(self, value=None, d=None):
        self.value = value
        self._d = d

    def get_dict(self):
        if self._d is None:
            raise TypeError("not a Dict output")
        return dict(self._d)


class _FakeCalcNode:
    """Just enough of an ``OrbgenCalcWorkChain`` node for ``_collect_grid_point``."""

    def __init__(self, outputs, pk=1, exit_status=0, extras=None):
        self.outputs = outputs
        self.pk = pk
        self.exit_status = exit_status
        self.is_finished_ok = True
        self.process_state = None
        self.process_label = "OrbgenCalcWorkChain"
        self.called = []
        self.base = type("B", (), {"extras": type("E", (), {"all": extras or {}})()})()

    @property
    def inputs(self):
        raise AttributeError("this fake has no inputs")


def _node_with_family(label_in_info, family_output=None):
    outputs = {
        "siab_info": _FakeOutput(d={
            "family_label": label_in_info, "lmax": 4, "rcut": 10.0,
        }),
        "energies": _FakeOutput(d={"delta_E_max_per_atom_meV": 109.95}),
    }
    if family_output is not None:
        outputs["pseudo_family"] = _FakeOutput(value=family_output)
    return _FakeCalcNode(outputs)


def test_report_flags_a_non_variational_lcao_energy():
    """E_lcao < E_pw cannot happen for a real basis — the report must say so.

    2026-09-30: a regenerated UPF made ABACUS's LCAO path sit 69 eV below its PW
    path, i.e. the ΔE tables were measuring a broken comparison rather than an
    unconverged basis.
    """
    from aiida_orbgen.utils.report.orbgen import render_report

    point = GridPoint(
        l_max=4, r_cut=10.0, pk=461051, exit_status=304, finished_ok=False,
        process_state="finished", tolerance_meV=4.2,
        delta_max_per_atom_meV=34692.955,
        per_struct=[
            {"folder": "U-dimer-2.75-10au", "n_atoms": 2,
             "E_pw": -4950.975407, "E_lcao_nsw": -5020.361317,
             "dE": 69.385910, "dE_per_atom": 34.692955},
        ],
    )
    summary = OrbgenRunSummary(
        node_pk=461057, node_uuid="303b7b46-78ab-474d-818e-ccd95d44583d",
        label="OrbgenGridSearchWorkChain", kind="gridsearch",
        status="Finished [404]", exit_status=404, process_state="finished",
        tolerance_meV=4.2, grid=[point],
    )
    text = render_report(summary, include_process_logs=False)
    assert "E_lcao_nsw < E_pw" in text
    assert "cannot be below the plane-wave reference" in text
    assert "number_of_proj" in text


def test_report_does_not_flag_a_normal_basis_error():
    """The ordinary case — LCAO slightly above PW — must stay quiet."""
    from aiida_orbgen.utils.report.orbgen import render_report

    point = GridPoint(
        l_max=4, r_cut=10.0, pk=410517, exit_status=304, finished_ok=True,
        process_state="finished", tolerance_meV=4.2, delta_max_per_atom_meV=86.418,
        per_struct=[
            {"folder": "U-dimer-2.75-10au", "n_atoms": 2,
             "E_pw": -3462.915097527, "E_lcao_nsw": -3462.742261675,
             "dE": 0.172836, "dE_per_atom": 0.086418},
        ],
    )
    summary = OrbgenRunSummary(
        node_pk=410274, node_uuid="0798b3e9-668c-485e-9479-fb8fcbf42c35",
        label="OrbgenGridSearchWorkChain", kind="gridsearch",
        status="Finished [404]", exit_status=404, process_state="finished",
        tolerance_meV=4.2, grid=[point],
    )
    text = render_report(summary, include_process_logs=False)
    assert "E_lcao_nsw < E_pw" not in text


def test_a_reference_tree_older_than_the_run_is_reported(tmp_path):
    """`static.dft_root` pointing at an old tree makes SIAB fit the wrong data.

    2026-09-30: `static.dft_root` was `project/u_14ve/`, where the trees of the
    *previous* attempt lived (broken pseudopotential, `E_lcao` 69 eV off); the report
    fitted those silently and SIAB additionally ran the missing monomer DFT itself.
    """
    import os
    from datetime import datetime, timedelta

    from aiida_orbgen.utils.report.orbitals import _reference_trees_older_than

    old_out = tmp_path / "U-dimer-2.75-9au" / "OUT.ABACUS"
    old_out.mkdir(parents=True)
    log = old_out / "running_scf.log"
    log.write_text("...")
    old_time = datetime(2026, 9, 30, 0, 17).timestamp()
    os.utime(log, (old_time, old_time))

    class Node:
        ctime = datetime(2026, 9, 30, 8, 5)

    message = _reference_trees_older_than(tmp_path, Node())
    assert message and "2026-09-30 00:17" in message and "2026-09-30 08:05" in message
    assert "cannot belong to it" in message

    # a tree written after the run started is fine
    fresh = datetime(2026, 9, 30, 8, 30).timestamp()
    os.utime(log, (fresh, fresh))
    assert _reference_trees_older_than(tmp_path, Node()) is None

    # and an empty root is not a finding either
    assert _reference_trees_older_than(tmp_path / "nothing", Node()) is None


def test_report_prefers_the_family_the_children_used():
    """siab_info carries the name SIAB implies; the output carries the real one."""
    from aiida_orbgen.utils.report.orbgen import _collect_grid_point

    implied = "siab-u-nr-pbe-z14-nsw-10au-150Ry-g"
    effective = implied + "-e6fb7f"
    point = _collect_grid_point(_node_with_family(implied, effective))

    assert point.siab_info["family_label"] == effective
    # other siab_info keys survive the override
    assert point.siab_info["lmax"] == 4
    assert point.l_max == 4 and point.r_cut == 10.0


def test_report_keeps_siab_info_when_there_is_no_family_output():
    """Nodes from before the output existed must read exactly as they did."""
    from aiida_orbgen.utils.report.orbgen import _collect_grid_point

    implied = "siab-u-nr-pbe-z14-nsw-10au-150Ry-g"
    point = _collect_grid_point(_node_with_family(implied))
    assert point.siab_info["family_label"] == implied


# ---------------------------------------------------------------------------
#  aiida-orbgen select — recording the chosen grid point
# ---------------------------------------------------------------------------
def _summary_with_points():
    def point(l_max, r_cut, pk, delta, tolerance=4.2):
        return GridPoint(
            l_max=l_max, r_cut=r_cut, pk=pk, exit_status=304, finished_ok=True,
            process_state="finished", tolerance_meV=tolerance,
            delta_max_per_atom_meV=delta, delta_max_meV=delta * 2,
        )

    return OrbgenRunSummary(
        node_pk=410274, node_uuid="0798b3e9-668c-485e-9479-fb8fcbf42c35",
        label="OrbgenGridSearchWorkChain", kind="gridsearch",
        status="Finished [404]", exit_status=404, process_state="finished",
        tolerance_meV=4.2,
        grid=[point(3, 9.0, 410363, 205.5), point(4, 9.0, 410492, 189.3),
              point(4, 10.0, 410517, 109.95)],
    )


def test_select_falls_back_to_the_workflow_choice():
    """No point meets the tolerance, so `select` repeats the workflow's own pick."""
    summary = _summary_with_points()
    chosen = choose_point(summary)
    assert chosen.pk == 410517                  # lowest Delta*E, not the first
    assert chosen.delta_max_per_atom_meV == 109.95

    # ... and says so, instead of pretending the choice is acceptable
    payload = selection_payload(summary, chosen)
    assert [p["acceptable"] for p in payload["selection"]["grid_points"]] == [
        False, False, False
    ]

    # When the workflow did find an acceptable point it records it in
    # ``summary.best`` (batch.py only sets it after a tolerance pass), and that
    # verdict outranks the raw Delta*E ranking.
    summary.best = {"calc_pk": 410363}
    assert choose_point(summary).pk == 410363

    # The bundle keeps every point's numbers, so the choice can be audited even
    # when it came from an explicit request rather than from the tolerance.
    payload = selection_payload(summary, choose_point(summary))
    chosen_row = payload["selection"]["grid_points"][0]
    assert (chosen_row["calc_pk"], chosen_row["delta_per_atom_meV"]) == (410363, 205.5)


def test_select_accepts_an_explicit_point():
    summary = _summary_with_points()
    assert choose_point(summary, l_max=4, r_cut=10.0).pk == 410517
    assert choose_point(summary, calc_pk=410492).pk == 410492
    assert choose_point(summary, l_max=5, r_cut=10.0) is None
    with pytest.raises(ValueError, match="together"):
        choose_point(summary, l_max=4)


def test_select_writes_a_reusable_bundle(tmp_path):
    summary = _summary_with_points()
    point = choose_point(summary, l_max=4, r_cut=10.0)
    payload = selection_payload(
        summary, point, chosen_by="--l-max 4 --r-cut 10",
        siab_config={"element": "U", "ecutjy": 150, "bessel_nao_rcut": [10]},
        base_input={"parameters": {"orbgen": {"u_14ve": "u_14ve"}},
                    "profile": "aiida_profile"},
    )
    written = write_selection(
        tmp_path, payload,
        siab_config=payload["siab_config"], point_name="lmax4_rcut10",
    )

    selected = json.loads(Path(written["selection"]).read_text())
    assert selected["selection"]["calc_pk"] == 410517
    assert selected["selection"]["delta_per_atom_meV"] == 109.95
    assert selected["selection"]["chosen_by"] == "--l-max 4 --r-cut 10"
    # every grid point is recorded, so the choice can be audited later
    assert [p["calc_pk"] for p in selected["selection"]["grid_points"]] == [
        410363, 410492, 410517
    ]
    assert selected["selection"]["grid_points"][0]["acceptable"] is False

    assert json.loads(Path(written["siab_config"]).read_text())["ecutjy"] == 150

    reuse = json.loads(Path(written["input_json"]).read_text())
    # The point's own config goes inline: ConfigLoader reads static.siab_config
    # instead of parameters.orbgen, so exactly one candidate exists and no preset
    # has to be edited.  `workflow` is left for the loader to decide (one candidate
    # -> orbgen.calc), which is what makes this file reusable as-is.
    assert reuse["static"]["siab_config"]["ecutjy"] == 150
    assert reuse["static"]["siab_config_name"] == "selected"
    assert "workflow" not in reuse
    assert reuse["static"]["selected_point"]["from_node"] == 410517
    assert reuse["parameters"] == {"orbgen": {"u_14ve": "u_14ve"}}   # untouched
