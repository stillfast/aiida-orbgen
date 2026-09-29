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
from aiida_orbgen.utils.config import (
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
    assert config["tolerance_meV"] == 4.2
    # scheduler options come from parameters/metadata.yml
    assert config["abacus"]["metadata"]["options"]["queue_name"] == "q_ysuan"

    # static.pseudo_path is injected as SIAB's pseudo_dir
    assert bundle.orbgen_presets[0].config["pseudo_dir"] == bundle.pseudo_path


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

_COMPLETE_SIAB = {
    "element": "U",
    "ecutwfc": 150,
    "ecutjy": 100,
    "bessel_nao_rcut": [9, 10],
    "geoms": [{"proto": "dimer", "pertkind": "stretch", "pertmags": "auto",
               "lmaxmax": 4}],
    "orbitals": [{"nzeta": [3, 2, 2, 1, 0], "geoms": [0]}],
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
    config["orbitals"] = [{"nzeta": [4, 3, 2, 2, 1], "lloc_min": 4,
                           "vloc_aux": "/tmp/U.UPF"}]
    with pytest.raises(ValueError, match="model_kwargs"):
        validate_siab_config(config)

    config["orbitals"] = [{"nzeta": [4, 3, 2, 2, 1],
                           "model_kwargs": {"lloc_min": 4,
                                            "vloc_aux": "/tmp/U.UPF"}}]
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
    assert "already up to date" in again["message"]

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


def test_primitive_export_falls_back_to_the_reference_tree(tmp_path):
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
    assert plan.output_dir.name == "lmax4_rcut9p0"


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
