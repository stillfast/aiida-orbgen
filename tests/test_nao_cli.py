"""``aiida-orbgen nao``: the CLI that contracts one NSW reference into NAO(s).

The reference run is the expensive half, and its own config already describes everything
the fit needs, so what has to be pinned down is: what config is used when the caller
names none, that a config from a *different* point is refused instead of quietly
triggering a fresh DFT, and that the summary says which scheme produced which file —
including the failure everyone hits once: a ``g`` channel requested without
``model_kwargs.vloc_aux`` comes out empty.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aiida_orbgen.utils.report.nao import (
    config_problems,
    default_config,
    describe_entries,
    render_summary,
    summarise_outcome,
)
from aiida_orbgen.utils.report.validate import nzeta_string

SIAB_CONFIG = {
    "element": "U",
    "pseudo_dir": "/tmp/U.pbe-n-nc.UPF",
    "fit_basis": "jy",
    "ecutwfc": 180,
    "ecutjy": 150,
    "bessel_nao_rcut": [11.0],
    "primitive_type": "reduced",
    "geoms": [{"proto": "dimer", "pertkind": "stretch", "pertmags": [2.1, 2.4, 2.8],
               "nbands": 40, "nspin": 1, "lmaxmax": 4, "celldm": 35}],
    "orbitals": [{"nzeta": [4, 3, 2, 2], "geoms": [0], "nbands": "occ",
                  "checkpoint": None}],
}

#: the two schemes the summary tests fit: ``[(name, nzeta), ...]``
SCHEMES = [("3s2p2d1f", [3, 2, 2, 1]), ("4s3p2d2f", [4, 3, 2, 2])]


def _scheme_config() -> dict:
    """A two-entry config: independent fits (``checkpoint: null``), as the docs advise."""
    return {"orbitals": [
        {"nzeta": nzeta, "geoms": [0], "nbands": "occ", "checkpoint": None}
        for _name, nzeta in SCHEMES
    ]}


def _orbital_name(scheme: str) -> str:
    """``U_gga_11au_150Ry_3s2p2d1f.orb`` — the name SIAB gives a scheme."""
    return f"U_gga_11au_150Ry_{scheme}.orb"


def _final_orbitals(directory: str, **overrides) -> dict:
    """A final-orbital outcome as ``report`` returns it for :data:`SCHEMES`."""
    files = [f"{directory}/{_orbital_name(name)}" for name, _ in SCHEMES]
    outcome = {
        "node_pk": 474021,
        "status": "ok",
        "dft_root": "/scratch/prod_r11_l4_j150/lmax4_rcut11",
        "files": files,
        "spillage": [9.72012586e-04, 4.70193629e-04],
        "validated": [{"file": files[index], "ok": True, "per_l": nzeta}
                      for index, (_name, nzeta) in enumerate(SCHEMES)],
    }
    outcome.update(overrides)
    return outcome


def _touch(paths) -> None:
    for path in paths:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"orb" * 10)


# ---------------------------------------------------------------------------
#  the config
# ---------------------------------------------------------------------------
def test_the_default_config_is_the_runs_own_with_the_grid_point_applied():
    """No `--config` means "fit what this run was built for", written out to be edited."""
    config = default_config(SIAB_CONFIG, l_max=3, r_cut=10.0)
    assert config["bessel_nao_rcut"] == [10.0]        # the grid point, not the preset's 11
    assert config["geoms"][0]["lmaxmax"] == 3
    assert config["orbitals"] == SIAB_CONFIG["orbitals"]
    # the input mapping is not modified
    assert SIAB_CONFIG["bessel_nao_rcut"] == [11.0]
    assert SIAB_CONFIG["geoms"][0]["lmaxmax"] == 4


def test_a_config_from_another_point_is_refused():
    """Otherwise SIAB looks for a different primitive orbital and recomputes the DFT."""
    assert config_problems(SIAB_CONFIG, l_max=4, r_cut=11.0) == []

    problems = config_problems(SIAB_CONFIG, l_max=4, r_cut=12.0)
    assert problems and "r_cut=12" in problems[0]

    problems = config_problems(SIAB_CONFIG, l_max=5, r_cut=11.0)
    assert problems and "lmaxmax" in problems[0]

    problems = config_problems({**SIAB_CONFIG, "orbitals": []}, l_max=4, r_cut=11.0)
    assert problems and "orbitals" in problems[0]

    problems = config_problems({"orbitals": [{"nzeta": [1]}]}, l_max=4, r_cut=11.0)
    assert any("bessel_nao_rcut" in problem for problem in problems)


def test_the_entries_are_described_including_the_g_trap():
    config = {"orbitals": [
        {"nzeta": [3, 2, 2, 1], "nbands": "occ", "checkpoint": None},
        {"nzeta": [4, 3, 3, 2, 1], "nbands": "occ*2", "checkpoint": 0,
         "model_kwargs": {"lloc_min": 4, "vloc_aux": "/tmp/U.upf"}},
        {"nzeta": [4, 3, 3, 2, 1], "nbands": "occ", "checkpoint": None},
    ]}
    lines = describe_entries(config)
    assert len(lines) == 3
    assert "3s2p2d1f" in lines[0] and "checkpoint=None" in lines[0]
    assert "checkpoint=0" in lines[1] and "vloc_aux" in lines[1]
    # the third asks for a g channel with nothing to fill it from
    assert "⚠" in lines[2] and "vloc_aux" in lines[2]


# ---------------------------------------------------------------------------
#  the summary
# ---------------------------------------------------------------------------
def test_the_summary_pairs_every_entry_with_its_file_and_spillage(tmp_path):
    """One row per `orbitals` entry, in the order of the config."""
    final = _final_orbitals(directory=str(tmp_path))
    _touch(final["files"])
    outcome = summarise_outcome(_scheme_config(), final, tmp_path)

    assert [entry["scheme"] for entry in outcome.entries] == [name for name, _ in SCHEMES]
    assert [entry["spillage"] for entry in outcome.entries] == [
        9.72012586e-04, 4.70193629e-04
    ]
    assert all(entry["ok"] for entry in outcome.entries)
    assert outcome.entries[0]["per_l"] == [3, 2, 2, 1]
    assert outcome.entries[0]["bytes"] == 30
    assert outcome.ok

    table = render_summary([outcome])
    assert "| scheme |" in table
    assert "9.7201e-04" in table and "4.7019e-04" in table
    assert "3 2 2 1" in table
    assert _orbital_name("3s2p2d1f") in table


def test_the_summary_reports_nothing_produced(tmp_path):
    """A fit that crashed (e.g. a shrinking checkpoint chain) has to be visible."""
    final = _final_orbitals(
        directory=str(tmp_path), status="failed", files=[],
        spillage=[9.72012586e-04], validated=[],
        message="orbgen exited 1 (axes don't match array)",
    )
    outcome = summarise_outcome(_scheme_config(), final, tmp_path)
    assert not outcome.ok
    assert all(entry["reason"] == "no orbital file was produced"
               for entry in outcome.entries)
    assert any("axes don't match" in warning for warning in outcome.warnings)
    assert "❌" in render_summary([outcome])


def test_the_summary_reports_an_empty_g_channel(tmp_path):
    """The g channel needs `model_kwargs.vloc_aux`, or the file fails validation.

    This is the shape a real run has: the second entry produced a file, and that file
    failed `validate_orbital` because `per l` came back ``[4, 3, 2, 2, 0]``.
    """
    reason = ("radial functions per l are [4, 3, 2, 2, 0] but the requested scheme "
              "4s3p2d2f needs [4, 3, 2, 2]")
    final = _final_orbitals(directory=str(tmp_path), validation_failed=reason)
    final["validated"][1].update({"ok": False, "per_l": [4, 3, 2, 2, 0],
                                  "reason": reason})
    _touch(final["files"])
    outcome = summarise_outcome(_scheme_config(), final, tmp_path)
    assert outcome.entries[0]["ok"] is True
    assert outcome.entries[1]["ok"] is False
    assert outcome.entries[1]["per_l"] == [4, 3, 2, 2, 0]
    assert "[4, 3, 2, 2, 0]" in outcome.entries[1]["reason"]
    assert "❌" in render_summary([outcome])


# ---------------------------------------------------------------------------
#  the CLI
# ---------------------------------------------------------------------------
class _FakeNode:
    """Just enough of an ``OrbgenCalcWorkChain`` for the command's bookkeeping."""

    pk = 474021
    uuid = "uuid"
    label = ""
    process_label = "OrbgenCalcWorkChain"
    process_state = type("S", (), {"value": "finished"})()
    exit_status = 0

    def __init__(self, siab_config: dict, *, l_max: int = 4, r_cut: float = 11.0,
                 config_path: str = "/scratch/tree/siab_config.json"):
        self._siab = siab_config
        self.inputs = type("I", (), {"siab_json": type("J", (), {
            "get_content": staticmethod(lambda: json.dumps(siab_config))})()})()
        self.outputs = type("O", (), {"siab_info": type("K", (), {"get_dict": staticmethod(
            lambda: {"lmax": l_max, "rcut": r_cut,
                     "nsw_filename": "U_gga_11au_150Ry_41s.orb",
                     "config_path": config_path})})()})()


def _install(monkeypatch, node):
    """Patch the two AiiDA entry points ``cmd_nao`` uses (it imports them locally)."""
    from aiida_orbgen.cli import run as cli_run

    monkeypatch.setattr("aiida.orm.load_node", lambda identifier: node)
    monkeypatch.setattr("aiida.load_profile", lambda *a, **k: None)
    return cli_run


def _write_output(tmp_path) -> Path:
    output = tmp_path / "output.json"
    output.write_text(json.dumps({"workflow": "orbgen.calc",
                                  "abacus": {"orbgen": {"prod_r11_l4_j150": "uuid"}}}))
    return output


def test_the_cli_writes_the_config_and_starts_one_fit(tmp_path, monkeypatch, capsys):
    """`aiida-orbgen nao -i output.json` must work with no config at all."""
    calls: dict = {}

    def fake_generate(identifier, out_dir, *, siab_config=None, **kwargs):
        calls["siab_config"] = siab_config
        calls["kwargs"] = kwargs
        config = json.loads(Path(siab_config).read_text())
        scheme = nzeta_string(config["orbitals"][0]["nzeta"])
        orb = Path(out_dir) / "lmax4_rcut11" / _orbital_name(scheme)
        orb.parent.mkdir(parents=True, exist_ok=True)
        orb.write_bytes(b"orb")
        final = _final_orbitals(directory=str(orb.parent), files=[str(orb)],
                                spillage=[4.70193629e-04],
                                validated=[{"file": str(orb), "ok": True,
                                            "per_l": config["orbitals"][0]["nzeta"]}])
        final["node_pk"] = 474021
        return type("R", (), {"final_orbitals": [final], "warnings": [],
                              "node_pk": 474021})(), "ok"

    cli_run = _install(monkeypatch, _FakeNode(SIAB_CONFIG))
    monkeypatch.setattr(cli_run, "generate_one_report", fake_generate)

    out_dir = tmp_path / "out"
    assert cli_run.main(["nao", "-i", str(_write_output(tmp_path)),
                         "-o", str(out_dir)]) == 0
    printed = capsys.readouterr().out
    assert "wrote the run's own config" in printed
    assert "4s3p2d2f" in printed and "4.7019e-04" in printed
    # the config really was written, and handed to the fit
    written = json.loads((out_dir / "orbgen.json").read_text())
    assert written["orbitals"][0]["nzeta"] == [4, 3, 2, 2]
    assert Path(calls["siab_config"]) == out_dir / "orbgen.json"
    assert calls["kwargs"]["run_final_orbital"] is True
    assert (out_dir / "nao.md").is_file()


def test_the_cli_fits_the_config_it_was_given(tmp_path, monkeypatch, capsys):
    """`--config` decides the schemes; several entries come out of one fit."""
    seen: dict = {}

    def fake_generate(identifier, out_dir, *, siab_config=None, **kwargs):
        config = json.loads(Path(siab_config).read_text())
        seen["n_entries"] = len(config["orbitals"])
        files, validated = [], []
        for entry in config["orbitals"]:
            orb = Path(out_dir) / "lmax4_rcut11" / _orbital_name(
                nzeta_string(entry["nzeta"]))
            orb.parent.mkdir(parents=True, exist_ok=True)
            orb.write_bytes(b"orb")
            files.append(str(orb))
            validated.append({"file": str(orb), "ok": True, "per_l": entry["nzeta"]})
        final = {"node_pk": 474021, "status": "ok", "dft_root": "/scratch/tree",
                 "files": files, "spillage": [9.7e-04, 4.7e-04], "validated": validated}
        return type("R", (), {"final_orbitals": [final], "warnings": [],
                              "node_pk": 474021})(), "ok"

    cli_run = _install(monkeypatch, _FakeNode(SIAB_CONFIG))
    monkeypatch.setattr(cli_run, "generate_one_report", fake_generate)
    config_path = tmp_path / "my_nao.json"
    config_path.write_text(
        json.dumps({**SIAB_CONFIG, "orbitals": _scheme_config()["orbitals"]})
    )

    out_dir = tmp_path / "out"
    assert cli_run.main(["nao", "-i", str(_write_output(tmp_path)),
                         "-c", str(config_path), "-o", str(out_dir)]) == 0
    assert seen["n_entries"] == 2
    printed = capsys.readouterr().out
    assert "3s2p2d1f" in printed and "4s3p2d2f" in printed
    assert not (out_dir / "orbgen.json").exists()      # its own config, not a copy


def test_the_cli_refuses_a_config_from_another_point(tmp_path, monkeypatch, capsys):
    """A mismatched config would silently become a fresh DFT; say so instead."""
    cli_run = _install(monkeypatch, _FakeNode(SIAB_CONFIG))
    mismatched = {**SIAB_CONFIG, "bessel_nao_rcut": [12.0]}
    config_path = tmp_path / "other.json"
    config_path.write_text(json.dumps(mismatched))
    assert cli_run.main(["nao", "-i", str(_write_output(tmp_path)),
                         "-c", str(config_path), "-o", str(tmp_path / "out")]) == 1
    assert "does not cover r_cut=11" in capsys.readouterr().err


def test_the_cli_skips_a_node_that_is_not_an_nsw_run(tmp_path, monkeypatch, capsys):
    node = _FakeNode(SIAB_CONFIG)
    node.process_label = "OrbgenBasisScanWorkChain"
    cli_run = _install(monkeypatch, node)
    assert cli_run.main(["nao", "-i", str(_write_output(tmp_path)),
                         "-o", str(tmp_path / "out")]) == 1
    assert "is not an NSW reference" in capsys.readouterr().err


def test_the_cli_dry_run_starts_no_fit(tmp_path, monkeypatch, capsys):
    cli_run = _install(monkeypatch, _FakeNode(SIAB_CONFIG))

    def explode(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("a dry run must not fit anything")

    monkeypatch.setattr(cli_run, "generate_one_report", explode)
    out_dir = tmp_path / "out"
    assert cli_run.main(["nao", "-i", str(_write_output(tmp_path)),
                         "-o", str(out_dir), "--dry-run"]) == 0
    assert "no fit started" in capsys.readouterr().out
    assert (out_dir / "orbgen.json").is_file()          # the config is still written


def test_the_cli_needs_an_input(tmp_path, monkeypatch):
    """`-i` is mandatory: argparse refuses, and a missing file is a clean error."""
    cli_run = _install(monkeypatch, _FakeNode(SIAB_CONFIG))
    with pytest.raises(SystemExit) as excinfo:
        cli_run.main(["nao", "-o", str(tmp_path / "out")])
    assert excinfo.value.code == 2
    assert cli_run.main(["nao", "-i", str(tmp_path / "missing.json"),
                         "-o", str(tmp_path / "out")]) == 1
