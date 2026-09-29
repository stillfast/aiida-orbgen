"""Reading children into energies: ``workflows/extract.py`` + the STRU guard.

These two pieces used to be the inside of ``OrbgenCalcWorkChain.extract_energies_step``,
where the only way to exercise them was to run a WorkChain (and, before that, a
real DFT).  They are pure node introspection, so fake nodes are enough — and the
cases that matter most are the ones a real run rarely produces: a child with no
``misc`` output, a soft-success (exit 304) child, and a child whose STRU points at
the wrong orbital file, which must be skipped instead of averaged in.
"""

from __future__ import annotations

import pytest

from aiida_orbgen.workflows.energies import describe_deltas
from aiida_orbgen.workflows.extract import collect_child_energies
from aiida_orbgen.workflows.siab import (
    expected_orbital_filename,
    find_abacus_child,
    verify_stru_orbital,
)


# ---------------------------------------------------------------------------
#  fakes: just enough AiiDA shape for the introspection code
# ---------------------------------------------------------------------------
class FakeRepository:
    def __init__(self, content):
        self.content = content

    def get_object_content(self, name):
        if name != "STRU" or self.content is None:
            raise FileNotFoundError(f"no {name} in the repository")
        return self.content


class FakeBase:
    def __init__(self, content=None, attributes=None):
        self.repository = FakeRepository(content)
        self.attributes = attributes or {}


class FakeOutputs:
    def __init__(self, misc=None):
        if misc is not None:
            self.misc = misc


class FakeMisc:
    def __init__(self, d):
        self._d = d

    def get_dict(self):
        return dict(self._d)


class FakeCalc:
    """Stands in for an ``AbacusCalculation`` (or any node)."""

    def __init__(self, pk=1, process_type="aiida.calculations:abacus.abacus",
                 stru=None, filename_second=None, total_energy=None,
                 has_misc=True, exit_status=0, is_finished_ok=True, called=()):
        self.pk = pk
        self.process_type = process_type
        self.base = FakeBase(stru, {"filename_second": filename_second})
        self.outputs = FakeOutputs(
            FakeMisc({"total_energy": total_energy}) if has_misc else None
        )
        self.exit_status = exit_status
        self.is_finished_ok = is_finished_ok
        self.called = called


class FakeInputs:
    def __init__(self, pseudos):
        self.pseudos = pseudos


class FakePseudo:
    def __init__(self, filename_second):
        self.base = FakeBase(attributes={"filename_second": filename_second})


STRU_WITH_ORBITAL = """\
ATOMIC_SPECIES
U 238.0289 U.upf

NUMERICAL_ORBITAL
{orbital}

LATTICE_CONSTANT
1.8897261254578281

LATTICE_VECTORS
10.0 0.0 0.0
0.0 10.0 0.0
0.0 0.0 10.0

ATOMIC_POSITIONS
Cartesian_angstrom_center_xyz
U
0.0
1
0.0 0.0 0.0 1 1 1
"""

STRU_PW_ONLY = """\
ATOMIC_SPECIES
U 238.0289 U.upf

LATTICE_CONSTANT
1.8897261254578281

ATOMIC_POSITIONS
Cartesian_angstrom_center_xyz
U
0.0
1
0.0 0.0 0.0 1 1 1
"""

GOOD_ORBITAL = "U_gga_10au_150Ry_4s3p2d2f1g.orb"
BAD_ORBITAL = "U_gga_10au_150Ry_2f.orb"


def _abacus_calc(pk=10, pseudo_orbital=GOOD_ORBITAL, stru_orbital=None):
    """The ``AbacusCalculation`` itself: it holds the STRU and the pseudo inputs.

    ``pseudo_orbital`` is the file name AiiDA injected (``filename_second``, i.e.
    the truth); ``stru_orbital`` is what the STRU generator wrote into
    ``NUMERICAL_ORBITAL``.  They differ exactly in the bug the guard catches.
    """
    if stru_orbital is None:
        stru_orbital = pseudo_orbital
    calc = FakeCalc(pk=pk, stru=STRU_WITH_ORBITAL.format(orbital=stru_orbital))
    calc.inputs = FakeInputs({"U": FakePseudo(pseudo_orbital)})
    return calc


def _pw_child(pk=20, energy=-1000.0, **kwargs):
    """A pw child workchain: an ``AbacusBaseWorkChain`` with an ABACUS child below."""
    return FakeCalc(pk=pk, total_energy=energy, called=(_abacus_calc(pk=pk * 10),),
                    **kwargs)


def _lcao_child(pk=21, energy=-999.0, stru_orbital=None, **kwargs):
    """An LCAO child workchain: the STRU-orbital guard looks at what it called."""
    return FakeCalc(pk=pk, total_energy=energy,
                    called=(_abacus_calc(pk=pk * 10, stru_orbital=stru_orbital),),
                    **kwargs)


def _child(calc, task="dimer", basis="lcao_nsw", n_atoms=2):
    """One ``ctx.children_info`` entry."""
    return {"node": calc, "task": task, "basis": basis, "n_atoms": n_atoms}


# ---------------------------------------------------------------------------
#  find_abacus_child / expected_orbital_filename
# ---------------------------------------------------------------------------
def test_find_abacus_child_returns_the_abacus_calculation():
    wanted = FakeCalc(pk=7)
    parent = FakeCalc(pk=3, called=(FakeCalc(pk=4, process_type="aiida.workflows:x"), wanted))
    assert find_abacus_child(parent) is wanted


def test_find_abacus_child_is_none_without_children():
    assert find_abacus_child(FakeCalc(pk=3)) is None
    assert find_abacus_child(FakeCalc(pk=3, called=())) is None


def test_expected_orbital_filename_accepts_any_species():
    """The lookup used to be hard-coded to uranium (``pseudos['U']``)."""
    calc = FakeCalc(pk=5)
    calc.inputs = FakeInputs({
        "Si": FakePseudo(None),
        "Fe": FakePseudo("Fe_gga_8au_100Ry_4s2p2d1f.orb"),
    })
    assert expected_orbital_filename(calc) == "Fe_gga_8au_100Ry_4s2p2d1f.orb"


def test_expected_orbital_filename_is_none_when_unset():
    calc = FakeCalc(pk=5)
    calc.inputs = FakeInputs({"U": FakePseudo(None)})
    assert expected_orbital_filename(calc) is None
    assert expected_orbital_filename(FakeCalc(pk=6)) is None


# ---------------------------------------------------------------------------
#  verify_stru_orbital
# ---------------------------------------------------------------------------
def test_verify_stru_orbital_accepts_a_matching_strU():
    reported = []
    assert verify_stru_orbital(_abacus_calc(), GOOD_ORBITAL, report=reported.append)
    assert reported == []


def test_verify_stru_orbital_rejects_the_f_only_stub():
    """The 2026-09-19 shape: STRU points at a stub, ABACUS returns a fake energy."""
    reported = []
    assert not verify_stru_orbital(
        _abacus_calc(stru_orbital=BAD_ORBITAL), GOOD_ORBITAL,
        report=reported.append
    )
    assert len(reported) == 1
    assert BAD_ORBITAL in reported[0] and GOOD_ORBITAL in reported[0]
    assert "STRU orbital mismatch" in reported[0]


def test_verify_stru_orbital_decodes_bytes_and_reads_only_the_orbital_line():
    calc = FakeCalc(pk=11, stru=STRU_WITH_ORBITAL.format(orbital=GOOD_ORBITAL).encode())
    assert verify_stru_orbital(calc, GOOD_ORBITAL)


def test_verify_stru_orbital_skips_pw_strus_and_non_abacus_nodes():
    # no NUMERICAL_ORBITAL block (a pw child)
    assert verify_stru_orbital(FakeCalc(pk=12, stru=STRU_PW_ONLY), GOOD_ORBITAL)
    # not an ABACUS node at all
    assert verify_stru_orbital(FakeCalc(pk=13, process_type="aiida.calculations:some.other"),
                               GOOD_ORBITAL)
    # no process_type attribute
    class Bare:
        base = FakeBase(STRU_WITH_ORBITAL.format(orbital=BAD_ORBITAL))
        pk = 14

    assert verify_stru_orbital(Bare(), GOOD_ORBITAL)


def test_verify_stru_orbital_tolerates_an_unreadable_strU():
    reported = []
    calc = FakeCalc(pk=15, stru=None)
    assert verify_stru_orbital(calc, GOOD_ORBITAL, report=reported.append)
    assert "cannot read STRU" in reported[0]


# ---------------------------------------------------------------------------
#  collect_child_energies
# ---------------------------------------------------------------------------
def _pair_of_children(pw_energy=-1000.0, lcao_energy=-999.0, n_atoms=2):
    pw = _child(_pw_child(pk=20, energy=pw_energy), task="dimer", basis="pw",
                n_atoms=n_atoms)
    lcao = _child(_lcao_child(pk=21, energy=lcao_energy), task="dimer",
                  basis="lcao_nsw", n_atoms=n_atoms)
    return [pw, lcao]


def test_collect_child_energies_pairs_pw_and_lcao():
    collected = collect_child_energies(_pair_of_children())

    assert collected.usable
    assert collected.n_pw == 1
    assert collected.n_lcao_total == 1
    assert collected.n_lcao_skipped == 0
    assert not collected.lcao_all_skipped
    # 1 eV over 2 atoms = 500 meV/atom
    assert collected.energies["delta_E_max_per_atom_meV"] == pytest.approx(500.0)
    assert [entry.basis for entry in collected.entries] == ["pw", "lcao"]
    assert [entry.n_atoms for entry in collected.entries] == [2, 2]


def test_collect_child_energies_counts_exit_304_as_usable():
    """304 (WARNING_TOLERANCE_EXCEEDED) is information for a grid search, not an error."""
    children = _pair_of_children()
    children[0]["node"].is_finished_ok = False
    children[0]["node"].exit_status = 304
    children[1]["node"].is_finished_ok = False
    children[1]["node"].exit_status = 304

    collected = collect_child_energies(children)
    assert len(collected.entries) == 2
    assert collected.energies["delta_E_max_per_atom_meV"] == pytest.approx(500.0)


def test_collect_child_energies_drops_a_failed_child():
    children = _pair_of_children()
    children[0]["node"].is_finished_ok = False
    children[0]["node"].exit_status = 400

    collected = collect_child_energies(children)
    assert [entry.basis for entry in collected.entries] == ["lcao"]
    assert collected.n_pw == 0


def test_collect_child_energies_skips_a_wrong_stru_orbital():
    """One bad LCAO child is dropped; the trustworthy ones are still paired."""
    children = _pair_of_children()
    children.append(_child(_lcao_child(pk=22, energy=-998.0, stru_orbital=BAD_ORBITAL),
                           task="trimer", basis="lcao_nsw", n_atoms=3))

    collected = collect_child_energies(children)
    assert collected.n_lcao_total == 2
    assert collected.n_lcao_skipped == 1
    assert collected.n_lcao_valid == 1
    assert not collected.lcao_all_skipped
    assert [entry.folder for entry in collected.entries] == ["dimer", "dimer"]
    assert any("⚠ 1/2 lcao children skipped" in note for note in collected.notes)


def test_collect_child_energies_flags_all_lcao_skipped():
    """Nothing trustworthy to compare with: the tolerance must not read as met."""
    pw = _child(_pw_child(pk=20, energy=-1000.0), task="dimer", basis="pw")
    bad = _child(_lcao_child(pk=21, energy=-999.0, stru_orbital=BAD_ORBITAL),
                 task="dimer", basis="lcao_nsw")
    bad2 = _child(_lcao_child(pk=22, energy=-998.0, stru_orbital=BAD_ORBITAL),
                  task="trimer", basis="lcao_nsw")

    collected = collect_child_energies([pw, bad, bad2])
    assert collected.lcao_all_skipped
    assert collected.n_lcao_skipped == 2
    assert collected.n_lcao_total == 2
    assert collected.n_lcao_valid == 0
    assert any(note.startswith("  ✗ all 2 lcao children") for note in collected.notes)


def test_collect_child_energies_reports_missing_misc_and_missing_energy():
    no_misc = _child(FakeCalc(pk=30, has_misc=False), task="a", basis="pw")
    no_energy = _child(FakeCalc(pk=31), task="b", basis="pw")

    collected = collect_child_energies([no_misc, no_energy])
    assert not collected.usable
    assert any("node 30 has no misc output" in note for note in collected.notes)
    assert any("node 31 misc has no total_energy" in note for note in collected.notes)


def test_collect_child_energies_accepts_an_injected_guard():
    """The guard is injectable so the caller can decide what "verified" means."""
    calls = []

    def refuse(node, expected):
        calls.append((node, expected))
        return False

    children = _pair_of_children()
    collected = collect_child_energies(children, verify_orbital=refuse)
    # it is handed the AbacusCalculation below the LCAO child, plus the file name it
    # must find in that calculation's STRU
    assert calls[0][0] is children[1]["node"].called[0]
    assert [expected for _, expected in calls] == [GOOD_ORBITAL]
    assert len(calls) == 1                       # the pw child is not verified
    assert collected.n_lcao_skipped == 1
    assert [entry.basis for entry in collected.entries] == ["pw"]


# ---------------------------------------------------------------------------
#  describe_deltas
# ---------------------------------------------------------------------------
def test_describe_deltas_renders_the_pairing():
    collected = collect_child_energies(_pair_of_children())
    lines = describe_deltas(collected.energies)

    assert len(lines) == 3                      # 2 maxima + 1 geometry
    assert "ΔE_max (per system)" in lines[0]
    assert "500.000 meV" in lines[1]            # 1 eV / 2 atoms
    assert "dimer" in lines[2] and "dE/atom=500.000 meV" in lines[2]


def test_describe_deltas_handles_no_pairs():
    lines = describe_deltas({"delta_E_max_per_atom_meV": 0.0})
    assert len(lines) == 2
    assert "0.000 meV" in lines[1]
