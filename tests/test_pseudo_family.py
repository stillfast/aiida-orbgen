"""A pseudo family is a name for one (UPF, ORB) pair — not just a name.

The bug this file guards against was found on the live profile on 2026-09-29 while
preparing a run with a freshly generated pseudopotential:

* `project/pseudo/U.pbe-n-nc.14ve.UPF` (md5 166049de…, the one the reference tree
  and the existing families were built from) and
* `project/u_14ve/U.pbe-n-nc.UPF` (md5 024f9a43…, the new file)

are different files that derive the **same** family label
`siab-u-nr-pbe-z14-nsw-10au-150Ry-g`, because the label is built from the element,
the UPF's xc/rel tokens, zval and the ORB name — none of which distinguishes two
generations of the same recipe.  `ensure_pseudo_family` then found the label in the
database (pk=220, holding the *old* UPF) and skipped creation, so the children would
have run with the previous pseudopotential while SIAB's reference DFT used the new
one.

`label_for_pair` closes that: same files → same label (reuse), different files →
content-suffixed label (register a new family).  All of it is pure, with the family
lookup injected, so it needs no AiiDA profile.
"""

from __future__ import annotations

import hashlib

import pytest

from aiida_orbgen.calculations.pseudo_family import (
    file_md5,
    label_for_pair,
)

BASE = "siab-u-nr-pbe-z14-nsw-10au-150Ry-g"


def _pair(tmp_path, upf_text="<UPF version='2.0.1'>old</UPF>", orb_text="orbital A"):
    upf = tmp_path / "U.pbe-n-nc.UPF"
    orb = tmp_path / "U_gga_10au_150Ry_4s3p2d2f1g.orb"
    upf.write_text(upf_text)
    orb.write_text(orb_text)
    return upf, orb


def _lookup(pairs):
    return lambda label: set(pairs)


# ---------------------------------------------------------------------------
#  file_md5
# ---------------------------------------------------------------------------
def test_file_md5_matches_hashlib(tmp_path):
    upf, _ = _pair(tmp_path)
    assert file_md5(upf) == hashlib.md5(upf.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
#  label_for_pair
# ---------------------------------------------------------------------------
def test_a_free_label_is_kept(tmp_path):
    upf, orb = _pair(tmp_path)
    assert label_for_pair(BASE, upf, orb, existing=_lookup([])) == (BASE, "")


def test_a_family_holding_the_same_files_is_reused(tmp_path):
    """The normal path: re-running the same point must not create new families."""
    upf, orb = _pair(tmp_path)
    stored = {(file_md5(upf), file_md5(orb))}
    label, note = label_for_pair(BASE, upf, orb, existing=_lookup(stored))
    assert (label, note) == (BASE, "")


def test_a_family_holding_other_files_forces_a_new_label(tmp_path):
    """The 2026-09-29 case: same name, different pseudopotential."""
    upf, orb = _pair(tmp_path, upf_text="<UPF version='2.0.1'>new</UPF>")
    stored_old_upf = hashlib.md5(b"<UPF version='2.0.1'>old</UPF>").hexdigest()

    label, note = label_for_pair(
        BASE, upf, orb, existing=_lookup([(stored_old_upf, "some-orb-md5")])
    )

    assert label != BASE
    assert label.startswith(BASE + "-")
    # The note has to name both sides: which family was in the way, and which file
    # the run was actually given.
    assert BASE in note
    assert upf.name in note
    assert stored_old_upf[:8] in note
    assert "previous pseudopotential" in note


def test_the_new_label_is_deterministic_and_content_derived(tmp_path):
    upf, orb = _pair(tmp_path)
    stored = _lookup([("someone-elses-upf", "someone-elses-orb")])

    first, _ = label_for_pair(BASE, upf, orb, existing=stored)
    again, _ = label_for_pair(BASE, upf, orb, existing=stored)
    assert first == again                      # a re-run reuses its own family

    # A different ORB (e.g. regenerated with another scheme) is a different family
    # too: it is the pair that is content-addressed, not just the UPF.
    other_orb = tmp_path / "other.orb"
    other_orb.write_text("orbital B")
    different, _ = label_for_pair(BASE, upf, other_orb, existing=stored)
    assert different != first

    # ... and once the suffixed label exists, it is reused rather than re-suffixed.
    label, note = label_for_pair(
        BASE, upf, orb, existing=_lookup([(file_md5(upf), file_md5(orb))])
    )
    assert (label, note) == (BASE, "")         # the original pair is still there


def test_the_suffix_is_short_enough_for_a_label(tmp_path):
    """AiiDA labels are limited; keep the added part small and readable."""
    upf, orb = _pair(tmp_path)
    label, _ = label_for_pair(
        BASE, upf, orb, existing=_lookup([("other-upf", "other-orb")])
    )
    suffix = label[len(BASE) + 1:]
    assert len(suffix) == 6
    assert suffix.isalnum()


def test_a_missing_upf_is_reported_as_such(tmp_path):
    """No silent fallback: an unreadable file must not be hashed as "empty"."""
    upf, orb = _pair(tmp_path)
    with pytest.raises(FileNotFoundError):
        label_for_pair(BASE, tmp_path / "nope.UPF", orb, existing=_lookup([]))
    with pytest.raises(FileNotFoundError):
        label_for_pair(BASE, upf, tmp_path / "nope.orb", existing=_lookup([]))
