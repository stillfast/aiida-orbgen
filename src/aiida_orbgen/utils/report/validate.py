"""Validate what a spillage run actually produced.

``generate_final_orbital`` used to accept any ``*.orb`` that appeared with a
fresh mtime.  That is not enough: on 2026-09-29 a run produced
``U_gga_10au_100Ry_4s3p2d2f1g.orb`` whose header said ``Number of Gorbital--> 0``
-- the file name promised a g channel and the file had none, because
``vloc_aux`` / ``lloc_min`` had been written at the top level of ``orbitals[1]``
where SIAB ignores them.  Nothing in the report noticed.

This module reads the produced orbital back and compares it with the scheme that
was requested, and scrapes the converged spillage value out of the log so the
report can show a quality number instead of just a file list.
"""

from __future__ import annotations

import re
from pathlib import Path

__all__ = [
    "SPECTRUM",
    "nzeta_string",
    "read_orbital",
    "scheme_of_name",
    "validate_orbital",
    "spillage_values",
]

#: SIAB's angular-momentum letters (``SIAB.io.convention.SPECTRUM``)
SPECTRUM = "spdfghijklmnopqrstuvwxyz"


def nzeta_string(nzeta) -> str:
    """``[4, 3, 2, 2, 1]`` → ``'4s3p2d2f1g'`` (SIAB's file-name spelling)."""
    return "".join(
        f"{int(nz)}{SPECTRUM[l]}" for l, nz in enumerate(nzeta) if int(nz) > 0
    )


def read_orbital(path: str | Path) -> dict:
    """Read a ``.orb`` file → ``{"elem", "rcut", "ecut", "nr", "per_l"}``.

    ``per_l`` is the number of radial functions per angular momentum, i.e. the
    list that has to match the requested ``nzeta``.
    """
    from SIAB.spillage.orbio import read_nao

    data = read_nao(str(path))
    chi = data["chi"]
    return {
        "elem": data.get("elem"),
        "rcut": float(data.get("rcut")),
        "ecut": float(data.get("ecut")),
        "nr": int(data.get("nr")),
        "per_l": [len(channel) for channel in chi],
    }


def scheme_of_name(path: str | Path, schemes) -> list[int] | None:
    """Which requested ``nzeta`` scheme a file name refers to, if any.

    SIAB names the file after the scheme (``…_4s3p2d2f1g.orb``), so the match is
    exact and order-insensitive.
    """
    name = Path(path).name
    for scheme in schemes:
        if nzeta_string(scheme) and nzeta_string(scheme) in name:
            return list(scheme)
    return None


def validate_orbital(path: str | Path, schemes) -> dict:
    """Check one produced ``.orb`` against the requested schemes.

    ``schemes`` is the list of ``nzeta`` lists from the SIAB config.  A file is
    ``ok`` when it can be read, its name matches one of the schemes and the
    radial functions per l match that scheme exactly.
    """
    result: dict = {"file": str(path), "ok": False}
    try:
        info = read_orbital(path)
    except Exception as exc:  # noqa: BLE001 — an unreadable product is a failure
        result["reason"] = f"cannot read: {exc}"
        return result
    result.update(info)

    expected = scheme_of_name(path, schemes)
    if expected is None:
        result["reason"] = (
            "name does not match any requested nzeta scheme "
            f"({[nzeta_string(s) for s in schemes]})"
        )
        return result
    result["expected"] = expected

    got = info["per_l"]
    # compare up to the requested l_max, padding the file's list with zeros
    width = max(len(expected), len(got))
    padded_got = got + [0] * (width - len(got))
    padded_expected = expected + [0] * (width - len(expected))
    if padded_got != padded_expected:
        result["reason"] = (
            f"radial functions per l are {got} but the requested scheme "
            f"{nzeta_string(expected)} needs {expected}"
        )
        return result
    result["ok"] = True
    return result


def spillage_values(log_path: str | Path) -> list[float]:
    """Converged spillage values recorded in a SIAB/``orbgen`` log.

    SIAB logs one line per cascade level::

        OrbgenCascade: orbital optimization ends with spillage = 1.20492740e-03
    """
    path = Path(log_path)
    if not path.is_file():
        return []
    pattern = re.compile(r"orbital optimization ends with spillage\s*=\s*([0-9.eE+-]+)")
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return []
    return [float(match) for match in pattern.findall(text)]
