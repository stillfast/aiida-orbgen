"""Assemble a spillage *reference tree* from AiiDA provenance.

SIAB's spillage minimisation needs, for every reference geometry
(the 5 dimers **and** the monomer), a directory laid out as::

    <jobdir>/INPUT, STRU
    <jobdir>/OUT.ABACUS/{running_scf.log, data-0-S, data-0-T,
                         istate.info, kpoints, WFC_NAO_GAMMA1.txt}

The AiiDA runs bring ``data-0-H/S/T``, ``istate.info``, ``running_scf.log``
and ``kpoints`` back to the cluster workdir. ``WFC_NAO_GAMMA1.txt`` is written
there too for runs submitted **after 2026-09-20** (``out_wfc_lcao`` is no longer
stripped by ``static/defaults.AIIDA_MANAGED_KEYS``); this module prefers that
file when it exists. For older runs — which dropped the key and therefore never
wrote wavefunctions — the SCF eigenvectors are rebuilt from the generalised
eigenproblem

    H C = S C eps

so ``C`` can be rebuilt exactly from the converged ``H`` and ``S`` (validated
against ``istate.info``: eigenvalues agree to ~1e-9 Ry, and the band space is
identical). Either way this module needs **no DFT** — it downloads the remote
files, rebuilds whatever is missing and checks every reconstruction.

Adapted from ``project/u_14ve/run_lmax4_rcut10/assemble_reference_from_aiida.py``
(where the algorithm and its validation were first worked out).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

__all__ = [
    "assemble_reference",
    "normalise_reference_for_siab",
    "siab_jobs_pending",
    "read_triu_abacus",
    "read_istate",
    "write_wfc_abacus",
    "reconstruct_wfc",
]

#: files taken from ``<remote>/OUT.<suffix>``
OUT_FILES = ("data-0-H", "data-0-S", "data-0-T",
             "istate.info", "running_scf.log", "kpoints", "INPUT")
#: real wavefunctions written by ABACUS when ``out_wfc_lcao`` survives
WFC_FILES = ("WFC_NAO_GAMMA1.txt",)
#: files taken from the remote workdir root
WORKDIR_FILES = ("INPUT", "STRU")
#: everything the spillage reads (needed to call a local folder complete)
SPILLAGE_FILES = ("running_scf.log", "data-0-S", "data-0-T",
                  "WFC_NAO_GAMMA1.txt")

RY2EV = 13.605693122994


# ---------------------------------------------------------------------------
#  ABACUS file formats
# ---------------------------------------------------------------------------


def read_triu_abacus(path: str | Path):
    """``data-{ik}-{H,S,T}``: token 0 is the dimension, then the upper triangle."""
    import numpy as np

    with open(path) as handle:
        tokens = handle.read().split()
    if not tokens:
        raise ValueError(f"{path}: empty matrix file")
    n = int(tokens[0])
    values = np.asarray(tokens[1:], dtype=float)
    expected = n * (n + 1) // 2
    if values.size != expected:
        raise ValueError(
            f"{path}: expected {expected} upper-triangle values for dim {n}, "
            f"got {values.size}"
        )
    mat = np.zeros((n, n))
    k = 0
    for i in range(n):
        row = values[k:k + n - i]
        k += n - i
        mat[i, i:] = row
        mat[i:, i] = row
    return mat


def read_istate(path: str | Path):
    """``istate.info`` → ``(bands, energies_eV, occupations)``."""
    import numpy as np

    bands, energies, occs = [], [], []
    with open(path) as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                bands.append(int(parts[0]))
                energies.append(float(parts[1]))
                occs.append(float(parts[2]))
            except ValueError:
                continue
    return np.array(bands), np.array(energies), np.array(occs)


def write_wfc_abacus(path: str | Path, eps_ry, occ, coef, nband: int,
                     per_line: int = 5) -> None:
    """Reproduce ABACUS' ``WFC_NAO_GAMMA1.txt`` layout (``read_wfc_lcao_txt``)."""
    norb = coef.shape[0]
    with open(path, "w") as handle:
        handle.write(f"{nband} (number of bands)\n")
        handle.write(f"{norb} (number of orbitals)\n")
        for ib in range(nband):
            handle.write(f"{ib + 1} (band)\n")
            handle.write(f"{eps_ry[ib]:.8e} (Ry)\n")
            handle.write(f"{occ[ib]:.8e} (Occupations)\n")
            column = coef[:, ib]
            for j in range(0, norb, per_line):
                handle.write(
                    "".join(f"{v:.8e} " for v in column[j:j + per_line]) + "\n"
                )


def reconstruct_wfc(out_dir: str | Path, *, tol_eV: float = 1e-3) -> dict:
    """Rebuild ``WFC_NAO_GAMMA1.txt`` from ``data-0-H``/``data-0-S``.

    Returns ``{"bands": n, "dev_eV": …, "occupied": n, "ok": bool}``; raises
    when the files are missing or the reconstruction does not reproduce
    ``istate.info``.
    """
    from scipy.linalg import eigh as generalized_eigh

    out_dir = Path(out_dir)
    ham = read_triu_abacus(out_dir / "data-0-H")
    ovl = read_triu_abacus(out_dir / "data-0-S")
    eps, coef = generalized_eigh(ham, ovl, lower=False)

    _, energies_eV, occupations = read_istate(out_dir / "istate.info")
    if energies_eV.size == 0:
        raise ValueError(f"{out_dir}/istate.info: no band lines found")

    nband = int(energies_eV.size)
    if eps.size < nband:
        raise ValueError(
            f"{out_dir}: H/S give {eps.size} states but istate.info lists {nband}"
        )
    dev = float(abs(eps[:nband] * RY2EV - energies_eV).max())
    if dev >= tol_eV:
        raise ValueError(
            f"{out_dir}: reconstructed eigenvalues deviate by {dev:.3e} eV "
            f"(> {tol_eV} eV) — refusing to write a wrong WFC"
        )
    write_wfc_abacus(out_dir / "WFC_NAO_GAMMA1.txt", eps[:nband],
                     occupations[:nband], coef[:, :nband], nband)
    return {
        "bands": nband,
        "occupied": int((occupations > 0.5).sum()),
        "dev_eV": dev,
        "ok": True,
    }


# ---------------------------------------------------------------------------
#  remote → local assembly
# ---------------------------------------------------------------------------


def _folder_is_complete(jobdir: Path, expected_nao: int | None = None) -> bool:
    """Files present, finished — and computed with the expected basis."""
    from aiida_orbgen.utils.report.orbitals import _running_scf_done

    if expected_nao is None:
        out_dirs = sorted(jobdir.glob("OUT.*")) if jobdir.is_dir() else []
        for out_dir in out_dirs:
            if not all((out_dir / name).is_file() for name in SPILLAGE_FILES):
                continue
            try:
                if "Finish Time" in (out_dir / "running_scf.log").read_text(errors="ignore"):
                    return True
            except OSError:
                continue
        return False
    done, _ = _running_scf_done(jobdir, expected_nao)
    return done


def _remote_suffix(transport, remote: str) -> str:
    """Suffix of the ABACUS output dir on the remote (usually ``aiida``)."""
    import re
    import tempfile

    local = os.path.join(tempfile.mkdtemp(prefix="orbgen_remote_"), "INPUT")
    try:
        transport.getfile(f"{remote}/INPUT", local)
        text = Path(local).read_text(errors="ignore")
    except Exception:  # noqa: BLE001 — fall back to the aiida default
        return "aiida"
    match = re.search(r"^\s*suffix\s+(\S+)", text, flags=re.MULTILINE)
    return match.group(1) if match else "aiida"


def _lcao_children(calc_node) -> dict[str, Any]:
    """``{folder: AbacusBaseWorkChain}`` for the LCAO jobs of one grid point."""
    found: dict[str, Any] = {}
    for child in getattr(calc_node, "called", []):
        if getattr(child, "process_label", "") != "AbacusBaseWorkChain":
            continue
        extras = child.base.extras.all
        if extras.get("basis") not in (None, "lcao_nsw"):
            continue
        folder = extras.get("task")
        if folder:
            found[folder] = child
    return found


# ---------------------------------------------------------------------------
#  SIAB consistency: never let the spillage step silently run a DFT
# ---------------------------------------------------------------------------


def siab_jobs_pending(dft_root: str | Path, config_path: str | Path) -> list[str]:
    """Folders SIAB's ``rundft`` would *run* with this tree and config.

    Calls the very same ``build_abacus_jobs`` the driver calls, so the answer is
    authoritative -- including SIAB's ``DuplicateCheck``, which compares the
    **whole** INPUT (``suffix``, ``ks_solver``, ``out_wfc_lcao``, ...) and not
    just "is there a finished log".  That is why an assembled tree, whose INPUT
    was written by the AiiDA plugin (``suffix aiida``), is classified as *new
    work* and triggers a real SCF.

    Note this call has the same side effect the driver's own call has: it
    rewrites INPUT/STRU of the folders it considers out of date.  That is
    harmless as long as the caller then either runs them (``run_missing="all"``)
    or normalises the tree with :func:`normalise_reference_for_siab`.
    """
    from SIAB.abacus.api import build_abacus_jobs
    from SIAB.driver.main import init

    dft_root = Path(dft_root).expanduser().resolve()
    # Resolve before chdir: a relative config path would break inside dft_root.
    config_path = Path(config_path).expanduser().resolve()

    # Mirror what `orbgen` itself sees: `generate_final_orbital` invokes it as
    # `cd <dft_root> && orbgen -i <config>`.  SIAB resolves a *relative*
    # `pseudo_dir` against the cwd (SIAB/io/param.ParamAssert) and
    # `build_abacus_jobs` creates its folders there too, so the check has to run
    # from the same directory -- otherwise a guard meant to protect the user
    # would fail for a config the real command accepts.
    cwd = os.getcwd()
    os.chdir(dft_root)
    try:
        glbparam, dftparam, spillparam, compparam, iop = init(str(config_path))
        jobs = build_abacus_jobs(
            atomspecies=[{
                "elem": glbparam["element"],
                # short-circuit on purpose: SIAB's own `rundft` evaluates
                # ``dftparam['ecutwfc']`` eagerly and raises KeyError for a
                # config that sets only ``ecutjy``.
                "ecutjy": spillparam.get("ecutjy") or dftparam.get("ecutwfc"),
                "zval": 0,
            }],
            geoms=spillparam["geoms"],
            rcuts=glbparam["bessel_nao_rcut"],
            dftparams=dftparam,
            spill_guess=spillparam.get("spill_guess"),
            compparam=compparam,
            **iop,
        )
    finally:
        os.chdir(cwd)
    return [str(job) for job in jobs if job is not None]


def _declared_suffix(jobdir: Path) -> str | None:
    """Suffix of the ABACUS output dir this folder's INPUT declares."""
    from SIAB.spillage.datparse import read_input_script

    input_path = jobdir / "INPUT"
    if not input_path.is_file():
        return None
    try:
        params = read_input_script(str(input_path))
    except Exception:  # noqa: BLE001 — unreadable INPUT: leave the folder alone
        return None
    return str(params.get("suffix") or "ABACUS")


def align_output_dirs(dft_root: str | Path) -> dict:
    """Rename ``OUT.<x>`` to the suffix each folder's INPUT declares.

    SIAB finds the reference data through the suffix in the INPUT
    (``_spil_bnd_autoset`` → ``OUT.<suffix>``), while an AiiDA-assembled tree
    keeps it under ``OUT.aiida``.  Aligning the two is what makes SIAB *use* the
    data instead of recomputing it -- and after ``build_abacus_jobs`` has
    rewritten the INPUT, ``DuplicateCheck`` reports "already done", so the job
    list alone cannot reveal the mismatch.

    Only folders with a single ``OUT.*`` directory are touched, and nothing is
    deleted: a folder with an unexpected layout is reported instead.
    """
    dft_root = Path(dft_root).expanduser().resolve()
    renamed: dict[str, str] = {}
    ambiguous: dict[str, list[str]] = {}
    for jobdir in sorted(p for p in dft_root.iterdir() if p.is_dir()):
        # ``<name>.stale`` is the report layer's quarantine of reference data
        # that was computed with another primitive basis; it is deliberately out
        # of the way and must stay untouched.
        if ".stale" in jobdir.name:
            continue
        if not (jobdir / "INPUT").is_file():
            continue
        wanted = _declared_suffix(jobdir)
        if wanted is None:
            continue
        target = jobdir / f"OUT.{wanted}"
        out_dirs = sorted(p for p in jobdir.glob("OUT.*") if p.is_dir())
        if target in out_dirs:
            continue
        if len(out_dirs) == 1:
            try:
                out_dirs[0].rename(target)
            except OSError as exc:  # pragma: no cover — permissions/races
                ambiguous[jobdir.name] = [f"rename failed: {exc}"]
                continue
            renamed[jobdir.name] = f"{out_dirs[0].name} -> {target.name}"
        elif out_dirs:
            ambiguous[jobdir.name] = [p.name for p in out_dirs]
    return {"renamed": renamed, "ambiguous": ambiguous}


def spillage_data_ok(jobdir: str | Path) -> bool:
    """``OUT.<suffix>`` holds everything the spillage reads.

    A folder can look up to date to SIAB (its INPUT matches what SIAB would
    write) and still contain no reference data at all -- that is exactly the
    state ``build_abacus_jobs`` leaves behind when it creates a folder for a
    geometry that was never computed.
    """
    jobdir = Path(jobdir)
    suffix = _declared_suffix(jobdir)
    if suffix is None:
        return False
    out = jobdir / f"OUT.{suffix}"
    return all((out / name).is_file() for name in SPILLAGE_FILES)


def normalise_reference_for_siab(
    dft_root: str | Path,
    config_path: str | Path,
    *,
    allow_run: bool = False,
    required: list[str] | None = None,
    optional: list[str] | None = None,
) -> dict:
    """Make SIAB accept an AiiDA-assembled tree, or explain why it will not.

    Two separate traps, both of which end in "a reference DFT you did not ask
    for" or "the reference data that was collected is ignored":

    1. ``SIAB.abacus.utils.DuplicateCheck`` compares the **whole** INPUT, so a
       folder whose INPUT was written by the AiiDA plugin (``suffix aiida``,
       ``ks_solver scalapack_gvx``, no ``out_wfc_lcao``) is classified as *new
       work* and ``rundft`` recomputes it.  Asking SIAB itself which jobs it
       would run (``build_abacus_jobs``) rewrites those INPUT/STRU with its own
       autoset -- after which the folders are up to date.
    2. SIAB locates the reference data through that suffix
       (``OUT.<suffix>/``), while an assembled tree keeps it in ``OUT.aiida``.
       :func:`align_output_dirs` renames the directory to match, so the
       wavefunctions that were downloaded are the ones the spillage reads.

    Returns ``{"ok", "checked", "jobs", "renamed", "ambiguous", "message"}``;
    ``ok`` is False when SIAB would still have to compute something, in which
    case the caller must refuse to start the spillage unless the user explicitly
    allowed a reference DFT (``run_missing="all"``).
    """
    dft_root = Path(dft_root).expanduser().resolve()

    jobs = siab_jobs_pending(dft_root, config_path)
    aligned = align_output_dirs(dft_root)
    renamed = dict(aligned["renamed"])
    if jobs:
        # the folders SIAB just rewrote are the ones whose data has to be realigned
        renamed_again = align_output_dirs(dft_root)["renamed"]
        renamed.update(renamed_again)

    jobs_after = siab_jobs_pending(dft_root, config_path)
    # Jobs for the *optional* folders (the monomer, i.e. the atomic initial
    # guess) are legitimately computed by SIAB and must not block the spillage;
    # everything else has to be refused unless the user allowed a DFT.
    allowed_extra = set(optional or [])
    blocking_jobs = [job for job in jobs_after if job not in allowed_extra]

    # A folder whose INPUT SIAB just rewrote looks "up to date" even when it holds
    # no data at all, so check the contents as well.
    if required is not None:
        candidates = [name for name in required if ".stale" not in name]
    else:
        candidates = [job for job in jobs_after
                      if job not in allowed_extra and ".stale" not in job]
        candidates += [p.name for p in sorted(dft_root.iterdir())
                       if p.is_dir() and ".stale" not in p.name
                       and (p / "INPUT").is_file()
                       and p.name not in allowed_extra]
    missing_data = [name for name in dict.fromkeys(candidates)
                    if not spillage_data_ok(dft_root / name)]

    result = {
        "checked": True,
        "jobs": jobs_after,
        "blocking_jobs": blocking_jobs,
        "missing_data": missing_data,
        "renamed": renamed,
        "ambiguous": aligned["ambiguous"],
    }
    if blocking_jobs or missing_data:
        result["ok"] = bool(allow_run)
        problems = []
        if blocking_jobs:
            problems.append("SIAB would run " + ", ".join(blocking_jobs))
        if missing_data:
            problems.append("no usable reference data in " + ", ".join(missing_data))
        result["message"] = (
            "; ".join(problems)
            + (" (allowed by run_missing='all')" if allow_run else
               " — re-run the LCAO reference jobs with the current presets, or "
               "pass --force-final-orbital to let SIAB compute them now")
        )
        return result

    result["ok"] = True
    if jobs_after:
        notes_extra = ["SIAB may still compute " + ", ".join(jobs_after)
                       + " (the optional initial guess)"]
    else:
        notes_extra = []
    notes = list(notes_extra)
    if renamed:
        notes.append("renamed " + ", ".join(f"{k}: {v}" for k, v in renamed.items()))
    if aligned["ambiguous"]:
        notes.append(
            "unexpected multiple output dirs: "
            + ", ".join(f"{k}={v}" for k, v in aligned["ambiguous"].items())
        )
    result["message"] = "SIAB would recompute nothing" + (
        f" ({'; '.join(notes)})" if notes else ""
    )
    return result



def assemble_reference(
    calc_node,
    dft_root: str | Path,
    *,
    folders: list[str] | None = None,
    profile: str | None = None,
    expected_nao: dict[str, int | None] | None = None,
) -> dict:
    """Build/extend a spillage reference tree from the AiiDA remote data.

    ``calc_node`` is the ``OrbgenCalcWorkChain`` of one grid point. For every
    LCAO child it downloads ``OUT.<suffix>/{data-0-H/S/T, istate.info,
    running_scf.log, kpoints, INPUT}`` plus the workdir ``INPUT``/``STRU``,
    rebuilds ``WFC_NAO_GAMMA1.txt`` and validates it. Folders that are already
    complete (including the monomer, which AiiDA never ran) are left alone.
    """
    from aiida import load_profile

    load_profile(profile)
    dft_root = Path(dft_root).expanduser().resolve()
    dft_root.mkdir(parents=True, exist_ok=True)

    children = _lcao_children(calc_node)
    if folders is not None:
        children = {name: node for name, node in children.items() if name in folders}
    if not children:
        return {
            "ok": False,
            "folders": {},
            "message": (
                f"OrbgenCalcWorkChain<{calc_node.pk}> has no LCAO child to "
                f"assemble from"
            ),
        }

    result: dict[str, Any] = {"ok": True, "dft_root": str(dft_root), "folders": {}}
    for folder, base in sorted(children.items()):
        jobdir = dft_root / folder
        entry: dict[str, Any] = {"folder": folder}
        if _folder_is_complete(jobdir, (expected_nao or {}).get(folder)):
            entry["status"] = "already-complete"
            result["folders"][folder] = entry
            continue

        try:
            remote_node = base.outputs.remote_folder
            remote = remote_node.get_remote_path()
        except Exception:  # noqa: BLE001 — fall back to the calcjob workdir
            remote, remote_node = None, None
            for descendant in base.called_descendants:
                if getattr(descendant, "process_label", "") == "AbacusCalculation":
                    remote = descendant.get_remote_workdir()
                    remote_node = descendant.computer
                    break
        if remote is None:
            entry.update(status="failed", message="no remote folder recorded")
            result["folders"][folder] = entry
            result["ok"] = False
            continue

        local_out = jobdir / "OUT.ABACUS"
        local_out.mkdir(parents=True, exist_ok=True)
        started = time.time()
        missing: list[str] = []
        computer = getattr(remote_node, "computer", remote_node)
        try:
            with computer.get_transport() as transport:
                if not transport.path_exists(remote):
                    entry.update(status="failed",
                                 message=f"remote workdir gone: {remote}")
                    result["folders"][folder] = entry
                    result["ok"] = False
                    continue
                suffix = _remote_suffix(transport, remote)
                entry["remote"] = remote
                entry["remote_suffix"] = suffix
                for name in OUT_FILES + WFC_FILES:
                    source = f"{remote}/OUT.{suffix}/{name}"
                    if not transport.isfile(source):
                        if name in OUT_FILES:
                            missing.append(name)
                        continue
                    transport.getfile(source, str(local_out / name))
                for name in WORKDIR_FILES:
                    source = f"{remote}/{name}"
                    if transport.isfile(source):
                        transport.getfile(source, str(jobdir / name))
        except Exception as exc:  # noqa: BLE001 — network/transport problems
            entry.update(status="failed", message=f"transfer failed: {exc}")
            result["folders"][folder] = entry
            result["ok"] = False
            continue

        entry["seconds"] = round(time.time() - started, 1)
        # ``runtime_scf.log`` names differ; running_scf.log is what we require
        if missing:
            entry["missing"] = missing
        if not (local_out / "running_scf.log").is_file():
            entry.update(status="failed", message="no running_scf.log on the remote")
            result["folders"][folder] = entry
            result["ok"] = False
            continue

        if (local_out / "WFC_NAO_GAMMA1.txt").is_file():
            # ABACUS wrote the wavefunctions itself (run submitted after the
            # out_wfc_lcao fix) — nothing to rebuild.
            entry["status"] = "fetched"
            result["folders"][folder] = entry
            continue
        try:
            info = reconstruct_wfc(local_out)
        except Exception as exc:  # noqa: BLE001 — mismatching/missing matrices
            entry.update(status="failed", message=str(exc))
            result["folders"][folder] = entry
            result["ok"] = False
            continue

        entry.update(status="rebuilt", **info)
        result["folders"][folder] = entry

    rebuilt = [f for f, e in result["folders"].items() if e["status"] == "rebuilt"]
    fetched = [f for f, e in result["folders"].items() if e["status"] == "fetched"]
    ready = [f for f, e in result["folders"].items() if e["status"] == "already-complete"]
    result["summary"] = (
        f"{len(rebuilt)} rebuilt from AiiDA"
        + (f", {len(fetched)} fetched" if fetched else "")
        + f", {len(ready)} already complete"
        + (f", {len(result['folders']) - len(rebuilt) - len(fetched) - len(ready)} failed"
           if len(rebuilt) + len(fetched) + len(ready) != len(result["folders"]) else "")
    )
    return result
