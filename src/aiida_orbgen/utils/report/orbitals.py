"""Orbital-file handling for ``aiida-orbgen report``.

Two independent jobs live here:

1. :func:`export_primitive_orbitals` — pull the *primitive NSW orbital*
   (the ``.orb`` a grid point was evaluated with) out of AiiDA provenance and
   write it next to the report. The authoritative source is the
   ``AtomicOrbitalData`` that ``ensure_pseudo_family`` stored (its
   ``filename_second``); a filesystem copy of ``siab_info['orb_path']`` is
   used as a fallback.

2. :func:`generate_final_orbital` — run the SIAB spillage minimisation
   (``orbgen -i <config> -o <outdir>``) for the best ``(l_max, r_cut)``
   point, which turns the reference DFT results into the *final* CSW-NAO
   basis (``.orb`` + ``.param``). DFT results are never re-run here unless
   ``force=True``: the step only proceeds when every reference geometry
   already carries a finished ABACUS log.
"""

from __future__ import annotations

import json
import shlex
import time
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aiida_orbgen.utils.report.orbgen import select_best_point
from aiida_orbgen.interfaces.nsw import (
    apply_grid_point,
    folder_rcut,
    legacy_point_dir_name,
    point_dir_name as point_dir_name_of,
)
from aiida_orbgen.utils.report.validate import spillage_values, validate_orbital

__all__ = [
    "OrbitalFile",
    "resolve_dft_root",
    "point_dir_name",
    "export_primitive_orbitals",
    "generate_final_orbital",
    "select_best_point",
    "job_folders",
    "dft_root_status",
    "choose_point_for_dft_root",
    "dft_root_from_input_json",
    "load_preset_config",
    "configs_differ",
]


@dataclass
class OrbitalFile:
    """One exported orbital file."""

    path: Path
    source: str
    family_label: str | None = None
    l_max: int | None = None
    r_cut: float | None = None
    kind: str = "primitive"
    pk: int | None = None


# ---------------------------------------------------------------------------
#  Primitive (NSW) orbitals
# ---------------------------------------------------------------------------


def _atomic_orbital_data_nodes(family_label: str, second_filename: str | None) -> list:
    """Find the ``AtomicOrbitalData`` behind ``family_label``.

    Looks at the ``AtomicOrbitalFamily`` group first (that is what the
    workchain builds), then falls back to a repository-wide search on the
    second file name.
    """
    from aiida import orm

    found: list = []
    try:
        qb = orm.QueryBuilder()
        qb.append(orm.Group, filters={"label": family_label}, tag="group")
        qb.append(orm.Data, with_group="group", tag="data")
        for (node,) in qb.iterall():
            if _has_second_file(node):
                found.append(node)
    except Exception:  # noqa: BLE001 — fall through to the next strategy
        found = []

    if not found and second_filename:
        try:
            qb = orm.QueryBuilder()
            qb.append(orm.Data)
            for (node,) in qb.iterall():
                if _second_filename(node) == second_filename:
                    found.append(node)
        except Exception:  # noqa: BLE001
            pass

    return found


def _has_second_file(node) -> bool:
    return bool(_second_filename(node))


def _second_filename(node) -> str | None:
    try:
        return node.base.attributes.get("filename_second")
    except Exception:  # noqa: BLE001
        return None


def _claim_path(out_dir: Path, name: str, point, claimed: dict) -> Path:
    """Target path for ``name``, prefixing **only** on an in-run conflict.

    Files left behind by an earlier ``report`` run are overwritten: they come
    from the same ``AtomicOrbitalData`` node, so prefixing them would just pile
    up identical ``lmax…_rcut…_`` duplicates. A prefix is only added when two
    different grid points want the same file name in the same run.
    """
    key = (name, getattr(point, "pk", None))
    if key in claimed:
        return claimed[key]

    candidate = out_dir / name
    if candidate in claimed.values():
        l_max = getattr(point, "l_max", None)
        r_cut = getattr(point, "r_cut", None)
        prefix = (
            f"lmax{l_max}_rcut{r_cut:g}_"
            if l_max is not None and r_cut is not None
            else "dup_"
        )
        candidate = out_dir / (prefix + name)
    claimed[key] = candidate
    return candidate


def export_primitive_orbitals(
    summary,
    out_dir: str | Path,
    *,
    include_upf: bool = False,
) -> tuple[list[OrbitalFile], list[str]]:
    """Export the primitive ``.orb`` (one per grid point) into ``out_dir``.

    Returns ``(files, warnings)`` — warnings are human-readable lines the CLI
    can print, so a missing orbital never aborts the report.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files: list[OrbitalFile] = []
    warnings: list[str] = []
    claimed: dict = {}

    for point in getattr(summary, "grid", []):
        info = point.siab_info or {}
        family_label = info.get("family_label")
        orb_name = info.get("nsw_filename") or (
            Path(info["nsw"]).name if info.get("nsw") else None
        )
        written = False

        # 1) AiiDA repository of the AtomicOrbitalData (UPF + ORB pair)
        if family_label or orb_name:
            try:
                nodes = _atomic_orbital_data_nodes(family_label, orb_name)
            except Exception as exc:  # noqa: BLE001
                nodes = []
                warnings.append(f"grid point <{point.pk}>: orbital query failed: {exc}")
            for node in nodes:
                name = _second_filename(node) or orb_name or f"orbital-{node.pk}.orb"
                target = _claim_path(out_dir, name, point, claimed)
                try:
                    content = node.get_content_second(mode="rb")
                    target.write_bytes(
                        content if isinstance(content, bytes) else content.encode()
                    )
                except Exception as exc:  # noqa: BLE001
                    warnings.append(
                        f"grid point <{point.pk}>: cannot read .orb from "
                        f"AtomicOrbitalData<{node.pk}>: {exc}"
                    )
                    continue
                files.append(OrbitalFile(
                    path=target,
                    source=f"aiida:AtomicOrbitalData<{node.pk}>",
                    family_label=family_label,
                    l_max=point.l_max,
                    r_cut=point.r_cut,
                    kind="primitive",
                    pk=node.pk,
                ))
                written = True

                if include_upf:
                    upf_name = getattr(node, "filename", None) or "pseudo.UPF"
                    upf_target = _claim_path(out_dir, upf_name, point, claimed)
                    try:
                        upf_target.write_bytes(node.get_content(mode="rb"))
                        files.append(OrbitalFile(
                            path=upf_target,
                            source=f"aiida:AtomicOrbitalData<{node.pk}>",
                            family_label=family_label,
                            l_max=point.l_max,
                            r_cut=point.r_cut,
                            kind="pseudo",
                            pk=node.pk,
                        ))
                    except Exception as exc:  # noqa: BLE001
                        warnings.append(f"grid point <{point.pk}>: cannot read UPF: {exc}")

        # 2) Filesystem fallback (SIAB wrote the primitive orbital locally)
        if not written:
            for candidate in (info.get("orb_path"), info.get("nsw")):
                if not candidate:
                    continue
                source_path = Path(candidate)
                if not source_path.is_file():
                    continue
                target = _claim_path(out_dir, source_path.name, point, claimed)
                shutil.copy2(source_path, target)
                files.append(OrbitalFile(
                    path=target,
                    source=f"filesystem:{source_path}",
                    family_label=family_label,
                    l_max=point.l_max,
                    r_cut=point.r_cut,
                    kind="primitive",
                ))
                written = True
                break

        if not written:
            warnings.append(
                f"grid point <{point.pk}> (l_max={point.l_max}, r_cut={point.r_cut}): "
                f"no .orb found in AiiDA (family {family_label!r}) and "
                f"{info.get('orb_path')!r} is not on this filesystem"
            )

    return files, warnings


# ---------------------------------------------------------------------------
#  Final CSW-NAO orbital (SIAB spillage minimisation)
# ---------------------------------------------------------------------------


def _read_siab_json(calc_node) -> dict:
    """The SIAB config of a grid point, straight from its ``siab_json`` input."""
    try:
        node = calc_node.inputs.siab_json
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"node <{calc_node.pk}> has no siab_json input: {exc}") from exc
    content = node.get_content()
    if isinstance(content, bytes):
        content = content.decode("utf-8")
    return json.loads(content)




def job_folders(siab_info: dict, l_max: int | None, r_cut: float) -> tuple[list[str], list[str]]:
    """Reference-geometry folders a spillage run expects.

    Returns ``(required, optional)``: the dimers listed in ``siab_info`` and
    the monomer used for the atomic initial guess (optional — SIAB adds it
    itself, but it must already have DFT data for the run to be complete).
    """
    dft_entries = siab_info.get("dft") or []
    required = [entry["folder"] for entry in dft_entries if entry.get("folder")]

    element = None
    for entry in dft_entries:
        folder = entry.get("folder") or ""
        if folder:
            element = folder.split("-")[0]
            break
    if element is None:
        orb_name = siab_info.get("nsw_filename") or ""
        element = orb_name.split("_")[0] if orb_name else None

    optional: list[str] = []
    if element:
        optional.append(f"{element}-monomer-{folder_rcut(r_cut)}au")
    return required, optional


def _nao_dimension(jobdir: Path) -> int | None:
    """Number of LCAO basis functions recorded in the reference data itself."""
    for out_dir in sorted(Path(jobdir).glob("OUT.*")):
        matrix = out_dir / "data-0-H"
        if not matrix.is_file():
            continue
        try:
            return int(matrix.read_text(errors="ignore").split()[0])
        except (OSError, ValueError, IndexError):
            continue
    return None


def _n_atoms(jobdir: Path) -> int | None:
    """Atom count from the geometry's STRU (basis-independent)."""
    stru = Path(jobdir) / "STRU"
    if not stru.is_file():
        return None
    try:
        from abacuslite.io.generalio import read_stru

        params = read_stru(str(stru))
    except Exception:  # noqa: BLE001 — the check is optional
        return None
    species = params.get("species", []) if isinstance(params, dict) else []
    total = sum(int(sp.get("natom", 0)) for sp in species if isinstance(sp, dict))
    if total <= 0:
        total = sum(len(sp.get("atom", [])) for sp in species if isinstance(sp, dict))
    return total or None


def expected_nao(config: dict, l_max: int, r_cut: float, n_atoms: int | None) -> int | None:
    """LCAO dimension a reference calculation *should* have for this config.

    ``SIAB`` builds the primitive basis from ``(r_cut, ecutjy, lmaxmax)``; the
    number of functions is ``Σ_l nzeta_l * (2l+1)`` per atom. Used to detect a
    reference tree that was computed with a *different* basis (e.g. before
    ``ecutjy`` was changed) — mixing the two makes SIAB fail deep inside
    ``basistrans.jy2ao`` with "len(coef[…]) should not exceed nbes[…]".
    """
    if n_atoms is None:
        return None
    ecut = config.get("ecutjy", config.get("ecutwfc"))
    if ecut is None:
        return None
    try:
        from aiida_orbgen.interfaces.nsw import compute_nbes_per_l

        nzeta = compute_nbes_per_l(
            float(r_cut), float(ecut), int(l_max),
            config.get("primitive_type", "reduced"),
        )
    except Exception:  # noqa: BLE001 — SIAB not importable
        return None
    return int(n_atoms) * sum(nz * (2 * ell + 1) for ell, nz in enumerate(nzeta))


def expected_nao_by_folder(
    root: Path, folders: list[str], config: dict, l_max: int, r_cut: float
) -> dict[str, int | None]:
    return {
        name: expected_nao(config, l_max, r_cut, _n_atoms(Path(root) / name))
        for name in folders
    }


def _running_scf_done(
    folder: Path, expected_nao: int | None = None
) -> tuple[bool, str | None]:
    """``True`` when *any* ``OUT.*/running_scf.log`` of ``folder`` finished.

    With ``expected_nao`` the reference data must also come from the same
    primitive basis as the current config, otherwise it counts as *stale*.
    """
    if not folder.is_dir():
        return False, "missing"
    logs = sorted(folder.glob("OUT.*/running_scf.log"))
    if not logs:
        return False, "no OUT.*/running_scf.log"
    for log in logs:
        try:
            text = log.read_text(errors="ignore")
        except OSError:
            continue
        if "Finish Time" not in text:
            continue
        if expected_nao is not None:
            nao = _nao_dimension(folder)
            if nao is not None and nao != expected_nao:
                return False, (
                    f"stale: reference data were computed with a different "
                    f"primitive basis ({nao} LCAO functions per cell, this "
                    f"config expects {expected_nao} — check ecutjy/bessel_nao_rcut)"
                )
        return True, None
    return False, f"{logs[0].name} has no 'Finish Time'"


def dft_root_status(
    dft_root: str | Path,
    folders: list[str],
    expected_nao: dict[str, int | None] | None = None,
) -> tuple[bool, dict[str, str]]:
    """Check whether every reference geometry has finished DFT results."""
    root = Path(dft_root)
    details: dict[str, str] = {}
    ready = True
    for name in folders:
        expect = (expected_nao or {}).get(name)
        done, reason = _running_scf_done(root / name, expect)
        details[name] = "done" if done else (reason or "not done")
        if not done:
            ready = False
    return ready, details


def _build_siab_config(calc_node, l_max: int, r_cut: float,
                       siab_config_path: str | Path | None = None) -> dict:
    """SIAB config for one grid point (mirrors ``run_siab_pipeline``).

    ``siab_config_path`` overrides the ``siab_json`` stored on the node — handy
    when the run was submitted before a preset fix, so the stored copy is stale.
    """
    if siab_config_path is not None:
        with open(siab_config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
    else:
        config = _read_siab_json(calc_node)
    # 覆盖逻辑只有一份实现 (interfaces.nsw.apply_grid_point): 这里以前自己拼
    # bessel_nao_rcut, 与 run_siab_pipeline 的 float 写法不一致, 于是同一个格
    # 网格点在两条路径上会得到 ``9au`` 和 ``9.0au`` 两个名字。
    return apply_grid_point(config, l_max, r_cut)


def _apply_grid_point(config: dict, l_max: int, r_cut: float) -> dict:
    """Same override ``run_siab_pipeline`` applies for one grid point."""
    return apply_grid_point(config, l_max, r_cut)


def load_preset_config(
    input_json: str | Path | None,
    l_max: int,
    r_cut: float,
) -> tuple[dict | None, str | None]:
    """Re-materialise the SIAB config from ``input.json`` + the current presets.

    Returns ``(config, error)``. ``(None, reason)`` whenever the input file or
    the presets cannot be read — the caller then just keeps the stored config.
    """
    if not input_json:
        return None, "output.json records no input.json"
    path = Path(input_json)
    if not path.is_file():
        return None, f"input.json not found: {path}"
    try:
        from aiida_orbgen.utils.config import ConfigLoader

        bundle = ConfigLoader(path).load_all()
    except Exception as exc:  # noqa: BLE001
        return None, f"cannot re-read the presets: {exc}"

    if len(bundle.orbgen_presets) != 1:
        return None, (
            f"input.json selects {len(bundle.orbgen_presets)} orbgen presets "
            f"— cannot tell which one this grid point used"
        )
    config = _apply_grid_point(bundle.orbgen_presets[0].config, l_max, r_cut)
    return config, None


def configs_differ(left: dict, right: dict) -> bool:
    """Structural comparison of two SIAB configs (order-insensitive)."""
    return json.dumps(left, sort_keys=True) != json.dumps(right, sort_keys=True)


def _static_of(input_json: str | Path | None) -> dict:
    import json as _json

    if not input_json:
        return {}
    path = Path(input_json)
    if not path.is_file():
        return {}
    try:
        data = _json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    static = data.get("static") if isinstance(data, dict) else None
    return static if isinstance(static, dict) else {}


def dft_root_from_input_json(
    input_json: str | Path | None,
    point=None,
) -> Path | None:
    """``static.dft_root`` (or per-point ``static.dft_roots``) for ``point``.

    ``static.dft_roots`` maps the produced directory name to a tree, e.g.
    ``{"lmax4_rcut9": "…/run_lmax4_rcut9", "lmax4_rcut10": "…/run_lmax4_rcut10"}``.
    """
    static = _static_of(input_json)
    roots = static.get("dft_roots")
    if point is not None and isinstance(roots, dict) and point.r_cut is not None:
        for key in (*point_dir_aliases(point), f"{point.l_max},{folder_rcut(point.r_cut)}"):
            if roots.get(key):
                return Path(str(roots[key])).expanduser()
    value = static.get("dft_root")
    return Path(str(value)).expanduser() if value else None


def point_dir_name(point) -> str:
    """Directory name of a grid point's orbital directory (``lmax4_rcut10``)."""
    return point_dir_name_of(point.l_max, point.r_cut)


def point_dir_aliases(point) -> list[str]:
    """Every spelling this grid point's directory may have on disk.

    The workflow wrote ``lmax4_rcut10p0`` before 2026-09-29; such trees must stay
    readable even though new runs use ``lmax4_rcut10``.
    """
    current = point_dir_name(point)
    legacy = legacy_point_dir_name(point.l_max, point.r_cut)
    return [current] if legacy == current else [current, legacy]


def _point_is_covered(root: Path, point) -> bool:
    """``True`` when ``root`` already holds finished data for ``point``.

    Basis-aware only when the point's config is known (``point.config``); the
    resolution step mainly needs "does any per-point tree exist here", so the
    cheaper file check is used when no config is attached.
    """
    if point.r_cut is None or not root.is_dir():
        return False
    required, optional = job_folders(point.siab_info or {}, point.l_max, point.r_cut)
    config = getattr(point, "config", None)
    expect = (
        expected_nao_by_folder(root, required + optional, config,
                               point.l_max, point.r_cut)
        if config else None
    )
    ready, _ = dft_root_status(root, required + optional, expect)
    return ready


def resolve_dft_root(
    point,
    candidate: str | Path | None,
    *,
    input_json: str | Path | None = None,
    dft_root: str | Path | None = None,
) -> tuple[Path, str]:
    """Where this point's reference DFT lives, and how it was found.

    Order: ``--dft-root`` → ``static.dft_roots[<dir name>]`` →
    ``static.dft_root`` → the point's own SIAB ``output_dir``. When the
    candidate is a directory that does **not** cover the point, its immediate
    subdirectories are searched once — that is what makes a single
    ``static.dft_root`` pointing at ``project/u_14ve/`` serve every grid point
    (``run_lmax4_rcut9/``, ``run_lmax4_rcut10/``).
    """
    if dft_root is not None:
        base, origin = Path(dft_root).expanduser(), "--dft-root"
    else:
        configured = dft_root_from_input_json(input_json, point)
        if configured is not None:
            origin = ("input.json:static.dft_roots[" + point_dir_name(point) + "]"
                      if (isinstance(_static_of(input_json).get("dft_roots"), dict)
                          and point_dir_name(point) in _static_of(input_json)["dft_roots"])
                      else "input.json:static.dft_root")
            base = configured
        else:
            base, origin = Path(point.output_dir or "."), "the point's output_dir"

    if _point_is_covered(base, point):
        return base, origin

    if base.is_dir():
        for child in sorted(base.iterdir()):
            if child.is_dir() and _point_is_covered(child, point):
                return child, f"{origin} → {child.name}/"

    return base, origin


def choose_point_for_dft_root(summary, dft_root: str | Path):
    """Pick the grid point whose reference DFT is complete under ``dft_root``.

    The reference folders embed the cutoff (``U-dimer-1.89-10au``), so a tree
    prepared for one ``(l_max, r_cut)`` only satisfies that point. Returns
    ``(point, status)`` where ``status`` maps each tried point to a short
    reason — ``(None, {})`` when no point is covered.
    """
    root = Path(dft_root)
    if not root.is_dir():
        return None, {}
    status: dict[str, str] = {}
    for point in getattr(summary, "grid", []):
        if point.r_cut is None:
            continue
        required, optional = job_folders(point.siab_info or {}, point.l_max, point.r_cut)
        ready, details = dft_root_status(root, required + optional)
        label = f"l_max={point.l_max}, r_cut={folder_rcut(point.r_cut)}au"
        status[label] = "complete" if ready else "incomplete"
        if ready:
            return point, status
    return None, status


def generate_final_orbital(
    summary,
    out_dir: str | Path,
    *,
    dft_root: str | Path | None = None,
    run_missing: str = "monomer",
    dry_run: bool = False,
    timeout: int | None = 3600,
    orbgen_command: str | None = None,
    point=None,
    siab_config: str | Path | None = None,
    input_json: str | Path | None = None,
    use_stored_config: bool = False,
    assemble: bool = True,
    redo: bool = False,
    normalise_inputs: bool = True,
) -> dict:
    """Run the SIAB spillage minimisation for one reference-DFT tree.

    ``run_missing`` decides what happens when the reference DFT is incomplete:

    * ``"monomer"`` (default) — proceed only when the *monomer* (the atomic
      initial guess, a one-atom job SIAB runs in minutes) is all that is
      missing; anything bigger is skipped with an explanation;
    * ``"all"`` — let SIAB compute whatever is missing (``--force-final-orbital``);
    * ``"none"`` — only use what is already there.

    Everything else is derived automatically:

    * the DFT tree — ``--dft-root`` → ``static.dft_roots[<point>]`` →
      ``static.dft_root`` → the point's own SIAB ``output_dir``; a candidate
      that does not cover the point is also searched one level down, so one
      ``static.dft_root`` can point at a directory of per-point runs;
    * missing dimer data — assembled from AiiDA when ``assemble=True``
      (``aiida-orbgen fetch-dft`` logic, no DFT);
    * the SIAB config — explicit ``siab_config`` → the config stored on the
      node, unless it is invalid or differs from the current
      ``parameters/`` presets, in which case the presets are re-read (and the
      substitution is reported).

    ``normalise_inputs`` (default True) first makes the reference tree
    acceptable to SIAB's ``DuplicateCheck`` — see
    :func:`aiida_orbgen.utils.report.assemble.normalise_reference_for_siab`.
    Without it a tree assembled from AiiDA looks like new work and ``orbgen``
    would start a real reference DFT instead of using what was collected.

    Returns a dict describing the attempt (``status``, ``files``, ...) which
    the report renders verbatim.
    """
    out_dir = Path(out_dir)
    result: dict[str, Any] = {"status": "skipped", "files": []}

    # ---- 1. grid point ----------------------------------------------------
    pinned = point is not None
    if point is None:
        point = select_best_point(summary)
    if point is None:
        result["message"] = "no grid point available"
        return result
    result["l_max"] = point.l_max
    result["r_cut"] = point.r_cut
    result["point_source"] = (
        f"pinned (OrbgenCalcWorkChain<{point.pk}>)" if pinned
        else f"OrbgenCalcWorkChain<{point.pk}> (auto-selected)"
    )

    # ---- 2. DFT tree (with one-level search for per-point subdirectories) --
    root, root_source = resolve_dft_root(
        point, None, input_json=input_json, dft_root=dft_root
    )
    result["dft_root"] = str(root)
    result["dft_root_source"] = root_source

    from aiida import orm

    calc_node = orm.load_node(point.pk)

    # ---- 3. SIAB config ---------------------------------------------------
    try:
        stored = _build_siab_config(calc_node, point.l_max, point.r_cut)
    except Exception as exc:  # noqa: BLE001
        result["message"] = f"cannot recover the SIAB config: {exc}"
        return result

    config = stored
    config_source = "stored on the node (siab_json)"
    notices: list[str] = []
    if siab_config is not None:
        try:
            with open(siab_config, "r", encoding="utf-8") as handle:
                config = _apply_grid_point(json.load(handle), point.l_max, point.r_cut)
        except Exception as exc:  # noqa: BLE001
            result["message"] = f"cannot read --siab-json {siab_config}: {exc}"
            return result
        config_source = f"--siab-json {siab_config}"
    elif not use_stored_config:
        fresh, reason = load_preset_config(input_json, point.l_max, point.r_cut)
        stored_problem = None
        try:
            from aiida_orbgen.utils.config import validate_siab_config

            validate_siab_config(stored, source="stored siab_json")
        except Exception as exc:  # noqa: BLE001
            stored_problem = str(exc)
        if fresh is not None and (stored_problem or configs_differ(stored, fresh)):
            config = fresh
            config_source = "re-read from input.json + parameters/ presets"
            notices.append(
                ("the stored siab_json is invalid: " + stored_problem)
                if stored_problem
                else "the stored siab_json differs from the current presets"
            )
        elif stored_problem:
            notices.append("the stored siab_json is invalid: " + stored_problem)
        elif fresh is None and reason:
            notices.append(f"kept the stored siab_json ({reason})")

    # Everything belonging to this grid point lands in its own directory, so a
    # report directory stays readable and several points can coexist:
    #   <out_dir>/lmax4_rcut10/{*.orb,*.param,*.png,orbgen_lmax4_rcut10.{json,log}}
    work_dir = out_dir / point_dir_name(point)
    work_dir.mkdir(parents=True, exist_ok=True)
    result["dir"] = str(work_dir)

    config_path = work_dir / f"orbgen_{point_dir_name(point)}.json"
    config_path.write_text(json.dumps(config, indent=4) + "\n", encoding="utf-8")
    result["config"] = str(config_path)
    result["config_source"] = config_source
    if notices:
        result["config_notice"] = "; ".join(notices)

    # ---- 3b. reference expectations + optional assembly from AiiDA --------
    required, optional = job_folders(point.siab_info or {}, point.l_max, point.r_cut)
    expect = expected_nao_by_folder(
        root, required + optional, config, point.l_max, point.r_cut
    )
    result["expected_nao"] = {k: v for k, v in expect.items() if v is not None}
    if assemble:
        from aiida_orbgen.utils.report.assemble import assemble_reference

        try:
            # ``expected_nao`` makes the assembly refetch geometries whose data
            # were computed with a different primitive basis.
            assembled = assemble_reference(calc_node, root, expected_nao=expect)
        except Exception as exc:  # noqa: BLE001
            assembled = {"ok": False, "message": f"assembly failed: {exc}"}
        result["assemble"] = assembled
        if not assembled.get("ok", True):
            notices.append(
                "reference assembly incomplete: "
                + str(assembled.get("message") or assembled.get("summary") or "")
            )
            result["config_notice"] = "; ".join(notices)

    # ---- 3c. SIAB consistency: make sure no reference DFT will be run ------
    # SIAB's DuplicateCheck compares the *whole* INPUT, so a tree assembled from
    # AiiDA (INPUT written by the plugin: ``suffix aiida``, no ``out_wfc_lcao``)
    # is classified as new work and ``rundft`` would start a real SCF — while the
    # reference data we just gathered sits in an ``OUT.*`` directory SIAB is not
    # looking at.  Normalising first (and asserting the job list is empty) closes
    # that trap; ``run_missing="all"`` is the only way to allow a DFT here.
    if normalise_inputs:
        from aiida_orbgen.utils.report.assemble import normalise_reference_for_siab

        allow_run = run_missing == "all"
        try:
            normalised = normalise_reference_for_siab(
                root, config_path, allow_run=allow_run,
                required=required, optional=optional,
            )
        except Exception as exc:  # noqa: BLE001
            # SIAB cannot even read the config: ``orbgen`` aborts in its very
            # first step, so no DFT can start.  Warn instead of refusing — the
            # command below surfaces the real error.
            normalised = {
                "ok": True,
                "checked": False,
                "jobs": [],
                "renamed": {},
                "message": f"cannot check the reference tree against SIAB: {exc}",
            }
        result["normalise"] = normalised
        if normalised.get("renamed"):
            notices.append(
                "renamed reference output dir(s) so SIAB reuses them: "
                + ", ".join(f"{k}: {v}" for k, v in normalised["renamed"].items())
            )
            result["config_notice"] = "; ".join(notices)
        if not normalised.get("ok", False):
            result["status"] = "refused"
            result["message"] = str(normalised.get("message") or "") or (
                "SIAB would run the reference DFT; refusing to start the "
                "spillage (pass --force-final-orbital to allow it)"
            )
            return result

    # ---- 4. reference-DFT completeness ------------------------------------
    if not root.is_dir():
        result["message"] = (
            f"DFT root {root} does not exist on this machine — set "
            f"static.dft_root in input.json or pass --dft-root <dir> "
            f"(it must hold {required or ['<elem>-dimer-*']} + the monomer)"
        )
        return result

    ready, details = dft_root_status(root, required + optional, expect)
    result["dft_status"] = details
    if not ready:
        incomplete = [name for name, state in details.items() if state != "done"]
        # Only the monomer (the one-atom atomic initial guess) may be computed
        # automatically; a full reference DFT needs --force-final-orbital.
        monomer_only = bool(optional) and set(incomplete) <= set(optional)
        allowed = run_missing == "all" or (run_missing == "monomer" and monomer_only)
        if not allowed:
            if run_missing == "monomer":
                why = ("this step only computes the missing monomer by itself; "
                       "pass --force-final-orbital to let SIAB run the missing "
                       "reference DFT")
            else:
                why = "reference DFT runs are disabled (run_missing='none')"
            result["message"] = (
                f"reference DFT incomplete ({len(incomplete)}/{len(details)} "
                f"geometries): "
                + ", ".join(f"{name}: {details[name]}" for name in incomplete[:5])
                + (" …" if len(incomplete) > 5 else "")
                + f" — {why}"
            )
            return result

    # ---- 3d. quarantine stale data SIAB would silently reuse ---------------
    # SIAB's own ``job_done`` only looks for a finished log, so reference data
    # computed with another primitive basis must be moved aside — otherwise
    # ``rundft`` would "skip" it and the spillage would mix two bases (the very
    # failure mode this guard exists for).
    quarantined: dict[str, str] = {}
    for name, state in details.items():
        if state == "done" or not state.startswith("stale"):
            continue
        folder = root / name
        if not folder.exists():
            continue
        target = folder.with_name(f"{folder.name}.stale")
        if target.exists():
            target = folder.with_name(f"{folder.name}.stale-{int(time.time())}")
        # The whole job directory goes aside: SIAB regenerates INPUT/STRU/KPT
        # from the config anyway, and leaving it in place would make both
        # ``job_done`` (finished log) and its suffix detection look at stale data.
        try:
            folder.rename(target)
            quarantined[name] = target.name
        except OSError:
            pass
    if quarantined:
        result["quarantined"] = quarantined

    # ---- 3c. make-style idempotency ---------------------------------------
    # Re-running ``report`` on an unchanged tree must not repeat a multi-minute
    # spillage: if the point's directory already holds orbital files that are
    # newer than the reference DFT they were fitted from, document them.
    work_outputs = sorted(
        path for path in work_dir.iterdir()
        if path.suffix in (".orb", ".param", ".png")
    ) if work_dir.is_dir() else []
    if work_outputs and not (run_missing == "all" or redo):
        try:
            newest_output = max(path.stat().st_mtime for path in work_outputs)
            newest_input = max(
                (path.stat().st_mtime for path in root.glob("*/OUT.*/running_scf.log")),
                default=0.0,
            )
        except OSError:
            newest_output, newest_input = 0.0, 1.0
        if newest_output >= newest_input:
            result["status"] = "already-present"
            result["files"] = [str(path) for path in work_outputs]
            result["message"] = (
                f"{len(work_outputs)} orbital file(s) already up to date "
                f"(older than the reference DFT) — pass --redo-final-orbital to "
                f"recompute"
            )
            return result

    environment = str(config.get("environment") or "").strip().rstrip(";")
    command = orbgen_command or "orbgen"
    log_path = work_dir / f"orbgen_lmax{point.l_max}_rcut{point.r_cut:g}.log"
    # ``cd … && { env; } && orbgen …`` — the brace group keeps the exports in
    # the current shell (a subshell would lose PATH) while still aborting the
    # whole chain when ``cd`` fails.
    parts = [f"cd {shlex.quote(str(root))}"]
    if environment:
        parts.append(f"{{ {environment}; }}")
    parts.append(
        f"{command} -i {shlex.quote(str(config_path))} "
        f"-o {shlex.quote(str(work_dir))} "
        f"-l {shlex.quote(str(log_path))}"
    )
    shell_command = " && ".join(parts)
    result["command"] = shell_command
    result["log"] = str(log_path)

    if dry_run:
        result["status"] = "dry-run"
        result["message"] = "command not executed (--dry-run)"
        return result

    if shutil.which(command) is None and "PATH" not in environment:
        result["message"] = (
            f"'{command}' is not on PATH and the preset carries no "
            f"'environment' block; pass --orbgen-command <path>"
        )
        return result

    result["status"] = "running"
    # A re-run into the same directory overwrites the previous .orb/.param/.png,
    # so a name-diff would wrongly report "empty": compare modification times.
    started = time.time()
    try:
        completed = subprocess.run(
            shell_command,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["message"] = f"orbgen did not finish within {timeout}s (see {log_path})"
        return result

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(f"\n$ {shell_command}\n")
        handle.write(completed.stdout or "")
        handle.write(completed.stderr or "")

    produced = sorted(
        path for path in work_dir.iterdir()
        if path.is_file()
        and path.suffix in (".orb", ".param", ".png")
        and path.stat().st_mtime >= started - 1
    )
    result["files"] = [str(path) for path in produced]

    # ---- 5. 校验产物内容 (不能只看扩展名) --------------------------------
    # 曾经出现过文件名写着 ``4s3p2d2f1g`` 而文件头 ``Number of Gorbital--> 0``
    # 的情况 (vloc_aux 写在 orbitals[i] 顶层被 SIAB 忽略), 只按扩展名是查不出来
    # 的, 所以这里把每个 .orb 读回来与请求的 nzeta 逐 l 对比.
    schemes = [orb.get("nzeta") for orb in (config.get("orbitals") or [])
               if isinstance(orb.get("nzeta"), (list, tuple))]
    checked = [
        validate_orbital(path, schemes)
        for path in produced if path.suffix == ".orb"
    ] if schemes else []
    if checked:
        result["validated"] = checked
        bad = [item for item in checked if not item["ok"]]
        if bad:
            result["status"] = "invalid"
            result["message"] = "; ".join(
                f"{Path(item['file']).name}: {item.get('reason', 'invalid')}"
                for item in bad
            )

    values = spillage_values(log_path)
    if values:
        result["spillage"] = values
        result["spillage_last"] = values[-1]

    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()
        result["status"] = "failed"
        result["message"] = (
            f"orbgen exited {completed.returncode}"
            + (f"; last output: {tail[-1]}" if tail else "")
        )
        return result

    if result["status"] not in ("invalid",):
        result["status"] = "ok" if produced else "empty"
    if result["status"] == "empty":
        result["message"] = (
            f"orbgen finished but produced no .orb/.param in {out_dir}; "
            f"check {log_path}"
        )
    return result
