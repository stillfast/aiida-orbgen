"""Everything that talks to SIAB (ABACUS-CSW-NAO) for one grid point.

Two responsibilities live here, and nowhere else:

* :func:`run_siab_pipeline` — run ``generate_all_from_json`` with the grid point
  applied, and report the generated job list (a ``@calcfunction``, so it is
  cached and the workchain can reuse it);
* :func:`build_abacus_child_inputs` — turn one generated job folder plus an
  ``abacus.json``-shaped config into the inputs of a single ``abacus.base``
  child (INPUT merge order, PW/LCAO key surgery, STRU → ``StructureData``,
  k-points, scheduler options).

Keeping them out of the WorkChain module is what makes the WorkChain readable:
the classes in :mod:`aiida_orbgen.workflows.batch` only orchestrate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from aiida import orm
from aiida.engine import calcfunction
from aiida.orm import (
    Dict,
    Int,
    Float,
    KpointsData,
    SinglefileData,
    Str,
    StructureData,
    load_code,
)

from aiida_orbgen.interfaces import (
    apply_grid_point,
    generate_all_from_json,
    params_stru_to_ase,
    parse_incar,
)
from aiida_orbgen.static.defaults import (
    DEFAULT_CODE_LABEL,
    DEFAULT_MAX_MEMORY_KB,
    DEFAULT_NUM_MPI,
    DEFAULT_QUEUE_NAME,
    DEFAULT_WALLCLOCK_SECONDS,
    apply_input_overrides,
)

# abacuslite 读 STRU → dict
from abacuslite.io.generalio import read_stru

__all__ = [
    "run_siab_pipeline",
    "build_abacus_child_inputs",
    "n_atoms_from_stru",
]


@calcfunction
def run_siab_pipeline(
    siab_json: SinglefileData,
    output_dir: Str,
    lmax: Int,
    rcut: Float,
) -> Dict:
    """在 worker 节点跑 ``generate_all_from_json``, 返回任务列表。

    Parameters
    ----------
    siab_json : SinglefileData
        pbe_orbgen.json 文件
    output_dir : Str
        SIAB 生成目录
    lmax : Int
        最高角动量 (强制覆盖到 orbgen.json)
    rcut : Float
        截断半径 (强制覆盖到 orbgen.json)

    Returns
    -------
    Dict
        ``{"nsw", "nsw_filename", "pertmags", "dft", "upf_path", "orb_path",
           "family_label", "lmax", "rcut", "config_path"}``

    Notes
    -----
    The grid-point-overridden config is written to ``<output_dir>/siab_config.json``
    rather than to a temporary file: it is the input SIAB actually ran with (so
    it is worth keeping), and it gives ``resolve_paths_from_json`` a stable base
    directory.  Previously the base was a ``/tmp`` tempfile, so a relative
    ``pseudo_dir`` such as the shipped examples' ``"./U.pbe-n-nc.upf"`` resolved
    to ``/tmp/.../U.pbe-n-nc.upf`` and the pseudo-family build failed far away
    from the cause.
    """
    import json as _json
    import os as _os

    # 1) 读出 JSON 内容并应用 (l_max, r_cut) —— 覆盖逻辑只有一份实现
    content = siab_json.get_content()
    if isinstance(content, bytes):
        content = content.decode("utf-8")
    cfg = apply_grid_point(_json.loads(content), int(lmax.value), float(rcut.value))

    run_dir = _os.path.abspath(output_dir.value)
    _os.makedirs(run_dir, exist_ok=True)
    local_json = _os.path.join(run_dir, "siab_config.json")
    with open(local_json, "w", encoding="utf-8") as f:
        _json.dump(cfg, f, indent=2)

    # 2) 跑 SIAB pipeline
    result = generate_all_from_json(local_json, output_root=run_dir)

    # 3) 解析 UPF / family label
    from aiida_orbgen.calculations.pseudo_family import (
        resolve_paths_from_json,
    )

    paths = resolve_paths_from_json(local_json, result)

    return Dict(dict={
        "nsw": result["nsw"],
        "nsw_filename": result["nsw_filename"],
        "pertmags": result["pertmags"],
        "dft": result["dft"],
        "upf_path": paths["upf_path"],
        "orb_path": paths["orb_path"],
        "family_label": paths["family_label"],
        "config_path": local_json,
        "lmax": int(lmax.value),
        "rcut": float(rcut.value),
    })


# ===========================================================================
#  单个 abacus.base 任务 inputs 构造
# ===========================================================================


def build_abacus_child_inputs(
    dft_entry: Dict[str, Any],
    *,
    basis: str,                      # "pw" or "lcao_nsw"
    code_label: str,
    family_label: str,
    parameters: Dict[str, Any],       # 来自 abacus.json: {"input": {...}}
    queue_name: str,
    num_mpi: int,
    wallclock: int,
    max_memory_kb: int,
) -> Dict[str, Any]:
    """构造一个 ``abacus.base`` workchain 的 inputs。

    Parameters
    ----------
    dft_entry : dict
        ``generate_all_from_json()`` 返回的 ``dft[i]`` 项
    basis : str
        ``"pw"`` (平面波) 或 ``"lcao_nsw"`` (数值原子轨道, nsw=原始 SIAB)
    code_label, family_label : str
    parameters : dict
        来自 abacus.json 的 ``parameters`` 字段, 含 ``"input"`` 子 dict
        (强制覆盖 SIAB 生成的 INPUT)
    queue_name, num_mpi, wallclock, max_memory_kb : scheduler 参数
    """
    input_path = dft_entry["input"]
    stru_path = dft_entry["stru"]

    # 1) INPUT → Dict (过滤 AiiDA 托管 key + apply overrides)
    siab_input = parse_incar(input_path)
    merged = apply_input_overrides(siab_input)
    # abacus.json 的 parameters.input 优先级最高
    user_input = parameters.get("input", {}) if parameters else {}
    merged.update(user_input)

    # 2) 根据 basis 切 basis_type 和 ks_solver
    if basis == "pw":
        merged["basis_type"] = "pw"
        # PW basis 不支持 scalapack_gvx，需要切换到 PW 支持的求解器
        if merged.get("ks_solver") == "scalapack_gvx":
            merged["ks_solver"] = "dav"
        # PW basis 不支持 out_wfc_lcao 参数
        merged.pop("out_wfc_lcao", None)
    elif basis == "lcao_nsw":
        merged["basis_type"] = "lcao"
        # LCAO 时需要 orbital_dir (AiiDA 从 pseudo_family 注入)
    else:
        raise ValueError(f"Unknown basis: {basis!r}")

    params = merged

    # 3) STRU → ASE Atoms → StructureData
    params_stru = read_stru(stru_path)
    ase_atoms = params_stru_to_ase(params_stru)
    structure = StructureData(ase=ase_atoms)

    # 4) KPT: Gamma-only 1 1 1
    kp = KpointsData()
    kp.set_kpoints_mesh([1, 1, 1], offset=[0, 0, 0])

    # 5) metadata.options
    options = {
        "resources": {
            "num_machines": 1,
            "num_mpiprocs_per_machine": num_mpi,
            "tot_num_mpiprocs": num_mpi,
        },
        "max_wallclock_seconds": wallclock,
        "max_memory_kb": max_memory_kb,
        "queue_name": queue_name,
        "withmpi": True,
    }

    return {
        "abacus": {
            "code": load_code(code_label),
            "parameters": Dict(dict={"input": params}),
            "structure": structure,
            "metadata": {
                "options": options,
                "label": f"abacus-{dft_entry['folder']}-{basis}",
                "description": (
                    f"OrbgenCalcWorkChain | {dft_entry['folder']} | "
                    f"basis={basis} | pert={dft_entry.get('pert')}"
                ),
            },
        },
        "kpoints": kp,
        "pseudo_family": Str(family_label),
    }


def n_atoms_from_stru(stru_path: str | None, report=None) -> int:
    """Atom count of a generated STRU (needed for the per-atom ΔE).

    Falls back to 1 (per-atom normalisation effectively disabled) and warns
    through ``report`` when the file is missing or unreadable, instead of
    raising inside a step.
    """
    def _warn(message: str) -> None:
        if report is not None:
            report(message)

    if not stru_path or not Path(stru_path).is_file():
        _warn(f"  WARNING: STRU file not found at {stru_path!r}, "
              f"n_atoms=1 fallback (per-atom disabled)")
        return 1
    try:
        params_stru = read_stru(stru_path)
    except Exception as exc:  # noqa: BLE001 - unreadable STRU must not abort a step
        _warn(f"  WARNING: failed to parse STRU {stru_path!r}: {exc}; "
              f"n_atoms=1 fallback")
        return 1
    species = params_stru.get("species", []) if isinstance(params_stru, dict) else []
    total = sum(int(sp.get("natom", 0)) for sp in species if isinstance(sp, dict))
    if total <= 0:
        total = sum(len(sp.get("atom", [])) for sp in species if isinstance(sp, dict))
    return total if total > 0 else 1
