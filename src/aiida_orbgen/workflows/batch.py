"""
OrbgenCalcWorkChain — 单个 (l_max, r_cut) 组合下的批量 ABACUS 提交

对应 flowchart 中 "OrbgenCalc: 根据 orbgen.json + abacus.json, 运行特定 l_max 和 r_cut 的任务"。

每个 OrbgenCalcWorkChain:
  1. 接收 orbgen.json (含 element / pseudo_dir / geoms / ...),
     abacus.json (含 basis / parameters.input / tolerance),
     以及固定的 (l_max, r_cut)
  2. 调 SIAB pipeline (calcfunction, 缓存) 生成 NSW + 5 个结构的 INPUT/STRU
  3. 对每个结构 × {PW, LCAO:nsw} = 10 个 abacus.base 任务
  4. 收集能量, 输出 Dict

入口点
------
* ``orbgen.calc`` — 在 ``pyproject.toml`` 注册
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from aiida import orm
from aiida.engine import WorkChain, ExitCode, calcfunction, append_, if_, while_
from aiida.orm import (
    Bool,
    Dict,
    Float,
    Int,
    KpointsData,
    List,
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
from aiida_orbgen.static.json_inputs import with_default_abacus
from aiida_orbgen.workflows._grid import (
    GridEntry,
    build_cartesian_grid,
    build_multi_json_grid,
    has_pending_iterative,
    work_dir_name,
)

# abacuslite 读 STRU → dict
from abacuslite.io.generalio import read_stru

__all__ = [
    "OrbgenCalcWorkChain",
    "OrbgenGridSearchWorkChain",
    "run_siab_pipeline",
    "build_abacus_child_inputs",
]


# ===========================================================================
#  CalcFunction: SIAB pipeline (AiiDA 自动缓存)
# ===========================================================================


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


# ===========================================================================
#  CalcFunction: 从 abacus.base outputs 提取 energy
# ===========================================================================


@calcfunction
def create_energies_dict(d: "Dict|dict|List") -> Dict:
    """创建一个 energies 结果的 Dict 对象（用于 WorkChain 输出）。
    
    Parameters
    ----------
    d : Dict, dict, or List
        包含 energies 结果的字典、AiiDA Dict 或 AiiDA List（包含能量数据）
        
    Returns
    -------
    Dict
        AiiDA Dict 对象
    """
    # 如果是 List，说明是从子节点收集的原始数据，需要先计算
    if hasattr(d, "__iter__") and not hasattr(d, "get_dict"):
        # d 是一个 AiiDA List，计算能量差
        energies_by_basis = {}
        for item in d:
            item_dict = item.get_dict() if hasattr(item, "get_dict") else dict(item)
            basis = item_dict.get("basis_type", "unknown")
            energy = item_dict.get("energy", item_dict.get("E_total"))
            folder = item_dict.get("folder", "unknown")
            pert = item_dict.get("pert")
            if energy is None:
                continue
            energies_by_basis.setdefault(basis, []).append({
                "folder": folder,
                "energy": float(energy),
                "pert": pert,
            })
        
        delta_per_struct = []
        if "pw" in energies_by_basis and "lcao" in energies_by_basis:
            pw_by_folder = {e["folder"]: e for e in energies_by_basis["pw"]}
            for e_lcao in energies_by_basis["lcao"]:
                folder = e_lcao["folder"]
                e_pw_entry = pw_by_folder.get(folder)
                if e_pw_entry is None:
                    continue
                dE = abs(e_lcao["energy"] - e_pw_entry["energy"])
                delta_per_struct.append({
                    "folder": folder,
                    "E_pw": e_pw_entry["energy"],
                    "E_lcao_nsw": e_lcao["energy"],
                    "dE": dE,
                })
        
        delta_max = max((x["dE"] for x in delta_per_struct), default=0.0)
        d_dict = {
            "energies": energies_by_basis,
            "delta_E_per_struct": delta_per_struct,
            "delta_E_max_eV": float(delta_max),
            "delta_E_max_meV": float(delta_max * 1000.0),
        }
    elif hasattr(d, "get_dict"):
        # 如果是 AiiDA Dict，直接获取字典
        d_dict = d.get_dict()
    else:
        # 如果是普通字典，直接使用
        d_dict = dict(d)
    
    return Dict(dict=d_dict)


@calcfunction
def create_final_results(
    l_max_val,
    r_cut_val,
    results_list,
) -> Dict:
    """创建最终结果的 Dict 对象（用于 WorkChain 输出）。

    Parameters
    ----------
    l_max_val : int or float
        最大角动量
    r_cut_val : int or float
        截断半径
    results_list : list
        子节点信息的列表

    Returns
    -------
    Dict
        AiiDA Dict 对象
    """
    # 提取值（如果是 AiiDA Data 类型）
    if hasattr(l_max_val, "value"):
        l_max_val = l_max_val.value
    if hasattr(r_cut_val, "value"):
        r_cut_val = r_cut_val.value
    # results_list 应该是 AiiDA List 节点（包含原始 dicts）
    # AiiDA 引擎会自动将传入的 list 包装为 List 节点
    if hasattr(results_list, "get_list"):
        children = list(results_list.get_list())
    elif isinstance(results_list, (list, tuple)):
        children = list(results_list)
    else:
        children = results_list

    return Dict(dict={
        "l_max": int(l_max_val) if isinstance(l_max_val, (int, float)) else l_max_val,
        "r_cut": float(r_cut_val) if isinstance(r_cut_val, (int, float)) else r_cut_val,
        "children": children,
    })


@calcfunction
def create_grid_all_results(
    grid_results_list,
    tolerance_meV,
    search_strategy,
) -> Dict:
    """创建 OrbgenGridSearchWorkChain 的 all_results Dict.

    Parameters
    ----------
    grid_results_list : list
        每项是 {l_max, r_cut, calc_pk, exit_status, is_finished_ok, ...} 字典
    tolerance_meV : float
    search_strategy : str

    Returns
    -------
    Dict
    """
    tol = tolerance_meV.value if hasattr(tolerance_meV, "value") else float(tolerance_meV)
    strat = search_strategy.value if hasattr(search_strategy, "value") else str(search_strategy)
    if hasattr(grid_results_list, "get_list"):
        grid = list(grid_results_list.get_list())
    elif isinstance(grid_results_list, (list, tuple)):
        grid = list(grid_results_list)
    else:
        grid = grid_results_list
    return Dict(dict={
        "grid": grid,
        "tolerance_meV": float(tol),
        "search_strategy": str(strat),
    })


@calcfunction
def create_grid_summary(
    n_grid_points,
    n_tried,
    n_passed,
    n_failed,
    tolerance_meV,
    search_strategy,
    best_l_max=None,
    best_r_cut=None,
    best_delta_per_atom_meV=None,
    best_calc_pk=None,
) -> Dict:
    """创建 OrbgenGridSearchWorkChain 的 grid_summary Dict."""
    def get_val(x, conv=None):
        if x is None:
            return None
        if hasattr(x, "value"):
            return x.value
        return conv(x) if conv else x

    return Dict(dict={
        "n_grid_points": int(get_val(n_grid_points, int)),
        "n_tried": int(get_val(n_tried, int)),
        "n_passed": int(get_val(n_passed, int)),
        "n_failed": int(get_val(n_failed, int)),
        "best_l_max": get_val(best_l_max),
        "best_r_cut": get_val(best_r_cut),
        "best_delta_per_atom_meV": (
            float(get_val(best_delta_per_atom_meV, float))
            if get_val(best_delta_per_atom_meV) is not None
            else None
        ),
        "best_calc_pk": get_val(best_calc_pk),
        "tolerance_meV": float(get_val(tolerance_meV, float)),
        "search_strategy": str(get_val(search_strategy)),
    })


# ===========================================================================
#  OrbgenCalcWorkChain
# ===========================================================================


class OrbgenCalcWorkChain(WorkChain):
    """单个 (l_max, r_cut) 组合下, 跑 PW + LCAO:nsw 两组基组 × 5 结构 = 10 个任务。

    Inputs
    ------
    siab_json : SinglefileData
        pbe_orbgen.json
    abacus_config : Dict
        abacus.json 内容 (含 basis / input_overrides / tolerance / scheduler)
    output_dir : Str
        SIAB 生成目录
    l_max : Int
        最高角动量
    r_cut : Float
        截断半径 (Å)
    code_label : Str, optional
    family_label : Str, optional
    build_family : Bool, optional
    only : Int, optional
    max_iterations : Int, optional

    Outputs
    -------
    siab_info : Dict
        SIAB pipeline 结果
    results : Dict
        每任务的 PK + 状态
    energies : Dict
        能量 + ΔE_max (basis_type 比较)

    Entry point
    -----------
    ``orbgen.batch``
    """

    _child_workchain_entry_point = "abacus.base"

    # ------------------------------------------------------------------
    # define
    # ------------------------------------------------------------------

    @classmethod
    def define(cls, spec):
        super().define(spec)

        # ---- JSON inputs ----
        spec.input("siab_json", valid_type=SinglefileData,
                   help="pbe_orbgen.json (SIAB config, contains UPF path).")
        spec.input("abacus_config", valid_type=Dict,
                   help="abacus.json (basis / input_overrides / tolerance / scheduler).")

        # ---- 固定的 (l_max, r_cut) ----
        spec.input("l_max", valid_type=Int,
                   help="最高角动量 (orbit gen 参数).")
        spec.input("r_cut", valid_type=Float,
                   help="截断半径, Å (orbit gen 参数).")

        # ---- 路径 ----
        spec.input("output_dir", valid_type=Str,
                   help="SIAB 生成目录 (worker 可写).")

        # ---- AiiDA / scheduler ----
        spec.input("code_label", valid_type=Str, required=False,
                   help="AiiDA code label (default: from static.defaults).")
        spec.input("family_label", valid_type=Str, required=False,
                   help="AiiDA pseudo family label (default: auto-infer).")
        spec.input("max_iterations", valid_type=Int, required=False,
                   help="Max abacus.base workchain retry.")

        # ---- 行为控制 ----
        spec.input("build_family", valid_type=Bool, required=False,
                   default=lambda: orm.Bool(True),
                   help="Auto-build pseudo_family if missing.")
        spec.input("only", valid_type=Int, required=False,
                   help="Only run N-th task (0-indexed, debug).")
        spec.input("dry_run", valid_type=Bool, required=False,
                   default=lambda: orm.Bool(False),
                   help="If True, only print, don't submit.")

        # ---- outputs ----
        spec.output("siab_info", valid_type=Dict, required=False,
                    help="SIAB pipeline result (nsw, family_label, dft, ...).")
        spec.output("results", valid_type=Dict, required=False,
                    help="Per-task PK + status.")
        spec.output("energies", valid_type=Dict, required=False,
                    help="Extracted energies + ΔE_max (eV, meV).")

        # ---- exit codes ----
        spec.exit_code(401, "ERROR_INVALID_ABACUS_CONFIG",
                       message="abacus_config invalid")
        spec.exit_code(402, "ERROR_NO_BASIS",
                       message="abacus_config.basis is empty")
        spec.exit_code(403, "ERROR_SIAB_FAILED",
                       message="SIAB pipeline failed")
        spec.exit_code(404, "ERROR_NO_DFT_JOBS",
                       message="SIAB produced 0 DFT jobs")
        spec.exit_code(405, "ERROR_BUILD_INPUTS",
                       message="Failed to build child inputs")
        spec.exit_code(301, "WARNING_PARTIAL_FAILURE",
                       message="Some children failed")
        spec.exit_code(302, "ERROR_ALL_FAILED",
                       message="All children failed")
        spec.exit_code(303, "WARNING_ENERGY_EXTRACT_FAILED",
                       message="Could not extract energies from outputs")
        spec.exit_code(304, "WARNING_TOLERANCE_EXCEEDED",
                       message="max |E_lcao_nsw - E_pw| exceeds tolerance (meV)")

        # ---- outline ----
        spec.outline(
            cls.validate_inputs,
            cls.run_siab_pipeline_step,
            cls.ensure_pseudo_family,
            cls.submit_children,
            cls.inspect_children,
            cls.extract_energies_step,
            cls.finalize,
        )

    # ------------------------------------------------------------------
    # dry run
    # ------------------------------------------------------------------

    @property
    def _dry_run(self) -> bool:
        """``dry_run`` 输入 (只出计划, 不落地任何东西).

        注意 AiiDA 的语义: step 返回 ``ExitCode(0)`` **不能**终止 outline --
        ``aiida/engine/processes/workchains/workchain.py`` 把 status 0 的
        ExitCode 映射成 ``None`` 后继续执行下一步. 所以每个 step 必须自己检查
        这个标志; 本类里曾经只有 submit/inspect/extract/finalize 检查, Step 1
        (SIAB pipeline) 和 Step 1.5 (建 pseudo family) 没有 -- dry-run 依然会
        启动 SIAB 子进程并在数据库里建 family.
        """
        return bool(self.inputs.get("dry_run", orm.Bool(False)).value)

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------

    def validate_inputs(self):
        """Check abacus_config has basis, etc."""
        cfg = self.inputs.abacus_config.get_dict()
        if not cfg:
            return self.exit_codes.ERROR_INVALID_ABACUS_CONFIG
        basis = cfg.get("basis", [])
        if not basis:
            return self.exit_codes.ERROR_NO_BASIS
        for b in basis:
            if b not in ("pw", "lcao_nsw"):
                self.report(f"ERROR: unknown basis {b!r}")
                return self.exit_codes.ERROR_INVALID_ABACUS_CONFIG
        # 应用 default
        self.ctx.abacus_cfg = with_default_abacus(cfg)
        self.report(
            f"abacus.json: basis={self.ctx.abacus_cfg['basis']}, "
            f"tolerance_meV={self.ctx.abacus_cfg['tolerance_meV']}"
        )
        return None

    # ------------------------------------------------------------------
    # Step 1: SIAB pipeline
    # ------------------------------------------------------------------

    def run_siab_pipeline_step(self):
        """Run ``generate_all_from_json`` with l_max/r_cut override."""
        self.report(
            f"Step 1: SIAB pipeline with l_max={self.inputs.l_max.value}, "
            f"r_cut={self.inputs.r_cut.value}"
        )
        if self._dry_run:
            # A dry run must not touch the filesystem: this step used to run
            # unconditionally, so "dry" runs still launched SIAB.
            self.report("  [DRY-RUN] SIAB pipeline not launched")
            return
        try:
            siab_result = run_siab_pipeline(
                self.inputs.siab_json,
                self.inputs.output_dir,
                self.inputs.l_max,
                self.inputs.r_cut,
            )
        except Exception as exc:
            self.report(f"ERROR: SIAB pipeline failed: {exc}")
            return self.exit_codes.ERROR_SIAB_FAILED

        info = siab_result.get_dict()
        n_dft = len(info.get("dft", []))
        if n_dft == 0:
            return self.exit_codes.ERROR_NO_DFT_JOBS

        self.report(
            f"  NSW: {info['nsw']}\n"
            f"  UPF: {info['upf_path']}\n"
            f"  ORB: {info['orb_path']}\n"
            f"  family: {info['family_label']}\n"
            f"  pertmags: {info['pertmags']}\n"
            f"  -> {n_dft} DFT job(s) generated"
        )
        self.ctx.siab_info = info
        self.out("siab_info", siab_result)

    # ------------------------------------------------------------------
    # Step 1.5: pseudo_family
    # ------------------------------------------------------------------

    def ensure_pseudo_family(self):
        from aiida_orbgen.calculations.pseudo_family import ensure_pseudo_family

        if self._dry_run:
            # Also a database side effect: creating a pseudo family is not
            # something a dry run may do.
            self.report("Step 1.5: [DRY-RUN] pseudo family not registered")
            return
        if (
            "build_family" in self.inputs
            and not self.inputs.build_family.value
        ):
            return
        family_label = self.ctx.siab_info["family_label"]
        upf_path = self.ctx.siab_info["upf_path"]
        orb_path = self.ctx.siab_info["orb_path"]
        self.report(f"Step 1.5: ensure_pseudo_family('{family_label}') ...")
        try:
            ensure_pseudo_family(
                upf_path, orb_path, family_label,
                build_if_missing=True,
                description="Built by OrbgenCalcWorkChain",
            )
        except Exception as exc:
            self.report(f"WARNING: ensure_pseudo_family failed: {exc}")

    # ------------------------------------------------------------------
    # Step 2: 提交子任务 (basis × dft_entry)
    # ------------------------------------------------------------------

    def submit_children(self):
        """对每个 (basis, dft_entry) 提交一个 abacus.base。"""
        if self._dry_run:
            # The plan is only known after the SIAB step, which a dry run
            # skips -- so report what the inputs describe instead.
            self.report(
                f"Step 2: [DRY-RUN] would submit "
                f"{len(self.ctx.abacus_cfg['basis'])} basis × N structures "
                f"(l_max={self.inputs.l_max.value}, "
                f"r_cut={self.inputs.r_cut.value})"
            )
            self.ctx.children_info = []
            return
        if not hasattr(self.ctx, "siab_info"):
            return self.exit_codes.ERROR_SIAB_FAILED

        dft_list = self.ctx.siab_info["dft"]
        only = self.inputs.get("only")
        if only is not None and only.value is not None:
            idx = only.value
            if 0 <= idx < len(dft_list):
                dft_list = [dft_list[idx]]
                self.report(f"  [only={idx}] 只跑第 {idx} 个任务")
            else:
                self.report(f"  WARNING: only={idx} 超出范围")

        basis_list = self.ctx.abacus_cfg["basis"]
        self.report(
            f"Step 2: 提交 {len(dft_list)} 结构 × {len(basis_list)} basis "
            f"= {len(dft_list) * len(basis_list)} 个 abacus.base"
        )

        # 从 abacus 键读取配置
        abacus_config = self.ctx.abacus_cfg.get("abacus", {})
        metadata_options = abacus_config.get("metadata", {}).get("options", {})
        
        queue_name = metadata_options.get("queue_name", DEFAULT_QUEUE_NAME)
        num_mpi = metadata_options.get("num_mpiprocs_per_machine", DEFAULT_NUM_MPI)
        if num_mpi is None:
            num_mpi = metadata_options.get("num_mpi", DEFAULT_NUM_MPI)
        wallclock = metadata_options.get("max_wallclock_seconds", DEFAULT_WALLCLOCK_SECONDS)
        max_memory_kb = metadata_options.get("max_memory_kb", DEFAULT_MAX_MEMORY_KB)
        code_label = (
            self.inputs.get("code_label").value
            if "code_label" in self.inputs
            else abacus_config.get("code", DEFAULT_CODE_LABEL)
        )
        family_label = (
            self.inputs.get("family_label").value
            if "family_label" in self.inputs
            else self.ctx.siab_info["family_label"]
        )
        input_overrides = abacus_config.get("parameters", {})

        self.ctx.children_info = []

        for dft_entry in dft_list:
            for basis in basis_list:
                try:
                    inputs = build_abacus_child_inputs(
                        dft_entry,
                        basis=basis,
                        code_label=code_label,
                        family_label=family_label,
                        parameters=input_overrides,
                        queue_name=queue_name,
                        num_mpi=num_mpi,
                        wallclock=wallclock,
                        max_memory_kb=max_memory_kb,
                    )
                except Exception as exc:
                    self.report(
                        f"  build_inputs FAILED for {dft_entry['folder']} "
                        f"| {basis}: {exc}"
                    )
                    continue

                from aiida.plugins import WorkflowFactory
                child_cls = WorkflowFactory(self._child_workchain_entry_point)

                # 读取 STRU 计算原子数 (用于 per-atom 能量归一化)
                n_atoms = self._read_n_atoms_from_stru(dft_entry)

                running = self.submit(child_cls, **inputs)
                running.base.extras.set_many({
                    "task": dft_entry["folder"],
                    "basis": basis,
                    "l_max": str(self.inputs.l_max.value),
                    "r_cut": str(self.inputs.r_cut.value),
                    "n_atoms": str(n_atoms),
                })
                self.ctx.children_info.append({
                    "task": dft_entry["folder"],
                    "basis": basis,
                    "node": running,
                    "n_atoms": n_atoms,
                })
                self.to_context(children=append_(running))

    # ------------------------------------------------------------------
    # Step 3: 等待子任务完成
    # ------------------------------------------------------------------

    def _read_n_atoms_from_stru(self, dft_entry: dict) -> int:
        """从 STRU 文件读取原子数 (用于 per-atom 能量归一化)."""
        stru_path = dft_entry.get("stru")
        if not stru_path or not Path(stru_path).is_file():
            self.report(
                f"  WARNING: STRU file not found at {stru_path!r}, "
                f"n_atoms=1 fallback (per-atom disabled)"
            )
            return 1
        try:
            params_stru = read_stru(stru_path)
        except Exception as exc:
            self.report(
                f"  WARNING: failed to parse STRU {stru_path!r}: {exc}; "
                f"n_atoms=1 fallback"
            )
            return 1
        species = params_stru.get("species", []) if isinstance(params_stru, dict) else []
        n = sum(int(sp.get("natom", 0)) for sp in species if isinstance(sp, dict))
        if n <= 0:
            # 回退到 atom 列表长度
            n = sum(
                len(sp.get("atom", [])) for sp in species if isinstance(sp, dict)
            )
        return n if n > 0 else 1

    def inspect_children(self):
        info = self.ctx.children_info
        # 304 (tolerance exceeded) 视为软成功, 不算 failure
        n_ok = sum(
            1 for i in info
            if i["node"].is_finished_ok or i["node"].exit_status == 304
        )
        n_fail = len(info) - n_ok
        self.report(f"  Children: {n_ok} OK, {n_fail} failed (of {len(info)})")
        return None  # 继续 extract_energies

    # ------------------------------------------------------------------
    # Step 4: 提取能量
    # ------------------------------------------------------------------

    def _verify_stru_orbital(self, node, expected_filename: str) -> bool:
        """检查 AbacusCalculation 的 STRU 中 NUMERICAL_ORBITAL 行是否引用
        真实的 orbital 文件 (而不是引用 f-only 的 stub 文件).

        这是 aiida-abacus STRU 生成的 workaround: 防止当 bug 导致 STRU
        引用错误的 orbital 文件时,返回假数据.

        Returns
        -------
        bool
            True if STRU references the correct file, False otherwise.
        """
        # 通过 process_type 判定 (AiiDA 把 plugin 注册成 aiida.calculations:abacus.abacus)
        if not hasattr(node, 'process_type'):
            return True
        if 'abacus' not in str(node.process_type).lower():
            return True
        try:
            content = node.base.repository.get_object_content('STRU')
        except Exception as exc:
            self.report(
                f"  WARNING: 无法读 STRU (PK {node.pk}): {exc}; 跳过验证"
            )
            return True
        if isinstance(content, bytes):
            content = content.decode("utf-8")
        for line in content.splitlines():
            if line.strip() == "NUMERICAL_ORBITAL":
                continue
            if line.strip().endswith('.orb'):
                actual = line.strip()
                if actual != expected_filename:
                    self.report(
                        f"  ✗ STRU orbital mismatch (PK {node.pk}):\n"
                        f"      STRU 引用 : {actual}\n"
                        f"      实际文件  : {expected_filename}\n"
                        f"      这通常意味着 aiida-abacus 的 STRU 生成有 bug, "
                        f"ABACUS 会找不到 STRU 引用的轨道文件并 fallback 到错误的能量.\n"
                        f"      跳过此 lcao 结果 (不返回假数据)."
                    )
                    return False
                return True
        # 没找到 NUMERICAL_ORBITAL 段 (pw 计算)
        return True

    def extract_energies_step(self):
        # 收集所有成功的子节点的 misc (total_energy)
        outputs = []
        for idx, item in enumerate(self.ctx.children_info):
            node = item["node"]
            if not node.is_finished_ok:
                # exit_status == 304 (WARNING_TOLERANCE_EXCEEDED) 视为软成功:
                # 子任务跑完了 SCF + energy extract, 只是 lcao 精度没达标,
                # 仍然有合法的 misc.total_energy, 继续提取.
                if node.exit_status != 304:
                    continue
            # 【新增】对 lcao 计算, 验证 STRU 引用的 orbital 文件名是否正确
            # 防止 aiida-abacus STRU 生成的 bug 导致假数据
            if str(item.get("basis", "")) == "lcao_nsw":
                # 找到 AbacusCalculation 节点
                abacus_calc = None
                if hasattr(node, 'called') and node.called:
                    for sub in node.called:
                        type_name = type(sub).__name__
                        if 'AbacusCalculation' in type_name or 'abacus' in str(getattr(sub, 'process_type', '')).lower():
                            abacus_calc = sub
                            break
                if abacus_calc is not None:
                    # 找实际的 orbital 文件名 (从 pseudos['U'].filename_second)
                    actual_filename = None
                    if hasattr(abacus_calc, 'inputs') and 'pseudos' in abacus_calc.inputs:
                        u_pseudo = abacus_calc.inputs.pseudos.get('U')
                        if u_pseudo:
                            actual_filename = u_pseudo.base.attributes.get('filename_second')
                    if actual_filename and not self._verify_stru_orbital(
                        abacus_calc, actual_filename
                    ):
                        continue  # 跳过, 不提取假数据
            # AbacusBaseWorkChain 输出的是 misc，不是 output_parameters
            misc = node.outputs.misc if hasattr(node.outputs, "misc") else None
            if misc is None:
                self.report(f"  WARNING: node {node.pk} has no misc output")
                continue
            # 从 misc 中提取 total_energy
            d = misc.get_dict()
            total_energy = d.get("total_energy")
            if total_energy is None:
                self.report(f"  WARNING: node {node.pk} misc has no total_energy")
                continue
            # 构造输出格式 (basis_type: "lcao_nsw" -> "lcao")
            # basis: "pw" 或 "lcao_nsw" -> "pw" 或 "lcao"
            basis_str = str(item["basis"])
            basis_type = "pw" if basis_str == "pw" else "lcao"
            task_str = str(item["task"])
            output_entry = {
                "folder": task_str,
                "basis_type": basis_type,
                "E_total": float(total_energy),
            }
            # 直接传递字典，不要包装成 Dict 对象
            outputs.append(output_entry)

        if not outputs:
            self.report("WARNING: no children outputs to extract")
            self.report(
                f"  (children_info = {len(self.ctx.children_info)} entries)"
            )
            return self.exit_codes.WARNING_ENERGY_EXTRACT_FAILED

        # 直接在 WorkChain 中计算，不使用 calcfunction

        try:
            # 建立 folder -> n_atoms 映射 (从 children_info)
            n_atoms_by_folder = {}
            for item in self.ctx.children_info:
                folder = str(item.get("task", ""))
                n_atoms_by_folder[folder] = int(item.get("n_atoms", 1))

            energies_by_basis = {}
            for entry in outputs:
                basis = entry.get("basis_type", "unknown")
                energy = entry.get("E_total")
                if energy is None:
                    continue
                folder = entry.get("folder", "unknown")
                n_atoms = n_atoms_by_folder.get(folder, 1)
                energies_by_basis.setdefault(basis, []).append({
                    "folder": folder,
                    "energy": float(energy),
                    "pert": None,
                    "n_atoms": n_atoms,
                })

            delta_per_struct = []
            if "pw" in energies_by_basis and "lcao" in energies_by_basis:
                pw_by_folder = {e["folder"]: e for e in energies_by_basis["pw"]}
                for e_lcao in energies_by_basis["lcao"]:
                    folder = e_lcao["folder"]
                    e_pw_entry = pw_by_folder.get(folder)
                    if e_pw_entry is None:
                        continue
                    n_atoms = max(1, e_lcao.get("n_atoms", 1))
                    dE_total = abs(e_lcao["energy"] - e_pw_entry["energy"])
                    dE_per_atom = dE_total / n_atoms
                    delta_per_struct.append({
                        "folder": folder,
                        "n_atoms": n_atoms,
                        "E_pw": e_pw_entry["energy"],
                        "E_lcao_nsw": e_lcao["energy"],
                        "dE": dE_total,
                        "dE_per_atom": dE_per_atom,
                    })

            # 【新增】检查 lcao 数据的完整性
            n_pw = len(energies_by_basis.get("pw", []))
            n_lcao_valid = len(energies_by_basis.get("lcao", []))
            n_lcao_total = sum(
                1 for item in self.ctx.children_info
                if str(item.get("basis", "")) == "lcao_nsw"
            )
            n_lcao_skipped = n_lcao_total - n_lcao_valid
            if n_lcao_total > 0 and n_lcao_valid == 0:
                self.report(
                    f"  ✗ 全部 {n_lcao_total} 个 lcao 子都因 STRU orbital 错而被跳过, "
                    f"lcao_nsw 数据完全不可用 (aiida-abacus STRU 生成有 bug). "
                    f"pw={n_pw} 仍然有效, 但 lcao:nsw 质量评估无法进行."
                )
                # 报告"未通过 tolerance" 因为 lcao 全错 = tolerance 实质不达标
                self.ctx.tolerance_exceeded = True
                self.ctx.lcao_all_skipped = True
            elif n_lcao_skipped > 0:
                self.report(
                    f"  ⚠ {n_lcao_skipped}/{n_lcao_total} 个 lcao 子因 STRU orbital 错而被跳过"
                )
                self.ctx.lcao_skipped_count = n_lcao_skipped

            delta_max = max((x["dE"] for x in delta_per_struct), default=0.0)
            delta_max_per_atom = max(
                (x["dE_per_atom"] for x in delta_per_struct), default=0.0
            )
            d = {
                "energies": energies_by_basis,
                "delta_E_per_struct": delta_per_struct,
                "delta_E_max_eV": float(delta_max),
                "delta_E_max_meV": float(delta_max * 1000.0),
                "delta_E_max_per_atom_eV": float(delta_max_per_atom),
                "delta_E_max_per_atom_meV": float(delta_max_per_atom * 1000.0),
            }
        except Exception as exc:
            import traceback
            self.report(f"ERROR: energy extraction failed: {exc}")
            self.report(traceback.format_exc())
            return self.exit_codes.WARNING_ENERGY_EXTRACT_FAILED

        self.report(
            f"  ΔE_max (per system)  = {d.get('delta_E_max_meV', '?'):.3f} meV "
            f"({d.get('delta_E_max_eV', '?'):.6f} eV)"
        )
        self.report(
            f"  ΔE_max (per atom)    = {d.get('delta_E_max_per_atom_meV', '?'):.3f} meV "
            f"({d.get('delta_E_max_per_atom_eV', '?'):.6f} eV)"
        )
        for entry in d.get("delta_E_per_struct", []):
            self.report(
                f"    {entry['folder']:30s}  N={entry['n_atoms']:2d}  "
                f"E_pw={entry['E_pw']:.6f}  "
                f"E_lcao_nsw={entry['E_lcao_nsw']:.6f}  "
                f"dE={entry['dE']*1000:.3f} meV  "
                f"dE/atom={entry['dE_per_atom']*1000:.3f} meV"
            )
        self.ctx.energies = d

        # 与 tolerance 比较: ΔE_per_atom_max < tolerance_meV ?
        # 0.1 kcal/mol ≈ 4.2 meV 是 **per atom** 标准 (化学精度)
        tolerance_meV = float(self.ctx.abacus_cfg.get("tolerance_meV", 4.2))
        delta_per_atom_meV = float(d.get("delta_E_max_per_atom_meV", 0.0))
        if "lcao" in energies_by_basis and "pw" in energies_by_basis:
            self.ctx.tolerance_meV = tolerance_meV
            self.ctx.delta_per_atom_meV = delta_per_atom_meV
            if delta_per_atom_meV > tolerance_meV:
                self.report(
                    f"  ✗ tolerance EXCEEDED: ΔE/atom_max={delta_per_atom_meV:.3f} meV "
                    f"> tolerance_meV={tolerance_meV:.3f} meV "
                    f"(0.1 kcal/mol/atom = 4.2 meV) -> lcao:nsw NOT acceptable"
                )
                self.ctx.tolerance_exceeded = True
            else:
                self.report(
                    f"  ✓ tolerance OK: ΔE/atom_max={delta_per_atom_meV:.3f} meV "
                    f"<= tolerance_meV={tolerance_meV:.3f} meV "
                    f"-> lcao:nsw is acceptable"
                )
                self.ctx.tolerance_exceeded = False

        # 注意：不在这里返回 exit code，把判断留给 finalize
        # 这样可以保证 results/energies 节点仍然能正常输出
        return None

    # ------------------------------------------------------------------
    # Step 5: final results
    # ------------------------------------------------------------------

    def finalize(self):
        info = self.ctx.children_info
        results = []
        n_ok = 0
        n_failed = 0
        for item in info:
            node = item["node"]
            # 304 (tolerance exceeded) 视为软成功, 不算 failure
            ok = node.is_finished_ok or node.exit_status == 304
            if ok:
                n_ok += 1
            else:
                n_failed += 1
            results.append({
                "task": item["task"],
                "basis": item["basis"],
                "pk": node.pk,
                "exit_status": node.exit_status,
                "ok": ok,
            })

        # 通过 calcfunction 创建结果 Dict
        results_node = create_final_results(
            l_max_val=self.inputs.l_max.value,
            r_cut_val=self.inputs.r_cut.value,
            results_list=results,
        )
        self.out("results", results_node)
        self.report(
            f"OrbgenCalcWorkChain Finished: {n_ok} OK, {n_failed} failed"
        )

        # 如果有 energies 结果，也通过 calcfunction 输出
        if hasattr(self.ctx, "energies") and self.ctx.energies:
            # 通过 calcfunction 创建 energies Dict
            energies_node = create_energies_dict(self.ctx.energies)
            self.out("energies", energies_node)

        # 决定最终 exit code (优先级: ERROR > tolerance > partial > 0)
        if n_ok == 0:
            return self.exit_codes.ERROR_ALL_FAILED
        # tolerance 优先于 partial: 即使有 partial failure, 也先报告 tolerance
        if getattr(self.ctx, "tolerance_exceeded", False):
            return self.exit_codes.WARNING_TOLERANCE_EXCEEDED
        if n_failed > 0:
            return self.exit_codes.WARNING_PARTIAL_FAILURE
        return ExitCode(0)


# ===========================================================================
# OrbgenGridSearchWorkChain
# ===========================================================================
class OrbgenGridSearchWorkChain(WorkChain):
    """网格搜索: 在多个 (l_max, r_cut) 候选上跑 OrbgenCalcWorkChain,
    选出**最小**满足 tolerance 的 (l_max, r_cut) 组合.

    Inputs
    ------
    - ``siab_json`` (SinglefileData)         - orbgen.json
    - ``abacus_config`` (Dict)                - abacus.json (与 OrbgenCalcWorkChain 相同)
    - ``l_max_candidates`` (List of Int)     - 候选 l_max 列表 (升序)
    - ``r_cut_candidates`` (List of Float)   - 候选 r_cut 列表 (升序)
    - ``output_dir`` (Str)                   - SIAB 生成的根目录 (每个组合一个子目录)
    - ``search_strategy`` (Str, optional)    - "iterative" (默认, 从小到大逐个尝试)
                                                或 "exhaustive" (全部并行跑)
    - ``code_label``, ``family_label`` 等    - 透传给 OrbgenCalcWorkChain

    Outputs
    -------
    - ``best_result`` (Dict)   - 最优 (l_max, r_cut) 的 energies Dict
    - ``all_results`` (Dict)   - 所有尝试的 (l_max, r_cut) -> result
    - ``grid_summary`` (Dict)  - 网格搜索总结 (best, n_tried, n_passed, ...)
    """

    _child_workchain_entry_point = "orbgen.calc"

    # ------------------------------------------------------------------
    # define
    # ------------------------------------------------------------------
    @classmethod
    def define(cls, spec):
        super().define(spec)

        # ---- JSON inputs ----
        spec.input("siab_json", valid_type=SinglefileData, required=False,
                   help="pbe_orbgen.json (SIAB config, contains UPF path). "
                        "与 orbgen_jsons 互斥.")
        spec.input("orbgen_jsons", valid_type=List, required=False,
                   help="List[SinglefileData]: 每个 JSON 一份配置 (不同 nzeta / "
                        "bessel_nao_rcut). l_max = len(nzeta[0]) - 1, "
                        "r_cut = bessel_nao_rcut[0]. "
                        "与 siab_json 互斥.")
        spec.input("abacus_config", valid_type=Dict,
                   help="abacus.json (basis / input_overrides / tolerance / scheduler).")

        # ---- 候选网格 ----
        spec.input("l_max_candidates", valid_type=List, required=False,
                   help="候选 l_max 列表 (升序排列). 与 orbgen_jsons 互斥.")
        spec.input("r_cut_candidates", valid_type=List, required=False,
                   help="候选 r_cut 列表 (升序排列, Å). 与 orbgen_jsons 互斥.")

        # ---- 路径 ----
        spec.input("output_dir", valid_type=Str,
                   help="SIAB 生成的根目录 (worker 可写).")

        # ---- 搜索策略 ----
        spec.input("search_strategy", valid_type=Str, required=False,
                   default=lambda: orm.Str("iterative"),
                   help='"iterative" (默认, 从小到大) 或 "exhaustive" (全部并行).')

        # ---- 透传给 OrbgenCalcWorkChain (optional) ----
        spec.input("code_label", valid_type=Str, required=False,
                   help="AiiDA code label.")
        spec.input("family_label", valid_type=Str, required=False,
                   help="AiiDA pseudo family label.")
        spec.input("max_iterations", valid_type=Int, required=False)
        spec.input("build_family", valid_type=Bool, required=False)
        spec.input("only", valid_type=Int, required=False,
                   help="仅供调试: 每个 OrbgenCalcWorkChain 内部只跑 N-th 结构.")
        spec.input("dry_run", valid_type=Bool, required=False,
                   default=lambda: orm.Bool(False),
                   help="If True, only print, don't submit.")

        # ---- outputs ----
        spec.output("best_result", valid_type=Dict, required=False,
                    help="最优 (l_max, r_cut) 的 energies Dict.")
        spec.output("all_results", valid_type=Dict, required=False,
                    help="所有尝试的 (l_max, r_cut) -> 结果.")
        spec.output("grid_summary", valid_type=Dict, required=False,
                    help="网格搜索总结.")

        # ---- exit codes ----
        spec.exit_code(401, "ERROR_INVALID_INPUT",
                       message="l_max_candidates or r_cut_candidates invalid")
        spec.exit_code(402, "ERROR_INVALID_ABACUS_CONFIG",
                       message="abacus_config invalid")
        spec.exit_code(403, "ERROR_EMPTY_GRID",
                       message="No (l_max, r_cut) combinations to try")
        spec.exit_code(404, "ERROR_NO_ACCEPTABLE_ORBITALS",
                       message="All (l_max, r_cut) candidates exceed tolerance")
        spec.exit_code(301, "WARNING_PARTIAL_FAILURE",
                       message="Some CalcWorkChains failed")

        # ---- outline ----
        spec.outline(
            cls.validate_inputs,
            cls.generate_grid_step,
            cls.launch_search_step,
            if_(cls.is_iterative)(
                while_(cls.has_pending_iterative)(
                    cls.iterate_step,
                ),
            ).else_(
                cls.collect_exhaustive_step,
            ),
            cls.finalize,
        )

    # ------------------------------------------------------------------
    # dry run
    # ------------------------------------------------------------------

    @property
    def _dry_run(self) -> bool:
        """``dry_run`` 输入 (只出计划, 不落地任何东西).

        注意 AiiDA 的语义: step 返回 ``ExitCode(0)`` **不能**终止 outline --
        ``aiida/engine/processes/workchains/workchain.py`` 把 status 0 的
        ExitCode 映射成 ``None`` 后继续执行下一步. 所以每个 step 必须自己检查
        这个标志; 本类里曾经只有 submit/inspect/extract/finalize 检查, Step 1
        (SIAB pipeline) 和 Step 1.5 (建 pseudo family) 没有 -- dry-run 依然会
        启动 SIAB 子进程并在数据库里建 family.
        """
        return bool(self.inputs.get("dry_run", orm.Bool(False)).value)

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------
    def validate_inputs(self):
        """验证输入."""
        cfg = self.inputs.abacus_config.get_dict()
        if not cfg:
            return self.exit_codes.ERROR_INVALID_ABACUS_CONFIG

        # 检查互斥: siab_json vs orbgen_jsons
        has_single = "siab_json" in self.inputs
        has_multi = "orbgen_jsons" in self.inputs
        if has_single and has_multi:
            self.report(
                "ERROR: siab_json 和 orbgen_jsons 互斥, 只能选一个"
            )
            return self.exit_codes.ERROR_INVALID_INPUT
        if not has_single and not has_multi:
            self.report(
                "ERROR: 必须提供 siab_json 或 orbgen_jsons 之一"
            )
            return self.exit_codes.ERROR_INVALID_INPUT

        # 多 JSON 模式: grid 由每个 JSON 自带 (l_max, r_cut) 决定
        if has_multi:
            json_nodes = self.inputs.orbgen_jsons.get_list()
            if not json_nodes:
                self.report("ERROR: orbgen_jsons 为空")
                return self.exit_codes.ERROR_INVALID_INPUT
            self.ctx.orbgen_jsons_list = list(json_nodes)
            self.ctx.use_multi_json = True
            self.report(
                f"Mode: multi-JSON ({len(json_nodes)} files), "
                "l_max/r_cut 从每个 JSON 自动提取"
            )
        else:
            self.ctx.use_multi_json = False
            # 读取 candidates
            l_max_list = self.inputs.l_max_candidates.get_list()
            r_cut_list = self.inputs.r_cut_candidates.get_list()
            if not l_max_list or not r_cut_list:
                self.report(
                    f"ERROR: empty candidates: l_max={l_max_list}, r_cut={r_cut_list}"
                )
                return self.exit_codes.ERROR_INVALID_INPUT
            # 排序 (升序, 优先小参数)
            self.ctx.l_max_list = sorted(set(int(x) for x in l_max_list))
            self.ctx.r_cut_list = sorted(set(float(x) for x in r_cut_list))
            self.report(
                f"Mode: Cartesian product, "
                f"l_max candidates = {self.ctx.l_max_list}, "
                f"r_cut candidates = {self.ctx.r_cut_list}"
            )

        self.ctx.tolerance_meV = float(cfg.get("tolerance_meV", 4.2))
        self.ctx.search_strategy = str(
            self.inputs.get("search_strategy").value
            if "search_strategy" in self.inputs
            else "iterative"
        )
        self.report(
            f"Strategy: {self.ctx.search_strategy}, "
            f"tolerance = {self.ctx.tolerance_meV} meV/atom"
        )
        return None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_lmax_rcut_from_orbgen_json(json_node: SinglefileData) -> tuple:
        """从 orbgen.json 提取 (l_max, r_cut).

        - l_max: orbitals[0].nzeta 的长度 - 1
                 (长度 6 = s/p/d/f/g/h → l_max=5)
        - r_cut: bessel_nao_rcut[0]

        Returns
        -------
        (l_max: int, r_cut: float)
        """
        import json as _json
        content = json_node.get_content()
        if isinstance(content, bytes):
            content = content.decode("utf-8")
        cfg = _json.loads(content)

        # nzeta: 优先 orbitals[0].nzeta, 然后 geoms[0].lmaxmax
        nzeta_list = cfg.get("orbitals", [{}])[0].get("nzeta")
        if nzeta_list:
            l_max = len(nzeta_list) - 1
        else:
            geoms = cfg.get("geoms", [])
            l_max = int(geoms[0].get("lmaxmax", 4)) if geoms else 4

        # r_cut: bessel_nao_rcut
        rcut_raw = cfg.get("bessel_nao_rcut", [9])
        if isinstance(rcut_raw, (int, float)):
            r_cut = float(rcut_raw)
        else:
            r_cut = float(rcut_raw[0])

        return l_max, r_cut

    # ------------------------------------------------------------------
    # Step 1: 生成网格
    # ------------------------------------------------------------------
    def generate_grid_step(self):
        """生成 ``GridEntry`` 候选列表 (见 ``workflows/_grid.py``)."""
        grid = []
        if self.ctx.use_multi_json:
            # 多 JSON 模式: 每个 JSON 一个 (l_max, r_cut) 组合
            pairs = []
            for json_node in self.ctx.orbgen_jsons_list:
                l_max, r_cut = self._extract_lmax_rcut_from_orbgen_json(json_node)
                pairs.append((int(l_max), float(r_cut), json_node))
                self.report(
                    f"  [multi-json] {json_node.filename}: l_max={l_max}, r_cut={r_cut}"
                )
            grid = build_multi_json_grid(pairs)
        else:
            # 单 JSON 模式: Cartesian product, 按 (l_max ↑, r_cut ↑) 排序
            grid = build_cartesian_grid(
                self.ctx.l_max_list, self.ctx.r_cut_list, self.inputs.siab_json
            )

        self.ctx.grid = grid
        self.report(
            f"Generated {len(grid)} grid points: "
            f"{[(entry.l_max, entry.r_cut) for entry in grid]}"
        )
        if not grid:
            return self.exit_codes.ERROR_EMPTY_GRID
        return None

    # ------------------------------------------------------------------
    # Iterative strategy: 顺序尝试, 满足即停
    # ------------------------------------------------------------------
    def is_iterative(self):
        return self.ctx.search_strategy == "iterative"

    def has_pending_iterative(self):
        """如果还有 grid point 没尝试, 继续循环."""
        return has_pending_iterative(
            grid=self.ctx.grid,
            n_done=len(getattr(self.ctx, "grid_results", [])),
            best=getattr(self.ctx, "best", None),
            dry_run=self._dry_run,
        )

    def launch_search_step(self):
        """根据 strategy 启动搜索."""
        self.ctx.grid_results = []
        self.ctx.best = None

        # dry-run 模式: 只打印, 不提交 (真正的 early return, 见 _dry_run)
        if self._dry_run:
            for entry in self.ctx.grid:
                self.report(
                    f"  [DRY-RUN] grid #{entry.index + 1}/{len(self.ctx.grid)}: "
                    f"{entry.label}"
                )
            self.report(
                f"[DRY-RUN] {len(self.ctx.grid)} candidates would be submitted "
                f"(strategy={self.ctx.search_strategy})"
            )
            # 用 dry-run 子 WC 列表占位, 让 while 不会循环
            self.ctx.calc = []
            self.ctx.calcs = []
            return None

        from aiida.plugins import WorkflowFactory
        calc_wc_cls = WorkflowFactory("orbgen.calc")

        if self.ctx.search_strategy == "iterative":
            # iterative: 提交第一个, 后续由 iterate_step 处理
            entry = self.ctx.grid[0]
            self.report(
                f"[iterative] try #{entry.index + 1}/{len(self.ctx.grid)}: "
                f"{entry.label}"
            )
            running = self.submit(calc_wc_cls, **self._build_calc_inputs(entry))
            running.base.extras.set_many({
                "l_max": str(entry.l_max),
                "r_cut": str(entry.r_cut),
                "grid_index": str(entry.index),
            })
            return self.to_context(calc=append_(running))

        # exhaustive: 并行提交所有
        running_list = []
        for entry in self.ctx.grid:
            self.report(
                f"[exhaustive] submit #{entry.index + 1}/{len(self.ctx.grid)}: "
                f"{entry.label}"
            )
            running = self.submit(calc_wc_cls, **self._build_calc_inputs(entry))
            running.base.extras.set_many({
                "l_max": str(entry.l_max),
                "r_cut": str(entry.r_cut),
                "grid_index": str(entry.index),
            })
            running_list.append(running)
            self.ctx.grid_results.append({
                "l_max": entry.l_max,
                "r_cut": entry.r_cut,
                "calc_pk": running.pk,
                "exit_status": None,
                "is_finished_ok": False,
            })
        # append_ 只接受 1 个参数: 在循环里直接 self.to_context, 不 return
        # AiiDA 会把所有 awaitable 收集到 ctx.calcs
        for r in running_list:
            self.to_context(calcs=append_(r))

    def iterate_step(self):
        """Iterative 策略的核心循环 step.

        每次调用 (由 while_ 触发):
          1. 检查 self.ctx.calc[-1] 刚完成的子
          2. 提取 energies, 判断 tolerance
          3. 如果通过或失败, has_pending_iterative 返回 False 退出循环
          4. 否则提交下一个

        Note: exit_status == 304 (WARNING_TOLERANCE_EXCEEDED) 视为软成功
        - energies 仍可提取, tolerance_ok=False, 不算 failure, 不中断循环.
        """
        calc_node = self.ctx.calc[-1]  # 最新完成
        entry = self.ctx.grid[len(self.ctx.grid_results)]
        l_max, r_cut = entry.l_max, entry.r_cut

        # 304 = WARNING_TOLERANCE_EXCEEDED: 子任务跑完了 SCF + energy extract,
        # 只是 lcao_nsw 精度没达标, 这对 grid search 是**有效信息**而非错误
        has_energies = hasattr(calc_node.outputs, "energies")
        is_tolerance_exceeded = (
            calc_node.exit_status == 304 and has_energies
        )
        treat_as_soft_success = (
            calc_node.is_finished_ok or is_tolerance_exceeded
        )

        result_entry = {
            "l_max": l_max,
            "r_cut": r_cut,
            "calc_pk": calc_node.pk,
            "exit_status": calc_node.exit_status,
            "is_finished_ok": calc_node.is_finished_ok,
            "tolerance_exceeded_only": is_tolerance_exceeded,
        }

        if treat_as_soft_success and has_energies:
            energies = calc_node.outputs.energies.get_dict()
            delta_per_atom_meV = float(
                energies.get("delta_E_max_per_atom_meV", float("inf"))
            )
            tolerance_ok = delta_per_atom_meV <= self.ctx.tolerance_meV
            result_entry["energies"] = energies
            result_entry["delta_per_atom_meV"] = delta_per_atom_meV
            result_entry["tolerance_ok"] = tolerance_ok

            tag = "⚠ TOLERANCE_EXCEEDED" if is_tolerance_exceeded else None
            self.report(
                f"  ΔE/atom = {delta_per_atom_meV:.3f} meV "
                f"(tolerance = {self.ctx.tolerance_meV:.3f} meV) "
                f"-> {'✓ OK' if tolerance_ok else '✗ EXCEEDED'}"
                + (f" [{tag}]" if tag else "")
            )
            if tolerance_ok and self.ctx.best is None:
                self.ctx.best = {
                    "l_max": l_max,
                    "r_cut": r_cut,
                    "calc_pk": calc_node.pk,
                    "delta_per_atom_meV": delta_per_atom_meV,
                    "energies": energies,
                }
                self.report(
                    f"  ★ found acceptable (l_max={l_max}, r_cut={r_cut})"
                )
        else:
            self.report(
                f"  ✗ OrbgenCalcWorkChain<{calc_node.pk}> failed "
                f"(exit_status={calc_node.exit_status})"
            )

        self.ctx.grid_results.append(result_entry)

        # 如果 best 找到或已是最后一个, has_pending_iterative 会返回 False
        if not self.has_pending_iterative():
            return None

        # 提交下一个
        from aiida.plugins import WorkflowFactory
        calc_wc_cls = WorkflowFactory("orbgen.calc")
        entry = self.ctx.grid[idx + 1]
        self.report(
            f"[iterative] try #{entry.index + 1}/{len(self.ctx.grid)}: "
            f"{entry.label}"
        )
        running = self.submit(calc_wc_cls, **self._build_calc_inputs(entry))
        running.base.extras.set_many({
            "l_max": str(entry.l_max),
            "r_cut": str(entry.r_cut),
            "grid_index": str(entry.index),
        })
        return self.to_context(calc=append_(running))

    def collect_exhaustive_step(self):
        """Exhaustive 策略: 等待所有完成后收集结果.

        Note: 同 ``iterate_step``, exit_status == 304 视为软成功.
        """
        if not hasattr(self.ctx, "calcs") or not self.ctx.calcs:
            return None

        best = None
        for idx, calc_node in enumerate(self.ctx.calcs):
            entry = self.ctx.grid_results[idx]
            entry["exit_status"] = calc_node.exit_status
            entry["is_finished_ok"] = calc_node.is_finished_ok

            has_energies = hasattr(calc_node.outputs, "energies")
            is_tolerance_exceeded = (
                calc_node.exit_status == 304 and has_energies
            )
            treat_as_soft_success = (
                calc_node.is_finished_ok or is_tolerance_exceeded
            )
            entry["tolerance_exceeded_only"] = is_tolerance_exceeded

            if treat_as_soft_success and has_energies:
                energies = calc_node.outputs.energies.get_dict()
                delta_per_atom_meV = float(
                    energies.get("delta_E_max_per_atom_meV", float("inf"))
                )
                tolerance_ok = delta_per_atom_meV <= self.ctx.tolerance_meV
                entry["energies"] = energies
                entry["delta_per_atom_meV"] = delta_per_atom_meV
                entry["tolerance_ok"] = tolerance_ok

                tag = "⚠ TOL_EXCEEDED" if is_tolerance_exceeded else ""
                self.report(
                    f"  (l_max={entry['l_max']}, r_cut={entry['r_cut']}): "
                    f"ΔE/atom = {delta_per_atom_meV:.3f} meV "
                    f"-> {'✓' if tolerance_ok else '✗'} {tag}".rstrip()
                )
                if tolerance_ok and best is None:
                    # 网格已按 l_max, r_cut 升序排列, 第一个匹配就是最优
                    best = {
                        "l_max": entry["l_max"],
                        "r_cut": entry["r_cut"],
                        "calc_pk": calc_node.pk,
                        "delta_per_atom_meV": delta_per_atom_meV,
                        "energies": energies,
                    }
            else:
                self.report(
                    f"  (l_max={entry['l_max']}, r_cut={entry['r_cut']}): "
                    f"FAILED (exit_status={calc_node.exit_status})"
                )

        self.ctx.best = best
        return None

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------
    def finalize(self):
        if self._dry_run:
            return None

        # 统计数据
        n_tried = len(self.ctx.grid_results)
        n_passed = sum(
            1 for e in self.ctx.grid_results
            if e.get("tolerance_ok", False)
        )
        # 真失败: 跑挂了, 不是 finish_ok 也不是 304 tolerance 软成功
        n_failed = sum(
            1 for e in self.ctx.grid_results
            if e.get("is_finished_ok", False) is False
            and not e.get("tolerance_exceeded_only", False)
        )

        try:
            # all_results: 通过 calcfunction 创建
            all_results_node = create_grid_all_results(
                grid_results_list=self.ctx.grid_results,
                tolerance_meV=self.ctx.tolerance_meV,
                search_strategy=self.ctx.search_strategy,
            )
            self.out("all_results", all_results_node)
        except Exception as exc:
            import traceback
            self.report(f"ERROR in finalize (all_results): {exc}")
            self.report(traceback.format_exc())
            return self.exit_codes.WARNING_PARTIAL_FAILURE

        # best_result: 通过 calcfunction 创建 (如果 best 存在)
        if self.ctx.best is not None:
            best_data = dict(self.ctx.best)
            best_data["tolerance_meV"] = self.ctx.tolerance_meV
            best_data["search_strategy"] = self.ctx.search_strategy
            best_node = Dict(dict=best_data).store()
            self.out("best_result", best_node)
            self.report(
                f"★ Best: l_max={self.ctx.best['l_max']}, "
                f"r_cut={self.ctx.best['r_cut']}, "
                f"ΔE/atom = {self.ctx.best['delta_per_atom_meV']:.3f} meV"
            )

        # grid_summary: 通过 calcfunction 创建
        best_l_max = self.ctx.best["l_max"] if self.ctx.best else None
        best_r_cut = self.ctx.best["r_cut"] if self.ctx.best else None
        best_dpa = (
            self.ctx.best["delta_per_atom_meV"] if self.ctx.best else None
        )
        best_pk = self.ctx.best["calc_pk"] if self.ctx.best else None
        summary_node = create_grid_summary(
            n_grid_points=len(self.ctx.grid),
            n_tried=n_tried,
            n_passed=n_passed,
            n_failed=n_failed,
            best_l_max=best_l_max,
            best_r_cut=best_r_cut,
            best_delta_per_atom_meV=best_dpa,
            best_calc_pk=best_pk,
            tolerance_meV=self.ctx.tolerance_meV,
            search_strategy=self.ctx.search_strategy,
        )
        self.out("grid_summary", summary_node)

        self.report(
            f"Grid search done: {n_tried} tried, {n_passed} passed, "
            f"{n_failed} failed"
        )

        if self.ctx.best is None:
            if n_failed == n_tried:
                return self.exit_codes.WARNING_PARTIAL_FAILURE
            return self.exit_codes.ERROR_NO_ACCEPTABLE_ORBITALS
        return ExitCode(0)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _build_calc_inputs(self, entry: GridEntry) -> dict:
        """为 ``OrbgenCalcWorkChain`` 构造 inputs.

        Parameters
        ----------
        entry : GridEntry
            当前候选 (l_max / r_cut / 对应的 SIAB config 节点).
        """
        # 每个组合使用独立的子目录, 避免 SIAB 输出互相覆盖
        output_dir_root = Path(self.inputs.output_dir.value)
        combo_dir = output_dir_root / work_dir_name(entry.l_max, entry.r_cut)
        combo_dir.mkdir(parents=True, exist_ok=True)

        siab_json = entry.siab_json if entry.siab_json is not None \
            else self.inputs.siab_json

        child_inputs = {
            "siab_json": siab_json,
            "abacus_config": self.inputs.abacus_config,
            "l_max": orm.Int(entry.l_max),
            "r_cut": orm.Float(entry.r_cut),
            "output_dir": orm.Str(str(combo_dir)),
        }
        # 透传 optional inputs
        for key in ("code_label", "family_label", "max_iterations",
                    "build_family", "only"):
            if key in self.inputs:
                child_inputs[key] = self.inputs[key]
        return child_inputs

