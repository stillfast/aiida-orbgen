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
from typing import Any, Dict, Optional

from aiida import orm
from aiida.engine import WorkChain, ExitCode, append_, if_, while_
from aiida.orm import (
    Bool,
    Dict,
    Float,
    Int,
    List,
    SinglefileData,
    Str,
)

from aiida_orbgen.interfaces import params_stru_to_ase
from aiida_orbgen.static.defaults import (
    DEFAULT_CODE_LABEL,
    DEFAULT_MAX_MEMORY_KB,
    DEFAULT_NUM_MPI,
    DEFAULT_QUEUE_NAME,
    DEFAULT_WALLCLOCK_SECONDS,
)
from aiida_orbgen.static.json_inputs import with_default_abacus
from aiida_orbgen.workflows.energies import (
    SOFT_SUCCESS_EXIT_STATUS,
    ChildEnergy,
    evaluate_energies,
    is_soft_success,
    pair_energies,
    tolerance_verdict,
)
from aiida_orbgen.workflows._grid import (
    GridEntry,
    build_cartesian_grid,
    build_explicit_grid,
    build_multi_json_grid,
    cap_grid,
    has_pending_iterative,
    work_dir_name,
)

# SIAB-facing code and result assembly live in their own modules; they are
# re-exported here because ``pyproject.toml`` points the entry points at this
# module and existing imports (tests, report layer) use these names.
from aiida_orbgen.workflows.results import (
    create_energies_dict,
    create_final_results,
    create_grid_all_results,
    create_grid_summary,
)
from aiida_orbgen.workflows.siab import (
    build_abacus_child_inputs,
    n_atoms_from_stru,
    run_siab_pipeline,
)

# abacuslite 读 STRU → dict

__all__ = [
    "OrbgenCalcWorkChain",
    "OrbgenGridSearchWorkChain",
    "run_siab_pipeline",
    "build_abacus_child_inputs",
]


# ===========================================================================
#  OrbgenCalcWorkChain
# ===========================================================================


def _validate_siab_json_inputs(inputs, report) -> str | None:
    """Validate every SIAB config a WorkChain was handed.

    Returns an error message (and reports warnings) or ``None``.  A WorkChain
    submitted directly through ``WorkflowFactory`` never passes
    ``ConfigLoader``, so without this the two classes of mistake that cost the
    most time would only show up inside SIAB:

    * an ``nzeta`` scheme the primitive basis cannot provide — discovered in
      ``basistrans`` *after* the reference DFT has been paid for;
    * ``vloc_aux`` / ``lloc_min`` written outside ``model_kwargs`` — silently
      dropped, so a requested g channel comes out empty.
    """
    import json as _json

    from aiida_orbgen.utils.config import validate_siab_config

    nodes = []
    if "siab_json" in inputs:
        nodes.append(inputs.siab_json)
    if "orbgen_jsons" in inputs:
        nodes.extend(inputs.orbgen_jsons.get_list())
    for node in nodes:
        try:
            content = node.get_content()
            if isinstance(content, bytes):
                content = content.decode("utf-8")
            config = _json.loads(content)
        except Exception as exc:  # noqa: BLE001 — unreadable JSON is a config error
            return f"cannot read siab_json<{node.pk}>: {exc}"
        try:
            warnings = validate_siab_config(
                config, source=f"siab_json<{node.pk}>"
            )
        except Exception as exc:  # noqa: BLE001 — ValueError from the validator
            return str(exc)
        for warning in warnings:
            report(f"  WARNING: {warning}")
    return None


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
        spec.output("primitive_orbital", valid_type=SinglefileData, required=False,
                    help="The primitive NSW .orb this grid point used, archived in "
                         "provenance (the paths in siab_info point at scratch space).")
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
        spec.exit_code(406, "ERROR_INVALID_SIAB_CONFIG",
                       message="siab_json is not a usable SIAB configuration")

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

        problem = _validate_siab_json_inputs(self.inputs, self.report)
        if problem:
            self.report(f"ERROR: invalid SIAB config: {problem}")
            return self.exit_codes.ERROR_INVALID_SIAB_CONFIG
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

        if "primitive_orbital" in siab_result:
            self.out("primitive_orbital", siab_result["primitive_orbital"])
        info = siab_result["info"].get_dict()
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
        self.out("siab_info", siab_result["info"])

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
        """Atom count from the generated STRU (per-atom ΔE normalisation)."""
        return n_atoms_from_stru(dft_entry.get("stru"), report=self.report)

    def inspect_children(self):
        info = self.ctx.children_info
        n_ok = sum(
            1 for i in info
            if is_soft_success(i["node"].is_finished_ok, i["node"].exit_status)
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
            # exit_status == 304 counts as a soft success (see
            # workflows/energies.is_soft_success): the child ran the whole SCF
            # and the energy extraction, only the LCAO accuracy missed.
            if not is_soft_success(node.is_finished_ok, node.exit_status):
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

            # 配对与 ΔE 计算只有一份实现 (workflows/energies.py)
            d = pair_energies([
                ChildEnergy(
                    folder=str(entry["folder"]),
                    basis=str(entry["basis_type"]),
                    energy=float(entry["E_total"]),
                    n_atoms=n_atoms_by_folder.get(str(entry["folder"]), 1),
                )
                for entry in outputs
                if entry.get("E_total") is not None
            ])
            energies_by_basis = d["energies"]
            n_pw = d["n_pw"]
            n_lcao_valid = d["n_lcao"]
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
            # 0.1 kcal/mol ≈ 4.2 meV 是 **per atom** 标准 (化学精度)
            acceptable, message = tolerance_verdict(delta_per_atom_meV, tolerance_meV)
            self.report(f"  {message}")
            self.ctx.tolerance_exceeded = not acceptable

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
            ok = is_soft_success(node.is_finished_ok, node.exit_status)
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
    - ``candidates`` (List, optional)        - 显式 [[l_max, r_cut], ...] (非笛卡尔积)
    - ``stop_on_first_valid`` (Bool, 默认 True) - iterative 找到即停; False = 跑完整个网格
    - abacus_config 里的 ``max_l_max`` / ``max_r_cut`` 会裁剪候选网格
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
        spec.exit_code(406, "ERROR_INVALID_SIAB_CONFIG",
                       message="a candidate siab_json is not a usable SIAB "
                               "configuration")
        spec.input("candidates", valid_type=List, required=False,
                   help="显式候选列表 [[l_max, r_cut], ...] (非笛卡尔积), 用于只在已"
                        "扫过的网格上补点. 与 l_max_candidates/r_cut_candidates 互斥.")

        # ---- 路径 ----
        spec.input("output_dir", valid_type=Str,
                   help="SIAB 生成的根目录 (worker 可写).")

        # ---- 搜索策略 ----
        spec.input("search_strategy", valid_type=Str, required=False,
                   default=lambda: orm.Str("iterative"),
                   help='"iterative" (默认, 从小到大) 或 "exhaustive" (全部并行).')
        spec.input("stop_on_first_valid", valid_type=Bool, required=False,
                   default=lambda: orm.Bool(True),
                   help="iterative 策略下找到第一个可接受组合就停 (默认). 设为 False "
                        "则跑完整个网格再选最优 —— 只有 exhaustive 才等价于此前的"
                        "行为, 现在两种策略都可以.")

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
        if "candidates" in self.inputs and (
            "l_max_candidates" in self.inputs or "r_cut_candidates" in self.inputs
        ):
            self.report(
                "ERROR: candidates 与 l_max_candidates/r_cut_candidates 互斥"
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

        problem = _validate_siab_json_inputs(self.inputs, self.report)
        if problem:
            self.report(f"ERROR: invalid SIAB config: {problem}")
            return self.exit_codes.ERROR_INVALID_SIAB_CONFIG

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
        if "candidates" in self.inputs:
            pairs = self.inputs.candidates.get_list()
            try:
                grid = build_explicit_grid(pairs, self.inputs.siab_json)
            except (TypeError, ValueError) as exc:
                self.report(f"ERROR: invalid candidates: {exc}")
                return self.exit_codes.ERROR_INVALID_INPUT
            self.report(f"Mode: explicit candidates ({len(grid)} points)")
        elif self.ctx.use_multi_json:
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

        # ``with_default_abacus`` forwards these two caps from abacus.json; only
        # advanced.py used to honour them, everything else silently ignored them.
        raw = self.inputs.abacus_config.get_dict()
        before = len(grid)
        grid = cap_grid(grid, raw.get("max_l_max"), raw.get("max_r_cut"))
        if grid and len(grid) != before:
            self.report(
                f"  caps: max_l_max={raw.get('max_l_max')}, "
                f"max_r_cut={raw.get('max_r_cut')} -> {before} -> {len(grid)} points"
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
            stop_on_first_valid=bool(
                self.inputs.get("stop_on_first_valid", orm.Bool(True)).value
            ),
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
        # 只是 lcao_nsw 精度没达标, 这对 grid search 是**有效信息**而非错误.
        # 分类与判定都取自 workflows/energies.py (唯一定义).
        has_energies = hasattr(calc_node.outputs, "energies")
        treat_as_soft_success = is_soft_success(
            calc_node.is_finished_ok, calc_node.exit_status, has_energies
        )
        is_tolerance_exceeded = (
            calc_node.exit_status == SOFT_SUCCESS_EXIT_STATUS and has_energies
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
            verdict = evaluate_energies(energies, self.ctx.tolerance_meV)
            delta_per_atom_meV = verdict["delta_per_atom_meV"]
            tolerance_ok = verdict["tolerance_ok"]
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
            treat_as_soft_success = is_soft_success(
                calc_node.is_finished_ok, calc_node.exit_status, has_energies
            )
            is_tolerance_exceeded = (
                calc_node.exit_status == SOFT_SUCCESS_EXIT_STATUS and has_energies
            )
            entry["tolerance_exceeded_only"] = is_tolerance_exceeded

            if treat_as_soft_success and has_energies:
                energies = calc_node.outputs.energies.get_dict()
                verdict = evaluate_energies(energies, self.ctx.tolerance_meV)
                delta_per_atom_meV = verdict["delta_per_atom_meV"]
                tolerance_ok = verdict["tolerance_ok"]
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

