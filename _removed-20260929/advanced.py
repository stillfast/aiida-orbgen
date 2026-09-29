"""LEGACY / UNREGISTERED — do not use for new work.

This module holds an older, differently structured ``OrbgenGridSearchWorkChain``.
It is **not** reachable from any entry point: ``pyproject.toml`` registers
``orbgen.gridsearch`` -> ``workflows.batch:OrbgenGridSearchWorkChain``, and
``workflows/__init__.py`` now re-exports that same class.  Before 2026-09-29 the
package ``__init__`` re-exported *this* class instead, so two implementations
with the same ``process_label`` and colliding exit-code numbers (301/401/402/403
mean different things here than in ``batch``) were both importable.

Kept only as a reference for three capabilities ``batch``'s grid search still
lacks; port them before deleting this file:

* ``stop_on_first_valid`` — ``batch`` hard-codes "stop as soon as an acceptable
  point is found", so "run the whole grid, then choose" is impossible there;
* an explicit ``candidates`` list (arbitrary ``(l_max, r_cut)`` pairs instead of
  a Cartesian product);
* the ``max_l_max`` / ``max_r_cut`` caps (``with_default_abacus`` forwards them,
  but ``batch`` never reads them).

Do **not** port its per-*system* ΔE comparison (``advanced.py`` compares
``delta_E_max_meV``): the rest of the stack, including the tolerance verdict,
uses ΔE per atom.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from aiida import orm
from aiida.engine import WorkChain, ExitCode, ToContext, while_
from aiida.orm import (
    Bool,
    Dict,
    Float,
    Int,
    List,
    SinglefileData,
    Str,
)
from aiida.plugins import WorkflowFactory

from aiida_orbgen.static.defaults import (
    DEFAULT_CODE_LABEL,
    DEFAULT_MAX_ITERATIONS,
)
from aiida_orbgen.static.json_inputs import (
    parse_lmax_rcut_candidates,
    with_default_abacus,
)

__all__ = [
    "OrbgenGridSearchWorkChain",
    "build_candidate_grid",
]


def build_candidate_grid(
    orbgen_cfg: Dict[str, Any],
    abacus_cfg: Dict[str, Any],
) -> List[Tuple[int, float]]:
    """构造 (l_max, r_cut) 候选列表。

    排序: 先按 l_max 升序, 再按 r_cut 升序 (从小到大, 找最小可用)。

    跳过规则 (可选, abacus.json 中可配):
      - ``"max_l_max"``: 不超过这个值
      - ``"max_r_cut"``: 不超过这个值
    """
    grid = parse_lmax_rcut_candidates(orbgen_cfg)
    max_lmax = abacus_cfg.get("max_l_max")
    max_rcut = abacus_cfg.get("max_r_cut")
    if max_lmax is not None:
        grid = [(lm, rc) for (lm, rc) in grid if lm <= int(max_lmax)]
    if max_rcut is not None:
        grid = [(lm, rc) for (lm, rc) in grid if rc <= float(max_rcut)]
    # 升序: 优先小 l_max, 同 l_max 优先小 r_cut
    grid = sorted(grid, key=lambda x: (x[0], x[1]))
    return grid


# ===========================================================================
#  WorkChain
# ===========================================================================


class OrbgenGridSearchWorkChain(WorkChain):
    """按 (l_max ↑, r_cut ↑) 顺序迭代 OrbgenCalcWorkChain, 找最小满足 tolerance 的组合。

    Inputs
    ------
    siab_json : SinglefileData
        pbe_orbgen.json
    abacus_config : Dict
        abacus.json (含 basis / parameters.input / tolerance / scheduler)
    output_dir : Str
        SIAB 生成目录
    candidates : List, optional
        显式指定 (l_max, r_cut) 候选列表; 缺省时从 orbgen.json 自动推断
    code_label : Str, optional
    family_label : Str, optional
    build_family : Bool, optional
    stop_on_first_valid : Bool, optional
        True (默认): 找到第一个 ΔE_max < tolerance 的组合就停
        False: 跑完所有候选, 输出全部 ΔE
    max_iterations : Int, optional

    Outputs
    -------
    best : Dict
        ``{"l_max", "r_cut", "delta_E_max_meV", "ok", "batch_pk"}`` 或空 dict
    all_results : List
        所有候选的 (l_max, r_cut, ΔE_max_meV, ok, batch_pk)
    best_batch_energies : Dict
        最佳组合的 energies (从 OrbgenCalcWorkChain 透传)
    """

    _child_workchain_entry_point = "orbgen.calc"

    # ------------------------------------------------------------------
    # define
    # ------------------------------------------------------------------

    @classmethod
    def define(cls, spec):
        super().define(spec)

        # ---- JSON inputs ----
        spec.input("siab_json", valid_type=SinglefileData,
                   help="pbe_orbgen.json (SIAB config).")
        spec.input("abacus_config", valid_type=Dict,
                   help="abacus.json (含 basis / parameters.input / tolerance_meV).")
        spec.input("output_dir", valid_type=Str,
                   help="SIAB 生成目录.")

        # ---- 候选参数 (可选) ----
        spec.input("candidates", valid_type=List, required=False,
                   help="显式指定候选列表: list of {lmax, rcut} dicts. "
                        "缺省时从 orbgen.json 自动推断.")

        # ---- AiiDA ----
        spec.input("code_label", valid_type=Str, required=False)
        spec.input("family_label", valid_type=Str, required=False)
        spec.input("build_family", valid_type=Bool, required=False,
                   default=lambda: orm.Bool(True))
        spec.input("stop_on_first_valid", valid_type=Bool, required=False,
                   default=lambda: orm.Bool(True),
                   help="True: 找到第一个满足 tolerance 的组合就停; "
                        "False: 跑完所有候选.")
        spec.input("max_iterations", valid_type=Int, required=False)
        spec.input("dry_run", valid_type=Bool, required=False,
                   default=lambda: orm.Bool(False))

        # ---- outputs ----
        spec.output("best", valid_type=Dict,
                    help="最佳 (l_max, r_cut) 组合.")
        spec.output("all_results", valid_type=Dict,
                    help="所有候选的结果 (Dict 形式: {results: [...]}).")
        spec.output("best_batch_energies", valid_type=Dict, required=False,
                    help="最佳 OrbgenCalcWorkChain 的 energies 透传.")

        # ---- exit codes ----
        spec.exit_code(401, "ERROR_INVALID_ABACUS_CONFIG",
                       message="abacus_config invalid")
        spec.exit_code(402, "ERROR_NO_CANDIDATES",
                       message="No (l_max, r_cut) candidates found")
        spec.exit_code(403, "ERROR_BATCH_SUBMIT_FAILED",
                       message="Failed to submit any OrbgenCalcWorkChain")
        spec.exit_code(301, "WARNING_NO_VALID",
                       message="No candidate satisfied tolerance")
        spec.exit_code(0, "SUCCESS",
                       message="Found valid (l_max, r_cut) within tolerance")

        # ---- outline: while loop ----
        spec.outline(
            cls.validate_inputs,
            cls.setup_candidates,
            while_(cls.should_continue)(
                cls.submit_one_batch,
                cls.inspect_batch,
            ),
            cls.finalize,
        )

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------

    def validate_inputs(self):
        cfg = self.inputs.abacus_config.get_dict()
        if not cfg:
            return self.exit_codes.ERROR_INVALID_ABACUS_CONFIG
        self.ctx.abacus_cfg = with_default_abacus(cfg)
        self.report(
            f"AdvancedWorkChain: tolerance_meV={self.ctx.abacus_cfg['tolerance_meV']}"
        )
        return None

    def setup_candidates(self):
        """构建候选列表。"""
        if "candidates" in self.inputs:
            cand = self.inputs.candidates.get_list()
            self.ctx.candidates = [
                (int(d["lmax"]), float(d["rcut"])) for d in cand
            ]
        else:
            # 从 siab_json 内容中读 orbgen.json, 推 candidates
            import json as _json
            import tempfile
            content = self.inputs.siab_json.get_content()
            if isinstance(content, bytes):
                content = content.decode("utf-8")
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, encoding="utf-8"
            ) as f:
                f.write(content)
                local_json = f.name
            with open(local_json, "r", encoding="utf-8") as f:
                orbgen_cfg = _json.load(f)
            self.ctx.candidates = build_candidate_grid(
                orbgen_cfg, self.ctx.abacus_cfg
            )

        if not self.ctx.candidates:
            return self.exit_codes.ERROR_NO_CANDIDATES

        self.report(
            f"Candidate grid (l_max × r_cut, sorted ascending): "
            f"{self.ctx.candidates}"
        )
        self.ctx.tolerance_meV = self.ctx.abacus_cfg["tolerance_meV"]
        self.ctx.all_results = []
        self.ctx.best = None
        self.ctx.best_batch_node = None
        self.ctx.idx = 0
        return None

    # ------------------------------------------------------------------
    # iteration: while_(should_continue)
    # ------------------------------------------------------------------

    def should_continue(self) -> bool:
        """``while_`` 循环条件: 还有 candidate + 还没找到 valid。"""
        if not hasattr(self.ctx, "idx"):
            self.ctx.idx = 0
        if not hasattr(self.ctx, "all_results"):
            self.ctx.all_results = []
        if not hasattr(self.ctx, "best"):
            self.ctx.best = None

        # 1) 跑完所有 candidates
        if self.ctx.idx >= len(self.ctx.candidates):
            return False

        # 2) 找到 valid (且 stop_on_first_valid=True)
        if self.ctx.best is not None:
            stop = self.inputs.get("stop_on_first_valid")
            if stop is None or stop.value:
                return False

        return True

    def submit_one_batch(self):
        """提交第 idx 个 OrbgenCalcWorkChain。"""
        lmax, rcut = self.ctx.candidates[self.ctx.idx]
        self.report(
            f"\n=== Candidate {self.ctx.idx+1}/{len(self.ctx.candidates)}: "
            f"l_max={lmax}, r_cut={rcut} ==="
        )

        if self.inputs.dry_run.value:
            self.report(f"  [DRY-RUN] Would submit OrbgenCalcWorkChain")
            self.ctx.all_results.append({
                "l_max": lmax, "r_cut": rcut,
                "ok": None, "delta_E_max_meV": None, "batch_pk": None,
            })
            self.ctx.idx += 1
            # dry-run 模式直接跳出 while_
            self.ctx.best = {"__dry_run_break__": True}
            return ExitCode(0)

        try:
            batch_node = self._submit_batch(lmax, rcut)
        except Exception as exc:
            self.report(f"  ERROR submitting batch: {exc}")
            self.ctx.all_results.append({
                "l_max": lmax, "r_cut": rcut,
                "ok": False, "delta_E_max_meV": None,
                "batch_pk": None, "error": str(exc),
            })
            self.ctx.idx += 1
            return None  # 继续 while_

        self.report(f"  Submitted OrbgenCalcWorkChain PK={batch_node.pk}")
        return ToContext(batch=batch_node)

    def inspect_batch(self):
        """刚完成的 batch 已被 ToContext 注入 ``ctx.batch``。"""
        if not hasattr(self.ctx, "batch") or self.ctx.batch is None:
            # dry-run 已经 break, 但 outline 仍会调一次, 直接返回
            return ExitCode(0)

        batch_node = self.ctx.batch
        idx = self.ctx.idx
        lmax, rcut = self.ctx.candidates[idx]

        ok = batch_node.is_finished_ok
        if not ok:
            self.report(
                f"  OrbgenCalcWorkChain {batch_node.pk} failed "
                f"(exit_status={batch_node.exit_status})"
            )
            self.ctx.all_results.append({
                "l_max": lmax, "r_cut": rcut,
                "ok": False, "delta_E_max_meV": None,
                "batch_pk": batch_node.pk,
            })
        else:
            energies = batch_node.outputs.get("energies", None)
            if energies is None:
                self.report(
                    f"  OrbgenCalcWorkChain {batch_node.pk} OK but no energies output"
                )
                dE_meV = float("inf")
            else:
                dE_meV = float(energies.get_dict().get(
                    "delta_E_max_meV", float("inf")
                ))

            self.report(
                f"  ΔE_max = {dE_meV:.3f} meV "
                f"(tol={self.ctx.tolerance_meV} meV)"
            )
            self.ctx.all_results.append({
                "l_max": lmax, "r_cut": rcut,
                "ok": True, "delta_E_max_meV": dE_meV,
                "batch_pk": batch_node.pk,
            })

            if dE_meV < self.ctx.tolerance_meV:
                self.ctx.best = {
                    "l_max": lmax, "r_cut": rcut,
                    "delta_E_max_meV": dE_meV,
                    "ok": True, "batch_pk": batch_node.pk,
                }
                self.ctx.best_batch_node = batch_node
                self.report(
                    f"  ✓ FOUND VALID: l_max={lmax}, r_cut={rcut}, "
                    f"ΔE_max={dE_meV:.3f} meV"
                )

        # 推进 idx
        self.ctx.idx += 1
        return None  # 让 while_ 重新判断 should_continue

    def _submit_batch(self, lmax: int, rcut: float):
        """构造并 submit 一个 OrbgenCalcWorkChain。"""
        batch_cls = WorkflowFactory(self._child_workchain_entry_point)

        batch_inputs: Dict[str, Any] = {
            "siab_json": self.inputs.siab_json,
            "abacus_config": self.inputs.abacus_config,
            "output_dir": self.inputs.output_dir,
            "l_max": Int(lmax),
            "r_cut": Float(rcut),
            "dry_run": Bool(False),
        }
        if "code_label" in self.inputs:
            batch_inputs["code_label"] = self.inputs.code_label
        if "family_label" in self.inputs:
            batch_inputs["family_label"] = self.inputs.family_label
        if "build_family" in self.inputs:
            batch_inputs["build_family"] = self.inputs.build_family
        if "max_iterations" in self.inputs:
            batch_inputs["max_iterations"] = self.inputs.max_iterations

        return self.submit(batch_cls, **batch_inputs)

    # ------------------------------------------------------------------
    # finalize
    # ------------------------------------------------------------------

    def finalize(self):
        # 把 all_results 写到 output
        all_results = self.ctx.get("all_results", [])
        # 强制转为 plain dict (避免 Dict 节点混入)
        clean_results = []
        for r in all_results:
            if hasattr(r, "get_dict"):
                clean_results.append(r.get_dict())
            else:
                clean_results.append(dict(r))
        all_dict = Dict(dict={"results": clean_results})
        all_dict.store()
        self.out("all_results", all_dict)

        # best
        best = self.ctx.get("best")
        if best and "l_max" in best:  # 跳过 dry-run marker
            best_node = Dict(dict=best)
            best_node.store()
            self.out("best", best_node)
            self.report(
                f"\n=== FINAL: best (l_max, r_cut) = "
                f"({best['l_max']}, {best['r_cut']}), "
                f"ΔE_max = {best['delta_E_max_meV']:.3f} meV ==="
            )

            # 透传 best batch 的 energies
            if self.ctx.get("best_batch_node") is not None:
                energies = self.ctx.best_batch_node.outputs.get("energies", None)
                if energies is not None:
                    self.out("best_batch_energies", energies)
        elif best and "__dry_run_break__" in best:
            # dry-run 模式: best 还没设, 但有 dummy marker
            self.report(
                f"\n=== FINAL (DRY-RUN): all {len(clean_results)} candidate(s) "
                f"simulated; would stop on first match ==="
            )
            empty_best = Dict(dict={})
            empty_best.store()
            self.out("best", empty_best)
        else:
            self.report(
                f"\n=== FINAL: no candidate satisfied tolerance "
                f"({self.ctx.tolerance_meV} meV) ==="
            )
            empty_best = Dict(dict={})
            empty_best.store()
            self.out("best", empty_best)
            return self.exit_codes.WARNING_NO_VALID

        return ExitCode(0)
