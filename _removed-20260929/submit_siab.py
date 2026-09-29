"""
``aiida-orbgen submit-siab`` — 两级 workchain 提交 CLI

根据 flowchart, aiida-orbgen 提供了两个 workchain:

* ``OrbgenCalcWorkChain`` (entry point: ``orbgen.calc``)
  - 跑特定 (l_max, r_cut) 下的 PW + LCAO:nsw × N 结构 = 2N 个 abacus 任务
  - 输出 energies 和 ΔE_max
  - l_max/r_cut 可从 orbgen.json 自动提取 (默认)

* ``OrbgenGridSearchWorkChain`` (entry point: ``orbgen.gridsearch``)
  - 迭代 (l_max, r_cut) 候选, 调 OrbgenCalcWorkChain
  - 选最小满足 tolerance 的组合

输入 (从 JSON 读取, 都在 static/examples/ 下)
--------------------------------------------

- ``orbgen.json``: SIAB 配置 (含 element / pseudo_dir / geoms /
  bessel_nao_rcut / lmaxmax)
- ``abacus.json``: ABACUS 配置 (含 basis / parameters.input /
  tolerance_meV / scheduler / abacus.code/parameters/metadata)

用法
----

.. code-block:: bash

    # OrbgenCalcWorkChain: 自动从 orbgen.json 提取 l_max/r_cut
    aiida-orbgen submit-siab calc \\
        --orbgen  project/pbe/orbgen.json \\
        --abacus  project/pbe/abacus.json

    # OrbgenCalcWorkChain: 显式指定 l_max/r_cut (覆盖 orbgen.json)
    aiida-orbgen submit-siab calc \\
        --orbgen  project/pbe/orbgen.json \\
        --abacus  project/pbe/abacus.json \\
        --l-max 4 --r-cut 9

    # OrbgenGridSearchWorkChain: 迭代 candidates, 找最小可用
    aiida-orbgen submit-siab gridsearch \\
        --orbgen  project/pbe/orbgen.json \\
        --abacus  project/pbe/abacus.json

所有默认值在 ``aiida_orbgen.static.defaults``。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


__all__ = ["add_submit_siab_parser", "cmd_submit_siab"]


def cmd_submit_siab(args) -> int:
    """``aiida-orbgen submit-siab`` 的入口。"""
    # ---- sub-command 选择 ----
    if args.subcmd == "calc":
        return _cmd_calc(args)
    if args.subcmd == "gridsearch":
        return _cmd_gridsearch(args)
    return 1


# ---------------------------------------------------------------------------
#  Batch
# ---------------------------------------------------------------------------


def _cmd_calc(args) -> int:
    """submit OrbgenCalcWorkChain (单个 l_max, r_cut 组合).

    l_max 和 r_cut 优先从 CLI 参数获取，若未指定则从 orbgen.json 自动提取:
    - r_cut 从 bessel_nao_rcut[0] 获取
    - l_max 从 geoms[0].lmaxmax 获取
    """
    from aiida import orm
    from aiida.engine import submit
    from aiida.orm import SinglefileData
    from aiida_orbgen.static.json_inputs import (
        load_abacus_config,
        load_orbgen_config,
        parse_lmax_rcut_candidates,
        with_default_abacus,
    )
    from aiida_orbgen.workflows import OrbgenCalcWorkChain

    if not args.orbgen.is_file():
        print(f"Error: orbgen.json not found: {args.orbgen}", file=sys.stderr)
        return 1
    if not args.abacus.is_file():
        print(f"Error: abacus.json not found: {args.abacus}", file=sys.stderr)
        return 1

    orbgen_cfg = load_orbgen_config(args.orbgen)
    abacus_cfg = with_default_abacus(load_abacus_config(args.abacus))
    
    # 从 orbgen.json 提取 l_max/r_cut（单组合情况）
    candidates = parse_lmax_rcut_candidates(orbgen_cfg)
    default_lmax, default_rcut = candidates[0] if candidates else (4, 9.0)
    
    # CLI 参数优先，否则使用 orbgen.json 中的值
    l_max = args.l_max if args.l_max is not None else default_lmax
    r_cut = args.r_cut if args.r_cut is not None else default_rcut

    print(f"orbgen.json: element={orbgen_cfg.get('element')}")
    print(f"  auto-detected l_max={default_lmax}, r_cut={default_rcut} from orbgen.json")
    print(f"abacus.json: basis={abacus_cfg['basis']}, "
          f"tolerance_meV={abacus_cfg['tolerance_meV']}")
    print(f"Running OrbgenCalcWorkChain with l_max={l_max}, r_cut={r_cut}")

    # 将相对路径转换为绝对路径
    orbgen_path = args.orbgen.resolve()
    output_root_path = args.output_root.resolve()

    inputs: dict = {
        "siab_json": SinglefileData(file=str(orbgen_path)),
        "abacus_config": orm.Dict(dict=abacus_cfg),
        "output_dir": orm.Str(str(output_root_path)),
        "l_max": orm.Int(l_max),
        "r_cut": orm.Float(r_cut),
    }
    if args.code_label:
        inputs["code_label"] = orm.Str(args.code_label)
    if args.family_label:
        inputs["family_label"] = orm.Str(args.family_label)
    if args.no_build_family:
        inputs["build_family"] = orm.Bool(False)
    if args.max_iterations:
        inputs["max_iterations"] = orm.Int(args.max_iterations)
    if args.only is not None:
        inputs["only"] = orm.Int(args.only)
    inputs["dry_run"] = orm.Bool(args.dry_run)

    node = submit(OrbgenCalcWorkChain, **inputs)
    print(f"Submitted OrbgenCalcWorkChain PK = {node.pk}, UUID = {node.uuid}")
    print(f"Inspect: verdi process show {node.pk}")
    return 0


# ---------------------------------------------------------------------------
#  GridSearch
# ---------------------------------------------------------------------------


def _cmd_gridsearch(args) -> int:
    """submit OrbgenGridSearchWorkChain (迭代 (l_max, r_cut) 候选)."""
    from aiida import orm
    from aiida.engine import submit
    from aiida.orm import SinglefileData
    from aiida_orbgen.static.json_inputs import (
        load_abacus_config,
        load_orbgen_config,
        with_default_abacus,
    )
    from aiida.plugins import WorkflowFactory
    OrbgenGridSearchWorkChain = WorkflowFactory("orbgen.gridsearch")

    if not args.orbgen.is_file():
        print(f"Error: orbgen.json not found: {args.orbgen}", file=sys.stderr)
        return 1
    if not args.abacus.is_file():
        print(f"Error: abacus.json not found: {args.abacus}", file=sys.stderr)
        return 1

    orbgen_cfg = load_orbgen_config(args.orbgen)
    abacus_cfg = with_default_abacus(load_abacus_config(args.abacus))

    # 候选 l_max 来自 orbgen.json 的 lmaxmax, r_cut 来自 bessel_nao_rcut
    lmax_max = orbgen_cfg.get("geoms", [{}])[0].get("lmaxmax", 4)
    bessel_rcut = orbgen_cfg.get("bessel_nao_rcut", 9.0)
    if args.lmax_candidates:
        lmax_candidates = [int(x) for x in args.lmax_candidates.split(",")]
    else:
        # 默认 [1, 2, ..., lmax_max]
        lmax_candidates = list(range(1, lmax_max + 1))
    if args.rcut_candidates:
        rcut_candidates = [float(x) for x in args.rcut_candidates.split(",")]
    else:
        # 默认 [6.0, 7.0, 8.0, 9.0, bessel_rcut]
        rcut_candidates = [6.0, 7.0, 8.0, 9.0, bessel_rcut]
        rcut_candidates = sorted(set(rcut_candidates))

    print(f"orbgen.json: element={orbgen_cfg.get('element')}")
    print(f"abacus.json: basis={abacus_cfg['basis']}, "
          f"tolerance_meV={abacus_cfg['tolerance_meV']}")
    print(f"l_max candidates: {lmax_candidates}")
    print(f"r_cut candidates: {rcut_candidates}")
    print(f"search_strategy: {args.strategy}")

    # 将相对路径转换为绝对路径
    orbgen_path = args.orbgen.resolve()
    output_root_path = args.output_root.resolve()

    inputs: dict = {
        "siab_json": SinglefileData(file=str(orbgen_path)),
        "abacus_config": orm.Dict(dict=abacus_cfg),
        "l_max_candidates": orm.List(list=lmax_candidates),
        "r_cut_candidates": orm.List(list=rcut_candidates),
        "output_dir": orm.Str(str(output_root_path)),
        "search_strategy": orm.Str(args.strategy),
    }
    if args.code_label:
        inputs["code_label"] = orm.Str(args.code_label)
    if args.family_label:
        inputs["family_label"] = orm.Str(args.family_label)
    if args.no_build_family:
        inputs["build_family"] = orm.Bool(False)
    if args.max_iterations:
        inputs["max_iterations"] = orm.Int(args.max_iterations)
    inputs["dry_run"] = orm.Bool(args.dry_run)

    node = submit(OrbgenGridSearchWorkChain, **inputs)
    print(f"Submitted OrbgenGridSearchWorkChain PK = {node.pk}, UUID = {node.uuid}")
    print(f"Inspect: verdi process show {node.pk}")
    return 0


# ---------------------------------------------------------------------------
#  Parser
# ---------------------------------------------------------------------------


def add_submit_siab_parser(sub) -> None:
    """注册 ``aiida-orbgen submit-siab`` 子命令。"""
    p = sub.add_parser(
        "submit-siab",
        help=(
            "Two-level workchain submission: "
            "'submit-siab calc' for fixed (l_max, r_cut); "
            "'submit-siab gridsearch' for iterating candidates."
        ),
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub_p = p.add_subparsers(dest="subcmd", required=True)

    # ----- calc -----
    p_c = sub_p.add_parser(
        "calc",
        help="OrbgenCalcWorkChain: 跑特定 (l_max, r_cut) 下的 2N 个 abacus 任务.",
        description=(
            "OrbgenCalcWorkChain: 根据 orbgen.json 和 abacus.json 运行特定 l_max 和 r_cut 的任务.\n\n"
            "l_max 和 r_cut 可从 orbgen.json 自动提取 (bessel_nao_rcut + geoms[0].lmaxmax),\n"
            "也可通过 --l-max/--r-cut 显式指定以覆盖默认值."
        ),
    )
    _add_common_args(p_c)
    p_c.add_argument("--l-max", type=int, default=None,
                     help="最高角动量 (默认从 orbgen.json 的 geoms[0].lmaxmax 提取).")
    p_c.add_argument("--r-cut", type=float, default=None,
                     help="截断半径, Å (默认从 orbgen.json 的 bessel_nao_rcut[0] 提取).")
    p_c.add_argument("--only", type=int, default=None,
                     help="只跑第 N 个结构 (0-indexed, 调试).")
    p_c.set_defaults(func=cmd_submit_siab)

    # ----- gridsearch -----
    p_g = sub_p.add_parser(
        "gridsearch",
        help="OrbgenGridSearchWorkChain: 迭代 (l_max, r_cut) 候选, 选最小满足 tolerance 的.",
    )
    _add_common_args(p_g)
    p_g.add_argument(
        "--lmax-candidates", dest="lmax_candidates", default=None,
        help="l_max 候选列表, 逗号分隔 (如 '2,3,4'). 默认: [1..lmaxmax].",
    )
    p_g.add_argument(
        "--rcut-candidates", dest="rcut_candidates", default=None,
        help="r_cut 候选列表, 逗号分隔 (如 '6,7,8,9'). 默认: [6,7,8,9,bessel_rcut].",
    )
    p_g.add_argument(
        "--strategy", dest="strategy", default="iterative",
        choices=["iterative", "exhaustive"],
        help="搜索策略: iterative (默认, 从小到大, 满足即停) "
             "或 exhaustive (全部并行).",
    )
    p_g.set_defaults(func=cmd_submit_siab)

    p.set_defaults(func=cmd_submit_siab)


def _add_common_args(p):
    """Common CLI args shared by batch/advanced."""
    p.add_argument(
        "--orbgen", type=Path, required=True,
        help="orbgen.json (SIAB config) 路径.",
    )
    p.add_argument(
        "--abacus", type=Path, required=True,
        help="abacus.json (ABACUS config) 路径.",
    )
    p.add_argument(
        "--output-root", type=Path, default=Path("/tmp/test_orbgen_output"),
        help="SIAB 生成目录 (default: /tmp/test_orbgen_output).",
    )
    p.add_argument("--code-label", default=None,
                   help="AiiDA code label (default: from static.defaults).")
    p.add_argument("--family-label", default=None,
                   help="AiiDA pseudo family label (default: auto-infer).")
    p.add_argument("--no-build-family", action="store_true",
                   help="Do not auto-build the pseudo_family if missing.")
    p.add_argument("--max-iterations", type=int, default=None,
                   help="Max abacus.base workchain retry.")
    p.add_argument("--dry-run", action="store_true",
                   help="设置 workchain 的 dry_run=True, 不真正 submit.")
