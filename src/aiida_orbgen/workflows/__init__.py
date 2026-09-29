"""WorkChain implementations for aiida_orbgen.

两级 workchain 架构 (flowchart):

- :class:`OrbgenCalcWorkChain`  (entry point: ``orbgen.calc``)
  — 一个 ``(l_max, r_cut)``: SIAB pipeline + {PW, LCAO:nsw} × N 结构, 输出 ΔE
- :class:`OrbgenGridSearchWorkChain`  (entry point: ``orbgen.gridsearch``)
  — 多个候选, 按 ΔE 选最小的可接受组合

两个类都定义在 :mod:`aiida_orbgen.workflows.batch` —— 也就是
``pyproject.toml`` 注册的那个模块。本模块以前却从 ``advanced`` 重新导出
``OrbgenGridSearchWorkChain``, 导致 ``WorkflowFactory("orbgen.gridsearch")`` 与
``from aiida_orbgen.workflows import OrbgenGridSearchWorkChain`` 拿到**两个不同
的实现**(退出码 301/401/402/403 含义还不同, ``process_label`` 却同名, 在
provenance 里无法区分)。现在只有一个来源。

``workflows/advanced.py`` 仅作为参考保留: batch 的网格搜索还缺它那三个能力
(``stop_on_first_valid``、显式 ``candidates`` 列表、``max_l_max`` / ``max_r_cut``
上限), 但它不对应任何 entry point。
"""

from aiida_orbgen.workflows.batch import (
    OrbgenCalcWorkChain,
    OrbgenGridSearchWorkChain,
    build_abacus_child_inputs,
    run_siab_pipeline,
)

__all__ = [
    "OrbgenCalcWorkChain",
    "OrbgenGridSearchWorkChain",
    "build_abacus_child_inputs",
    "run_siab_pipeline",
]
