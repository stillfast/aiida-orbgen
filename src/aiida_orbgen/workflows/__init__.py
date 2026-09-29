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

模块划分
--------

``batch.py``     两个 WorkChain 本身 (只做编排: submit → collect → decide)
``_grid.py``     网格候选与停止规则 (``GridEntry`` 等纯逻辑)
``siab.py``      所有与 SIAB 打交道、生成/解析其文件的代码
``results.py``   输出 Dict 的组装 (4 个 calcfunction)

``advanced.py`` 已于 2026-09-29 删除: 它的三个独有能力和
(``stop_on_first_valid``、显式 ``candidates``、``max_l_max`` / ``max_r_cut``)
已移植进 batch 的网格搜索, 见 ``_removed-20260929/``。
"""

from aiida_orbgen.workflows.batch import (
    OrbgenCalcWorkChain,
    OrbgenGridSearchWorkChain,
)
from aiida_orbgen.workflows.siab import (
    build_abacus_child_inputs,
    n_atoms_from_stru,
    run_siab_pipeline,
)

__all__ = [
    "OrbgenCalcWorkChain",
    "OrbgenGridSearchWorkChain",
    "build_abacus_child_inputs",
    "n_atoms_from_stru",
    "run_siab_pipeline",
]
