"""
aiida_orbgen.static.defaults - 默认参数

集中存放 ``aiida-orbgen`` 的所有可调参数。CLI / Python API 调用方
可以:

- 直接 ``import`` 当作只读常量
- 在调用入口 (``build_submission_config`` 等) 用 kwargs 覆盖
- 用户自定义: 复制本文件修改, 通过 ``--config-file`` 提供 (TODO)

修改本文件后, 跑 ``pytest tests/`` 确认没有破坏。
"""
from __future__ import annotations

from typing import Any, Dict


# ---------------------------------------------------------------------------
# AiiDA / scheduler
# ---------------------------------------------------------------------------

# AiiDA 中已注册的 code 标签 (verdi code list)
DEFAULT_CODE_LABEL: str = "abacus_lts@yeesuan"

# Slurm 队列名
DEFAULT_QUEUE_NAME: str = "q_ysuan"

# 每个节点 MPI 进程数
DEFAULT_NUM_MPI: int = 56

# 单节点 wallclock (秒)
DEFAULT_WALLCLOCK_SECONDS: int = 6 * 60 * 60  # 6 h

DEFAULT_MAX_MEMORY_KB: int = 350 * 1024 * 1024

# 工作流最大重试次数
DEFAULT_MAX_ITERATIONS: int = 3


# ---------------------------------------------------------------------------
# resources 子字典 (填入 metadata.options)
# ---------------------------------------------------------------------------

DEFAULT_RESOURCES: Dict[str, Any] = {
    "num_machines": 1,
    "num_mpiprocs_per_machine": DEFAULT_NUM_MPI,
    "tot_num_mpiprocs": DEFAULT_NUM_MPI,
}


# ---------------------------------------------------------------------------
# metadata.options 完整模板
# ---------------------------------------------------------------------------

DEFAULT_METADATA_OPTIONS: Dict[str, Any] = {
    "resources": dict(DEFAULT_RESOURCES),
    "max_wallclock_seconds": DEFAULT_WALLCLOCK_SECONDS,
    "max_memory_kb": DEFAULT_MAX_MEMORY_KB,
    "queue_name": DEFAULT_QUEUE_NAME,
    "withmpi": True,
}


# ---------------------------------------------------------------------------
# AiiDA-managed INPUT keys
#
# 这些 key 不传给 abacus.base workchain 的 ``parameters``,
# 因为它们由 AiiDA / pseudo_family 接管:
#   - pseudo_dir / orbital_dir  : AiiDA 自动注入 (从 pseudo_family)
#   - stru_file  / kpoint_file  : 我们用 StructureData / KpointsData 显式传
#   - wannier_card / suffix     : 不需要
#   - calculation               : workflow 决定 (默认 scf)
#
# ``out_wfc_lcao`` is deliberately **not** filtered any more (2026-09-20): the
# SIAB pipeline sets it to 1 so that ABACUS writes ``WFC_NAO_GAMMA1.txt``, which
# ``SIAB.spillage`` needs to fit the reference orbitals. Dropping it here meant
# every AiiDA-run LCAO job silently produced no wavefunctions, and the spillage
# step then had to rebuild them from ``data-0-H``/``data-0-S``
# (``aiida_orbgen.utils.report.assemble``). PW jobs never see the key:
# ``build_abacus_child_inputs`` pops it for ``basis == "pw"``.
# ---------------------------------------------------------------------------

AIIDA_MANAGED_KEYS: frozenset = frozenset({
    "pseudo_dir",
    "orbital_dir",
    "stru_file",
    "kpoint_file",
    "wannier_card",
    "suffix",
    "calculation",
    "basis_type",  # 由 WorkChain 根据 basis 列表动态设置
    "bessel_nao_rcut",
    "bessel_nao_lmax",
    "bessel_nao_tolerence",
    "bessel_descriptor",
    "bessel_smooth",
    "bessel_ecut",
    "bessel_rcut",
    "bessel_sigma",
    "bessel_screen_coeff",
})


# ---------------------------------------------------------------------------
# 默认 family label 模板
#
# 例: siab-u-nr-pbe-z6-nsw-9au-100Ry-g
#
# 占位符在 ``resolve_family_label()`` 中填充:
#   {element}  {rel}  {pp_xc}  {zval}  {rcut}  {ecut}  {lmax}
#   zval: 价电子数 (从 UPF PP_HEADER 的 z_valence 提取), 用于区分不同赝势
# ---------------------------------------------------------------------------

DEFAULT_FAMILY_LABEL_TEMPLATE: str = (
    "siab-{element}-{rel}-{pp_xc}-z{zval}-nsw-{rcut}au-{ecut}Ry-{lmax}"
)

# family label 字段顺序 (用于从 ORB 文件名 / UPF 文件名提取)
FAMILY_LABEL_KEYS: tuple = (
    "element", "rel", "pp_xc", "zval", "rcut", "ecut", "lmax",
)


# ---------------------------------------------------------------------------
# INPUT 参数覆盖
#
# 这些字段会在 SIAB ``generate_incar`` 写出 INPUT 之后被强制覆盖,
# 用于纠正 SIAB 默认值 (例如把 ``ks_solver=genelpa`` 改为 ``scalapack_gvx``)。
#
# 键 = INPUT 字段名 (ABACUS 标准), 值 = 覆盖值。
# 修改后, 下次跑 ``run`` / ``report`` 就会用新值生成 INPUT。
# 注意: 这些值会出现在最终 abacus.base 提交的 ``parameters.input`` 中。
# ---------------------------------------------------------------------------

INPUT_OVERRIDES: Dict[str, Any] = {
    "ks_solver": "scalapack_gvx",
}


def apply_input_overrides(params: Dict[str, Any]) -> Dict[str, Any]:
    """用 ``INPUT_OVERRIDES`` 覆盖 ``params`` (in-place + return)。

    同时过滤掉被 AiiDA 托管的 key (pseudo_dir / orbital_dir / ...)。
    """
    from aiida_orbgen.static.defaults import AIIDA_MANAGED_KEYS
    out = {k: v for k, v in params.items() if k not in AIIDA_MANAGED_KEYS}
    out.update(INPUT_OVERRIDES)
    return out


# ---------------------------------------------------------------------------
# 顶层 ``DEFAULTS`` 视图 (方便 ``from aiida_orbgen.static import DEFAULTS``)
# ---------------------------------------------------------------------------

DEFAULTS: Dict[str, Any] = {
    "code_label": DEFAULT_CODE_LABEL,
    "queue_name": DEFAULT_QUEUE_NAME,
    "num_mpi": DEFAULT_NUM_MPI,
    "wallclock_seconds": DEFAULT_WALLCLOCK_SECONDS,
    "max_memory_kb": DEFAULT_MAX_MEMORY_KB,
    "max_iterations": DEFAULT_MAX_ITERATIONS,
    "resources": DEFAULT_RESOURCES,
    "metadata_options": DEFAULT_METADATA_OPTIONS,
    "aiida_managed_keys": AIIDA_MANAGED_KEYS,
    "family_label_template": DEFAULT_FAMILY_LABEL_TEMPLATE,
}
