"""
aiida_orbgen.static - 静态配置 (默认参数模板 + JSON 加载)

将 ``aiida-orbgen`` 命令行工具 / Python API 使用的所有默认参数集中放在这里, 方便:

- 一处修改, 全局生效
- 用户可以 ``from aiida_orbgen.static import DEFAULTS`` 引用
- CLI 标志覆盖默认值
"""

from aiida_orbgen.static.defaults import (
    DEFAULTS,
    AIIDA_MANAGED_KEYS,
    DEFAULT_FAMILY_LABEL_TEMPLATE,
    DEFAULT_RESOURCES,
    DEFAULT_METADATA_OPTIONS,
    FAMILY_LABEL_KEYS,
    INPUT_OVERRIDES,
    apply_input_overrides,
)
from aiida_orbgen.static.json_inputs import (
    DEFAULT_ABACUS_CONFIG,
    load_abacus_config,
    load_orbgen_config,
    merge_input_overrides,
    merge_scheduler,
    parse_lmax_rcut_candidates,
    with_default_abacus,
)

__all__ = [
    # defaults.py
    "DEFAULTS",
    "AIIDA_MANAGED_KEYS",
    "DEFAULT_FAMILY_LABEL_TEMPLATE",
    "DEFAULT_RESOURCES",
    "DEFAULT_METADATA_OPTIONS",
    "FAMILY_LABEL_KEYS",
    "INPUT_OVERRIDES",
    "apply_input_overrides",
    # json_inputs.py
    "DEFAULT_ABACUS_CONFIG",
    "load_abacus_config",
    "load_orbgen_config",
    "merge_input_overrides",
    "merge_scheduler",
    "parse_lmax_rcut_candidates",
    "with_default_abacus",
]
