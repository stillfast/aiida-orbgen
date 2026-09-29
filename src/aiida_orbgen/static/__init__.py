"""
aiida_orbgen.static - static configuration (default parameter templates + JSON loading)

All defaults used by the ``aiida-orbgen`` command line tool / Python API live
here, so that:

- A single edit takes effect everywhere
- Users can reference them via ``from aiida_orbgen.static import DEFAULTS``
- CLI flags override the defaults
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
