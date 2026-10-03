"""ConfigLoader: ``input.json`` + the ``parameters/`` preset tree.

This is the ``aiida-orbgen`` counterpart of
``aiida_uranium_workflow.utils.config.ConfigLoader``: the CLI never reads a
YAML preset itself, it hands an ``input.json`` to this loader and receives a
fully resolved :class:`ParamBundle`.

Layout of ``input.json``
------------------------

.. code-block:: json

    {
      "parameters": {
        "abacus": {"test": "test"},
        "orbgen": {"test": "test"}
      },
      "static": {
        "pseudo_path": "/abs/path/U.pbe-n-nc.14ve.UPF",
        "metadata": "yeesuan"
      },
      "profile": "aiida_profile",
      "code": {"abacus": "abacus_lts@yeesuan"}
    }

``parameters``
    One slot per *level*, so a run is composed instead of written out:

    * ``abacus`` → ``parameters/abacus/<category>.yml`` — ABACUS / scheduler
      presets (canonicalised into the ``abacus.json`` shape the workchains
      expect: ``basis`` / ``abacus.parameters.input`` /
      ``abacus.metadata.options`` / ``tolerance_meV``).
    * ``orbgen`` → ``parameters/orbgen/<category>.yml`` — SIAB (CSW-NAO)
      presets, i.e. the ``orbgen.json`` content (element / geoms / orbitals /
      bessel_nao_rcut / ecutjy / ...).  For a value-selection scan this preset is
      the **reference point** the ladder reduces from.
    * ``scan`` → ``parameters/scan/<category>.yml`` — the ladder and the criterion
      of a scan.  A scan preset *is* the ``input.json["scan"]`` section under a
      name; an inline section is merged over it (inline wins), so a single value can
      be varied for one run without writing a new preset.

    Each slot accepts three spellings::

        "abacus": "u_14ve"                     # parameters/abacus/abacus.yml
        "abacus": ["test", "u_14ve"]           # same file, two presets
        "abacus": {"test": "test"}             # parameters/abacus/test.yml

    The same three spellings apply to every slot, e.g.
    ``"scan": {"u_14ve": "basis_ladder"}`` for ``parameters/scan/u_14ve.yml``.

    The dict form is what the reference ``input.json`` uses: the key is the
    *file* (category) and the value is the preset name inside it.

``static``
    ``pseudo_path`` (absolute path of the UPF; injected as the SIAB
    ``pseudo_dir``), ``metadata`` (a scheduler-options preset name inside
    ``parameters/metadata.yml``), ``output_dir`` (SIAB run root, optional),
    ``abacus_input`` (ABACUS INPUT keys merged into every child of this run, last word:
    ``{"scf_thr": 1e-4, "mixing_beta": 0.1}``), ``tolerance_meV`` (the per-run criterion,
    overriding the preset's),
    ``dft_root`` / ``dft_roots`` (reference DFT trees, used by ``report``),
    ``siab_config`` (an inline SIAB config, see below).

    ``siab_config`` replaces the ``parameters.orbgen`` slot with the config
    itself, and ``siab_config_name`` (default ``"inline"``) is the preset name it
    is known by — that name becomes the run directory
    (``<output_dir>/<name>/lmax4_rcut10``).  ``aiida-orbgen select`` writes both so
    that "run the point the grid search chose" needs no preset edit.  The config is
    canonicalised and validated exactly like a YAML preset, and because it carries
    that point's ``bessel_nao_rcut`` / ``lmaxmax`` it resolves to a single
    candidate (→ ``orbgen.calc``).

The ``parameters/`` tree is **user input**: the plugin only ever *reads* it.
Nothing here (or anywhere else in the package) rewrites a preset — values such
as ``ecutjy``, ``bessel_nao_rcut`` or ``device`` always come from the YAML the
user wrote, and a change there propagates to the next ``run``/``report``
(a reference tree that no longer matches is detected and refetched instead of
being silently mixed).
``code``
    ``{"abacus": "<AiiDA code label>"}``.
``workflow``
    Optional. ``"orbgen.calc"`` (single ``(l_max, r_cut)``),
    ``"orbgen.gridsearch"`` (whole candidate grid), or one of the two
    value-selection scans ``"orbgen.ecutwfc"`` / ``"orbgen.basis"``. When absent the
    loader decides: a non-empty ``scan`` section says which scan its keys describe
    (:data:`SCAN_KEYS`), otherwise the candidate grid does (one candidate →
    ``orbgen.calc``, more than one → ``orbgen.gridsearch``).
``scan``
    Optional.  Normally it is not written out at all but *named* through the ``scan``
    preset slot (``parameters/scan/<category>.yml``); whatever is written here is
    merged over that preset, key by key.  Its keys are listed in :data:`SCAN_KEYS` and
    every one of them is validated, so a mistyped ladder fails in ``check`` instead of
    submitting a different scan:

    .. code-block:: json

        "scan": {
          "ecutwfc_values": [100, 120, 150, 180, 200],
          "with_lcao": true
        }

    or, for ``orbgen.basis`` (``l_max``/``r_cut`` stay in the orbgen preset — they are
    its candidate grid, i.e. the *reference point* of the ladder):

    .. code-block:: json

        "scan": {
          "reference_ecutjy": 150.0,
          "ecutjy_values": [125, 100],
          "l_max_values": [3],
          "r_cut_values": [11, 10],
          "ecutwfc": 180,
          "atomization_tolerance_meV": 50,
          "pw_reference_pk": 469028
        }
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from aiida_orbgen.utils.yamlio import read_json, read_yaml

__all__ = [
    "PARAMETERS_DIR",
    "METADATA_FILE",
    "WORKFLOW_CALC",
    "WORKFLOW_GRIDSEARCH",
    "WORKFLOW_ECUTWFC",
    "WORKFLOW_BASIS",
    "SUPPORTED_WORKFLOWS",
    "SCAN_WORKFLOWS",
    "SCAN_KEYS",
    "canonical_scan_config",
    "scan_families",
    "PresetEntry",
    "ParamBundle",
    "ConfigLoader",
    "canonical_abacus_config",
    "canonical_orbgen_config",
    "candidates_from_orbgen",
    "validate_siab_config",
    "validate_abacus_input",
    "SIAB_ONLY_INPUT_KEYS",
]

#: ``<package>/parameters`` — the preset tree shipped with the plugin.
PARAMETERS_DIR = Path(__file__).resolve().parent.parent / "parameters"
METADATA_FILE = PARAMETERS_DIR / "metadata.yml"

WORKFLOW_CALC = "orbgen.calc"
WORKFLOW_GRIDSEARCH = "orbgen.gridsearch"
WORKFLOW_ECUTWFC = "orbgen.ecutwfc"
WORKFLOW_BASIS = "orbgen.basis"
SUPPORTED_WORKFLOWS = (
    WORKFLOW_CALC,
    WORKFLOW_GRIDSEARCH,
    WORKFLOW_ECUTWFC,
    WORKFLOW_BASIS,
)

#: The two ``scan`` workflows choose *parameters*: they produce a decision, not an
#: orbital flat, so ``report`` refuses them.
SCAN_WORKFLOWS = (WORKFLOW_ECUTWFC, WORKFLOW_BASIS)

#: slot -> (sub-directory, default preset file inside it).  One slot per level:
#: which DFT to run (``abacus``), which reference basis and geometries to fit
#: (``orbgen``), and which ladder/criterion a value-selection scan follows (``scan``).
SLOTS: dict[str, tuple[str, str]] = {
    "abacus": ("abacus", "abacus.yml"),
    "orbgen": ("orbgen", "orbgen.yml"),
    "scan": ("scan", "scan.yml"),
}

DEFAULT_TOLERANCE_MEV = 100.0
DEFAULT_BASIS = ["pw", "lcao_nsw"]

#: Keys the AiiDA-side SIAB pipeline (``generate_all_from_json`` →
#: ``generate_nsw`` / ``generate_incar`` / ``generate_stru``) actually reads.
#: SIAB declares ``abacus_command`` / ``pseudo_dir`` / ``element`` /
#: ``bessel_nao_rcut`` / ``geoms`` / ``orbitals`` compulsory in
#: ``SIAB.io.param.ParamAssert``; ``pseudo_dir`` is injected from
#: ``static.pseudo_path`` and ``abacus_command`` is only used by SIAB's own DFT
#: runner, so neither is required here. ``ecutjy`` *is* required: ``generate_nsw``
#: indexes it directly (missing ⇒ ``KeyError: 'ecutjy'`` inside the daemon).
SIAB_REQUIRED_KEYS = ("element", "bessel_nao_rcut", "ecutjy", "geoms", "orbitals")
SIAB_REQUIRED_GEOM_KEYS = ("proto", "pertkind", "pertmags", "lmaxmax")
#: Needed only by the final CSW-NAO orbital step, so their absence is a warning.
SIAB_RECOMMENDED_KEYS = ("ecutwfc", "abacus_command", "environment", "mpi_command")

#: Keys that belong to SIAB / the orbital generator, never to an ABACUS INPUT.
#: ``ecutjy`` is the notorious one: ABACUS 3.10.0 aborts with
#: "THE PARAMETER NAME 'ecutjy' IS NOT USED! ... Bad parameter", which makes
#: every AbacusCalculation Excepted (observed 2026-09-19).
SIAB_ONLY_INPUT_KEYS = ("ecutjy", "vloc_aux", "primitive_type", "nzeta")


#: ``input.json["scan"]`` — the parameters of the two value-selection workflows.
#: ``key -> (workflow family, python type, what it does)``; the family is what the
#: loader uses to tell "this input.json wants an ecutwfc scan" from "… a basis scan".
SCAN_KEYS: dict[str, tuple[str, type, str]] = {
    # -- orbgen.ecutwfc -----------------------------------------------------
    "ecutwfc_values": ("ecutwfc", list, "the ladder of PW cutoffs (Ry)"),
    "ecutwfc_scale": ("ecutwfc", list,
                      "multipliers of baseline_ecutwfc (used without ecutwfc_values)"),
    "baseline_ecutwfc": ("ecutwfc", (int, float),
                         "cutoff the reference was generated at"),
    "reference_ecutwfc": ("ecutwfc", (int, float),
                          "cutoff that counts as converged (default: the largest)"),
    "with_lcao": ("ecutwfc", bool,
                  "also run one LCAO child per geometry at reference_ecutwfc"),
    # -- orbgen.basis -------------------------------------------------------
    "reference_ecutjy": ("basis", (int, float), "ecutjy of the reference point (Ry)"),
    "ecutjy_values": ("basis", list, "ecutjy candidates to reduce to (Ry)"),
    "l_max_values": ("basis", list, "l_max candidates to reduce to"),
    "r_cut_values": ("basis", list, "r_cut candidates to reduce to (Bohr)"),
    "strategy": ("basis", str, "'ladder' (default) or 'exhaustive'"),
    "ecutwfc": ("basis", (int, float), "PW cutoff of the reference children (Ry)"),
    "stop_on_first_pass": ("basis", bool,
                           "stop at the cheapest candidate inside the tolerance"),
    "atomization_tolerance_meV": ("basis", (int, float),
                                  "also keep |d(atomization energy)| inside this"),
    "by": ("basis", str, "cost key of 'cheapest': seconds | cost | nchi"),
    "pw_reference": ("basis", dict,
                     "{geometry: {energy, n_atoms}}: reuse a computed PW reference"),
    "pw_reference_pk": ("basis", int,
                        "PK of a scan whose PW reference to reuse (no PW child runs)"),
}

#: Keys that select a workflow rather than configure it.
SCAN_SELECTOR_KEYS: dict[str, str] = {
    key: family for key, (family, _type, _help) in SCAN_KEYS.items()
}


def canonical_scan_config(scan: dict | None, *, source: str = "input.json['scan']") -> dict:
    """Validate ``input.json['scan']`` and return it as the workflows want it.

    The section holds the *parameters of the scan* (the ladder, the criterion, the
    reference point) — never code: both workflows are generic and everything they do
    differently comes from here.  Unknown keys are an error rather than a silent
    no-op, because a mistyped ladder would otherwise submit a valid but different
    scan; a key that belongs to the other scan family is reported the same way, since
    ``ecutwfc_values`` next to ``r_cut_values`` is almost always a leftover.
    """
    if scan is None:
        return {}
    if not isinstance(scan, dict):
        raise TypeError(
            f"{source} must be an object mapping the scan parameters to values, "
            f"got {type(scan).__name__}"
        )
    unknown = [key for key in scan if key not in SCAN_KEYS]
    if unknown:
        raise KeyError(
            f"{source} has unknown key(s) {sorted(unknown)}; supported: "
            f"{sorted(SCAN_KEYS)}"
        )
    out: dict[str, Any] = {}
    families: set[str] = set()
    for key, value in scan.items():
        family, expected, help_text = SCAN_KEYS[key]
        families.add(family)
        if value is None:
            continue
        if isinstance(expected, tuple):
            if isinstance(value, bool) or not isinstance(value, expected):
                raise TypeError(
                    f"{source}.{key} must be a number, got {value!r} ({help_text})"
                )
        elif expected is bool:
            if not isinstance(value, bool):
                raise TypeError(
                    f"{source}.{key} must be true or false, got {value!r} "
                    f"({help_text})"
                )
        elif expected is list:
            if not isinstance(value, list) or not value:
                raise TypeError(
                    f"{source}.{key} must be a non-empty list, got {value!r} "
                    f"({help_text})"
                )
            for item in value:
                if isinstance(item, bool) or not isinstance(item, (int, float)):
                    raise TypeError(
                        f"{source}.{key} may only contain numbers, got {item!r}"
                    )
        elif not isinstance(value, expected):
            raise TypeError(
                f"{source}.{key} must be a {expected.__name__}, got {value!r} "
                f"({help_text})"
            )
        if key in ("strategy",) and value not in ("ladder", "exhaustive"):
            raise ValueError(
                f"{source}.{key}={value!r} is not one of 'ladder', 'exhaustive'"
            )
        if key == "by" and value not in ("seconds", "cost", "nchi"):
            raise ValueError(
                f"{source}.{key}={value!r} is not one of 'seconds', 'cost', 'nchi'"
            )
        if key in ("l_max_values",):
            value = [int(item) for item in value]
        out[key] = value
    return out


def scan_families(scan: dict) -> set[str]:
    """Which scan family the keys of ``scan`` belong to (see :data:`SCAN_KEYS`)."""
    return {SCAN_KEYS[key][0] for key in scan if key in SCAN_KEYS}


def validate_siab_config(config: dict, *, source: str = "orbgen preset") -> list[str]:
    """Check an orbgen preset *before* it reaches a daemon.

    Fatal problems raise ``ValueError``; everything else comes back as warnings.
    The checks themselves live in :mod:`aiida_orbgen.spec` (pydantic), which is
    also what the WorkChain boundary uses, so a preset cannot pass here and fail
    there.  Two classes of mistake are caught that used to reach SIAB:

    * keys SIAB reads only from an orbital's ``model_kwargs`` (``vloc_aux``,
      ``lloc_min``) written at the top level — silently dropped, so a requested g
      channel came out empty;
    * an ``nzeta`` scheme the primitive basis cannot provide, which SIAB only
      notices after the reference DFT has been paid for.
    """
    from aiida_orbgen.spec import OrbgenSpec

    # SIAB's `ParamAssert`/`OrbitalAssert` demand these four keys in *every* orbital
    # entry (SIAB/io/param.py: ORBITAL_COMPULSORY_).  Our model fills the missing ones
    # with defaults, so a config can pass here and still be rejected by SIAB — which
    # only happens at the very end of `report`, after the whole grid has been computed
    # (2026-09-30: "orbital 0 does not have all the compulsory keys").
    required = ("nzeta", "geoms", "nbands", "checkpoint")
    for index, orbital in enumerate(config.get("orbitals") or []):
        if not isinstance(orbital, dict):
            continue
        missing = [key for key in required if key not in orbital]
        if missing:
            raise ValueError(
                f"{source}: orbitals[{index}] is missing {missing}; SIAB requires all "
                f"of {list(required)} (use `checkpoint: null` and `nbands: \"occ\"` if "
                f"you mean the defaults)"
            )

    flat = OrbgenSpec.flat_vloc_aux(config)
    if flat:
        raise ValueError(
            f"{source}: {flat[0]}, where SIAB ignores them — nest them under "
            f"'model_kwargs' (model_kwargs: {{lloc_min: 4, vloc_aux: /abs/path.UPF}})."
        )

    try:
        spec = OrbgenSpec.model_validate(config)
    except Exception as exc:                     # pydantic ValidationError
        raise ValueError(f"{source}: {exc}") from exc

    warnings = [
        f"{source}: '{key}' is not set — needed by the final CSW-NAO orbital "
        f"step (`aiida-orbgen report`), not by the DFT/workchain path"
        for key in SIAB_RECOMMENDED_KEYS
        if not config.get(key)
    ]
    warnings.extend(f"{source}: {warning}" for warning in spec.warnings())
    return warnings


# ---------------------------------------------------------------------------
#  Bundle
# ---------------------------------------------------------------------------


@dataclass
class PresetEntry:
    """One resolved YAML preset."""

    slot: str          # "abacus" | "orbgen"
    name: str          # preset name inside the file
    config: dict       # resolved (canonicalised) configuration
    source: str        # "<file>#<name>", for error messages / provenance

    @property
    def label(self) -> str:
        return self.name


@dataclass
class ParamBundle:
    """Everything ``run`` needs: resolved presets, codes, scheduler options."""

    input_params: dict
    workflow: str
    workflow_explicit: bool
    abacus_presets: list[PresetEntry]
    orbgen_presets: list[PresetEntry]
    metadata: dict
    metadata_name: str
    static: dict
    codes: dict
    profile: str | None
    pseudo_path: str | None
    output_root: Path | None
    input_json: Path | None = None
    search_strategy: str = "exhaustive"
    #: ``input.json["scan"]``: the parameters of ``orbgen.ecutwfc`` / ``orbgen.basis``
    scan: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    # -- derived -----------------------------------------------------------
    @property
    def tolerance_meV(self) -> float:
        """Tolerance of the first ABACUS preset (all presets share it)."""
        for entry in self.abacus_presets:
            return float(entry.config.get("tolerance_meV", DEFAULT_TOLERANCE_MEV))
        return DEFAULT_TOLERANCE_MEV

    @property
    def code_label(self) -> str | None:
        return self.codes.get("abacus")

    def pairs(self) -> list[tuple[PresetEntry, PresetEntry]]:
        """Cartesian product ``abacus preset × orbgen preset``."""
        return [
            (abacus, orbgen)
            for abacus in self.abacus_presets
            for orbgen in self.orbgen_presets
        ]

    def candidates(self, orbgen: PresetEntry | None = None) -> list[tuple[int, float]]:
        """``(l_max, r_cut)`` candidates of one orbgen preset."""
        if orbgen is None:
            orbgen = self.orbgen_presets[0]
        limits = self.abacus_presets[0].config if self.abacus_presets else {}
        return candidates_from_orbgen(orbgen.config, limits)


# ---------------------------------------------------------------------------
#  Preset canonicalisation
# ---------------------------------------------------------------------------


def _deep_update(base: dict, override: dict) -> dict:
    """Recursive ``dict.update`` (override wins, dicts merged)."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def canonical_abacus_config(
    preset: dict,
    *,
    code: str | None = None,
    options: dict | None = None,
    tolerance_meV: float | None = None,
    input_extra: dict | None = None,
) -> dict:
    """Turn a ``parameters/abacus/*.yml`` preset into the ``abacus.json`` shape.

    Two spellings are accepted:

    * flat — ``{parameters: {input: {...}, max_iterations: N}, basis: [...]}``
      (what ``parameters/abacus/test.yml`` uses), optionally with
      ``metadata.options`` at the top level;
    * nested — ``{abacus: {code: ..., parameters: {...}, metadata: {...}},
      basis: [...]}`` (the ``abacus.json`` spelling itself).

    The result always looks like::

        {
          "basis": ["pw", "lcao_nsw"],
          "abacus": {
            "code": "<label>",
            "parameters": {"input": {...}},
            "metadata": {"options": {...}}
          },
          "tolerance_meV": 100.0,
          "max_l_max": ..., "max_r_cut": ...
        }
    """
    preset = preset or {}
    nested = preset.get("abacus") if isinstance(preset.get("abacus"), dict) else {}
    nested = copy.deepcopy(nested)

    params = copy.deepcopy(preset.get("parameters") or nested.get("parameters") or {})
    input_overrides = copy.deepcopy(params.get("input") or {})

    # scheduler options: metadata.yml < top-level preset < nested preset
    merged_options = copy.deepcopy(options or {})
    merged_options = _deep_update(merged_options, preset.get("metadata", {}).get("options", {}))
    merged_options = _deep_update(merged_options, nested.get("metadata", {}).get("options", {}))

    config: dict[str, Any] = {
        "basis": list(preset.get("basis") or nested.get("basis") or DEFAULT_BASIS),
        "abacus": {
            "parameters": {"input": input_overrides},
            "metadata": {"options": merged_options},
        },
    }

    code_label = code or nested.get("code") or preset.get("code")
    if code_label:
        config["abacus"]["code"] = code_label

    # metadata label/description are optional passthroughs
    for key in ("label", "description"):
        for source in (nested.get("metadata", {}), preset.get("metadata", {})):
            if key in source:
                config["abacus"]["metadata"][key] = source[key]

    max_iterations = (
        preset.get("max_iterations")
        or params.get("max_iterations")
        or nested.get("max_iterations")
    )
    if max_iterations is not None:
        config["abacus"]["max_iterations"] = int(max_iterations)

    # `input.json["static"]["abacus_input"]`: the last word on the ABACUS INPUT of every
    # child of this run -- the per-run counterpart of a preset's `parameters.input`, so a
    # `scf_thr`/`mixing_beta` for one difficult system does not need a new preset file in
    # the plugin tree.  It goes through the same validation as a preset (SIAB-only keys
    # are rejected) because `validate_abacus_input` reads this very dict.
    if input_extra:
        if not isinstance(input_extra, dict):
            raise TypeError(
                "input.json['static']['abacus_input'] must be an object of ABACUS "
                f"INPUT keys, got {type(input_extra).__name__}"
            )
        config["abacus"]["parameters"]["input"] = _deep_update(
            input_overrides, input_extra
        )

    config["tolerance_meV"] = float(
        tolerance_meV
        if tolerance_meV is not None
        else preset.get("tolerance_meV", nested.get("tolerance_meV", DEFAULT_TOLERANCE_MEV))
    )

    # candidate-grid limits (OrbgenGridSearchWorkChain honours these)
    for key in ("max_l_max", "max_r_cut"):
        value = preset.get(key, nested.get(key))
        if value is not None:
            config[key] = value

    return config


def canonical_orbgen_config(
    preset: dict,
    *,
    pseudo_path: str | None = None,
) -> dict:
    """Turn a ``parameters/orbgen/*.yml`` preset into a SIAB JSON config.

    ``static.pseudo_path`` becomes SIAB's ``pseudo_dir`` (the key
    ``resolve_paths_from_json`` uses to locate the UPF); a ``pseudo_dir``
    written inside the preset survives only when ``pseudo_path`` is unset.
    Everything else is passed through verbatim.
    """
    config = copy.deepcopy(preset or {})
    if pseudo_path:
        config["pseudo_dir"] = str(Path(pseudo_path).expanduser().resolve())
    elif "pseudo_dir" in config:
        config["pseudo_dir"] = str(Path(config["pseudo_dir"]).expanduser())
    return config


def validate_abacus_input(config: dict, *, source: str = "abacus preset") -> list[str]:
    """Check the ABACUS-side ``parameters.input`` of a preset.

    Fatal: SIAB-only keys (``ecutjy`` …) — ABACUS refuses the whole INPUT
    ("THE PARAMETER NAME 'ecutjy' IS NOT USED! ... Bad parameter") and every
    child calculation is Excepted.  Warnings: AiiDA-managed keys (the workchain
    drops them, so setting them has no effect) and LCAO output switches that the
    spillage step needs being switched *off*.
    """
    from aiida_orbgen.spec import AbacusSpec

    spec = AbacusSpec(
        basis=list(config.get("basis") or DEFAULT_BASIS),
        tolerance_meV=float(config.get("tolerance_meV", DEFAULT_TOLERANCE_MEV)),
        parameters_input=(
            config.get("abacus", {}).get("parameters", {}).get("input", {}) or {}
        ),
    )
    errors = spec.errors()
    if errors:
        raise ValueError(f"{source}: {errors[0]}")
    return [f"{source}: {warning}" for warning in spec.warnings()]


def candidates_from_orbgen(orbgen_cfg: dict, limits: dict | None = None) -> list[tuple[int, float]]:
    """``(l_max, r_cut)`` candidate grid of a SIAB config.

    Delegates to :func:`aiida_orbgen.static.json_inputs.parse_lmax_rcut_candidates`
    (which understands ``basis_candidates`` / ``lmax_candidates`` /
    ``rcut_candidates`` / ``bessel_nao_rcut`` + ``geoms[0].lmaxmax``) and then
    applies the optional ``max_l_max`` / ``max_r_cut`` caps.
    """
    from aiida_orbgen.static.json_inputs import parse_lmax_rcut_candidates

    grid = parse_lmax_rcut_candidates(orbgen_cfg or {})
    limits = limits or {}
    max_lmax = limits.get("max_l_max")
    max_rcut = limits.get("max_r_cut")
    if max_lmax is not None:
        grid = [(lm, rc) for (lm, rc) in grid if lm <= int(max_lmax)]
    if max_rcut is not None:
        grid = [(lm, rc) for (lm, rc) in grid if rc <= float(max_rcut)]
    return sorted(set(grid), key=lambda item: (item[0], item[1]))


# ---------------------------------------------------------------------------
#  Loader
# ---------------------------------------------------------------------------


class ConfigLoader:
    """Read an ``input.json`` and the ``parameters/`` YAML files it points at."""

    #: Keys ``input.json`` must contain. ``profile`` / ``code`` may be left out
    #: (they can come from the presets' ``abacus.code`` and the default
    #: profile), but the reference layout always carries them.
    REQUIRED_KEYS = ("parameters", "static")

    def __init__(self, input_json_path: str | Path) -> None:
        self.input_json_path = Path(input_json_path)
        self.input_params: dict[str, Any] = read_json(self.input_json_path)
        self._warnings: list[str] = []

    # -- public API --------------------------------------------------------
    def load_all(self) -> ParamBundle:
        """Resolve every preset referenced by ``input.json``."""
        self._validate()
        self._warnings = []

        static = self.input_params.get("static") or {}
        codes = self._codes()
        profile = self.input_params.get("profile")
        pseudo_path = static.get("pseudo_path")
        if pseudo_path and not Path(str(pseudo_path)).expanduser().is_file():
            raise FileNotFoundError(
                f"static.pseudo_path does not exist: {pseudo_path}"
            )
        if pseudo_path:
            # Cheapest possible place to catch a broken pseudopotential: `check` and
            # `run` both come through here, and the problems found this way (duplicate
            # angular-momentum channels, mis-sized blocks) only show up as tens-of-eV
            # nonsense — or an ABACUS assertion — after a full grid has been paid for.
            from aiida_orbgen.utils.upf import upf_warnings

            self._warnings.extend(upf_warnings(str(pseudo_path)))

        metadata, metadata_name = self._metadata()

        abacus_presets = self._load_slot(
            "abacus",
            code=codes.get("abacus"),
            options=metadata.get("options", {}),
            # `static.tolerance_meV` wins over the preset's own value; the code
            # default (100 meV/atom) is the fallback in canonical_abacus_config.
            tolerance_meV=static.get("tolerance_meV"),
            input_extra=static.get("abacus_input"),
        )
        orbgen_presets = self._inline_orbgen_preset(pseudo_path=pseudo_path)
        if orbgen_presets:
            if "orbgen" in (self.input_params.get("parameters") or {}):
                self._warnings.append(
                    "static.siab_config is set, so parameters['orbgen'] is ignored "
                    "(the inline config is the whole SIAB configuration)"
                )
        else:
            orbgen_presets = self._load_slot("orbgen", pseudo_path=pseudo_path)

        if not abacus_presets:
            raise KeyError(
                f"input.json['parameters'] resolved to no ABACUS preset; "
                f"check the 'abacus' slot of {self.input_json_path}"
            )
        if not orbgen_presets:
            raise KeyError(
                f"input.json['parameters'] resolved to no orbgen preset; "
                f"check the 'orbgen' slot of {self.input_json_path}"
            )

        output_root = static.get("output_dir")
        scan = self._resolve_scan()

        workflow, explicit = self._resolve_workflow(abacus_presets, orbgen_presets, scan)

        return ParamBundle(
            input_params=self.input_params,
            workflow=workflow,
            workflow_explicit=explicit,
            abacus_presets=abacus_presets,
            orbgen_presets=orbgen_presets,
            metadata=metadata,
            metadata_name=metadata_name,
            static=static,
            codes=codes,
            profile=profile,
            pseudo_path=str(pseudo_path) if pseudo_path else None,
            output_root=Path(output_root).expanduser() if output_root else None,
            input_json=self.input_json_path,
            search_strategy=str(
                self.input_params.get("search_strategy")
                or static.get("search_strategy")
                or "exhaustive"
            ),
            scan=scan,
            warnings=list(self._warnings),
        )

    # -- validation --------------------------------------------------------
    def _validate(self) -> None:
        if not isinstance(self.input_params, dict):
            raise TypeError(f"{self.input_json_path}: expected a JSON object")
        for key in self.REQUIRED_KEYS:
            if key not in self.input_params:
                raise KeyError(
                    f"input.json is missing required key '{key}': "
                    f"{self.input_json_path}"
                )
        parameters = self.input_params["parameters"]
        if not isinstance(parameters, dict) or not parameters:
            raise TypeError("input.json['parameters'] must be a non-empty object")
        unknown = [slot for slot in parameters if slot not in SLOTS]
        if unknown:
            raise KeyError(
                f"input.json['parameters'] has unknown slot(s) {unknown}; "
                f"supported: {sorted(SLOTS)}"
            )
        static = self.input_params["static"]
        if not isinstance(static, dict):
            raise TypeError("input.json['static'] must be an object")

    def _codes(self) -> dict:
        raw = self.input_params.get("code") or {}
        codes: dict[str, str] = {}
        if isinstance(raw, str):
            codes["abacus"] = raw
        elif isinstance(raw, dict):
            codes = {str(k): str(v) for k, v in raw.items()}
        else:
            raise TypeError("input.json['code'] must be a string or an object")
        if "abacus" not in codes:
            raise KeyError(
                "input.json['code'] must provide the 'abacus' code label "
                "(e.g. \"code\": {\"abacus\": \"abacus_lts@yeesuan\"})"
            )
        return codes

    def _metadata(self) -> tuple[dict, str]:
        name = (self.input_params.get("static") or {}).get("metadata")
        if not name:
            return {}, ""
        table = read_yaml(METADATA_FILE)
        if name not in table:
            raise KeyError(
                f"Metadata preset '{name}' not found in {METADATA_FILE}; "
                f"available: {sorted(table)}"
            )
        entry = table[name] or {}
        if not isinstance(entry, dict):
            raise TypeError(f"metadata preset '{name}' must be a mapping")
        return entry, str(name)

    # -- preset loading ----------------------------------------------------
    def _inline_orbgen_preset(self, **canonical_kwargs) -> list[PresetEntry]:
        """The ``static.siab_config`` override, as a one-entry preset list.

        ``aiida-orbgen select`` writes it: a SIAB config that already ran (and was
        validated) for one grid point, so re-running *that* point needs no preset
        edit.  Because the config carries the point's own ``bessel_nao_rcut`` and
        ``lmaxmax``, the candidate grid it yields has exactly one entry and the
        loader picks ``orbgen.calc`` by itself.

        The override goes through the same canonicalisation and validation as a
        YAML preset — a hand-written one is not a way to skip the checks.
        """
        static = self.input_params.get("static") or {}
        raw = static.get("siab_config")
        if raw is None:
            return []
        if not isinstance(raw, dict) or not raw:
            raise TypeError(
                f"input.json['static']['siab_config'] must be the SIAB config "
                f"itself (a non-empty object), got {type(raw).__name__}"
            )
        # The preset name is not cosmetic: it names the run directory
        # (``<output_dir>/<name>/lmax4_rcut10``) and appears in the provenance, so
        # ``static.siab_config_name`` lets a project call it something meaningful
        # instead of the key it happened to be written under.
        name = static.get("siab_config_name") or "inline"
        if not isinstance(name, str) or not name.strip():
            raise TypeError(
                "input.json['static']['siab_config_name'] must be a non-empty string"
            )
        source = f"{self.input_json_path}#static.siab_config"
        config = canonical_orbgen_config(raw, **canonical_kwargs)
        self._warnings.extend(validate_siab_config(config, source=source))
        return [PresetEntry(
            slot="orbgen", name=str(name).strip(), config=config, source=source
        )]

    def _resolve_scan(self) -> dict:
        """The ``scan`` section: the preset(s) of the ``scan`` slot, plus the inline one.

        A scan preset holds the ladder and the criterion of one scan (e.g.
        ``parameters/scan/u_14ve.yml#basis_ladder``); ``input.json["scan"]`` then only has
        to say *what to change for this run* — the two are merged with the inline keys
        winning, so a project can keep its ladders in the preset tree and still vary one
        value (``{"by": "nchi"}``) without a new preset.
        """
        scan = {}
        for entry in self._load_slot("scan"):
            clash = sorted(set(scan) & set(entry.config))
            if clash:
                raise KeyError(
                    f"scan presets {clash} overlap: {entry.source} sets keys another "
                    f"selected preset already set ({clash})"
                )
            scan.update(entry.config)
        inline = canonical_scan_config(
            self.input_params.get("scan"), source=f"{self.input_json_path}#scan"
        )
        if inline and scan:
            note = (
                f"{self.input_json_path}#scan overrides "
                f"{sorted(set(inline) & set(scan))} of the scan preset"
                if set(inline) & set(scan) else None
            )
            if note:
                self._warnings.append(note)
        scan.update(inline)
        return scan

    def _load_slot(self, slot: str, **canonical_kwargs) -> list[PresetEntry]:
        """Resolve one ``parameters[<slot>]`` entry into a list of presets."""
        parameters = self.input_params["parameters"]
        if slot not in parameters:
            return []
        sub_dir, default_file = SLOTS[slot]
        value = parameters[slot]

        # dict form: {"<category-file>": preset | [presets], ...}
        if isinstance(value, dict):
            entries: list[PresetEntry] = []
            for category, preset_value in value.items():
                if not isinstance(category, str) or not category:
                    raise TypeError(
                        f"parameters['{slot}'] keys must be non-empty strings, "
                        f"got {category!r}"
                    )
                table_path = PARAMETERS_DIR / sub_dir / f"{category}.yml"
                table = read_yaml(table_path)
                entries.extend(
                    self._resolve(table, self._names(slot, preset_value),
                                  slot=slot, source=table_path, **canonical_kwargs)
                )
            return entries

        table_path = PARAMETERS_DIR / sub_dir / default_file
        table = read_yaml(table_path)
        return self._resolve(
            table, self._names(slot, value), slot=slot, source=table_path,
            **canonical_kwargs,
        )

    @staticmethod
    def _names(slot: str, value: Any) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple)):
            if not all(isinstance(item, str) for item in value):
                raise TypeError(
                    f"parameters['{slot}'] must be a string or a list of "
                    f"strings, got {value!r}"
                )
            return list(value)
        raise TypeError(
            f"parameters['{slot}'] must be a preset name, a list of preset "
            f"names, or a {{file: preset}} object, got {value!r}"
        )

    def _resolve(
        self,
        table: dict,
        names: Iterable[str],
        *,
        slot: str,
        source: Path,
        **canonical_kwargs,
    ) -> list[PresetEntry]:
        entries: list[PresetEntry] = []
        for name in names:
            if name not in table:
                raise KeyError(
                    f"Preset '{name}' not found in {source}; "
                    f"available: {sorted(table)}"
                )
            raw = table[name] or {}
            if not isinstance(raw, dict):
                raise TypeError(
                    f"Preset '{name}' in {source} must be a mapping, "
                    f"got {type(raw).__name__}"
                )
            if slot == "abacus":
                config = canonical_abacus_config(raw, **canonical_kwargs)
                self._warnings.extend(
                    validate_abacus_input(config, source=f"{source}#{name}")
                )
            elif slot == "scan":
                # a scan preset *is* the `scan` section: same schema, same validation
                config = canonical_scan_config(raw, source=f"{source}#{name}")
            else:
                config = canonical_orbgen_config(raw, **canonical_kwargs)
                self._warnings.extend(
                    validate_siab_config(config, source=f"{source}#{name}")
                )
            entries.append(
                PresetEntry(
                    slot=slot,
                    name=str(name),
                    config=config,
                    source=f"{source}#{name}",
                )
            )
        return entries

    # -- workflow selection ------------------------------------------------
    def _resolve_workflow(
        self,
        abacus_presets: list[PresetEntry],
        orbgen_presets: list[PresetEntry],
        scan: dict | None = None,
    ) -> tuple[str, bool]:
        """``(workflow, explicit)``: the ``workflow`` key, else what the inputs imply.

        ``input.json["workflow"]`` always wins.  Otherwise ``scan`` decides — its keys
        say which of the two value-selection scans is meant (``ecutwfc_values`` =
        ``orbgen.ecutwfc``; ``ecutjy_values``/``r_cut_values``/… = ``orbgen.basis``) —
        and without a ``scan`` section the candidate grid does, as before.
        """
        explicit = self.input_params.get("workflow")
        if explicit:
            if explicit not in SUPPORTED_WORKFLOWS:
                raise ValueError(
                    f"input.json['workflow']={explicit!r} is not supported; "
                    f"choose one of {list(SUPPORTED_WORKFLOWS)}"
                )
            return str(explicit), True

        families = scan_families(scan or {})
        if len(families) > 1:
            raise ValueError(
                "input.json['scan'] mixes the keys of two scans "
                f"({sorted(families)}): the ecutwfc ladder and the basis ladder cannot "
                "be configured in one run — split them into two input.json files, or "
                "set 'workflow' explicitly"
            )
        if families == {"ecutwfc"}:
            return WORKFLOW_ECUTWFC, False
        if families == {"basis"}:
            return WORKFLOW_BASIS, False

        # Auto: a single (l_max, r_cut) candidate needs no search.
        limits = abacus_presets[0].config if abacus_presets else {}
        for orbgen in orbgen_presets:
            if len(candidates_from_orbgen(orbgen.config, limits)) > 1:
                return WORKFLOW_GRIDSEARCH, False
        return WORKFLOW_CALC, False
