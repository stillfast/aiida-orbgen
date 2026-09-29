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
    One slot per preset family. Two slots exist:

    * ``abacus`` → ``parameters/abacus/<category>.yml`` — ABACUS / scheduler
      presets (canonicalised into the ``abacus.json`` shape the workchains
      expect: ``basis`` / ``abacus.parameters.input`` /
      ``abacus.metadata.options`` / ``tolerance_meV``).
    * ``orbgen`` → ``parameters/orbgen/<category>.yml`` — SIAB (CSW-NAO)
      presets, i.e. the ``orbgen.json`` content (element / geoms / orbitals /
      bessel_nao_rcut / ...).

    Each slot accepts three spellings::

        "abacus": "u_14ve"                     # parameters/abacus/abacus.yml
        "abacus": ["test", "u_14ve"]           # same file, two presets
        "abacus": {"test": "test"}             # parameters/abacus/test.yml

    The dict form is what the reference ``input.json`` uses: the key is the
    *file* (category) and the value is the preset name inside it.

``static``
    ``pseudo_path`` (absolute path of the UPF; injected as the SIAB
    ``pseudo_dir``), ``metadata`` (a scheduler-options preset name inside
    ``parameters/metadata.yml``), ``output_dir`` (SIAB run root, optional),
    ``dft_root`` / ``dft_roots`` (reference DFT trees, used by ``report``).

The ``parameters/`` tree is **user input**: the plugin only ever *reads* it.
Nothing here (or anywhere else in the package) rewrites a preset — values such
as ``ecutjy``, ``bessel_nao_rcut`` or ``device`` always come from the YAML the
user wrote, and a change there propagates to the next ``run``/``report``
(a reference tree that no longer matches is detected and refetched instead of
being silently mixed).
``code``
    ``{"abacus": "<AiiDA code label>"}``.
``workflow``
    Optional. ``"orbgen.calc"`` (single ``(l_max, r_cut)``) or
    ``"orbgen.gridsearch"`` (whole candidate grid). When absent the loader
    decides from the candidate grid: one candidate → ``orbgen.calc``,
    more than one → ``orbgen.gridsearch``.
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
    "SUPPORTED_WORKFLOWS",
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
SUPPORTED_WORKFLOWS = (WORKFLOW_CALC, WORKFLOW_GRIDSEARCH)

#: slot -> (sub-directory, default preset file inside it)
SLOTS: dict[str, tuple[str, str]] = {
    "abacus": ("abacus", "abacus.yml"),
    "orbgen": ("orbgen", "orbgen.yml"),
}

DEFAULT_TOLERANCE_MEV = 4.2
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


def validate_siab_config(config: dict, *, source: str = "orbgen preset") -> list[str]:
    """Check an orbgen preset *before* it reaches a daemon.

    Missing keys the pipeline reads are fatal — they used to surface only as
    ``run_siab_pipeline`` Excepted (exit 403) in the daemon; keys that only the
    final-orbital step needs are returned as warnings.
    """
    missing = [key for key in SIAB_REQUIRED_KEYS if config.get(key) is None]
    if missing:
        raise ValueError(
            f"{source}: missing required SIAB key(s) {missing}. "
            f"The pipeline needs {list(SIAB_REQUIRED_KEYS)} — see "
            f"parameters/orbgen/orbgen.yml for a complete example."
        )

    rcut = config["bessel_nao_rcut"]
    if not isinstance(rcut, (list, tuple)) or not rcut:
        raise ValueError(
            f"{source}: 'bessel_nao_rcut' must be a non-empty list "
            f"(e.g. [9, 10]), got {rcut!r}"
        )

    geoms = config["geoms"]
    if not isinstance(geoms, (list, tuple)) or not geoms:
        raise ValueError(f"{source}: 'geoms' must be a non-empty list")
    for index, geom in enumerate(geoms):
        if not isinstance(geom, dict):
            raise ValueError(f"{source}: geoms[{index}] must be a mapping")
        absent = [key for key in SIAB_REQUIRED_GEOM_KEYS if key not in geom]
        if absent:
            raise ValueError(
                f"{source}: geoms[{index}] is missing {absent}; "
                f"required: {list(SIAB_REQUIRED_GEOM_KEYS)}"
            )

    if not config["orbitals"]:
        raise ValueError(f"{source}: 'orbitals' must be a non-empty list")

    # SIAB reads vloc_aux / lloc_min only from ``model_kwargs``
    # (``orbital_model_required_keys['atomic']``); written flat they are
    # silently ignored, which quietly changes the fitted orbital.
    for index, orbital in enumerate(config["orbitals"]):
        if not isinstance(orbital, dict):
            raise ValueError(f"{source}: orbitals[{index}] must be a mapping")
        flat = [key for key in ("vloc_aux", "lloc_min") if key in orbital]
        if flat:
            raise ValueError(
                f"{source}: orbitals[{index}] sets {flat} at the top level, "
                f"where SIAB ignores them — nest them under 'model_kwargs' "
                f"(model_kwargs: {{lloc_min: 4, vloc_aux: /abs/path.UPF}})."
            )

    warnings = [
        f"{source}: '{key}' is not set — needed by the final CSW-NAO orbital "
        f"step (`aiida-orbgen report`), not by the DFT/workchain path"
        for key in SIAB_RECOMMENDED_KEYS
        if not config.get(key)
    ]
    if not config.get("pseudo_dir"):
        warnings.append(
            f"{source}: 'pseudo_dir' is not set and input.json['static'] has "
            f"no 'pseudo_path'"
        )
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
          "tolerance_meV": 4.2,
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

    Catches the two ways a preset can poison the INPUT that reaches ABACUS:

    * SIAB-only keys (``ecutjy`` …) — fatal: ABACUS refuses the whole INPUT
      ("THE PARAMETER NAME 'ecutjy' IS NOT USED! ... Bad parameter") and every
      child calculation is Excepted;
    * AiiDA-managed keys (``pseudo_dir`` / ``basis_type`` / ``bessel_nao_rcut`` …)
      — warned about, because the workchain drops them, so setting them here
      has no effect (and hides where the real value comes from).
    """
    from aiida_orbgen.static.defaults import AIIDA_MANAGED_KEYS

    overrides = (
        config.get("abacus", {}).get("parameters", {}).get("input", {}) or {}
    )
    fatal = sorted(set(overrides) & set(SIAB_ONLY_INPUT_KEYS))
    if fatal:
        raise ValueError(
            f"{source}: parameters.input contains SIAB-only key(s) {fatal} — "
            f"ABACUS rejects unknown parameters and aborts the whole INPUT. "
            f"Put them in the orbgen preset instead (e.g. `ecutjy` belongs "
            f"next to `bessel_nao_rcut`)."
        )

    managed = sorted(set(overrides) & set(AIIDA_MANAGED_KEYS))
    if managed:
        return [
            f"{source}: parameters.input sets AiiDA-managed key(s) {managed}; "
            f"the workchain drops them (pseudo_family / basis handling decides "
            f"their values)"
        ]
    return []


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

        metadata, metadata_name = self._metadata()

        abacus_presets = self._load_slot(
            "abacus",
            code=codes.get("abacus"),
            options=metadata.get("options", {}),
        )
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
        workflow, explicit = self._resolve_workflow(abacus_presets, orbgen_presets)

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
    ) -> tuple[str, bool]:
        explicit = self.input_params.get("workflow")
        if explicit:
            if explicit not in SUPPORTED_WORKFLOWS:
                raise ValueError(
                    f"input.json['workflow']={explicit!r} is not supported; "
                    f"choose one of {list(SUPPORTED_WORKFLOWS)}"
                )
            return str(explicit), True

        # Auto: a single (l_max, r_cut) candidate needs no search.
        limits = abacus_presets[0].config if abacus_presets else {}
        for orbgen in orbgen_presets:
            if len(candidates_from_orbgen(orbgen.config, limits)) > 1:
                return WORKFLOW_GRIDSEARCH, False
        return WORKFLOW_CALC, False
