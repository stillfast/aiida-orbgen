"""Typed, validated view of one orbital-generation run.

Why this exists
---------------
The parameters of a run used to be validated in several places, each knowing a
different subset: ``utils/config.py`` (presets), ``static/json_inputs``
(``abacus.json``), and nothing at all for a WorkChain submitted directly through
``WorkflowFactory``. Two of the 2026-09-19 incidents were exactly this class of
problem:

* ``vloc_aux`` / ``lloc_min`` written at the top level of ``orbitals[i]`` — SIAB
  reads them only from that orbital's ``model_kwargs``, so the request for a g
  channel was silently dropped and the delivered file said ``1g`` while its
  header said ``Gorbital = 0``;
* a ``nzeta`` scheme that the primitive basis cannot provide — SIAB only notices
  inside ``basistrans`` *after* the reference DFT has been paid for.

This module is the single place that answers "is this run describable?".
:class:`OrbgenSpec` models the SIAB side, :class:`AbacusSpec` the ABACUS side,
and both are strict where it matters (orbital keys are closed per
initialization model, geometry keys are required).  The adapters in
``utils/config.py`` and the WorkChain boundary call into them; validation
*answers* (errors vs warnings) are returned as data, not raised, so each caller
can decide how loud to be.

Deliberately **not** modelled: the full SIAB key space.  Unknown *top-level*
keys are passed through (SIAB rejects what it does not know) but reported, since
a typo there is invisible otherwise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "OrbgenSpec",
    "AbacusSpec",
    "GeomSpec",
    "OrbitalSpec",
    "SIAB_ONLY_INPUT_KEYS",
    "AIIDA_MANAGED_INPUT_KEYS",
    "describe_problems",
]

#: Keys that belong to SIAB / the orbital generator and never to an ABACUS INPUT
#: (ABACUS aborts with "THE PARAMETER NAME 'ecutjy' IS NOT USED! ... Bad
#: parameter", which excepts every child calculation).
SIAB_ONLY_INPUT_KEYS = ("ecutjy", "vloc_aux", "primitive_type", "nzeta")

#: ``parameters.input`` keys the workchain/AiiDA decides, so writing them has no
#: effect (silently, before this module warned about it).
AIIDA_MANAGED_INPUT_KEYS = (
    "pseudo_dir", "orbital_dir", "stru_file", "kpoint_file", "wannier_card",
    "suffix", "calculation", "basis_type", "bessel_nao_rcut", "bessel_nao_lmax",
    "bessel_nao_tolerence", "bessel_descriptor", "bessel_smooth",
    "bessel_ecut", "bessel_rcut", "bessel_sigma", "bessel_screen_coeff",
)

#: Output switches the spillage step needs from an LCAO reference run.
LCAO_REQUIRED_OUTPUTS = {
    "out_wfc_lcao": "the wavefunctions SIAB fits the orbitals against",
    "out_mat_hs": "the H/S matrices used to rebuild wavefunctions of older runs",
}

#: Keys SIAB's own runner reads (used only to warn about typos).
KNOWN_SIAB_KEYS = frozenset({
    "environment", "mpi_command", "abacus_command", "element", "pseudo_dir",
    "xc", "device", "fit_basis", "ecutwfc", "ecutjy", "bessel_nao_rcut",
    "primitive_type", "smearing_method", "smearing_sigma", "mixing_type",
    "mixing_beta", "mixing_ndim", "mixing_gg0", "spill_guess", "optimizer",
    "max_steps", "nthreads_rcut", "print_every", "verbose", "ftol", "gtol",
    "maxcor", "geoms", "orbitals", "gamma_only", "nspin", "nbands",
    "scf_thr", "scf_nmax", "ks_solver", "ecutrho",
})

MODEL_KWARGS_KEYS = {
    "ones": frozenset(),
    "random": frozenset({"seed"}),
    "atomic": frozenset({"jobdir", "vloc_aux", "lloc_min"}),
    "hydrogen": frozenset({"slater", "otherelem"}),
    "pretrained": frozenset({"pretrained"}),
}


def describe_problems(problems: list[str]) -> str:
    return "; ".join(problems)


class GeomSpec(BaseModel):
    """One reference geometry series (dimers/trimers of perturbed bond lengths).

    Extra keys are allowed on purpose: SIAB forwards every ABACUS parameter found
    in a geom entry to that geometry's job (``dftspecific``), so ``scf_thr`` or a
    per-geometry ``ecutwfc`` are legitimate here.
    """

    model_config = ConfigDict(extra="allow")

    proto: Literal["dimer", "trimer", "monomer"]
    pertkind: Literal["stretch"]
    pertmags: list[float] | Literal["auto", "scan"]
    lmaxmax: int = Field(ge=0, le=8)
    # SIAB's autoset fills nbands with 'auto' when a geom does not set it
    nbands: int | None = Field(default=None, gt=0)
    nspin: Literal[1, 2] = 1
    celldm: float | None = None


class OrbitalSpec(BaseModel):
    """One level of the orbital cascade.

    ``model_kwargs`` is closed per ``model``: ``vloc_aux`` / ``lloc_min`` written
    anywhere else are ignored by SIAB, so accepting them would be a lie.
    """

    model_config = ConfigDict(extra="forbid")

    nzeta: list[int] = Field(min_length=1)
    geoms: list[int] = Field(min_length=1)
    nbands: int | Literal["occ", "occ*2", "all"] = "occ"
    checkpoint: int | None = None
    fix_components: list[list[int]] | None = None
    model: Literal["ones", "random", "atomic", "hydrogen", "pretrained"] = "atomic"
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    filename: str | None = None
    greedygrow: bool = False
    nzeta_max: list[int] | None = None

    @model_validator(mode="after")
    def _check_model_kwargs(self) -> "OrbitalSpec":
        allowed = MODEL_KWARGS_KEYS[self.model]
        unknown = sorted(set(self.model_kwargs) - allowed)
        if unknown:
            raise ValueError(
                f"model_kwargs {unknown} are not read for model "
                f"'{self.model}' (allowed: {sorted(allowed)}). "
                f"In particular vloc_aux/lloc_min must be *inside* model_kwargs."
            )
        if any(nz < 0 for nz in self.nzeta):
            raise ValueError(f"nzeta must not contain negatives: {self.nzeta}")
        if not any(nz > 0 for nz in self.nzeta):
            raise ValueError(f"nzeta {self.nzeta} requests no orbital at all")
        return self

    @property
    def l_max(self) -> int:
        """Highest l with a non-zero zeta count (``[3,2,2,1,0]`` → 3)."""
        return max(l for l, nz in enumerate(self.nzeta) if nz > 0)


class OrbgenSpec(BaseModel):
    """The SIAB (CSW-NAO) side of one run."""

    model_config = ConfigDict(extra="allow")   # unknown top-level keys only warn

    element: str
    ecutjy: float = Field(gt=0)
    ecutwfc: float | None = Field(default=None, gt=0)
    bessel_nao_rcut: list[float] = Field(min_length=1)
    primitive_type: Literal["reduced", "normalized"] = "reduced"
    pseudo_dir: str | None = None
    fit_basis: Literal["jy", "pw"] = "jy"
    geoms: list[GeomSpec] = Field(min_length=1)
    orbitals: list[OrbitalSpec] = Field(min_length=1)

    # ---- cross checks ----------------------------------------------------
    @model_validator(mode="after")
    def _check_consistency(self) -> "OrbgenSpec":
        problems: list[str] = []

        if any(rcut <= 0 for rcut in self.bessel_nao_rcut):
            problems.append(f"bessel_nao_rcut must be positive: {self.bessel_nao_rcut}")

        # The scheme must fit inside the primitive basis: SIAB only discovers a
        # too-large nzeta inside `basistrans`, long after the reference DFT ran.
        available = self.achievable_nzeta
        for index, orbital in enumerate(self.orbitals):
            if orbital.l_max > self.lmaxmax:
                problems.append(
                    f"orbitals[{index}] asks for l_max={orbital.l_max} but "
                    f"geoms[0].lmaxmax={self.lmaxmax}"
                )
                continue
            too_many = [
                f"l={l}: {orbital.nzeta[l]} > {available[l]}"
                for l in range(len(orbital.nzeta))
                if l < len(available) and orbital.nzeta[l] > available[l]
            ]
            if too_many:
                problems.append(
                    f"orbitals[{index}].nzeta {orbital.nzeta} is not reachable from "
                    f"(r_cut={self.bessel_nao_rcut[0]}, ecutjy={self.ecutjy}, "
                    f"{self.primitive_type}): available {available} "
                    f"({describe_problems(too_many)}). "
                    f"Raise ecutjy / r_cut or lower the scheme."
                )
            for geom_index in orbital.geoms:
                if not 0 <= geom_index < len(self.geoms):
                    problems.append(
                        f"orbitals[{index}].geoms refers to geoms[{geom_index}], "
                        f"which does not exist (only {len(self.geoms)} defined)"
                    )

        if self.ecutwfc is not None and self.ecutjy > self.ecutwfc:
            problems.append(
                f"ecutjy ({self.ecutjy}) is above ecutwfc ({self.ecutwfc}); the "
                f"reference DFT would be computed with a coarser plane-wave "
                f"cutoff than the basis it is supposed to resolve"
            )

        # Every geometry used by a cascade level must be listed before it.
        for index, orbital in enumerate(self.orbitals):
            if orbital.checkpoint is not None and not 0 <= orbital.checkpoint < index:
                problems.append(
                    f"orbitals[{index}].checkpoint={orbital.checkpoint} must refer "
                    f"to an earlier level (0..{index - 1}) or be null"
                )

        if problems:
            raise ValueError(describe_problems(problems))

        # ``vloc_aux`` / ``lloc_min`` written flat cannot be represented by
        # OrbitalSpec (extra="forbid"), but the message above would be cryptic
        # for a user of the old spelling; make it explicit instead.
        return self

    # ---- derived ---------------------------------------------------------
    @property
    def lmaxmax(self) -> int:
        return max((geom.lmaxmax for geom in self.geoms), default=0)

    @property
    def achievable_nzeta(self) -> list[int]:
        """Radial functions per l that (r_cut, ecutjy, primitive_type) provides."""
        from aiida_orbgen.interfaces.nsw import compute_nbes_per_l

        rcut = float(self.bessel_nao_rcut[0])
        return list(compute_nbes_per_l(rcut, float(self.ecutjy), self.lmaxmax,
                                       self.primitive_type))

    def grid_points(self) -> list[tuple[int, float]]:
        """``(l_max, r_cut)`` candidates: one per requested r_cut."""
        l_max = max(orbital.l_max for orbital in self.orbitals)
        return [(l_max, float(rcut)) for rcut in self.bessel_nao_rcut]

    def to_siab_config(self) -> dict[str, Any]:
        """The dict SIAB reads (round-trips the validated model)."""
        return self.model_dump(exclude_none=True, mode="json")

    # ---- reporting -------------------------------------------------------
    def warnings(self, base_dir: str | Path | None = None) -> list[str]:
        """Non-fatal problems, phrased for a human.

        ``base_dir`` is the directory of the config being validated; a relative
        ``vloc_aux`` / ``pseudo_dir`` is resolved against it (that is what SIAB
        does) before complaining that the file is missing.
        """
        problems: list[str] = []
        extra = sorted(
            key for key in set(self.model_extra or {}) - KNOWN_SIAB_KEYS
            if not key.startswith(("torch.", "scipy."))
        )
        if extra:
            problems.append(
                f"unknown SIAB key(s) {extra} — SIAB ignores what it does not know "
                f"(a typo here is silent)"
            )
        if not self.pseudo_dir:
            problems.append("'pseudo_dir' is not set (static.pseudo_path missing?)")
        for index, orbital in enumerate(self.orbitals):
            vloc = orbital.model_kwargs.get("vloc_aux")
            vloc_usable = False
            if vloc:
                candidate = Path(str(vloc)).expanduser()
                if not candidate.is_absolute() and base_dir is not None:
                    candidate = Path(base_dir).expanduser() / candidate
                if candidate.is_file():
                    vloc_usable = True
                else:
                    problems.append(
                        f"orbitals[{index}]: vloc_aux={vloc} does not exist "
                        f"(resolved to {candidate}) — the high-l channels would "
                        f"fall back to the plain atomic guess"
                    )
            if not vloc_usable and orbital.l_max >= orbital.model_kwargs.get("lloc_min", 4):
                problems.append(
                    f"orbitals[{index}] requests l_max={orbital.l_max} without "
                    f"vloc_aux: the high-l channels are only initialized from the "
                    f"auxiliary potential"
                )
        return problems

    @classmethod
    def flat_vloc_aux(cls, config: dict) -> list[str]:
        """``orbitals[i]`` keys SIAB never reads (the 2026-09-19 g-channel bug)."""
        offenders = []
        for index, orbital in enumerate(config.get("orbitals") or []):
            if not isinstance(orbital, dict):
                continue
            flat = [key for key in ("vloc_aux", "lloc_min") if key in orbital]
            if flat:
                offenders.append(f"orbitals[{index}] sets {flat} at the top level")
        return offenders


class AbacusSpec(BaseModel):
    """The ABACUS side of one run (``abacus.json`` / the abacus preset)."""

    model_config = ConfigDict(extra="allow")

    basis: list[Literal["pw", "lcao_nsw"]] = Field(min_length=1)
    tolerance_meV: float = Field(default=4.2, gt=0)
    parameters_input: dict[str, Any] = Field(default_factory=dict)

    def errors(self) -> list[str]:
        problems: list[str] = []
        fatal = sorted(set(self.parameters_input) & set(SIAB_ONLY_INPUT_KEYS))
        if fatal:
            problems.append(
                f"parameters.input contains SIAB-only key(s) {fatal} — ABACUS "
                f"rejects unknown parameters and aborts the whole INPUT. Put them "
                f"in the orbgen preset instead (e.g. `ecutjy` next to "
                f"`bessel_nao_rcut`)."
            )
        return problems

    def warnings(self) -> list[str]:
        problems: list[str] = []
        managed = sorted(set(self.parameters_input) & set(AIIDA_MANAGED_INPUT_KEYS))
        if managed:
            problems.append(
                f"parameters.input sets AiiDA-managed key(s) {managed}; the "
                f"workchain drops them (pseudo_family / basis handling decides "
                f"their values)"
            )
        if "lcao_nsw" in self.basis:
            for key, why in LCAO_REQUIRED_OUTPUTS.items():
                value = self.parameters_input.get(key)
                if value is None:
                    continue          # SIAB's autoset provides the defaults
                if str(value).strip() in ("0", "false", "False"):
                    problems.append(
                        f"parameters.input disables {key} (= {value}) but the "
                        f"reference LCAO runs need it: {why}"
                    )
        return problems
