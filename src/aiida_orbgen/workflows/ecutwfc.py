"""``orbgen.ecutwfc`` -- converge the plane-wave reference of the basis fit.

The plane-wave cutoff is the one basis parameter that has nothing to do with the
numerical atomic orbitals: it defines the *reference* total energy the spillage fit and
the ``lcao``-vs-``pw`` comparison are measured against.  It therefore has to be settled
before any ``(r_cut, l_max, ecutjy)`` is compared, and the only defensible criterion is
the one the paper uses for ``\\epsilon_{\\mathrm{NSW}}^{\\mathrm{PW}}``: raise the cutoff
until the total energy of the reference geometries stops moving.

This WorkChain runs the same geometries at a ladder of cutoffs and reports the smallest
cutoff from which every further step is inside ``tolerance_meV`` per atom
(:func:`~aiida_orbgen.workflows.ladder.pw_convergence`).  Nothing is compared against an
NAO basis, so the decision is independent of the orbital parameters:

* ``with_lcao`` (default: follow ``abacus.json``) additionally runs **one** LCAO child
  per geometry at the reference cutoff.  That child is not part of the cutoff decision
  -- it is the sanity check that the contracted basis reproduces the converged PW
  limit on the same geometries, and it is the number to quote when the basis is
  published.
* The ladder is *nested*: the same geometries at ever higher ``ecutwfc``.  A cutoff
  that converged the total energy can be reused as the ``ecutwfc`` input of
  :class:`~aiida_orbgen.workflows.basis.OrbgenBasisScanWorkChain`, and its PW energies
  as that scan's ``pw_reference``, so the expensive reference is paid for once.
"""

from __future__ import annotations

import json
from typing import Any

from aiida import orm
from aiida.engine import ExitCode

from aiida_orbgen.utils.config import DEFAULT_TOLERANCE_MEV
from aiida_orbgen.workflows._children import (
    child_options,
    geometry_key,
    record_child,
    submit_child,
    with_input_overrides,
    write_child_results,
)
from aiida_orbgen.workflows.batch import OrbgenCalcWorkChain
from aiida_orbgen.workflows.extract import collect_child_energies
from aiida_orbgen.workflows.ladder import pw_convergence
from aiida_orbgen.workflows.results import create_dict

__all__ = ["OrbgenEcutwfcWorkChain", "DEFAULT_ECUTWFC_SCALE"]

#: The ladder used when ``ecutwfc_values`` is not given: multipliers of the baseline
#: cutoff the SIAB reference was generated at.  The paper's ``E_c`` = 100 Ry is a
#: *spillage* cutoff, so the reference cutoff of a production fit is usually a multiple
#: of it -- 1.8x (180 Ry) is where the U reference used here converged to 5.9 meV/atom.
DEFAULT_ECUTWFC_SCALE: tuple[float, ...] = (1.0, 1.25, 1.5, 1.8, 2.0, 2.5, 3.0)


class OrbgenEcutwfcWorkChain(OrbgenCalcWorkChain):
    """Decide ``ecutwfc`` from PW total-energy convergence.

    Inherits the SIAB/pseudo-family steps of
    :class:`~aiida_orbgen.workflows.batch.OrbgenCalcWorkChain` (one orbital tree,
    generated at the ``l_max``/``r_cut`` inputs) and replaces the child submission and
    the analysis: instead of ``{pw, lcao} x structure`` at one cutoff, it runs
    ``structure x cutoff`` PW children, plus optionally one LCAO child per structure.

    Inputs
    ------
    siab_json, abacus_config, l_max, r_cut, output_dir
        as in ``orbgen.calc`` (the SIAB tree is generated but only its geometries and
        pseudopotential are used by the PW children).
    ecutwfc_values : List, optional
        explicit ladder in Ry (e.g. ``[100, 120, 150, 180, 200]``).
    ecutwfc_scale : List, optional
        multipliers of ``baseline_ecutwfc`` used when ``ecutwfc_values`` is absent
        (default :data:`DEFAULT_ECUTWFC_SCALE`).
    baseline_ecutwfc : Float, optional
        the cutoff the reference was generated at; default: ``abacus.json``'s
        ``parameters.input.ecutwfc``, else the ``ecutwfc`` of the SIAB JSON, else 100 Ry.
    reference_ecutwfc : Float, optional
        the cutoff that counts as "the converged limit" -- the one the optional LCAO
        child runs at and the one its energies are compared against.  Default: the
        largest value of the ladder.  Must be a value of the ladder.
    with_lcao : Bool, optional
        run one LCAO child per geometry at ``reference_ecutwfc``.  Default: whatever
        ``abacus.json``'s ``basis`` says (``"lcao_nsw"`` in it -> True).
    code_label, family_label, build_family, dry_run, max_iterations
        as in ``orbgen.calc``.  ``only`` is ignored (the ladder is the point).

    Outputs
    -------
    ecutwfc_decision : Dict
        ``pw_convergence``'s result (``values``, ``steps``, ``chosen``, ``converged``,
        ``per_geometry``) plus ``baseline_ecutwfc``, ``reference_ecutwfc``, the
        ``tolerance_meV`` used, ``pw_reference`` (``{"ecutwfc": …, "geometries":
        {geometry: {energy, n_atoms}}}`` at the reference cutoff, ready to be handed to
        ``orbgen.basis`` via ``input.json["scan"]["pw_reference_pk"]``), and -- when the LCAO children ran --
        ``lcao_vs_pw`` (per-geometry ``dE_per_atom_meV`` and its maximum).
    siab_info, primitive_orbital, pseudo_family, results
        as in ``orbgen.calc``.

    Exit codes
    ----------
    0     the ladder converged (and the optional LCAO check is inside the tolerance)
    301   some children failed (the ladder may have holes)
    302   every child failed
    303   no energy could be extracted
    304   the LCAO children are outside the tolerance against the converged PW energy
    407   no cutoff of the ladder is converged: run a wider ladder
    408   the ladder has fewer than two cutoffs, or a non-positive one

    Entry point
    -----------
    ``orbgen.ecutwfc``
    """

    # ------------------------------------------------------------------
    # define
    # ------------------------------------------------------------------
    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("ecutwfc_values", valid_type=orm.List, required=False,
                   help="Explicit ladder of PW cutoffs in Ry, ascending.")
        spec.input("ecutwfc_scale", valid_type=orm.List, required=False,
                   default=lambda: orm.List(list=list(DEFAULT_ECUTWFC_SCALE)),
                   help="Multipliers of baseline_ecutwfc (used without ecutwfc_values).")
        spec.input("baseline_ecutwfc", valid_type=orm.Float, required=False,
                   help="Cutoff the reference was generated at (default: abacus.json).")
        spec.input("reference_ecutwfc", valid_type=orm.Float, required=False,
                   help="Cutoff that counts as converged (default: the largest one).")
        spec.input("with_lcao", valid_type=orm.Bool, required=False,
                   help="Also run one LCAO child per geometry at reference_ecutwfc "
                        "(default: follow abacus.json's basis list).")

        spec.output("ecutwfc_decision", valid_type=orm.Dict, required=False,
                    help="The ladder, the step sizes, the chosen cutoff and -- with "
                         "with_lcao -- the LCAO-vs-PW difference at that cutoff.")

        spec.exit_code(407, "WARNING_ECUTWFC_NOT_CONVERGED",
                       message="no cutoff of the ladder is converged within the "
                               "tolerance: run a wider ladder")
        spec.exit_code(408, "ERROR_ECUTWFC_LADDER",
                       message="ecutwfc_values must hold at least two positive cutoffs")

    # ------------------------------------------------------------------
    # input handling
    # ------------------------------------------------------------------
    def _baseline_ecutwfc(self) -> tuple[float, str]:
        """``(cutoff, source)`` the ladder is scaled from."""
        if "baseline_ecutwfc" in self.inputs:
            return float(self.inputs.baseline_ecutwfc.value), "input baseline_ecutwfc"
        parameters = (self.ctx.abacus_cfg.get("abacus", {}) or {}).get("parameters", {}) or {}
        cutoff = (parameters.get("input", {}) or {}).get("ecutwfc")
        if cutoff:
            return float(cutoff), "abacus.json parameters.input.ecutwfc"
        try:
            content = self.inputs.siab_json.get_content()
            if isinstance(content, bytes):
                content = content.decode("utf-8")
            cutoff = json.loads(content).get("ecutwfc")
        except Exception:  # noqa: BLE001 -- the JSON is validated elsewhere
            cutoff = None
        if cutoff:
            return float(cutoff), "SIAB json ecutwfc"
        return 100.0, "default 100 Ry"

    def _ecutwfc_values(self) -> list[float]:
        """The ladder, ascending and deduplicated."""
        if "ecutwfc_values" in self.inputs:
            raw = list(self.inputs.ecutwfc_values.get_list())
        else:
            baseline, _ = self._baseline_ecutwfc()
            scale = self.inputs.ecutwfc_scale.get_list()
            raw = [baseline * float(s) for s in scale]
        return sorted({float(v) for v in raw if v is not None})

    def _reference_ecutwfc(self, values: list[float]) -> float:
        if "reference_ecutwfc" in self.inputs:
            return float(self.inputs.reference_ecutwfc.value)
        return values[-1]

    def _with_lcao(self) -> bool:
        """Whether the LCAO children run: the flag, else ``abacus.json``'s basis."""
        if "with_lcao" in self.inputs:
            return bool(self.inputs.with_lcao.value)
        return "lcao_nsw" in (self.ctx.abacus_cfg.get("basis") or [])

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------
    def validate_inputs(self):
        result = super().validate_inputs()
        if result:
            return result

        values = self._ecutwfc_values()
        if len(values) < 2 or values[0] <= 0.0:
            self.report(
                f"ERROR: the ecutwfc ladder needs >= 2 positive cutoffs, got {values}"
            )
            return self.exit_codes.ERROR_ECUTWFC_LADDER
        baseline, source = self._baseline_ecutwfc()
        reference = self._reference_ecutwfc(values)
        if reference not in values:
            self.report(
                f"  WARNING: reference_ecutwfc={reference:g} Ry is not in the ladder "
                f"{values}; the LCAO comparison will use {values[-1]:g} Ry"
            )
            reference = values[-1]
        self.ctx.ecutwfc_values = values
        self.ctx.reference_ecutwfc = reference
        self.report(
            f"ecutwfc ladder: {values} Ry (baseline {baseline:g} Ry from {source}; "
            f"reference {reference:g} Ry; "
            f"tolerance {self.ctx.abacus_cfg.get('tolerance_meV', DEFAULT_TOLERANCE_MEV):g} "
            f"meV/atom)"
        )
        return None

    # ------------------------------------------------------------------
    # Step 2: submit structure x cutoff (+ one LCAO per structure)
    # ------------------------------------------------------------------
    def submit_children(self):
        values = list(getattr(self.ctx, "ecutwfc_values", []) or self._ecutwfc_values())
        reference = float(getattr(self.ctx, "reference_ecutwfc", values[-1]))
        with_lcao = self._with_lcao()

        if self._dry_run:
            plan = ", ".join(f"{v:g}" for v in values)
            self.report(
                f"Step 2: [DRY-RUN] would submit PW children at ecutwfc = {plan} Ry "
                f"for every structure"
                + (f", plus one LCAO child at {reference:g} Ry" if with_lcao else "")
            )
            self.ctx.children_info = []
            return None

        if not hasattr(self.ctx, "siab_info"):
            return self.exit_codes.ERROR_SIAB_FAILED

        options = child_options(self)
        jobs: list[tuple[str, float]] = [("pw", value) for value in values]
        if with_lcao:
            jobs.append(("lcao_nsw", reference))

        structures = self.ctx.siab_info["dft"]
        self.report(
            f"Step 2: {len(structures)} structures x {len(jobs)} jobs = "
            f"{len(structures) * len(jobs)} abacus.base runs "
            f"({options['num_mpi']} MPI procs, queue {options['queue_name']})"
        )
        self.ctx.children_info = []
        n_submitted = 0
        for dft_entry in structures:
            geometry = geometry_key(dft_entry)
            for basis, cutoff in jobs:
                parameters = with_input_overrides(options, {"ecutwfc": float(cutoff)})
                label = f"pw@{cutoff:g}" if basis == "pw" else "lcao"
                try:
                    node, n_atoms = submit_child(
                        self, dft_entry=dft_entry, basis=basis, options=options,
                        parameters=parameters,
                    )
                except Exception as exc:  # noqa: BLE001
                    # one geometry must not kill the ladder: it simply does not
                    # take part in it
                    self.report(
                        f"  ERROR: could not build the {basis} child of {geometry} "
                        f"at {cutoff:g} Ry: {exc}"
                    )
                    continue
                record_child(self, node=node, label=label, geometry=geometry,
                             basis=basis, n_atoms=n_atoms)
                n_submitted += 1
        self.report(f"  {n_submitted} children submitted")
        if n_submitted == 0:
            return self.exit_codes.ERROR_BUILD_INPUTS
        return None

    # ------------------------------------------------------------------
    # Step 4: the convergence decision
    # ------------------------------------------------------------------
    def extract_energies_step(self):
        """Turn the children into ``{cutoff: {geometry: energy}}`` and decide."""
        try:
            collected = collect_child_energies(
                self.ctx.children_info, report=self.report
            )
        except Exception as exc:  # noqa: BLE001 -- a broken child must not abort the WC
            import traceback
            self.report(f"ERROR: energy extraction failed: {exc}")
            self.report(traceback.format_exc())
            return self.exit_codes.WARNING_ENERGY_EXTRACT_FAILED

        for note in collected.notes:
            self.report(note)

        if self._dry_run:
            return None
        if not collected.usable:
            self.report("WARNING: no child produced an energy")
            return self.exit_codes.WARNING_ENERGY_EXTRACT_FAILED

        from aiida_orbgen.workflows._children import split_task

        curve: dict[float, dict[str, Any]] = {}
        lcao: dict[str, Any] = {}
        for entry in collected.entries:
            label, geometry = split_task(entry.folder)
            if label.startswith("pw@"):
                try:
                    cutoff = float(label.split("@", 1)[1])
                except ValueError:
                    continue
                curve.setdefault(cutoff, {})[geometry] = {
                    "energy": float(entry.energy),
                    "n_atoms": int(entry.n_atoms),
                }
            else:
                lcao[geometry] = {
                    "energy": float(entry.energy),
                    "n_atoms": int(entry.n_atoms),
                }

        tolerance_meV = float(
            self.ctx.abacus_cfg.get("tolerance_meV", DEFAULT_TOLERANCE_MEV)
        )
        if not curve:
            self.report("WARNING: no PW child produced an energy")
            return self.exit_codes.WARNING_ENERGY_EXTRACT_FAILED

        decision = pw_convergence(curve, tolerance_meV)
        decision["baseline_ecutwfc"] = self._baseline_ecutwfc()[0]
        decision["reference_ecutwfc"] = float(getattr(self.ctx, "reference_ecutwfc", 0.0))
        # The PW reference itself, at the cutoff that counts as converged: this is what
        # `orbgen.basis` wants as its `pw_reference` input (and what
        # `input.json["scan"]["pw_reference_pk"]` reads), so the expensive PW side of a
        # basis comparison is paid for exactly once.
        if curve.get(decision["reference_ecutwfc"]):
            # Wrapped under "geometries" on purpose: AiiDA refuses *top-level* Dict keys
            # containing a dot, and the geometry names of a perturbed dimer are
            # "dimer-2.8".  Nested keys have no such rule, so one level of structure
            # makes the block storable (and `scan.pw_reference` takes the same shape).
            decision["pw_reference"] = {
                "ecutwfc": float(decision["reference_ecutwfc"]),
                "geometries": {
                    geometry: {"energy": float(item["energy"]),
                               "n_atoms": int(item["n_atoms"])}
                    for geometry, item in curve[decision["reference_ecutwfc"]].items()
                },
            }
        decision["n_geometries"] = len(
            {geom for entry in curve.values() for geom in entry}
        )

        self.report(
            f"  PW total energy vs ecutwfc (tolerance {tolerance_meV:g} meV/atom):"
        )
        for step in decision["steps"]:
            worst = step["max_meV"]
            mark = "✓" if step["within_tolerance"] else "✗"
            shown = "n/a" if worst is None else f"{worst:8.2f} meV/atom"
            self.report(
                f"    {step['from']:>5g} -> {step['to']:<5g} Ry : {shown} {mark}"
            )
        if decision["converged"]:
            self.report(
                f"  ✓ ecutwfc = {decision['chosen']:g} Ry: every further step is inside "
                f"the tolerance"
            )
        else:
            self.report(
                f"  ✗ NOT converged within the ladder: the largest cutoff "
                f"({decision['chosen']:g} Ry) still moves by "
                f"{(decision['steps'][-1]['max_meV'] if decision['steps'] else float('nan')):.2f} "
                f"meV/atom per step"
            )

        # The optional LCAO check: a fixed NAO basis against the converged PW energy.
        if lcao:
            reference = float(decision["reference_ecutwfc"])
            pw_at_reference = curve.get(reference)
            if pw_at_reference is None:
                self.report(f"  WARNING: no PW energy at the reference cutoff {reference:g} Ry")
            else:
                rows = {}
                for geometry, item in lcao.items():
                    pw_item = pw_at_reference.get(geometry)
                    if pw_item is None:
                        continue
                    n_atoms = max(1, int(item.get("n_atoms") or 1))
                    rows[geometry] = {
                        "e_nsw": float(item["energy"]),
                        "e_pw": float(pw_item["energy"]),
                        "n_atoms": n_atoms,
                        "dE_per_atom_meV": (
                            (float(item["energy"]) - float(pw_item["energy"]))
                            / n_atoms * 1000.0
                        ),
                    }
                if rows:
                    worst = max(rows.items(), key=lambda kv: abs(kv[1]["dE_per_atom_meV"]))
                    decision["lcao_vs_pw"] = {
                        "reference_ecutwfc": reference,
                        "per_geometry": rows,
                        "max_abs_per_atom_meV": abs(worst[1]["dE_per_atom_meV"]),
                        "worst_geometry": worst[0],
                        "tolerance_ok": abs(worst[1]["dE_per_atom_meV"]) <= tolerance_meV,
                    }
                    for geometry, row in sorted(rows.items()):
                        self.report(
                            f"    LCAO-vs-PW {geometry:<14s} "
                            f"{row['dE_per_atom_meV']:+9.2f} meV/atom"
                        )
                    ok = decision["lcao_vs_pw"]["tolerance_ok"]
                    self.report(
                        f"  {'✓' if ok else '✗'} LCAO basis vs the PW energy at "
                        f"{reference:g} Ry: max |ΔE|/atom = "
                        f"{decision['lcao_vs_pw']['max_abs_per_atom_meV']:.2f} meV "
                        f"({worst[0]})"
                    )

        self.ctx.ecutwfc_decision = decision
        self.ctx.ecutwfc_tolerance_ok = bool(
            (decision.get("lcao_vs_pw") or {}).get("tolerance_ok", True)
        )
        return None

    # ------------------------------------------------------------------
    # Step 5: outputs
    # ------------------------------------------------------------------
    def finalize(self):
        if self._dry_run:
            self.report("Step 5: [DRY-RUN] nothing submitted, no outputs")
            return ExitCode(0)

        n_ok, n_failed = write_child_results(self)
        decision = getattr(self.ctx, "ecutwfc_decision", None)
        if decision:
            self.out("ecutwfc_decision", create_dict(decision))

        if n_ok == 0:
            self.report("OrbgenEcutwfcWorkChain: every child failed")
            return self.exit_codes.ERROR_ALL_FAILED
        if n_failed:
            self.report(
                f"OrbgenEcutwfcWorkChain: {n_ok} OK, {n_failed} failed -- the ladder "
                f"may have holes"
            )
            return self.exit_codes.WARNING_PARTIAL_FAILURE
        if decision and not decision.get("converged"):
            return self.exit_codes.WARNING_ECUTWFC_NOT_CONVERGED
        if not self.ctx.ecutwfc_tolerance_ok:
            return self.exit_codes.WARNING_TOLERANCE_EXCEEDED
        chosen = decision.get("chosen") if decision else None
        tail = f", ecutwfc = {chosen:g} Ry" if chosen is not None else ""
        self.report(
            f"OrbgenEcutwfcWorkChain Finished: {n_ok} children OK{tail}"
        )
        return ExitCode(0)
