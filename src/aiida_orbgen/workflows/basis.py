"""``orbgen.basis`` -- pick ``(r_cut, l_max, ecutjy)`` of the primitive NSW basis.

The criterion is the paper's ``\\epsilon_{\\mathrm{NSW}}^{\\mathrm{PW}}``: the total
energy of the *same* reference geometries in the primitive numerical-orbital basis
against the converged plane-wave energy, per atom, at the ``tolerance_meV`` of the run.
Among the candidates that meet it the cheapest one wins -- that is what makes the
resulting basis usable in the large cells (defect supercells) it is being built for.

Why this is not a grid search over ``orbgen.calc``
-------------------------------------------------

``OrbgenCalcWorkChain`` compares ``{pw, lcao}`` at **one** grid point, so a grid of ``N``
candidates pays for ``N`` *identical* PW references, and PW is the expensive side.  This
WorkChain instead:

* computes the PW reference **once** (:meth:`~OrbgenBasisScanWorkChain.submit_pw_step`)
  and reuses it for every candidate, or takes one from an earlier run through the
  ``pw_reference`` input (the natural companion of ``orbgen.ecutwfc``);
* walks a **ladder** rather than a Cartesian product
  (:func:`~aiida_orbgen.workflows.ladder.plan_candidates`): one super-parameter at a
  time, always *downwards* from the reference (``ecutjy`` first -- the dominant knob --
  then ``l_max``, then ``r_cut``), so every candidate is cheaper than the point it was
  derived from;
* **evaluates cheapest first** (:func:`~aiida_orbgen.workflows.ladder.plan_evaluation_order`)
  and stops at the first candidate inside the tolerance (``stop_on_first_pass``, default
  true): the candidates are walked cheapest first and the reference point -- the most
  expensive one -- is evaluated *last* (or first when the atomization gate needs it as
  the baseline) and never counts as the reduction, because a scan that stopped there
  would go back to the most expensive basis it knows.  The first candidate that passes
  therefore *is* the cheapest one that passes, unless the atomization gate rejects it,
  in which case the scan continues and says so;
* reports the cost of every candidate (measured wall-clock ``seconds`` and the ``nchi``
  proxy), so the choice can be re-made with a different criterion without rerunning
  anything;
* refuses an ``r_cut`` larger than half the smallest cell edge of the reference
  structures: past that point the two-centre tables describe neighbours that cannot
  exist, and the "basis" is silently a different, worse one.

The geometries are the ones the SIAB config already contains -- the perturbed dimers
*and* the monomer -- so the atomization energy (``E_dimer - 2 E_monomer``) comes for
free and ``atomization_tolerance_meV`` can apply the stricter, defect-like criterion.

Notes
-----
``abacus.json``'s ``basis`` list is ignored here: a basis scan is ``lcao`` children
against a PW reference by definition.  (Use the ``lcao_only.yml`` preset if the
validation of an unused key bothers you.)
"""

from __future__ import annotations

import json
import os
from typing import Any

from aiida import orm
from aiida.engine import ExitCode, if_, while_

from aiida_orbgen.utils.config import DEFAULT_TOLERANCE_MEV
from aiida_orbgen.workflows._children import (
    cell_edges_bohr,
    child_options,
    geometry_key,
    geometries_from_entries,
    n_primitive_functions,
    rcut_fits_cell,
    record_child,
    seconds_of,
    siab_cache_hint,
    siab_tree_problem,
    split_task,
    submit_child,
    with_input_overrides,
    write_child_results,
)
from aiida_orbgen.workflows.batch import OrbgenCalcWorkChain
from aiida_orbgen.workflows.extract import collect_child_energies
from aiida_orbgen.workflows.ladder import (
    basis_table,
    candidate_cost,
    candidate_label,
    describe_row,
    no_basis_message,
    pick_cheapest,
    plan_candidates,
    plan_evaluation_order,
)
from aiida_orbgen.workflows.results import create_dict
from aiida_orbgen.workflows.siab import run_siab_pipeline, siab_code_digest

__all__ = ["OrbgenBasisScanWorkChain"]


class OrbgenBasisScanWorkChain(OrbgenCalcWorkChain):
    """Rank ``(r_cut, l_max, ecutjy)`` candidates by ``|E_nsw - E_pw|``, pick the
    cheapest acceptable one.

    Inputs
    ------
    siab_json, abacus_config, output_dir, l_max, r_cut
        as in ``orbgen.calc``.  ``l_max`` / ``r_cut`` (with ``reference_ecutjy``) are the
        **reference point** of the ladder: the most expensive basis the scan may
        consider, and the first candidate it tries.
    ecutjy_values, l_max_values, r_cut_values : List, optional
        values to *reduce* the reference to.  An absent/empty list means "do not vary
        this parameter".
    strategy : Str, optional
        ``"ladder"`` (default: one parameter at a time, cheapest first) or
        ``"exhaustive"`` (the Cartesian product of the three lists).
    reference_ecutjy : Float, optional
        ``ecutjy`` of the reference point (default: the SIAB JSON's ``ecutjy``, the
        cutoff the reference orbitals were fitted with).
    ecutwfc : Float, optional
        PW cutoff of the reference children (default: ``abacus.json``).  Pass the value
        :class:`~aiida_orbgen.workflows.ecutwfc.OrbgenEcutwfcWorkChain` converged to.
    pw_reference : Dict, optional
        ``{geometry: {"energy": eV, "n_atoms": N}}`` of an already computed PW
        reference: with it the scan runs **no** PW child at all.
    stop_on_first_pass : Bool, optional
        stop at the first candidate inside the tolerance (default True).  The candidates
        are walked cheapest first, so this stops at the cheapest acceptable basis.
    atomization_tolerance_meV : Float, optional
        additionally require ``|d(atomization energy)|`` against the reference candidate
        to stay inside this many meV.  Without it the criterion is the plain per-atom
        total-energy difference.
    by : Str, optional
        cost key behind "cheapest": ``"seconds"`` (default, measured), ``"cost"`` (the
        ``nchi**2 * r_cut**3`` proxy) or ``"nchi"``.
    code_label, family_label, build_family, dry_run, max_iterations
        as in ``orbgen.calc``.

    Outputs
    -------
    basis_decision : Dict
        the whole table (per candidate: ``dE_per_atom_meV`` per geometry,
        ``dE_max_abs_meV``, ``atomization_meV``, ``atomization_vs_reference_meV``,
        ``seconds``, ``nchi``, ``cost``, ``tolerance_ok``), ``best``, ``reference``, the
        tolerance and the gate used, and what was planned versus what actually ran.
    chosen_basis : Dict
        the winner alone: ``r_cut``, ``l_max``, ``ecutjy``, its numbers, its orbital
        filename and the pseudo family built from it.
    primitive_orbital : SinglefileData
        the primitive ``.orb`` of the winner, archived in the provenance (the paths in
        ``basis_decision`` point into scratch space).
    results, pseudo_family
        as in ``orbgen.calc``, over the children of the whole scan.

    Exit codes
    ----------
    0     the winner exists and every child ran
    301   some children failed
    302   no child produced an energy at all
    303   no candidate could be evaluated
    409   no candidate meets the tolerance: widen the ladder
    410   the reference ``r_cut`` is larger than half the smallest cell edge
    411   the PW reference could not be produced
    412   the SIAB pipeline result came from the cache and its files are gone

    Entry point
    -----------
    ``orbgen.basis``
    """

    # ------------------------------------------------------------------
    # define
    # ------------------------------------------------------------------
    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("ecutjy_values", valid_type=orm.List, required=False,
                   help="ecutjy candidates to reduce to (Ry), ascending.")
        spec.input("l_max_values", valid_type=orm.List, required=False,
                   help="l_max candidates to reduce to, ascending.")
        spec.input("r_cut_values", valid_type=orm.List, required=False,
                   help="r_cut candidates to reduce to (Bohr), ascending.")
        spec.input("strategy", valid_type=orm.Str, required=False,
                   default=lambda: orm.Str("ladder"),
                   help="'ladder' (one parameter at a time, cheapest first) or "
                        "'exhaustive' (Cartesian product).")
        spec.input("reference_ecutjy", valid_type=orm.Float, required=False,
                   help="ecutjy of the reference point (default: the SIAB JSON's).")
        spec.input("ecutwfc", valid_type=orm.Float, required=False,
                   help="PW cutoff of the reference children (default: abacus.json).")
        spec.input("pw_reference", valid_type=orm.Dict, required=False,
                   help="Already computed PW reference "
                        "{geometry: {'energy': eV, 'n_atoms': N}}: no PW child is run.")
        spec.input("stop_on_first_pass", valid_type=orm.Bool, required=False,
                   default=lambda: orm.Bool(True),
                   help="Stop at the first candidate inside the tolerance.")
        spec.input("atomization_tolerance_meV", valid_type=orm.Float, required=False,
                   help="Also keep the atomization-energy difference against the "
                        "reference candidate inside this many meV.")
        spec.input("by", valid_type=orm.Str, required=False,
                   default=lambda: orm.Str("seconds"),
                   help="Cost key of 'cheapest': seconds | cost | nchi.")

        spec.output("basis_decision", valid_type=orm.Dict, required=False,
                    help="The comparison table, the winner and the book-keeping.")
        spec.output("chosen_basis", valid_type=orm.Dict, required=False,
                    help="The winning (r_cut, l_max, ecutjy) and its numbers.")

        spec.exit_code(409, "WARNING_NO_BASIS_WITHIN_TOLERANCE",
                       message="no candidate of the scan meets the tolerance: widen "
                               "the ladder")
        spec.exit_code(410, "ERROR_RCUT_TOO_LARGE",
                       message="r_cut exceeds half the smallest cell edge of the "
                               "reference structures")
        spec.exit_code(411, "ERROR_NO_PW_REFERENCE",
                       message="the PW reference is neither given nor obtainable")
        spec.exit_code(412, "ERROR_SIAB_TREE_MISSING",
                       message="the SIAB result is cached but its files are gone: use a "
                               "fresh output_dir")

        spec.outline(
            cls.validate_inputs,
            cls.plan_step,
            cls.siab_step,
            cls.family_step,
            if_(cls.needs_pw_reference)(cls.submit_pw_step, cls.collect_pw_step),
            while_(cls.candidates_left)(
                cls.submit_candidate_step,
                cls.inspect_candidate_step,
                cls.evaluate_candidate_step,
            ),
            cls.finalize,
        )

    # ------------------------------------------------------------------
    # small helpers
    # ------------------------------------------------------------------
    def _tolerance_meV(self) -> float:
        return float(self.ctx.abacus_cfg.get("tolerance_meV", DEFAULT_TOLERANCE_MEV))

    def _atomization_gate(self) -> float | None:
        if "atomization_tolerance_meV" in self.inputs:
            return float(self.inputs.atomization_tolerance_meV.value)
        return None

    def _reference_ecutjy(self) -> tuple[float, str]:
        """``(ecutjy, source)`` of the reference point."""
        if "reference_ecutjy" in self.inputs:
            return float(self.inputs.reference_ecutjy.value), "input reference_ecutjy"
        try:
            content = self.inputs.siab_json.get_content()
            if isinstance(content, bytes):
                content = content.decode("utf-8")
            value = json.loads(content).get("ecutjy")
        except Exception:  # noqa: BLE001 -- the SIAB JSON is validated elsewhere
            value = None
        if value:
            return float(value), "SIAB json ecutjy"
        return 100.0, "default 100 Ry"

    def _pw_reference(self) -> dict[str, Any]:
        """The PW energies the candidates are compared against."""
        if "pw_reference" in self.inputs:
            return {
                str(key): dict(value)
                for key, value in self.inputs.pw_reference.get_dict().items()
            }
        return dict(getattr(self.ctx, "pw_reference", {}) or {})

    # ------------------------------------------------------------------
    # Step 0: validation and the candidate plan
    # ------------------------------------------------------------------
    def validate_inputs(self):
        result = super().validate_inputs()
        if result:
            return result

        ecutjy, source = self._reference_ecutjy()
        strategy = str(self.inputs.strategy.value) if "strategy" in self.inputs else "ladder"
        if strategy not in ("ladder", "exhaustive"):
            self.report(f"ERROR: unknown strategy {strategy!r}")
            return self.exit_codes.ERROR_INVALID_ABACUS_CONFIG
        if str(self.inputs.by.value if "by" in self.inputs else "seconds") not in (
            "seconds", "cost", "nchi"
        ):
            self.report(f"ERROR: unknown cost key 'by'")
            return self.exit_codes.ERROR_INVALID_ABACUS_CONFIG

        def _values(name: str) -> list[Any]:
            if name not in self.inputs:
                return []
            return list(self.inputs[name].get_list())

        reference = {
            "r_cut": float(self.inputs.r_cut.value),
            "l_max": int(self.inputs.l_max.value),
            "ecutjy": ecutjy,
        }
        try:
            candidates = plan_candidates(
                r_cut_values=_values("r_cut_values"),
                l_max_values=_values("l_max_values"),
                ecutjy_values=_values("ecutjy_values"),
                reference=reference,
                strategy=strategy,
            )
        except Exception as exc:  # noqa: BLE001 -- a bad ladder is a user error
            self.report(f"ERROR: could not build the candidate list: {exc}")
            return self.exit_codes.ERROR_INVALID_ABACUS_CONFIG

        gate = self._atomization_gate()
        # The reference point is the "do not reduce anything" candidate and the baseline
        # of the atomization gate.  An exhaustive product need not contain it at all, and
        # then it would be neither the fallback answer nor a baseline -- so it is added.
        self.ctx.reference_candidate = dict(reference)
        self.ctx.reference_label = candidate_label(reference)
        if self.ctx.reference_label not in {candidate_label(e) for e in candidates}:
            candidates = [dict(reference)] + candidates
            self.report(
                f"  the exhaustive product does not contain the reference point: "
                f"adding {self.ctx.reference_label}"
            )
        # Cheapest first, reference last -- or first when the atomization gate needs it
        # as the baseline.  The reference is the most expensive candidate, so it must
        # never be the candidate the early exit stops at: that would end every scan at
        # the basis it was meant to reduce.
        costs = {
            candidate_label(entry): candidate_cost(
                n_primitive_functions(entry["r_cut"], entry["ecutjy"], entry["l_max"]),
                entry["r_cut"],
            )
            for entry in candidates
        }
        self.ctx.candidates = plan_evaluation_order(
            candidates, costs=costs, reference_label=self.ctx.reference_label,
            baseline_first=gate is not None,
        )
        self.ctx.cursor = 0
        self.ctx.done = False
        self.ctx.rows = {}
        self.ctx.candidate_ok = {}
        self.ctx.all_children = []
        self.report(
            f"reference point: r_cut={reference['r_cut']:g} au, "
            f"l_max={reference['l_max']}, ecutjy={ecutjy:g} Ry ({source}); "
            f"strategy={strategy}"
        )
        if not bool(self.inputs.get("stop_on_first_pass", orm.Bool(True)).value):
            self.report("  every candidate will be evaluated (stop_on_first_pass=False)")
        if gate is not None:
            self.report(f"  the atomization baseline runs first (gate {gate:g} meV)")
        for entry in self.ctx.candidates:
            label = candidate_label(entry)
            self.report(
                f"  candidate {label}: r_cut={entry['r_cut']:g} au, "
                f"l_max={entry['l_max']}, ecutjy={entry['ecutjy']:g} Ry, "
                f"nchi={n_primitive_functions(entry['r_cut'], entry['ecutjy'], entry['l_max'])}, "
                f"cost={costs.get(label)}"
                + ("  <- reference point" if label == self.ctx.reference_label else "")
            )
        return None

    # ------------------------------------------------------------------
    # Step 1: the plan (a dry run stops after this)
    # ------------------------------------------------------------------
    def plan_step(self):
        tolerance = self._tolerance_meV()
        gate = self._atomization_gate()
        self.report(
            f"Step 1: {len(self.ctx.candidates)} candidate(s), tolerance "
            f"{tolerance:g} meV/atom on max |E_nsw - E_pw|"
            + (f", atomization gate {gate:g} meV" if gate is not None else "")
        )
        if self._dry_run:
            pw = ("reused from the pw_reference input" if "pw_reference" in self.inputs
                  else "computed once for the reference tree")
            self.report(
                f"  [DRY-RUN] PW reference: {pw}; then one LCAO run per geometry per "
                f"candidate; nothing submitted"
            )
        return None

    # ------------------------------------------------------------------
    # Step 2: one SIAB tree per candidate
    # ------------------------------------------------------------------
    def siab_step(self):
        """Generate the primitive basis and the DFT job list of every candidate.

        Local and cheap (no scheduler involved), and doing it for the whole ladder up
        front is what makes the plan checkable: the spillage step that would *need* LCAO
        results is not part of it -- ``orbgen`` runs that later, on the chosen basis.
        """
        if self._dry_run:
            self.report("Step 2: [DRY-RUN] SIAB not launched")
            return None

        self.ctx.siab = {}
        self.ctx.orbitals = {}
        self.ctx.family_by_label = {}
        root = os.path.abspath(str(self.inputs.output_dir.value))
        digest = siab_code_digest()
        for entry in self.ctx.candidates:
            label = candidate_label(entry)
            run_dir = os.path.join(root, label)
            try:
                result = run_siab_pipeline(
                    self.inputs.siab_json,
                    orm.Str(run_dir),
                    orm.Int(int(entry["l_max"])),
                    orm.Float(float(entry["r_cut"])),
                    orm.Str(digest),
                    orm.Float(float(entry["ecutjy"])),
                )
            except Exception as exc:  # noqa: BLE001 -- one bad candidate is not fatal
                self.report(f"  ERROR: SIAB failed for {label}: {exc}")
                continue
            info = result["info"].get_dict()
            if not info.get("dft"):
                self.report(f"  ERROR: SIAB produced no DFT job for {label}")
                continue
            problem = siab_tree_problem(info)
            if problem:
                # a cached pipeline result whose files are gone: every child built from
                # it would fail on a missing INPUT/STRU, with nothing pointing at the
                # cache.  Skip the candidate and say so.
                self.report(f"  ERROR: {label}: {problem}")
                self.report(f"         {siab_cache_hint(run_dir)}")
                self.ctx.stale_trees = True
                continue
            self.ctx.siab[label] = info
            orbital = result.get("primitive_orbital")
            if orbital is not None:
                self.ctx.orbitals[label] = orbital
            self.report(
                f"  {label}: {len(info['dft'])} geometries, primitive "
                f"{os.path.basename(str(info.get('nsw_filename') or info.get('orb_path')))}"
            )
        if not self.ctx.siab:
            if getattr(self.ctx, "stale_trees", False):
                return self.exit_codes.ERROR_SIAB_TREE_MISSING
            return self.exit_codes.ERROR_SIAB_FAILED

        # the reference candidate defines the geometry set and the pseudopotential
        self.ctx.siab_info = self.ctx.siab[self.ctx.reference_label]

        problem = self._check_rcut_fits()
        if problem:
            self.report(f"ERROR: {problem}")
            return self.exit_codes.ERROR_RCUT_TOO_LARGE
        return None

    def _check_rcut_fits(self) -> str | None:
        """``r_cut <= half the smallest cell edge`` of the reference structures.

        The largest ``r_cut`` of the ladder is what matters: the ladder only ever
        reduces it, so a reference point that fits keeps every candidate legal.
        """
        from aiida_orbgen.interfaces.stru import parse_stru

        r_cut = float(self.inputs.r_cut.value)
        for dft_entry in self.ctx.siab_info.get("dft", []):
            stru_path = dft_entry.get("stru")
            if not stru_path or not os.path.isfile(str(stru_path)):
                continue
            try:
                edges = cell_edges_bohr(parse_stru(str(stru_path)))
            except Exception:  # noqa: BLE001 -- an unreadable STRU is reported elsewhere
                continue
            problem = rcut_fits_cell(
                r_cut, edges, whats=str(dft_entry.get("folder") or "")
            )
            if problem:
                return problem
        return None

    # ------------------------------------------------------------------
    # Step 3: one pseudo family per candidate (its own orbital + the pseudopotential)
    # ------------------------------------------------------------------
    def family_step(self):
        from aiida_orbgen.calculations.pseudo_family import ensure_pseudo_family

        if self._dry_run:
            self.report("Step 3: [DRY-RUN] pseudo families not registered")
            return None

        build = ("build_family" not in self.inputs) or bool(self.inputs.build_family.value)
        self.ctx.family_labels = []
        for label, info in self.ctx.siab.items():
            base_label = str(info["family_label"])
            resolved = base_label
            if build:
                try:
                    resolved = ensure_pseudo_family(
                        str(info["upf_path"]), str(info["orb_path"]), base_label,
                        build_if_missing=True,
                        description=f"OrbgenBasisScanWorkChain candidate {label}",
                    )
                except Exception as exc:  # noqa: BLE001
                    self.report(
                        f"  WARNING: ensure_pseudo_family failed for {label}: {exc}"
                    )
                    resolved = base_label
            self.ctx.family_by_label[label] = str(resolved)
            self.ctx.family_labels.append(str(resolved))
            if str(resolved) != base_label:
                self.report(
                    f"  ⚠ {label}: family label '{base_label}' was taken -> "
                    f"'{resolved}'"
                )
        self.report(f"Step 3: {len(self.ctx.family_by_label)} pseudo families ready")
        return None

    def _family_for(self, label: str) -> str:
        families = getattr(self.ctx, "family_by_label", {}) or {}
        if label in families:
            return str(families[label])
        known = getattr(self.ctx, "family_labels", []) or [""]
        return str(known[-1])

    # ------------------------------------------------------------------
    # Step 4: the PW reference, computed once
    # ------------------------------------------------------------------
    def needs_pw_reference(self) -> bool:
        """True when this run has to compute the PW reference itself."""
        if self._dry_run or "pw_reference" in self.inputs:
            return False
        return bool(getattr(self.ctx, "siab", None))

    def submit_pw_step(self):
        """One PW child per reference geometry -- the only PW run of the scan."""
        if not hasattr(self.ctx, "siab_info"):
            return self.exit_codes.ERROR_SIAB_FAILED
        options = child_options(
            self, family_label=self._family_for(self.ctx.reference_label)
        )
        overrides: dict[str, Any] = {}
        if "ecutwfc" in self.inputs:
            overrides["ecutwfc"] = float(self.inputs.ecutwfc.value)
        parameters = with_input_overrides(options, overrides) if overrides else None

        self.ctx.children_info = []
        n_submitted = 0
        for dft_entry in self.ctx.siab_info["dft"]:
            geometry = geometry_key(dft_entry)
            try:
                node, n_atoms = submit_child(
                    self, dft_entry=dft_entry, basis="pw", options=options,
                    parameters=parameters,
                )
            except Exception as exc:  # noqa: BLE001
                self.report(f"  ERROR: could not build the PW child of {geometry}: {exc}")
                continue
            entry = record_child(self, node=node, label="pw", geometry=geometry,
                                 basis="pw", n_atoms=n_atoms)
            self.ctx.all_children.append(entry)
            n_submitted += 1
        self.report(
            f"Step 4: {n_submitted} PW reference children "
            f"(ecutwfc={self._pw_ecutwfc()} Ry"
            + ("" if overrides or "ecutwfc" in self.inputs else " from the SIAB input")
            + ")"
        )
        if n_submitted == 0:
            return self.exit_codes.ERROR_NO_PW_REFERENCE
        return None

    def collect_pw_step(self):
        """Turn the PW children into ``ctx.pw_reference`` (geometry -> energy)."""
        try:
            collected = collect_child_energies(
                self.ctx.children_info, report=self.report
            )
        except Exception as exc:  # noqa: BLE001
            self.report(f"ERROR: PW reference extraction failed: {exc}")
            return self.exit_codes.WARNING_ENERGY_EXTRACT_FAILED
        for note in collected.notes:
            self.report(note)

        reference = {}
        for entry in collected.entries:
            if entry.basis != "pw":
                continue
            _, geometry = split_task(entry.folder)
            reference[geometry] = {
                "energy": float(entry.energy),
                "n_atoms": int(entry.n_atoms),
            }
        if not reference:
            self.report("ERROR: the PW reference produced no energy")
            return self.exit_codes.ERROR_NO_PW_REFERENCE
        self.ctx.pw_reference = reference
        self.ctx.pw_source = "computed"
        self.report(
            f"  PW reference: {len(reference)} geometries "
            f"({', '.join(sorted(reference))})"
        )
        return None

    # ------------------------------------------------------------------
    # Step 5: the candidates, cheapest first
    # ------------------------------------------------------------------
    def candidates_left(self) -> bool:
        if self._dry_run or getattr(self.ctx, "done", False):
            return False
        return int(getattr(self.ctx, "cursor", 0)) < len(
            getattr(self.ctx, "candidates", []) or []
        )

    def submit_candidate_step(self):
        entry = self.ctx.candidates[self.ctx.cursor]
        label = candidate_label(entry)
        info = self.ctx.siab.get(label)
        self.ctx.children_info = []
        if info is None:
            # SIAB refused this candidate: it simply has no row in the table
            self.report(f"Step 5: {label} skipped (no SIAB tree)")
            self.ctx.current = {"label": label, "entry": dict(entry), "nodes": []}
            return None

        options = child_options(self, family_label=self._family_for(label))
        nodes = []
        n_skipped = 0
        for dft_entry in info["dft"]:
            geometry = geometry_key(dft_entry)
            try:
                node, n_atoms = submit_child(
                    self, dft_entry=dft_entry, basis="lcao_nsw", options=options,
                )
            except Exception as exc:  # noqa: BLE001
                self.report(
                    f"  ERROR: could not build the LCAO child of {geometry}: {exc}"
                )
                n_skipped += 1
                continue
            record = record_child(self, node=node, label=label, geometry=geometry,
                                  basis="lcao_nsw", n_atoms=n_atoms)
            self.ctx.all_children.append(record)
            nodes.append(node)
        self.ctx.current = {"label": label, "entry": dict(entry), "nodes": nodes}
        self.report(
            f"Step 5: candidate {label} ({self.ctx.cursor + 1}/"
            f"{len(self.ctx.candidates)}): {len(nodes)} LCAO children submitted"
            + (f", {n_skipped} not built" if n_skipped else "")
        )
        return None

    def inspect_candidate_step(self):
        info = self.ctx.children_info
        n_ok = sum(
            1 for item in info
            if item["node"].is_finished_ok or item["node"].exit_status == 304
        )
        self.report(
            f"  children: {n_ok} usable, {len(info) - n_ok} failed (of {len(info)})"
        )
        return None

    def evaluate_candidate_step(self):
        """One row of the table -- and, when it passes, the end of the scan."""
        current = self.ctx.current
        label: str = current["label"]
        entry: dict = current["entry"]
        nodes: list = current["nodes"]

        def _skip() -> None:
            """No verdict for this candidate: it gets no row and the scan goes on."""
            self.ctx.candidate_ok[label] = False
            self.ctx.cursor += 1

        if not nodes:
            _skip()
            return None

        try:
            collected = collect_child_energies(
                self.ctx.children_info, report=self.report
            )
        except Exception as exc:  # noqa: BLE001
            self.report(f"  ERROR: energy extraction failed for {label}: {exc}")
            _skip()
            return None
        for note in collected.notes:
            self.report(note)

        try:
            geometry_rows = geometries_from_entries(
                collected.entries, pw_reference=self._pw_reference()
            )
        except Exception as exc:  # noqa: BLE001
            self.report(f"  ERROR: could not pair the energies of {label}: {exc}")
            _skip()
            return None

        seconds = [seconds_of(node) for node in nodes]
        measured = sum(s for s in seconds if s is not None) or None
        row = {
            "candidate": dict(entry),
            "geometries": geometry_rows.get(label, {}),
            "seconds": measured,
            "nchi": n_primitive_functions(
                entry["r_cut"], entry["ecutjy"], entry["l_max"]
            ),
        }
        self.ctx.rows[label] = row

        # The verdict needs the reference candidate's row for the atomization
        # difference; the reference is candidates[0], so it is always available by the
        # time any other candidate is evaluated.
        window = {label: row}
        reference_label = self.ctx.reference_label
        if reference_label != label and reference_label in self.ctx.rows:
            window[reference_label] = self.ctx.rows[reference_label]
        table = basis_table(window, self._tolerance_meV(), reference=reference_label)
        row_info = next((r for r in table["rows"] if r["label"] == label), None)

        if row_info is None:
            self.report(f"  {label}: no comparable geometry (a missing PW energy?)")
            _skip()
            return None

        acceptable = bool(row_info["tolerance_ok"])
        reason = ""
        gate = self._atomization_gate()
        drift = row_info.get("atomization_vs_reference_meV")
        if acceptable and gate is not None and drift is not None and abs(drift) > gate:
            acceptable = False
            reason = f"  <- rejected by the atomization gate ({drift:+.1f} meV > {gate:g})"
        is_reference = label == self.ctx.reference_label
        self.ctx.candidate_ok[label] = acceptable
        self.report("  " + describe_row(row_info, self._tolerance_meV()) + reason)
        if is_reference:
            self.report(
                "  (the reference point is the starting point of the ladder, not a "
                "reduction: it is only the answer if nothing cheaper passes)"
            )
            self.ctx.cursor += 1
            return None

        if acceptable:
            stop = bool(self.inputs.get("stop_on_first_pass", orm.Bool(True)).value)
            if stop:
                self.ctx.done = True
                self.report(
                    "  ✓ the cheapest candidate of the ladder is inside the tolerance: "
                    "stopping (nothing cheaper can be better)"
                )
        self.ctx.cursor += 1
        return None

    # ------------------------------------------------------------------
    # Step 6: the table, the winner, the outputs
    # ------------------------------------------------------------------
    def finalize(self):
        if self._dry_run:
            self.report("Step 6: [DRY-RUN] nothing submitted, no outputs")
            return ExitCode(0)

        self.ctx.children_info = list(getattr(self.ctx, "all_children", []) or [])
        n_ok, n_failed = write_child_results(self)

        tolerance = self._tolerance_meV()
        by = str(self.inputs.by.value) if "by" in self.inputs else "seconds"
        gate = self._atomization_gate()
        rows = dict(getattr(self.ctx, "rows", {}) or {})
        reference_label = getattr(self.ctx, "reference_label", None)
        table = (
            basis_table(rows, tolerance, reference=reference_label) if rows
            else {"rows": [], "reference": reference_label, "within_tolerance": [],
                  "tolerance_meV": tolerance}
        )
        best = pick_cheapest(
            table, by=by, require_atomization=gate is not None,
            atomization_tolerance_meV=gate,
        )

        self.report("  " + "-" * 72)
        for row in sorted(table["rows"], key=lambda r: r.get("cost") or 0.0):
            self.report("  " + describe_row(row, tolerance))
        self.report("  " + "-" * 72)

        planned = [candidate_label(e) for e in getattr(self.ctx, "candidates", [])]
        decision = {
            "reference": table.get("reference"),
            "reference_point": getattr(self.ctx, "reference_candidate", None),
            "evaluation_order": [candidate_label(e)
                                 for e in getattr(self.ctx, "candidates", [])],
            "reduction_found": bool(
                best and best["label"] != getattr(self.ctx, "reference_label", None)
            ),
            "tolerance_meV": tolerance,
            "atomization_tolerance_meV": gate,
            "by": by,
            "table": table["rows"],
            "within_tolerance": table.get("within_tolerance", []),
            "best": best["label"] if best else None,
            "best_row": best,
            "planned": planned,
            "ran": list(rows),
            "planned_but_not_run": [label for label in planned if label not in rows],
            "stopped_early": bool(getattr(self.ctx, "done", False)),
            "pw_reference": {
                "source": ("input pw_reference" if "pw_reference" in self.inputs
                           else getattr(self.ctx, "pw_source", "none")),
                "ecutwfc": self._pw_ecutwfc(),
                "n_geometries": len(self._pw_reference()),
            },
        }
        self.out("basis_decision", create_dict(decision))

        if best:
            label = best["label"]
            chosen = dict(best["candidate"])
            chosen.update({
                "label": label,
                "dE_max_abs_meV": best.get("dE_max_abs_meV"),
                "dE_per_atom_meV": best.get("dE_per_atom_meV"),
                "atomization_meV": best.get("atomization_meV"),
                "atomization_vs_reference_meV": best.get("atomization_vs_reference_meV"),
                "seconds": best.get("seconds"),
                "nchi": best.get("nchi"),
            })
            info = (getattr(self.ctx, "siab", {}) or {}).get(label) or {}
            for key in ("nsw_filename", "orb_path", "family_label"):
                if info.get(key):
                    chosen[key] = str(info[key])
            if label in (getattr(self.ctx, "family_by_label", {}) or {}):
                chosen["family_label"] = self.ctx.family_by_label[label]
            self.out("chosen_basis", create_dict(chosen))
            orbital = (getattr(self.ctx, "orbitals", {}) or {}).get(label)
            if orbital is not None:
                self.out("primitive_orbital", orbital)
            self.report(
                f"  best: {label} ({chosen.get('nsw_filename', 'n/a')}) -- "
                f"dE={best.get('dE_max_abs_meV')} meV/atom"
                + (f", dA={best.get('atomization_vs_reference_meV')} meV"
                   if best.get("atomization_vs_reference_meV") is not None else "")
            )
            if label == getattr(self.ctx, "reference_label", None):
                self.report(
                    "  WARNING: the answer is the reference point -- no cheaper candidate "
                    "of this ladder met the tolerance (widen it, or relax the criterion: "
                    "tolerance_meV / atomization_tolerance_meV)"
                )

        if n_ok == 0:
            self.report("OrbgenBasisScanWorkChain: no child produced an energy")
            return self.exit_codes.ERROR_ALL_FAILED
        if not rows:
            self.report("OrbgenBasisScanWorkChain: no candidate could be evaluated")
            return self.exit_codes.WARNING_ENERGY_EXTRACT_FAILED
        if best is None:
            self.report("OrbgenBasisScanWorkChain: " + no_basis_message(
                table, tolerance, reference=reference_label,
                atomization_tolerance_meV=gate,
            ))
            return self.exit_codes.WARNING_NO_BASIS_WITHIN_TOLERANCE
        if n_failed:
            self.report(
                f"OrbgenBasisScanWorkChain: chose {best['label']}, but {n_failed} "
                f"child(ren) failed"
            )
            return self.exit_codes.WARNING_PARTIAL_FAILURE
        self.report(
            f"OrbgenBasisScanWorkChain Finished: chose {best['label']} out of "
            f"{len(rows)} evaluated candidate(s)"
        )
        return ExitCode(0)

    def _pw_ecutwfc(self) -> Any:
        """The PW cutoff the reference children run with.

        ``abacus.json`` usually does not carry ``ecutwfc`` at all (the SIAB-generated
        INPUT does), so the JSON reading mirrors
        :meth:`OrbgenEcutwfcWorkChain._baseline_ecutwfc` -- reporting "abacus.json Ry"
        instead of a number is useless when the question is which cutoff was used.
        """
        if "ecutwfc" in self.inputs:
            return float(self.inputs.ecutwfc.value)
        parameters = (self.ctx.abacus_cfg.get("abacus", {}) or {}).get("parameters", {}) or {}
        value = (parameters.get("input", {}) or {}).get("ecutwfc")
        if value:
            return value
        try:
            content = self.inputs.siab_json.get_content()
            if isinstance(content, bytes):
                content = content.decode("utf-8")
            return json.loads(content).get("ecutwfc")
        except Exception:  # noqa: BLE001 -- the SIAB JSON is validated elsewhere
            return None
