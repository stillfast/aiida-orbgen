"""
OrbgenCalcWorkChain — batch ABACUS submission for one (l_max, r_cut) pair

Corresponds to the "OrbgenCalc" box in the flowchart: with orbgen.json +
abacus.json, run the tasks for one specific l_max and r_cut.

Each OrbgenCalcWorkChain:
  1. Takes orbgen.json (element / pseudo_dir / geoms / ...),
     abacus.json (basis / parameters.input / tolerance),
     plus the fixed (l_max, r_cut) pair.
  2. Calls the SIAB pipeline (a cached calcfunction) to generate the NSW and the
     INPUT/STRU files of the 5 structures.
  3. Runs one abacus.base task per structure x {PW, LCAO:nsw} = 10 tasks.
  4. Collects the energies and outputs a Dict.

Entry point
-----------
* ``orbgen.calc`` — registered in ``pyproject.toml``
"""
from __future__ import annotations

from pathlib import Path
from aiida import orm
from aiida.engine import WorkChain, ExitCode, append_, if_, while_
from aiida.orm import (
    Bool,
    Dict,
    Float,
    Int,
    List,
    SinglefileData,
    Str,
)

from aiida_orbgen.static.defaults import (
    DEFAULT_CODE_LABEL,
    DEFAULT_MAX_MEMORY_KB,
    DEFAULT_NUM_MPI,
    DEFAULT_QUEUE_NAME,
    DEFAULT_WALLCLOCK_SECONDS,
)
from aiida_orbgen.static.json_inputs import with_default_abacus
from aiida_orbgen.workflows.energies import (
    SOFT_SUCCESS_EXIT_STATUS,
    describe_deltas,
    evaluate_energies,
    is_soft_success,
)
from aiida_orbgen.workflows.extract import collect_child_energies
from aiida_orbgen.workflows._grid import (
    GridEntry,
    build_cartesian_grid,
    build_explicit_grid,
    build_multi_json_grid,
    cap_grid,
    has_pending_iterative,
    work_dir_name,
)

# SIAB-facing code and result assembly live in their own modules; they are
# re-exported here because ``pyproject.toml`` points the entry points at this
# module and existing imports (tests, report layer) use these names.
from aiida_orbgen.workflows.results import (
    create_energies_dict,
    create_family_label,
    create_final_results,
    create_grid_all_results,
    create_grid_summary,
)
from aiida_orbgen.workflows.siab import (
    build_abacus_child_inputs,
    siab_code_digest,
    n_atoms_from_stru,
    run_siab_pipeline,
)

# abacuslite reads STRU into a dict

__all__ = [
    "OrbgenCalcWorkChain",
    "OrbgenGridSearchWorkChain",
    "run_siab_pipeline",
    "build_abacus_child_inputs",
]


# ===========================================================================
#  OrbgenCalcWorkChain
# ===========================================================================


def _validate_siab_json_inputs(inputs, report) -> str | None:
    """Validate every SIAB config a WorkChain was handed.

    Returns an error message (and reports warnings) or ``None``.  A WorkChain
    submitted directly through ``WorkflowFactory`` never passes
    ``ConfigLoader``, so without this the two classes of mistake that cost the
    most time would only show up inside SIAB:

    * an ``nzeta`` scheme the primitive basis cannot provide — discovered in
      ``basistrans`` *after* the reference DFT has been paid for;
    * ``vloc_aux`` / ``lloc_min`` written outside ``model_kwargs`` — silently
      dropped, so a requested g channel comes out empty.
    """
    import json as _json

    from aiida_orbgen.utils.config import validate_siab_config

    nodes = []
    if "siab_json" in inputs:
        nodes.append(inputs.siab_json)
    if "orbgen_jsons" in inputs:
        nodes.extend(inputs.orbgen_jsons.get_list())
    for node in nodes:
        try:
            content = node.get_content()
            if isinstance(content, bytes):
                content = content.decode("utf-8")
            config = _json.loads(content)
        except Exception as exc:  # noqa: BLE001 — unreadable JSON is a config error
            return f"cannot read siab_json<{node.pk}>: {exc}"
        try:
            warnings = validate_siab_config(
                config, source=f"siab_json<{node.pk}>"
            )
        except Exception as exc:  # noqa: BLE001 — ValueError from the validator
            return str(exc)
        for warning in warnings:
            report(f"  WARNING: {warning}")
    return None


class OrbgenCalcWorkChain(WorkChain):
    """Run PW + LCAO:nsw over 5 structures = 10 tasks for one (l_max, r_cut) pair.

    Inputs
    ------
    siab_json : SinglefileData
        pbe_orbgen.json
    abacus_config : Dict
        Contents of abacus.json (basis / input_overrides / tolerance / scheduler)
    output_dir : Str
        Directory SIAB writes its output into
    l_max : Int
        Highest angular momentum
    r_cut : Float
        Cutoff radius (Å)
    code_label : Str, optional
    family_label : Str, optional
    build_family : Bool, optional
    only : Int, optional
    max_iterations : Int, optional

    Outputs
    -------
    siab_info : Dict
        SIAB pipeline result
    results : Dict
        Per-task PK + status
    energies : Dict
        Energies + ΔE_max (comparison by basis_type)

    Entry point
    -----------
    ``orbgen.batch``
    """

    _child_workchain_entry_point = "abacus.base"

    # ------------------------------------------------------------------
    # define
    # ------------------------------------------------------------------

    @classmethod
    def define(cls, spec):
        super().define(spec)

        # ---- JSON inputs ----
        spec.input("siab_json", valid_type=SinglefileData,
                   help="pbe_orbgen.json (SIAB config, contains UPF path).")
        spec.input("abacus_config", valid_type=Dict,
                   help="abacus.json (basis / input_overrides / tolerance / scheduler).")

        # ---- fixed (l_max, r_cut) ----
        spec.input("l_max", valid_type=Int,
                   help="Highest angular momentum (orbit gen parameter).")
        spec.input("r_cut", valid_type=Float,
                   help="Cutoff radius, Å (orbit gen parameter).")

        # ---- paths ----
        spec.input("output_dir", valid_type=Str,
                   help="SIAB output directory (must be writable by the worker).")

        # ---- AiiDA / scheduler ----
        spec.input("code_label", valid_type=Str, required=False,
                   help="AiiDA code label (default: from static.defaults).")
        spec.input("family_label", valid_type=Str, required=False,
                   help="AiiDA pseudo family label (default: auto-infer).")
        spec.input("max_iterations", valid_type=Int, required=False,
                   help="Max abacus.base workchain retry.")

        # ---- behaviour control ----
        spec.input("build_family", valid_type=Bool, required=False,
                   default=lambda: orm.Bool(True),
                   help="Auto-build pseudo_family if missing.")
        spec.input("only", valid_type=Int, required=False,
                   help="Only run N-th task (0-indexed, debug).")
        spec.input("dry_run", valid_type=Bool, required=False,
                   default=lambda: orm.Bool(False),
                   help="If True, only print, don't submit.")

        # ---- outputs ----
        spec.output("siab_info", valid_type=Dict, required=False,
                    help="SIAB pipeline result (nsw, family_label, dft, ...).")
        spec.output("primitive_orbital", valid_type=SinglefileData, required=False,
                    help="The primitive NSW .orb this grid point used, archived in "
                         "provenance (the paths in siab_info point at scratch space).")
        spec.output("pseudo_family", valid_type=Str, required=False,
                    help="Pseudo family the children were given. Differs from "
                         "siab_info.family_label when that name already belonged to a "
                         "family built from another pseudopotential, in which case a "
                         "content-suffixed label is registered instead.")
        spec.output("results", valid_type=Dict, required=False,
                    help="Per-task PK + status.")
        spec.output("energies", valid_type=Dict, required=False,
                    help="Extracted energies + ΔE_max (eV, meV).")

        # ---- exit codes ----
        spec.exit_code(401, "ERROR_INVALID_ABACUS_CONFIG",
                       message="abacus_config invalid")
        spec.exit_code(402, "ERROR_NO_BASIS",
                       message="abacus_config.basis is empty")
        spec.exit_code(403, "ERROR_SIAB_FAILED",
                       message="SIAB pipeline failed")
        spec.exit_code(404, "ERROR_NO_DFT_JOBS",
                       message="SIAB produced 0 DFT jobs")
        spec.exit_code(405, "ERROR_BUILD_INPUTS",
                       message="Failed to build child inputs")
        spec.exit_code(301, "WARNING_PARTIAL_FAILURE",
                       message="Some children failed")
        spec.exit_code(302, "ERROR_ALL_FAILED",
                       message="All children failed")
        spec.exit_code(303, "WARNING_ENERGY_EXTRACT_FAILED",
                       message="Could not extract energies from outputs")
        spec.exit_code(304, "WARNING_TOLERANCE_EXCEEDED",
                       message="max |E_lcao_nsw - E_pw| exceeds tolerance (meV)")
        spec.exit_code(406, "ERROR_INVALID_SIAB_CONFIG",
                       message="siab_json is not a usable SIAB configuration")

        # ---- outline ----
        spec.outline(
            cls.validate_inputs,
            cls.run_siab_pipeline_step,
            cls.ensure_pseudo_family,
            cls.submit_children,
            cls.inspect_children,
            cls.extract_energies_step,
            cls.finalize,
        )

    # ------------------------------------------------------------------
    # dry run
    # ------------------------------------------------------------------

    @property
    def _dry_run(self) -> bool:
        """The ``dry_run`` input (produce a plan only, touch nothing).

        Note the AiiDA semantics: a step returning ``ExitCode(0)`` does **not**
        terminate the outline -- ``aiida/engine/processes/workchains/workchain.py``
        maps a status-0 ExitCode to ``None`` and carries on with the next step.
        Every step therefore has to check this flag itself; in this class only
        submit/inspect/extract/finalize used to check it, while Step 1 (SIAB
        pipeline) and Step 1.5 (building the pseudo family) did not -- so a dry
        run still launched the SIAB subprocess and created the pseudo family in
        the database.
        """
        return bool(self.inputs.get("dry_run", orm.Bool(False)).value)

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------

    def validate_inputs(self):
        """Check abacus_config has basis, etc."""
        cfg = self.inputs.abacus_config.get_dict()
        if not cfg:
            return self.exit_codes.ERROR_INVALID_ABACUS_CONFIG
        basis = cfg.get("basis", [])
        if not basis:
            return self.exit_codes.ERROR_NO_BASIS
        for b in basis:
            if b not in ("pw", "lcao_nsw"):
                self.report(f"ERROR: unknown basis {b!r}")
                return self.exit_codes.ERROR_INVALID_ABACUS_CONFIG
        # Apply the defaults
        self.ctx.abacus_cfg = with_default_abacus(cfg)
        self.report(
            f"abacus.json: basis={self.ctx.abacus_cfg['basis']}, "
            f"tolerance_meV={self.ctx.abacus_cfg['tolerance_meV']}"
        )

        problem = _validate_siab_json_inputs(self.inputs, self.report)
        if problem:
            self.report(f"ERROR: invalid SIAB config: {problem}")
            return self.exit_codes.ERROR_INVALID_SIAB_CONFIG
        return None

    # ------------------------------------------------------------------
    # Step 1: SIAB pipeline
    # ------------------------------------------------------------------

    def run_siab_pipeline_step(self):
        """Run ``generate_all_from_json`` with l_max/r_cut override."""
        self.report(
            f"Step 1: SIAB pipeline with l_max={self.inputs.l_max.value}, "
            f"r_cut={self.inputs.r_cut.value}"
        )
        if self._dry_run:
            # A dry run must not touch the filesystem: this step used to run
            # unconditionally, so "dry" runs still launched SIAB.
            self.report("  [DRY-RUN] SIAB pipeline not launched")
            return
        try:
            siab_result = run_siab_pipeline(
                self.inputs.siab_json,
                self.inputs.output_dir,
                self.inputs.l_max,
                self.inputs.r_cut,
                # cache key: AiiDA would otherwise hand back a job list computed by an
                # older checkout (see siab.siab_code_digest)
                siab_code_digest(),
            )
        except Exception as exc:
            self.report(f"ERROR: SIAB pipeline failed: {exc}")
            return self.exit_codes.ERROR_SIAB_FAILED

        if "primitive_orbital" in siab_result:
            self.out("primitive_orbital", siab_result["primitive_orbital"])
        info = siab_result["info"].get_dict()
        n_dft = len(info.get("dft", []))
        if n_dft == 0:
            return self.exit_codes.ERROR_NO_DFT_JOBS

        self.report(
            f"  NSW: {info['nsw']}\n"
            f"  UPF: {info['upf_path']}\n"
            f"  ORB: {info['orb_path']}\n"
            f"  family: {info['family_label']}\n"
            f"  pertmags: {info['pertmags']}\n"
            f"  -> {n_dft} DFT job(s) generated"
        )
        self.ctx.siab_info = info
        self.out("siab_info", siab_result["info"])

    # ------------------------------------------------------------------
    # Step 1.5: pseudo_family
    # ------------------------------------------------------------------

    def ensure_pseudo_family(self):
        from aiida_orbgen.calculations.pseudo_family import ensure_pseudo_family

        if self._dry_run:
            # Also a database side effect: creating a pseudo family is not
            # something a dry run may do.
            self.report("Step 1.5: [DRY-RUN] pseudo family not registered")
            return
        if (
            "build_family" in self.inputs
            and not self.inputs.build_family.value
        ):
            return
        base_label = self._base_family_label()
        upf_path = self.ctx.siab_info["upf_path"]
        orb_path = self.ctx.siab_info["orb_path"]
        self.report(f"Step 1.5: ensure_pseudo_family('{base_label}') ...")
        try:
            # The label that comes back is the one the children must use: when the
            # derived name already belongs to a family built from *other* files,
            # ensure_pseudo_family registers a content-suffixed one instead of
            # silently reusing the old pseudopotential (see
            # calculations/pseudo_family.label_for_pair).
            label = ensure_pseudo_family(
                upf_path, orb_path, base_label,
                build_if_missing=True,
                description="Built by OrbgenCalcWorkChain",
            )
            self.ctx.pseudo_family_label = label
            if label != base_label:
                self.report(
                    f"  ⚠ family label '{base_label}' was taken by another "
                    f"pseudopotential -> using '{label}'"
                )
        except Exception as exc:
            self.report(f"WARNING: ensure_pseudo_family failed: {exc}")

    def _base_family_label(self) -> str:
        """The family label *before* the content check.

        An explicit ``family_label`` input wins — it used to be ignored when the
        family was registered (only the children honoured it), so a run could build
        one label and hand the children another.
        """
        if "family_label" in self.inputs:
            return str(self.inputs.family_label.value)
        return str(self.ctx.siab_info["family_label"])

    def _effective_family_label(self) -> str:
        """The label the children must reference (built family first)."""
        return str(
            getattr(self.ctx, "pseudo_family_label", None)
            or self._base_family_label()
        )

    # ------------------------------------------------------------------
    # Step 2: submit children (basis x dft_entry)
    # ------------------------------------------------------------------

    def submit_children(self):
        """Submit one abacus.base workchain per (basis, dft_entry)."""
        if self._dry_run:
            # The plan is only known after the SIAB step, which a dry run
            # skips -- so report what the inputs describe instead.
            self.report(
                f"Step 2: [DRY-RUN] would submit "
                f"{len(self.ctx.abacus_cfg['basis'])} basis × N structures "
                f"(l_max={self.inputs.l_max.value}, "
                f"r_cut={self.inputs.r_cut.value})"
            )
            self.ctx.children_info = []
            return
        if not hasattr(self.ctx, "siab_info"):
            return self.exit_codes.ERROR_SIAB_FAILED

        dft_list = self.ctx.siab_info["dft"]
        only = self.inputs.get("only")
        if only is not None and only.value is not None:
            idx = only.value
            if 0 <= idx < len(dft_list):
                dft_list = [dft_list[idx]]
                self.report(f"  [only={idx}] running task #{idx} only")
            else:
                self.report(f"  WARNING: only={idx} is out of range")

        basis_list = self.ctx.abacus_cfg["basis"]
        self.report(
            f"Step 2: submitting {len(dft_list)} structures x {len(basis_list)} "
            f"basis = {len(dft_list) * len(basis_list)} abacus.base runs"
        )

        # Read the configuration from the "abacus" key
        abacus_config = self.ctx.abacus_cfg.get("abacus", {})
        metadata_options = abacus_config.get("metadata", {}).get("options", {})
        
        queue_name = metadata_options.get("queue_name", DEFAULT_QUEUE_NAME)
        num_mpi = metadata_options.get("num_mpiprocs_per_machine", DEFAULT_NUM_MPI)
        if num_mpi is None:
            num_mpi = metadata_options.get("num_mpi", DEFAULT_NUM_MPI)
        wallclock = metadata_options.get("max_wallclock_seconds", DEFAULT_WALLCLOCK_SECONDS)
        max_memory_kb = metadata_options.get("max_memory_kb", DEFAULT_MAX_MEMORY_KB)
        code_label = (
            self.inputs.get("code_label").value
            if "code_label" in self.inputs
            else abacus_config.get("code", DEFAULT_CODE_LABEL)
        )
        family_label = self._effective_family_label()
        input_overrides = abacus_config.get("parameters", {})

        self.ctx.children_info = []

        for dft_entry in dft_list:
            for basis in basis_list:
                try:
                    inputs = build_abacus_child_inputs(
                        dft_entry,
                        basis=basis,
                        code_label=code_label,
                        family_label=family_label,
                        parameters=input_overrides,
                        queue_name=queue_name,
                        num_mpi=num_mpi,
                        wallclock=wallclock,
                        max_memory_kb=max_memory_kb,
                    )
                except Exception as exc:
                    self.report(
                        f"  build_inputs FAILED for {dft_entry['folder']} "
                        f"| {basis}: {exc}"
                    )
                    continue

                from aiida.plugins import WorkflowFactory
                child_cls = WorkflowFactory(self._child_workchain_entry_point)

                # Read the STRU to count atoms (for per-atom energy normalisation)
                n_atoms = self._read_n_atoms_from_stru(dft_entry)

                running = self.submit(child_cls, **inputs)
                running.base.extras.set_many({
                    "task": dft_entry["folder"],
                    "basis": basis,
                    "l_max": str(self.inputs.l_max.value),
                    "r_cut": str(self.inputs.r_cut.value),
                    "n_atoms": str(n_atoms),
                })
                self.ctx.children_info.append({
                    "task": dft_entry["folder"],
                    "basis": basis,
                    "node": running,
                    "n_atoms": n_atoms,
                })
                self.to_context(children=append_(running))

    # ------------------------------------------------------------------
    # Step 3: wait for the children
    # ------------------------------------------------------------------

    def _read_n_atoms_from_stru(self, dft_entry: dict) -> int:
        """Atom count from the generated STRU (per-atom ΔE normalisation)."""
        return n_atoms_from_stru(dft_entry.get("stru"), report=self.report)

    def inspect_children(self):
        info = self.ctx.children_info
        n_ok = sum(
            1 for i in info
            if is_soft_success(i["node"].is_finished_ok, i["node"].exit_status)
        )
        n_fail = len(info) - n_ok
        self.report(f"  Children: {n_ok} OK, {n_fail} failed (of {len(info)})")
        return None  # carry on to extract_energies

    # ------------------------------------------------------------------
    # Step 4: extract energies
    # ------------------------------------------------------------------

    def extract_energies_step(self):
        """ΔE of every usable child, plus the tolerance verdict.

        Node walking lives in ``workflows/extract.py`` and the arithmetic in
        ``workflows/energies.py``; this step only narrates the result and turns it
        into context flags — no exit code, so that the results/energies nodes are
        still written out by ``finalize``.
        """
        try:
            collected = collect_child_energies(
                self.ctx.children_info, report=self.report
            )
        except Exception as exc:  # noqa: BLE001 — a broken child must not abort the WC
            import traceback
            self.report(f"ERROR: energy extraction failed: {exc}")
            self.report(traceback.format_exc())
            return self.exit_codes.WARNING_ENERGY_EXTRACT_FAILED

        for note in collected.notes:
            self.report(note)

        if not collected.usable:
            self.report("WARNING: no children outputs to extract")
            self.report(
                f"  (children_info = {len(self.ctx.children_info)} entries)"
            )
            return self.exit_codes.WARNING_ENERGY_EXTRACT_FAILED

        if collected.lcao_all_skipped:
            # Not one LCAO result is trustworthy, so the tolerance is effectively
            # not met — `finalize` turns this into WARNING_TOLERANCE_EXCEEDED.
            self.ctx.tolerance_exceeded = True
            self.ctx.lcao_all_skipped = True
        elif collected.n_lcao_skipped:
            self.ctx.lcao_skipped_count = collected.n_lcao_skipped

        energies = collected.energies
        self.ctx.energies = energies
        for line in describe_deltas(energies):
            self.report(line)

        # ΔE/atom against the tolerance: 0.1 kcal/mol ≈ 4.2 meV/atom (chemical
        # accuracy) is the per-atom standard.  The verdict has one implementation
        # (energies.evaluate_energies), the same one the grid search uses.
        if "lcao" in energies.get("energies", {}) and "pw" in energies.get("energies", {}):
            tolerance_meV = float(self.ctx.abacus_cfg.get("tolerance_meV", 4.2))
            verdict = evaluate_energies(energies, tolerance_meV)
            self.ctx.tolerance_meV = tolerance_meV
            self.ctx.delta_per_atom_meV = verdict["delta_per_atom_meV"]
            self.report(f"  {verdict['message']}")
            self.ctx.tolerance_exceeded = not verdict["tolerance_ok"]

        return None

    # ------------------------------------------------------------------
    # Step 5: final results
    # ------------------------------------------------------------------

    def finalize(self):
        info = self.ctx.children_info
        results = []
        n_ok = 0
        n_failed = 0
        for item in info:
            node = item["node"]
            ok = is_soft_success(node.is_finished_ok, node.exit_status)
            if ok:
                n_ok += 1
            else:
                n_failed += 1
            results.append({
                "task": item["task"],
                "basis": item["basis"],
                "pk": node.pk,
                "exit_status": node.exit_status,
                "ok": ok,
            })

        # Create the results Dict through a calcfunction
        results_node = create_final_results(
            l_max_val=self.inputs.l_max.value,
            r_cut_val=self.inputs.r_cut.value,
            results_list=results,
        )
        self.out("results", results_node)
        # Record the family the children actually used, so a later report does not
        # have to re-derive it (and cannot pick the other pseudopotential's family).
        if getattr(self.ctx, "pseudo_family_label", None):
            self.out("pseudo_family",
                     create_family_label(self.ctx.pseudo_family_label))
        self.report(
            f"OrbgenCalcWorkChain Finished: {n_ok} OK, {n_failed} failed"
        )

        # If there are energies, output those through a calcfunction too
        if hasattr(self.ctx, "energies") and self.ctx.energies:
            # Create the energies Dict through a calcfunction
            energies_node = create_energies_dict(self.ctx.energies)
            self.out("energies", energies_node)

        # Decide the final exit code (priority: ERROR > tolerance > partial > 0)
        if n_ok == 0:
            return self.exit_codes.ERROR_ALL_FAILED
        # tolerance outranks partial: report tolerance first even when a child failed
        if getattr(self.ctx, "tolerance_exceeded", False):
            return self.exit_codes.WARNING_TOLERANCE_EXCEEDED
        if n_failed > 0:
            return self.exit_codes.WARNING_PARTIAL_FAILURE
        return ExitCode(0)


# ===========================================================================
# OrbgenGridSearchWorkChain
# ===========================================================================
class OrbgenGridSearchWorkChain(WorkChain):
    """Grid search: run OrbgenCalcWorkChain over several (l_max, r_cut) candidates
    and pick the **smallest** combination that meets the tolerance.

    Inputs
    ------
    - ``siab_json`` (SinglefileData)         - orbgen.json
    - ``abacus_config`` (Dict)                - abacus.json (same as OrbgenCalcWorkChain)
    - ``l_max_candidates`` (List of Int)     - candidate l_max values (ascending)
    - ``r_cut_candidates`` (List of Float)   - candidate r_cut values (ascending)
    - ``candidates`` (List, optional)        - explicit [[l_max, r_cut], ...] (not a
                                                Cartesian product)
    - ``stop_on_first_valid`` (Bool, default True) - in iterative mode stop at the
                                                first acceptable combination; False =
                                                run the whole grid
    - ``max_l_max`` / ``max_r_cut`` in abacus_config crop the candidate grid
    - ``output_dir`` (Str)                   - root SIAB output directory (one
                                                subdirectory per combination)
    - ``search_strategy`` (Str, optional)    - "iterative" (default, try from small
                                                to large) or "exhaustive" (all in
                                                parallel)
    - ``code_label``, ``family_label`` etc.  - forwarded to OrbgenCalcWorkChain

    Outputs
    -------
    - ``best_result`` (Dict)   - energies Dict of the best (l_max, r_cut)
    - ``all_results`` (Dict)   - every tried (l_max, r_cut) -> result
    - ``grid_summary`` (Dict)  - grid search summary (best, n_tried, n_passed, ...)
    """

    _child_workchain_entry_point = "orbgen.calc"

    # ------------------------------------------------------------------
    # define
    # ------------------------------------------------------------------
    @classmethod
    def define(cls, spec):
        super().define(spec)

        # ---- JSON inputs ----
        spec.input("siab_json", valid_type=SinglefileData, required=False,
                   help="pbe_orbgen.json (SIAB config, contains UPF path). "
                        "Mutually exclusive with orbgen_jsons.")
        spec.input("orbgen_jsons", valid_type=List, required=False,
                   help="List[SinglefileData]: one config per JSON (different nzeta / "
                        "bessel_nao_rcut). l_max = len(nzeta[0]) - 1, "
                        "r_cut = bessel_nao_rcut[0]. "
                        "Mutually exclusive with siab_json.")
        spec.input("abacus_config", valid_type=Dict,
                   help="abacus.json (basis / input_overrides / tolerance / scheduler).")

        # ---- candidate grid ----
        spec.input("l_max_candidates", valid_type=List, required=False,
                   help="Candidate l_max values (ascending). Mutually exclusive "
                        "with orbgen_jsons.")
        spec.input("r_cut_candidates", valid_type=List, required=False,
                   help="Candidate r_cut values (ascending, Å). Mutually exclusive "
                        "with orbgen_jsons.")
        spec.exit_code(406, "ERROR_INVALID_SIAB_CONFIG",
                       message="a candidate siab_json is not a usable SIAB "
                               "configuration")
        spec.input("candidates", valid_type=List, required=False,
                   help="Explicit candidate list [[l_max, r_cut], ...] (not a Cartesian "
                        "product), for adding points to an already scanned grid. Mutually "
                        "exclusive with l_max_candidates/r_cut_candidates.")

        # ---- paths ----
        spec.input("output_dir", valid_type=Str,
                   help="Root SIAB output directory (must be writable by the worker).")

        # ---- search strategy ----
        spec.input("search_strategy", valid_type=Str, required=False,
                   default=lambda: orm.Str("iterative"),
                   help='"iterative" (default, small to large) or "exhaustive" '
                        '(all in parallel).')
        spec.input("stop_on_first_valid", valid_type=Bool, required=False,
                   default=lambda: orm.Bool(True),
                   help="In the iterative strategy, stop at the first acceptable "
                        "combination (default). Set to False to scan the whole grid and then "
                        "pick the best - that used to be exhaustive-only behaviour, now both "
                        "strategies support it.")

        # ---- forwarded to OrbgenCalcWorkChain (optional) ----
        spec.input("code_label", valid_type=Str, required=False,
                   help="AiiDA code label.")
        spec.input("family_label", valid_type=Str, required=False,
                   help="AiiDA pseudo family label.")
        spec.input("max_iterations", valid_type=Int, required=False)
        spec.input("build_family", valid_type=Bool, required=False)
        spec.input("only", valid_type=Int, required=False,
                   help="Debug only: run only the N-th structure inside each OrbgenCalcWorkChain.")
        spec.input("dry_run", valid_type=Bool, required=False,
                   default=lambda: orm.Bool(False),
                   help="If True, only print, don't submit.")

        # ---- outputs ----
        spec.output("best_result", valid_type=Dict, required=False,
                    help="Energies Dict of the best (l_max, r_cut).")
        spec.output("all_results", valid_type=Dict, required=False,
                    help="Every tried (l_max, r_cut) -> result.")
        spec.output("grid_summary", valid_type=Dict, required=False,
                    help="Grid search summary.")

        # ---- exit codes ----
        spec.exit_code(401, "ERROR_INVALID_INPUT",
                       message="l_max_candidates or r_cut_candidates invalid")
        spec.exit_code(402, "ERROR_INVALID_ABACUS_CONFIG",
                       message="abacus_config invalid")
        spec.exit_code(403, "ERROR_EMPTY_GRID",
                       message="No (l_max, r_cut) combinations to try")
        spec.exit_code(404, "ERROR_NO_ACCEPTABLE_ORBITALS",
                       message="All (l_max, r_cut) candidates exceed tolerance")
        spec.exit_code(301, "WARNING_PARTIAL_FAILURE",
                       message="Some CalcWorkChains failed")

        # ---- outline ----
        spec.outline(
            cls.validate_inputs,
            cls.generate_grid_step,
            cls.launch_search_step,
            if_(cls.is_iterative)(
                while_(cls.has_pending_iterative)(
                    cls.iterate_step,
                ),
            ).else_(
                cls.collect_exhaustive_step,
            ),
            cls.finalize,
        )

    # ------------------------------------------------------------------
    # dry run
    # ------------------------------------------------------------------

    @property
    def _dry_run(self) -> bool:
        """The ``dry_run`` input (produce a plan only, touch nothing).

        Note the AiiDA semantics: a step returning ``ExitCode(0)`` does **not**
        terminate the outline -- ``aiida/engine/processes/workchains/workchain.py``
        maps a status-0 ExitCode to ``None`` and carries on with the next step.
        Every step therefore has to check this flag itself; in this class only
        submit/inspect/extract/finalize used to check it, while Step 1 (SIAB
        pipeline) and Step 1.5 (building the pseudo family) did not -- so a dry
        run still launched the SIAB subprocess and created the pseudo family in
        the database.
        """
        return bool(self.inputs.get("dry_run", orm.Bool(False)).value)

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------
    def validate_inputs(self):
        """Validate the inputs."""
        cfg = self.inputs.abacus_config.get_dict()
        if not cfg:
            return self.exit_codes.ERROR_INVALID_ABACUS_CONFIG

        # Check mutual exclusivity: siab_json vs orbgen_jsons
        has_single = "siab_json" in self.inputs
        has_multi = "orbgen_jsons" in self.inputs
        if has_single and has_multi:
            self.report(
                "ERROR: siab_json and orbgen_jsons are mutually exclusive, pick one"
            )
            return self.exit_codes.ERROR_INVALID_INPUT
        if not has_single and not has_multi:
            self.report(
                "ERROR: one of siab_json or orbgen_jsons must be provided"
            )
            return self.exit_codes.ERROR_INVALID_INPUT
        if "candidates" in self.inputs and (
            "l_max_candidates" in self.inputs or "r_cut_candidates" in self.inputs
        ):
            self.report(
                "ERROR: candidates is mutually exclusive with l_max_candidates/r_cut_candidates"
            )
            return self.exit_codes.ERROR_INVALID_INPUT

        # Multi-JSON mode: the grid comes from the (l_max, r_cut) of each JSON
        if has_multi:
            json_nodes = self.inputs.orbgen_jsons.get_list()
            if not json_nodes:
                self.report("ERROR: orbgen_jsons is empty")
                return self.exit_codes.ERROR_INVALID_INPUT
            self.ctx.orbgen_jsons_list = list(json_nodes)
            self.ctx.use_multi_json = True
            self.report(
                f"Mode: multi-JSON ({len(json_nodes)} files), "
                "l_max/r_cut extracted from each JSON"
            )
        else:
            self.ctx.use_multi_json = False
            # Read the candidates
            l_max_list = self.inputs.l_max_candidates.get_list()
            r_cut_list = self.inputs.r_cut_candidates.get_list()
            if not l_max_list or not r_cut_list:
                self.report(
                    f"ERROR: empty candidates: l_max={l_max_list}, r_cut={r_cut_list}"
                )
                return self.exit_codes.ERROR_INVALID_INPUT
            # Sort ascending so the smaller parameters come first
            self.ctx.l_max_list = sorted(set(int(x) for x in l_max_list))
            self.ctx.r_cut_list = sorted(set(float(x) for x in r_cut_list))
            self.report(
                f"Mode: Cartesian product, "
                f"l_max candidates = {self.ctx.l_max_list}, "
                f"r_cut candidates = {self.ctx.r_cut_list}"
            )

        problem = _validate_siab_json_inputs(self.inputs, self.report)
        if problem:
            self.report(f"ERROR: invalid SIAB config: {problem}")
            return self.exit_codes.ERROR_INVALID_SIAB_CONFIG

        self.ctx.tolerance_meV = float(cfg.get("tolerance_meV", 4.2))
        self.ctx.search_strategy = str(
            self.inputs.get("search_strategy").value
            if "search_strategy" in self.inputs
            else "iterative"
        )
        self.report(
            f"Strategy: {self.ctx.search_strategy}, "
            f"tolerance = {self.ctx.tolerance_meV} meV/atom"
        )
        return None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_lmax_rcut_from_orbgen_json(json_node: SinglefileData) -> tuple:
        """Extract (l_max, r_cut) from an orbgen.json.

        - l_max: len(orbitals[0].nzeta) - 1
                 (a length of 6 = s/p/d/f/g/h -> l_max=5)
        - r_cut: bessel_nao_rcut[0]

        Returns
        -------
        (l_max: int, r_cut: float)
        """
        import json as _json
        content = json_node.get_content()
        if isinstance(content, bytes):
            content = content.decode("utf-8")
        cfg = _json.loads(content)

        # nzeta: prefer orbitals[0].nzeta, then geoms[0].lmaxmax
        nzeta_list = cfg.get("orbitals", [{}])[0].get("nzeta")
        if nzeta_list:
            l_max = len(nzeta_list) - 1
        else:
            geoms = cfg.get("geoms", [])
            l_max = int(geoms[0].get("lmaxmax", 4)) if geoms else 4

        # r_cut: bessel_nao_rcut
        rcut_raw = cfg.get("bessel_nao_rcut", [9])
        if isinstance(rcut_raw, (int, float)):
            r_cut = float(rcut_raw)
        else:
            r_cut = float(rcut_raw[0])

        return l_max, r_cut

    # ------------------------------------------------------------------
    # Step 1: build the grid
    # ------------------------------------------------------------------
    def generate_grid_step(self):
        """Build the list of ``GridEntry`` candidates (see ``workflows/_grid.py``)."""
        grid = []
        if "candidates" in self.inputs:
            pairs = self.inputs.candidates.get_list()
            try:
                grid = build_explicit_grid(pairs, self.inputs.siab_json)
            except (TypeError, ValueError) as exc:
                self.report(f"ERROR: invalid candidates: {exc}")
                return self.exit_codes.ERROR_INVALID_INPUT
            self.report(f"Mode: explicit candidates ({len(grid)} points)")
        elif self.ctx.use_multi_json:
            # Multi-JSON mode: one (l_max, r_cut) combination per JSON
            pairs = []
            for json_node in self.ctx.orbgen_jsons_list:
                l_max, r_cut = self._extract_lmax_rcut_from_orbgen_json(json_node)
                pairs.append((int(l_max), float(r_cut), json_node))
                self.report(
                    f"  [multi-json] {json_node.filename}: l_max={l_max}, r_cut={r_cut}"
                )
            grid = build_multi_json_grid(pairs)
        else:
            # Single-JSON mode: Cartesian product, sorted by (l_max ↑, r_cut ↑)
            grid = build_cartesian_grid(
                self.ctx.l_max_list, self.ctx.r_cut_list, self.inputs.siab_json
            )

        # ``with_default_abacus`` forwards these two caps from abacus.json; only
        # advanced.py used to honour them, everything else silently ignored them.
        raw = self.inputs.abacus_config.get_dict()
        before = len(grid)
        grid = cap_grid(grid, raw.get("max_l_max"), raw.get("max_r_cut"))
        if grid and len(grid) != before:
            self.report(
                f"  caps: max_l_max={raw.get('max_l_max')}, "
                f"max_r_cut={raw.get('max_r_cut')} -> {before} -> {len(grid)} points"
            )

        self.ctx.grid = grid
        self.report(
            f"Generated {len(grid)} grid points: "
            f"{[(entry.l_max, entry.r_cut) for entry in grid]}"
        )
        if not grid:
            return self.exit_codes.ERROR_EMPTY_GRID
        return None

    # ------------------------------------------------------------------
    # Iterative strategy: try in order, stop as soon as one is acceptable
    # ------------------------------------------------------------------
    def is_iterative(self):
        return self.ctx.search_strategy == "iterative"

    def has_pending_iterative(self):
        """Keep looping while grid points remain untried."""
        return has_pending_iterative(
            grid=self.ctx.grid,
            n_done=len(getattr(self.ctx, "grid_results", [])),
            best=getattr(self.ctx, "best", None),
            dry_run=self._dry_run,
            stop_on_first_valid=bool(
                self.inputs.get("stop_on_first_valid", orm.Bool(True)).value
            ),
        )

    def launch_search_step(self):
        """Start the search according to the chosen strategy."""
        self.ctx.grid_results = []
        self.ctx.best = None

        # dry-run mode: print only, submit nothing (a real early return, see _dry_run)
        if self._dry_run:
            for entry in self.ctx.grid:
                self.report(
                    f"  [DRY-RUN] grid #{entry.index + 1}/{len(self.ctx.grid)}: "
                    f"{entry.label}"
                )
            self.report(
                f"[DRY-RUN] {len(self.ctx.grid)} candidates would be submitted "
                f"(strategy={self.ctx.search_strategy})"
            )
            # Placeholder child WC lists so that the while loop does not iterate
            self.ctx.calc = []
            self.ctx.calcs = []
            return None

        from aiida.plugins import WorkflowFactory
        calc_wc_cls = WorkflowFactory("orbgen.calc")

        if self.ctx.search_strategy == "iterative":
            # iterative: submit the first one, the rest is handled by iterate_step
            entry = self.ctx.grid[0]
            self.report(
                f"[iterative] try #{entry.index + 1}/{len(self.ctx.grid)}: "
                f"{entry.label}"
            )
            running = self.submit(calc_wc_cls, **self._build_calc_inputs(entry))
            running.base.extras.set_many({
                "l_max": str(entry.l_max),
                "r_cut": str(entry.r_cut),
                "grid_index": str(entry.index),
            })
            return self.to_context(calc=append_(running))

        # exhaustive: submit them all in parallel
        running_list = []
        for entry in self.ctx.grid:
            self.report(
                f"[exhaustive] submit #{entry.index + 1}/{len(self.ctx.grid)}: "
                f"{entry.label}"
            )
            running = self.submit(calc_wc_cls, **self._build_calc_inputs(entry))
            running.base.extras.set_many({
                "l_max": str(entry.l_max),
                "r_cut": str(entry.r_cut),
                "grid_index": str(entry.index),
            })
            running_list.append(running)
            self.ctx.grid_results.append({
                "l_max": entry.l_max,
                "r_cut": entry.r_cut,
                "calc_pk": running.pk,
                "exit_status": None,
                "is_finished_ok": False,
            })
        # append_ takes only one argument: call self.to_context directly inside the
        # loop without returning; AiiDA collects every awaitable into ctx.calcs
        for r in running_list:
            self.to_context(calcs=append_(r))

    def iterate_step(self):
        """Core loop step of the iterative strategy.

        Each call (triggered by while_):
          1. Inspect self.ctx.calc[-1], the child that just finished.
          2. Extract the energies and evaluate the tolerance.
          3. On pass or failure, has_pending_iterative returns False and the loop exits.
          4. Otherwise submit the next one.

        Note: exit_status == 304 (WARNING_TOLERANCE_EXCEEDED) counts as a soft success
        - the energies can still be extracted, tolerance_ok=False, it is not a failure
        and it does not interrupt the loop.
        """
        calc_node = self.ctx.calc[-1]  # the most recently finished child
        entry = self.ctx.grid[len(self.ctx.grid_results)]
        l_max, r_cut = entry.l_max, entry.r_cut

        # 304 = WARNING_TOLERANCE_EXCEEDED: the child finished both the SCF and the
        # energy extraction, only the lcao_nsw accuracy missed the target. For a grid
        # search that is **useful information** rather than an error. Both the
        # classification and the verdict come from workflows/energies.py (the single
        # definition).
        has_energies = hasattr(calc_node.outputs, "energies")
        treat_as_soft_success = is_soft_success(
            calc_node.is_finished_ok, calc_node.exit_status, has_energies
        )
        is_tolerance_exceeded = (
            calc_node.exit_status == SOFT_SUCCESS_EXIT_STATUS and has_energies
        )

        result_entry = {
            "l_max": l_max,
            "r_cut": r_cut,
            "calc_pk": calc_node.pk,
            "exit_status": calc_node.exit_status,
            "is_finished_ok": calc_node.is_finished_ok,
            "tolerance_exceeded_only": is_tolerance_exceeded,
        }

        if treat_as_soft_success and has_energies:
            energies = calc_node.outputs.energies.get_dict()
            verdict = evaluate_energies(energies, self.ctx.tolerance_meV)
            delta_per_atom_meV = verdict["delta_per_atom_meV"]
            tolerance_ok = verdict["tolerance_ok"]
            result_entry["energies"] = energies
            result_entry["delta_per_atom_meV"] = delta_per_atom_meV
            result_entry["tolerance_ok"] = tolerance_ok

            tag = "⚠ TOLERANCE_EXCEEDED" if is_tolerance_exceeded else None
            self.report(
                f"  ΔE/atom = {delta_per_atom_meV:.3f} meV "
                f"(tolerance = {self.ctx.tolerance_meV:.3f} meV) "
                f"-> {'✓ OK' if tolerance_ok else '✗ EXCEEDED'}"
                + (f" [{tag}]" if tag else "")
            )
            if tolerance_ok and self.ctx.best is None:
                self.ctx.best = {
                    "l_max": l_max,
                    "r_cut": r_cut,
                    "calc_pk": calc_node.pk,
                    "delta_per_atom_meV": delta_per_atom_meV,
                    "energies": energies,
                }
                self.report(
                    f"  ★ found acceptable (l_max={l_max}, r_cut={r_cut})"
                )
        else:
            self.report(
                f"  ✗ OrbgenCalcWorkChain<{calc_node.pk}> failed "
                f"(exit_status={calc_node.exit_status})"
            )

        self.ctx.grid_results.append(result_entry)

        # Once best is found or the last point is reached, has_pending_iterative
        # returns False
        if not self.has_pending_iterative():
            return None

        # Submit the next one
        from aiida.plugins import WorkflowFactory
        calc_wc_cls = WorkflowFactory("orbgen.calc")
        # grid_results holds one row per finished child, so len(grid_results) is
        # the index of the next untried point (the same counter has_pending_iterative
        # uses).  This used to be `self.ctx.grid[idx + 1]`, with an undefined `idx`.
        next_index = len(self.ctx.grid_results)
        if next_index >= len(self.ctx.grid):
            return None
        entry = self.ctx.grid[next_index]
        self.report(
            f"[iterative] try #{entry.index + 1}/{len(self.ctx.grid)}: "
            f"{entry.label}"
        )
        running = self.submit(calc_wc_cls, **self._build_calc_inputs(entry))
        running.base.extras.set_many({
            "l_max": str(entry.l_max),
            "r_cut": str(entry.r_cut),
            "grid_index": str(entry.index),
        })
        return self.to_context(calc=append_(running))

    def collect_exhaustive_step(self):
        """Exhaustive strategy: wait for everything to finish, then collect results.

        Note: as in ``iterate_step``, exit_status == 304 counts as a soft success.
        """
        if not hasattr(self.ctx, "calcs") or not self.ctx.calcs:
            return None

        best = None
        for idx, calc_node in enumerate(self.ctx.calcs):
            entry = self.ctx.grid_results[idx]
            entry["exit_status"] = calc_node.exit_status
            entry["is_finished_ok"] = calc_node.is_finished_ok

            has_energies = hasattr(calc_node.outputs, "energies")
            treat_as_soft_success = is_soft_success(
                calc_node.is_finished_ok, calc_node.exit_status, has_energies
            )
            is_tolerance_exceeded = (
                calc_node.exit_status == SOFT_SUCCESS_EXIT_STATUS and has_energies
            )
            entry["tolerance_exceeded_only"] = is_tolerance_exceeded

            if treat_as_soft_success and has_energies:
                energies = calc_node.outputs.energies.get_dict()
                verdict = evaluate_energies(energies, self.ctx.tolerance_meV)
                delta_per_atom_meV = verdict["delta_per_atom_meV"]
                tolerance_ok = verdict["tolerance_ok"]
                entry["energies"] = energies
                entry["delta_per_atom_meV"] = delta_per_atom_meV
                entry["tolerance_ok"] = tolerance_ok

                tag = "⚠ TOL_EXCEEDED" if is_tolerance_exceeded else ""
                self.report(
                    f"  (l_max={entry['l_max']}, r_cut={entry['r_cut']}): "
                    f"ΔE/atom = {delta_per_atom_meV:.3f} meV "
                    f"-> {'✓' if tolerance_ok else '✗'} {tag}".rstrip()
                )
                if tolerance_ok and best is None:
                    # The grid is sorted by ascending l_max, r_cut, so the first match
                    # is the best
                    best = {
                        "l_max": entry["l_max"],
                        "r_cut": entry["r_cut"],
                        "calc_pk": calc_node.pk,
                        "delta_per_atom_meV": delta_per_atom_meV,
                        "energies": energies,
                    }
            else:
                self.report(
                    f"  (l_max={entry['l_max']}, r_cut={entry['r_cut']}): "
                    f"FAILED (exit_status={calc_node.exit_status})"
                )

        self.ctx.best = best
        return None

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------
    def finalize(self):
        if self._dry_run:
            return None

        # Gather the statistics
        n_tried = len(self.ctx.grid_results)
        n_passed = sum(
            1 for e in self.ctx.grid_results
            if e.get("tolerance_ok", False)
        )
        # A real failure: it crashed, i.e. it is neither finish_ok nor the 304 soft
        # tolerance success
        n_failed = sum(
            1 for e in self.ctx.grid_results
            if e.get("is_finished_ok", False) is False
            and not e.get("tolerance_exceeded_only", False)
        )

        try:
            # all_results: created through a calcfunction
            all_results_node = create_grid_all_results(
                grid_results_list=self.ctx.grid_results,
                tolerance_meV=self.ctx.tolerance_meV,
                search_strategy=self.ctx.search_strategy,
            )
            self.out("all_results", all_results_node)
        except Exception as exc:
            import traceback
            self.report(f"ERROR in finalize (all_results): {exc}")
            self.report(traceback.format_exc())
            return self.exit_codes.WARNING_PARTIAL_FAILURE

        # best_result: created when a best candidate exists
        if self.ctx.best is not None:
            best_data = dict(self.ctx.best)
            best_data["tolerance_meV"] = self.ctx.tolerance_meV
            best_data["search_strategy"] = self.ctx.search_strategy
            best_node = Dict(dict=best_data).store()
            self.out("best_result", best_node)
            self.report(
                f"★ Best: l_max={self.ctx.best['l_max']}, "
                f"r_cut={self.ctx.best['r_cut']}, "
                f"ΔE/atom = {self.ctx.best['delta_per_atom_meV']:.3f} meV"
            )

        # grid_summary: created through a calcfunction
        best_l_max = self.ctx.best["l_max"] if self.ctx.best else None
        best_r_cut = self.ctx.best["r_cut"] if self.ctx.best else None
        best_dpa = (
            self.ctx.best["delta_per_atom_meV"] if self.ctx.best else None
        )
        best_pk = self.ctx.best["calc_pk"] if self.ctx.best else None
        summary_node = create_grid_summary(
            n_grid_points=len(self.ctx.grid),
            n_tried=n_tried,
            n_passed=n_passed,
            n_failed=n_failed,
            best_l_max=best_l_max,
            best_r_cut=best_r_cut,
            best_delta_per_atom_meV=best_dpa,
            best_calc_pk=best_pk,
            tolerance_meV=self.ctx.tolerance_meV,
            search_strategy=self.ctx.search_strategy,
        )
        self.out("grid_summary", summary_node)

        self.report(
            f"Grid search done: {n_tried} tried, {n_passed} passed, "
            f"{n_failed} failed"
        )

        if self.ctx.best is None:
            if n_failed == n_tried:
                return self.exit_codes.WARNING_PARTIAL_FAILURE
            return self.exit_codes.ERROR_NO_ACCEPTABLE_ORBITALS
        return ExitCode(0)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _build_calc_inputs(self, entry: GridEntry) -> dict:
        """Build the inputs for ``OrbgenCalcWorkChain``.

        Parameters
        ----------
        entry : GridEntry
            The current candidate (l_max / r_cut / matching SIAB config node).
        """
        # Give each combination its own subdirectory so that SIAB outputs cannot
        # overwrite each other
        output_dir_root = Path(self.inputs.output_dir.value)
        combo_dir = output_dir_root / work_dir_name(entry.l_max, entry.r_cut)
        combo_dir.mkdir(parents=True, exist_ok=True)

        siab_json = entry.siab_json if entry.siab_json is not None \
            else self.inputs.siab_json

        child_inputs = {
            "siab_json": siab_json,
            "abacus_config": self.inputs.abacus_config,
            "l_max": orm.Int(entry.l_max),
            "r_cut": orm.Float(entry.r_cut),
            "output_dir": orm.Str(str(combo_dir)),
        }
        # Forward the optional inputs
        for key in ("code_label", "family_label", "max_iterations",
                    "build_family", "only"):
            if key in self.inputs:
                child_inputs[key] = self.inputs[key]
        return child_inputs

