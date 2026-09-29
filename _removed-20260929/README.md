# Modules removed on 2026-09-29 (batch 1 of the refactor)

Three modules had **zero remaining callers**. Their source files are no longer in
the tree (they were dropped once the repository was under git, so they stay
recoverable with `git show <commit-before-the-drop>:<path>`, and in
`.refactor-backup-*.tar.gz`); this table keeps the *reason*, which is the part
worth reading later.

| file | why it was removed | what replaced it |
|---|---|---|
| `submit_siab.py` | the legacy `aiida-orbgen submit-siab` sub-command: assembled workchain inputs itself, so it never called `apply_input_overrides` (AiiDA-managed keys such as `suffix`/`stru_file`/`pseudo_dir`/`basis_type` leaked into the ABACUS INPUT — see `dry_run_submissions/*/dry_run_inputs.json`) and never added `out_wfc_lcao`, i.e. every LCAO job it submitted silently produced no `WFC_NAO_*`. It also bypassed every `validate_*` check and duplicated `run`. | `cli/run.py` (`run` / `report` / `check` / `fetch-dft`) + `utils/config.py` |
| `advanced.py` | the second ``OrbgenGridSearchWorkChain`` (same ``process_label``, colliding exit codes, unreachable from any entry point). Its three unique capabilities were ported into the live grid search first: ``stop_on_first_valid``, an explicit ``candidates`` list, and the ``max_l_max`` / ``max_r_cut`` caps. Its per-*system* ΔE comparison was deliberately **not** ported (the rest of the stack uses per-atom). | `workflows/batch.py` + `workflows/_grid.py` |
| `orbgen_report.py` | superseded by the `utils/report/` package (`orbgen.py` + `orbitals.py` + `assemble.py` + `validate.py`); nothing imported it any more, and it read a `tolerance_meV` input that no spec defines | `utils/report/` |

Restore with `git show <commit>:<path> > <path>` (or from the backup tarball) if
one of them is ever needed again.
