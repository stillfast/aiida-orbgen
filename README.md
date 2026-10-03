# aiida-orbgen

AiiDA plugin for ABACUS orbital generation using CSW-NAO methodology.

## Overview

This plugin integrates the ABACUS-CSW-NAO (Contracted Spherical Wave Numerical Atomic Orbital) workflow into AiiDA, enabling automated orbital generation with full provenance tracking.

## Features

- **OrbgenWorkChain**: Complete workflow combining NSW generation and orbital optimization
- **CswCalcJob**: Direct wrapper for CSW-NAO calculations
- **CswParser**: Parser for orbital generation outputs
- **Full AiiDA Integration**: Works with AiiDA's provenance system

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/aiida-orbgen.git
cd aiida-orbgen

# Install in development mode
pip install -e ".[dev]"

# Register the plugin with AiiDA
reentry scan -r aiida
```

## Usage

### Command line: `run` → `report` (parameter presets)

The CLI is driven by an `input.json` plus the preset tree in
`src/aiida_orbgen/parameters/` (the same idea as `aiida-uranium-workflow`):

```
parameters/
├── metadata.yml          # scheduler-options presets       (static.metadata)
├── abacus/
│   ├── abacus.yml        # default preset file of the "abacus" slot
│   └── test.yml          # parameters.abacus = {"test": "test"}
└── orbgen/
    ├── orbgen.yml        # default preset file of the "orbgen" slot (SIAB config)
    └── test.yml          # parameters.orbgen = {"test": "test"}
```

`examples/input.json`:

```json
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
```

Each slot accepts `"preset"`, `["preset", …]` (same file) or
`{"file": "preset"}` (a category file such as `parameters/abacus/test.yml`).
`static.pseudo_path` is injected into the SIAB config as `pseudo_dir`, so one
orbgen preset works with any pseudopotential; `static.metadata` picks the
scheduler options; `static.output_dir` (or `--output-root`) sets the SIAB run
root (default `<input_dir>/run`).

```bash
# 1. validate + see the plan (offline, nothing submitted)
aiida-orbgen check -i examples/input.json

# 2. submit; writes <input_dir>/output.json
aiida-orbgen run -i examples/input.json

# 3. report.md + orbital files in the current directory
aiida-orbgen report -i examples/output.json -o ./
```

`run` picks the workflow automatically: **one** `(l_max, r_cut)` candidate
⇒ `OrbgenCalcWorkChain`; **several** (here `bessel_nao_rcut: [9, 10]`)
⇒ `OrbgenGridSearchWorkChain`. Force it with `"workflow": "orbgen.calc"` /
`"orbgen.gridsearch"` in `input.json`, or with `--workflow` on the CLI.
`output.json` records the WorkChain UUIDs:

```json
{
    "workflow": "orbgen.gridsearch",
    "input_json": "/abs/path/input.json",
    "abacus": {"orbgen": {"test": "0798b3e9-…"}}
}
```

`report` writes, into `-o DIR`:

```
DIR/
├── report.md                     # status, ΔE matrices, per-dimer energies, basis inventory, logs
├── primitive/                    # one primitive NSW .orb per grid point of the scan
│   ├── U_gga_9au_100Ry_27s27p26d26f25g.orb
│   └── U_gga_10au_100Ry_30s30p29d29f28g.orb
└── lmax4_rcut10/                 # the selected grid point's final CSW-NAO orbital
    ├── U_gga_10au_100Ry_3s2p2d1f.orb / .param / .png
    ├── U_gga_10au_100Ry_4s3p2d2f1g.orb / .param / .png
    ├── orbgen_lmax4_rcut10.json   # exact SIAB config used
    └── orbgen_lmax4_rcut10.log    # full SIAB log (DFT skips + optimiser output)
```

| output | meaning |
| --- | --- |
| `report.md` | the report itself |
| `primitive/*.orb` | the primitive NSW orbital each grid point was evaluated with, taken from the `AtomicOrbitalData` of the pseudo family (falls back to `siab_info['orb_path']`) |
| `lmax<L>_rcut<R>/*.orb` | the **final CSW-NAO orbital(s)**: `report` runs SIAB's spillage minimisation for one grid point; each further point (`--calc-pk`) gets its own directory. Products newer than the reference DFT are reused (`already-present`) — `--redo-final-orbital` forces a recompute |

With several nodes in one `output.json`, every node gets its own subdirectory
(`DIR/<short-uuid>/…`) so the `primitive/` and `lmax*_rcut*` names never collide.

`report` finishes **every grid point of the run**, assembling whatever reference
data is missing and computing a missing monomer on the way — so one command is
usually all you need:

```bash
aiida-orbgen report -i output.json -o ./
```

What it does per grid point, automatically:

1. **Finds the reference DFT tree** — `--dft-root` → `static.dft_roots[<point>]`
   → `static.dft_root` → the point's own SIAB `output_dir`. A candidate that does
   not cover the point is also searched one level down, so a single
   `static.dft_root` pointing at a *directory of per-point runs* serves all of
   them:

   ```json
   "static": {
       "pseudo_path": "…/U.pbe-n-nc.14ve.UPF",
       "metadata": "yeesuan",
       "dft_root": "/abs/path/u_14ve"          /* holds run_lmax4_rcut9/, run_lmax4_rcut10/ */
       /* or explicitly: "dft_roots": {"lmax4_rcut9": "…", "lmax4_rcut10": "…"} */
   }
   ```

2. **Assembles missing dimer data from AiiDA** (`--no-assemble-dft` to avoid
   touching the cluster): downloads `OUT.<suffix>/{data-0-H/S/T, istate.info,
   running_scf.log, kpoints, INPUT, WFC_NAO_GAMMA1.txt}` and rebuilds the LCAO
   wavefunctions from `H C = S C eps` when ABACUS did not write them (validated
   against `istate.info`, typically ~1e-5 eV). No DFT is submitted.
3. **Computes a missing monomer** — the one-atom atomic initial guess SIAB needs
   for `spill_guess: atomic`. Anything bigger is *not* run implicitly; pass
   `--force-final-orbital` for that. Keep `device: cpu` in the orbgen preset:
   the conda `abacus` is a CUDA build and `mpirun -np N` over a shared GPU dies
   with `cudaErrorMemoryAllocation`.
4. **Runs the spillage** and writes `lmax<L>_rcut<R>/`. Products newer than the
   reference DFT are reused (`already-present`) instead of recomputed —
   `--redo-final-orbital` forces a recompute.

The `parameters/` tree is **user input** — the plugin only reads it and never
rewrites a preset, so `ecutjy`, `bessel_nao_rcut`, `device`, … are always the
values you wrote; editing one simply takes effect on the next `run`/`report`.

Reference data is only accepted when it was computed with the **same primitive
basis** as the current config: the LCAO dimension recorded in the data
(`OUT.*/data-0-H` = `Σ_l nzeta_l·(2l+1)` per atom, with `nzeta` from
`bessel_nao_rcut`/`ecutjy`/`lmaxmax`) is compared with what the config implies.
A mismatch (e.g. `ecutjy` changed between runs) is reported as *stale* instead
of being silently mixed — that mismatch otherwise surfaces deep inside SIAB as
`ValueError: len(coef[itype][l][zeta]) should not exceed nbes[itype][l]`.
Stale geometries are refetched from AiiDA, and stale ones SIAB would otherwise
"skip" because a finished log exists (the monomer) are moved to
`<folder>.stale/` first so they get recomputed cleanly.

Every step is recorded in `report.md` (section 6 has one subsection per grid
point, with status, reference tree, config source and produced files). Points
that cannot be finished are listed with the reason and do not stop the others.

Escape hatches: `--calc-pk PK` (only that grid point), `--no-final-orbital`,
`--no-orbitals`, `--with-upf`, `--no-assemble-dft`, `--force-final-orbital`,
`--redo-final-orbital`, `--siab-json`, `--input-config`, `--use-stored-config`,
`--orbgen-command`, `--final-orbital-timeout`, `--dry-run`.

`fetch-dft` does step 1–2 alone (useful to prepare a tree without running
anything):

```bash
aiida-orbgen fetch-dft -i output.json                 # every point
aiida-orbgen fetch-dft -i output.json --calc-pk 412925 --dft-root ../run_lmax4_rcut9
```

> Runs submitted **after 2026-09-20** bring real wavefunctions: `out_wfc_lcao` is
> no longer stripped from the INPUT by `AIIDA_MANAGED_KEYS` (it was, and every
> AiiDA LCAO job silently wrote no `WFC_NAO_GAMMA1.txt`; PW jobs still drop the
> key in `build_abacus_child_inputs`). That change lives in the **daemon-side**
> `workflows/`+`static/` code, so it needs `verdi daemon restart` once.

> Preset detail: SIAB reads `vloc_aux` / `lloc_min` **only** from an orbital's
> `model_kwargs` block (`SIAB/orb/orb_jy.py`,
> `orbital_model_required_keys['atomic']`). Written flat they are silently
> ignored, so `validate_siab_config()` rejects the flat spelling.

### Choosing the parameters: the four WorkChains

Before any orbital is produced, four parameters have to be chosen: the plane-wave
reference cutoff (`ecutwfc`) and the three parameters of the primitive NSW basis
(`r_cut`, `l_max`, `ecutjy`). Two WorkChains answer those questions, and both write
their whole comparison table into the provenance, so the choice can be re-made later
without rerunning anything.

| entry point | question | criterion |
| --- | --- | --- |
| `orbgen.ecutwfc` | which `ecutwfc` is the PW reference converged at? | the total energy of the reference geometries stops moving between neighbouring cutoffs (`tolerance_meV` per atom) |
| `orbgen.basis` | which `(r_cut, l_max, ecutjy)` is the cheapest basis that is accurate enough? | `max |E_nsw - E_pw|` per atom over the same geometries, optionally plus the atomization-energy drift against the reference candidate |

```python
from aiida import load_profile, orm
from aiida.engine import submit
from aiida.plugins import WorkflowFactory

load_profile()

# 1) converge the PW reference (one ladder, one SIAB tree)
b = WorkflowFactory("orbgen.ecutwfc").get_builder()
b.siab_json = orm.SinglefileData(file="orbgen.json")   # geometries + pseudopotential
b.abacus_config = orm.Dict(dict=abacus_config)
b.l_max, b.r_cut = orm.Int(4), orm.Float(11.0)         # the basis to reference
b.ecutwfc_values = orm.List(list=[100.0, 120.0, 150.0, 180.0, 200.0])
b.output_dir = orm.Str("/scratch/ecutwfc")
ecutwfc_node = submit(b)          # -> outputs.ecutwfc_decision.chosen

# 2) walk the basis ladder against that reference
b = WorkflowFactory("orbgen.basis").get_builder()
b.siab_json = orm.SinglefileData(file="orbgen.json")
b.abacus_config = orm.Dict(dict=abacus_config)
b.l_max, b.r_cut = orm.Int(4), orm.Float(12.0)         # reference point (the largest)
b.reference_ecutjy = orm.Float(150.0)
b.ecutjy_values = orm.List(list=[125.0, 100.0])        # one super-parameter at a time
b.l_max_values = orm.List(list=[3])
b.r_cut_values = orm.List(list=[11.0, 10.0])
b.ecutwfc = orm.Float(180.0)                           # from step 1
b.atomization_tolerance_meV = orm.Float(50.0)          # the stricter criterion
b.output_dir = orm.Str("/scratch/basis")
basis_node = submit(b)            # -> outputs.chosen_basis, outputs.primitive_orbital
```

What the two scans are careful about (all of it is in the reports):

* **the PW reference is paid for once** -- `orbgen.basis` computes it on the reference
  tree and reuses it for every candidate, or takes one from an earlier run through the
  `pw_reference` input. A grid search over `orbgen.calc` would run the same PW children
  once per candidate, and PW is the expensive side;
* **the ladder, not the Cartesian product** -- `strategy="ladder"` (default) reduces one
  super-parameter at a time (`ecutjy` first, then `l_max`, then `r_cut`), so every
  candidate is cheaper than the point it came from; `strategy="exhaustive"` takes the
  product;
* **stop at the first passing candidate** (`stop_on_first_pass`, default true): the
  ladder is cost-ordered, so the first candidate inside the tolerance *is* the cheapest
  accurate basis -- unless the `atomization_tolerance_meV` gate rejects it, in which case
  the scan continues and reports the rejection;
* **cost is measured, not guessed** -- every row carries the wall-clock seconds of its
  children and the `nchi` proxy, and `by="seconds" | "cost" | "nchi"` chooses what
  "cheapest" means;
* **`r_cut` is checked against the cell** -- an `r_cut` above half the smallest cell edge
  is refused (exit 410) instead of silently producing two-centre tables for neighbours
  that cannot exist;
* `abacus.json`'s `basis` list is ignored by both scans (children are `pw` for the
  cutoff ladder, `lcao_nsw` against the reference for the basis ladder); the
  `parameters/abacus/pw.yml` and `lcao_only.yml` presets exist for exactly these two
  modes.

Exit codes worth knowing: `409` no candidate inside the tolerance (widen the ladder),
`407` the `ecutwfc` ladder did not converge, `304` the LCAO check of `orbgen.ecutwfc`
missed the tolerance, `301` some children failed.

### Python API

```python
from aiida import load_profile
from aiida.orm import Dict
from aiida_orbgen/workflows.orbgen import OrbgenWorkChain

load_profile()

# Define input parameters (see examples/mock_data/)
inputs = {
    "json_config": Dict(dict={
        "element": "Si",
        "pseudo_dir": "./Si.upf",
        "bessel_nao_rcut": [7],
        # ... other parameters
    })
}

# Submit the workflow
result = submit(OrbgenWorkChain, **inputs)
```

### Command Line

See [Usage → `run` → `report`](#command-line-run--report-parameter-presets) above.
The full sub-command list is:

```bash
aiida-orbgen run         -i input.json [--dry-run] [--workflow W] [-o output.json]
aiida-orbgen report      -i output.json -o DIR [--no-final-orbital] [--dft-root DIR]
aiida-orbgen check       -i input.json
aiida-orbgen fetch-dft   --dft-root DIR -o DIR
```

> The legacy `submit-siab {calc,gridsearch}` sub-command was removed on
> 2026-09-29: it assembled its own workchain inputs, so it never applied
> `apply_input_overrides` (AiiDA-managed keys leaked into the ABACUS INPUT and
> `out_wfc_lcao` was missing, i.e. LCAO jobs silently produced no `WFC_NAO_*`)
> and it bypassed every `validate_*` check. Use `run`, which resolves the same
> `orbgen.json` + `abacus.json` through `parameters/` presets.

## Project Structure

```
aiida_orbgen/
├── src/aiida_orbgen/    # Source code
│   ├── calculations/     # CalcJob implementations
│   ├── parsers/        # Output parsers
│   ├── workflows/       # WorkChain implementations
│   ├── data/           # Custom data nodes
│   └── cli/            # CLI tools
├── examples/            # Usage examples
├── tests/              # Unit tests
└── docs/               # Documentation
```

## Requirements

- Python >= 3.9
- AiiDA >= 2.0
- ABACUS >= 3.7.5
- ABACUS-CSW-NAO (SIAB)

## License

MIT License - see [LICENSE](LICENSE) for details.

## References

- ABACUS-CSW-NAO: https://github.com/kirk0830/ABACUS-CSW-NAO
- AiiDA: https://www.aiida.net
