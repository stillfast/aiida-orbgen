"""
SIAB (ABACUS-CSW-NAO) Python API reference

This document describes how to call the various functional modules of the SIAB library through the Python API.
It covers two core workflows:
1. JSON config -> NSW generation -> INPUT file generation
2. DFT results + NSW -> Orbgen optimization -> orbital files

Library path: /home/liguozhou/install/ABACUS-CSW-NAO
"""

# =============================================================================
# Part 1: JSON config parsing and parameter validation
# =============================================================================

"""
Module: SIAB.io.param

Core features:
- Read the JSON input configuration file
- Validate and group parameters
- Associate geometries with orbital parameters
"""

from SIAB.io.param import read, ParamAssert, orb_link_geom, group

# Usage example:
"""
# Read the JSON config file
glbparams, dftparams, spillparams, compute, iop = read('input.json')

# Return values:
# - glbparams: global parameters, including element, bessel_nao_rcut
# - dftparams: DFT calculation parameters (ABACUS INPUT parameters)
# - spillparams: Spillage optimization parameters (geoms, orbitals, primitive_type, etc.)
# - compute: computing environment parameters (abacus_command, mpi_command, etc.)
# - iop: internal optimization parameters (prefixed with __iop)
"""

# =============================================================================
# Part 2: NSW generation and ABACUS job construction
# =============================================================================

"""
Module: SIAB.abacus.api

Core features:
- Build ABACUS DFT calculation jobs
- Generate the INPUT and STRU input files
- Define atomic structures
"""

from SIAB.abacus.api import (
    build_abacus_jobs,
    job_done,
    _build_case,
    _build_atomspecies,
    _cal_nzeta,
)

from SIAB.abacus.io import (
    dftparam_to_text,
    structure_to_text,
    autoset,
    KPOINTS,
    ABACUS_PARAMS,
)

from SIAB.io.convention import dft_folder

# Usage example:
"""
# 1. Build atomic species information
atomspecies = _build_atomspecies(
    elem='Si',
    pp='Si.upf',
    ecut=40,
    rcut=7,
    lmaxmax=2,
    primitive_type='reduced'
)

# 2. Build a single DFT calculation case
folder = _build_case(
    proto='dimer',
    pertkind='stretch',
    pertmag=2.0,
    atomspecies={'Si': {'pp': 'Si.upf', 'orb': 'Si.orb'}},
    dftshared={'ecutwfc': 60, 'basis_type': 'lcao'},
    dftspecific={'gamma_only': '1'}
)

# 3. Build ABACUS jobs in batch
jobs = build_abacus_jobs(
    atomspecies=[{'elem': 'Si', 'ecutjy': 40, 'zval': 0}],
    rcuts=[7],
    dftparams={'ecutwfc': 60, 'basis_type': 'lcao'},
    geoms=[{'proto': 'dimer', 'pertkind': 'stretch', 'pertmags': [1.5, 2.0, 2.5]}],
    spill_guess='atomic'
)

# 4. Compute nzeta (number of zeta functions per angular momentum)
nzeta, ecut, lmaxmax = _cal_nzeta(
    rcut=7,
    ecut=40,
    lmaxmax=2,
    less_dof=0
)

# 5. Generate the INPUT file content
input_text = dftparam_to_text(dftparam)

# 6. Generate the STRU file content
stru_text, natom = structure_to_text(
    shape='dimer',
    element='Si',
    mass=1,
    fpseudo='Si.upf',
    lattice_constant=30,
    bond_length=2.0,
    nspin=1,
    forb=None
)

# 7. Auto-set DFT parameters
dftparam = autoset({
    'ecutwfc': 60,
    'basis_type': 'lcao',
    'nspin': 1
})
"""

# =============================================================================
# Part 3: Orbital optimization and Orbgen
# =============================================================================

"""
Module: SIAB.orb.api

Core features:
- Create an orbital cascade optimization instance
- Manage orbital initialization and the optimization workflow
"""

from SIAB.orb.api import (
    GetOrbCascadeInstance,
    DeriveCascadeInstance,
)

from SIAB.orb.cascade import OrbgenCascade

# Usage example:
"""
# Create an orbital cascade instance
cascade = GetOrbCascadeInstance(
    elem='Si',
    rcut=7,
    ecut=40,
    primitive_type='reduced',
    initializer={'model': 'atomic', 'model_kwargs': {'jobdir': 'Si-monomer-0-7au'}},
    orbparam=[
        {
            'nzeta': [1, 1, 0],
            'folders': ['Si-dimer-1.75-7au', 'Si-dimer-2.0-7au'],
            'nbnds': [20, 20],
            'iorb_frozen': None
        },
        {
            'nzeta': [1, 1, 1],
            'folders': ['Si-dimer-1.75-7au'],
            'nbnds': [20],
            'iorb_frozen': 0,  # freeze the coefficients of the first orbital
        }
    ],
    mode='jy',  # or 'pw'
    optimizer='torch.swats'  # or 'scipy.bfgs'
)

# Run the optimization
cascade, spillage_values = cascade.opt(
    diagnosis=True,
    options={'maxiter': 100},
    nthreads=4
)

# Export the orbital files
cascade.to_file(outdir='./output')

# Derive a new instance from an existing cascade (adding more orbitals)
new_cascade = DeriveCascadeInstance(
    elem='Si',
    rcut=7,
    ecut=40,
    primitive_type='reduced',
    cascade=old_cascade,
    orbparam=[{
        'nzeta': [1, 1, 1, 1],
        'folders': ['Si-dimer-1.75-7au'],
        'nbnds': [20],
        'iorb_frozen': 1,
    }]
)
"""

# =============================================================================
# Part 4: Spillage optimization core
# =============================================================================

"""
Module: SIAB.spillage.spillage

Core features:
- Extract reference data from DFT results
- Generate the initial orbital guess
- Compute the spillage value
"""

from SIAB.spillage.spillage import (
    initgen_jy,
    initgen_pw,
    _jy_data_extract,
)

from SIAB.spillage.spilltorch import SpillTorch_jy, SpillTorch_pw

# Usage example:
"""
# 1. Extract data from JY-mode DFT output
data = _jy_data_extract('OUT.ABACUS')
# Returns: {natom, nzeta, wk, S, T, C}

# 2. Generate initial orbital coefficients from the atomic calculation
coef = initgen_jy(
    outdir='Si-monomer-0-7au/OUT.ABACUS',
    nzeta=[1, 1, 0],
    ibands='all',
    nbes_gen=None,
    diagnosis=True
)
# Returns: coefficients in coef[l][zeta][q] form

# 3. Generate initial orbital coefficients from the PW-mode calculation
coef = initgen_pw(
    orb_mat='orb_matrix.0.dat',
    nzeta=[1, 1, 0],
    ibands='all'
)

# 4. Run spillage optimization with the Torch optimizer
minimizer = SpillTorch_jy()
minimizer.config_add('OUT.ABACUS', weight=(0, 1.0))
coefs_opt, spillage = minimizer.opt(
    coef_init=[initial_coef],
    coef_frozen=None,
    bounds=None,
    iconfs=[0],
    ibands=range(20),
    options={'lr': 0.001, 'maxiter': 100},
    nthreads=4
)
"""

# =============================================================================
# Part 5: Complete workflow interface
# =============================================================================

"""
Module: SIAB.driver.main

Core features:
- Complete ABACUS-ORBGEN workflow
- One-stop solution from the JSON config to the orbital files
"""

from SIAB.driver.main import (
    init,
    rundft,
    minimize_spillage,
)

# Usage example:
"""
# 1. Initialize the workflow (read parameters)
glbparams, dftparams, spillparams, compute, iop = init('input.json')

# 2. Run the DFT calculation
jobs = rundft(
    atomspecies=[{'elem': 'Si', 'ecutjy': 40, 'zval': 0}],
    rcuts=[7],
    dftparam={'ecutwfc': 60, 'basis_type': 'lcao'},
    geoms=spillparams['geoms'],
    spillguess='atomic',
    compparam={'abacus_command': 'mpirun -np 8 abacus'}
)

# 3. Run the spillage optimization
minimize_spillage(
    elem='Si',
    ecut=40,
    rcuts=[7],
    primitive_type='reduced',
    scheme=spillparams['orbitals'],
    dft_root='.',
    run_mode='jy',
    outdir='./output',
    max_steps=9000,
    spill_guess='atomic'
)
"""

# =============================================================================
# Part 6: Utility functions
# =============================================================================

"""
Module: SIAB.io.convention

Core features:
- File and folder naming conventions
- Orbital parameter string generation
"""

from SIAB.io.convention import (
    dft_folder,
    orb_folder,
    orb as orb_filename,
    nzeta_string,
)

# Usage example:
"""
# 1. Generate a DFT calculation folder name
folder = dft_folder('Si', 'dimer', 2.0, 7)
# Returns: 'Si-dimer-2.00-7au'

# 2. Generate an orbital folder name
folder = orb_folder('Si', [1, 1, 0])
# Returns: 'Si_s1p1'

# 3. Generate an orbital file name
filename = orb_filename('Si', 7, 40, [1, 1, 0])
# Returns: 'Si_gga_7au_40Ry_1s1p.orb'

# 4. Generate the nzeta string
nz_str = nzeta_string([1, 1, 1, 0])
# Returns: '1s1p1d'
"""

"""
Module: SIAB.spillage.radial

Core features:
- Spherical Bessel function related calculations
- Radial integrals
"""

from SIAB.spillage.radial import (
    _nbes,  # number of Bessel functions for the given l, rcut, ecut
    jl_reduce,  # Bessel function reduction
)

# Usage example:
"""
# Compute the number of Bessel functions for l=0, rcut=7, ecut=40
n = _nbes(0, 7, 40)
# Returns: the number of Bessel functions

# Reduce the Bessel functions
reduced = jl_reduce(l=0, nbes=21, rcut=7)
"""

# =============================================================================
# Complete API usage example
# =============================================================================

"""
Below is a complete Python script showing how to use the API:

```python
import os
import numpy as np
from SIAB.io.param import read
from SIAB.abacus.api import build_abacus_jobs, _build_atomspecies
from SIAB.abacus.io import autoset, dftparam_to_text, structure_to_text
from SIAB.io.convention import dft_folder
from SIAB.orb.api import GetOrbCascadeInstance
from SIAB.orb.cascade import OrbgenCascade
from SIAB.spillage.spillage import initgen_jy

# Step 1: read the configuration
glbparams, dftparams, spillparams, compparam, iop = read('config.json')

# Step 2: build and run the DFT jobs
atomspecies = _build_atomspecies(
    elem=glbparams['element'],
    pp=os.path.join(spillparams['pseudo_dir'], f'{glbparams["element"]}.upf'),
    ecut=spillparams.get('ecutjy', dftparams['ecutwfc']),
    rcut=glbparams['bessel_nao_rcut'][0],
    lmaxmax=2,
    primitive_type=spillparams['primitive_type']
)

# Step 3: build the ABACUS jobs
jobs = build_abacus_jobs(
    atomspecies=[{'elem': glbparams['element'], 
                  'ecutjy': spillparams.get('ecutjy', dftparams['ecutwfc']),
                  'zval': 0}],
    rcuts=glbparams['bessel_nao_rcut'],
    dftparams=dftparams,
    geoms=spillparams['geoms'],
    spill_guess=spillparams.get('spill_guess')
)

# Step 4: once the DFT calculations finish, create the orbital optimization cascade
cascade = GetOrbCascadeInstance(
    elem=glbparams['element'],
    rcut=glbparams['bessel_nao_rcut'][0],
    ecut=spillparams.get('ecutjy', dftparams['ecutwfc']),
    primitive_type=spillparams['primitive_type'],
    initializer={'model': 'atomic', 'model_kwargs': 
                {'jobdir': dft_folder(glbparams['element'], 'monomer', 0)}},
    orbparam=spillparams['orbitals'],
    mode=spillparams['fit_basis'],
    optimizer=spillparams.get('optimizer', 'torch.swats')
)

# Step 5: run the optimization
cascade, spillages = cascade.opt(diagnosis=True)

# Step 6: export the orbital files
cascade.to_file(outdir='./orbitals')

print(f"Orbital optimization complete, final spillage: {spillages}")
```
"""

# =============================================================================
# JSON input configuration format
# =============================================================================

"""
Example JSON input file:

{
    // computing environment configuration
    "environment": "",
    "mpi_command": "mpirun -np 8",
    "abacus_command": "abacus",

    // element information
    "element": "Si",
    "pseudo_dir": "Si.upf",

    // basis type: jy (spherical Bessel) or pw (plane wave)
    "fit_basis": "jy",

    // plane-wave kinetic energy cutoff (Ry)
    "ecutwfc": 60,

    // spherical Bessel kinetic energy cutoff (Ry)
    "ecutjy": 40,

    // orbital cutoff radius (au)
    "bessel_nao_rcut": [7, 10],

    // primitive basis type: reduced or normalized
    "primitive_type": "reduced",

    // orbital initial guess method
    "spill_guess": "atomic",

    // optimizer configuration
    "optimizer": "scipy.bfgs",
    "max_steps": 9000,
    "nthreads_rcut": 4,

    // geometry configuration
    "geoms": [
        {
            "proto": "dimer",           // structure prototype: dimer, trimer, monomer, etc.
            "pertkind": "stretch",       // perturbation kind: stretch, shear, twist
            "pertmags": [1.62, 1.82, 2.22, 2.72, 3.22],  // list of bond lengths (Angstrom)
            "nbands": 20,               // number of bands used in the calculation
            "nspin": 1,                 // spin polarization: 1 or 2
            "lmaxmax": 2,               // maximum angular momentum
            "celldm": 30                // lattice constant (Bohr)
        }
    ],

    // orbital generation configuration
    "orbitals": [
        {
            "nzeta": [1, 1, 0],         // number of zeta functions per angular momentum [s, p, d, ...]
            "geoms": [0],               // indices of the referenced geometries
            "nbands": "occ",            // number of bands involved in the optimization, "occ" or an integer
            "checkpoint": null,          // index of the frozen inner orbital
            "greedygrow": false         // whether to enable the greedy growth algorithm
        }
    ]
}
"""
