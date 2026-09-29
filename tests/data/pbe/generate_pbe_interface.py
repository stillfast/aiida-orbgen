"""
generate_pbe_interface.py

Use the aiida-orbgen interfaces to generate the following in the project/pbe_interface/ directory:
- primitive_jy/U_gga_9au_100Ry_27s27p26d26f25g.orb  (NSW raw orbital)
- U-dimer-1.89-9au/INPUT                          (ABACUS input)
- U-dimer-1.89-9au/STRU                           (ABACUS structure)

They can then be compared against the actual files in project/pbe/ with tools such as diff.

How to run:
    cd /home/liguozhou/abacus/calculations/orbgen
    python aiida-orbgen/tests/data/pbe/generate_pbe_interface.py
"""

import json
import os
import sys
from pathlib import Path

# Add aiida-orbgen/src to path
ROOT = Path("/home/liguozhou/abacus/calculations/orbgen")
sys.path.insert(0, str(ROOT / "aiida-orbgen" / "src"))
sys.path.insert(0, str(ROOT / "aiida-orbgen"))

# Add the SIAB (ABACUS-CSW-NAO) library to path
SIAB_PATH = Path("/home/liguozhou/install/ABACUS-CSW-NAO")
if SIAB_PATH.exists() and str(SIAB_PATH) not in sys.path:
    sys.path.insert(0, str(SIAB_PATH))

from aiida_orbgen.interfaces.nsw import generate_nsw
from aiida_orbgen.interfaces.incar import generate_incar
from aiida_orbgen.interfaces.stru import generate_stru


def main():
    """Main function: generate the NSW, INPUT and STRU files in the project/pbe_interface/ directory."""

    # Configure the paths
    config_path = ROOT / "aiida-orbgen" / "tests" / "data" / "pbe" / "pbe_orbgen.json"
    output_root = ROOT / "project" / "pbe_interface"
    job_folder = output_root / "U-dimer-1.89-9au"
    primitive_jy_dir = output_root / "primitive_jy"

    # Create the output directories
    primitive_jy_dir.mkdir(parents=True, exist_ok=True)
    job_folder.mkdir(parents=True, exist_ok=True)

    # Read the JSON configuration
    with open(config_path) as f:
        config = json.load(f)

    print("=" * 70)
    print("aiida-orbgen interface file generation test")
    print("=" * 70)
    print(f"Configuration file: {config_path}")
    print(f"Output directory: {output_root}")
    print()

    # 1. Generate the raw NSW orbital
    print("[1/3] Generating the raw NSW orbital...")
    nsw_path = generate_nsw(
        config,
        output_dir=str(primitive_jy_dir),
        lmaxmax=config["geoms"][0]["lmaxmax"],
    )
    print(f"      NSW file: {nsw_path}")
    nsw_filename = os.path.basename(nsw_path)
    print()

    # 2. Generate the INPUT file
    print("[2/3] Generating the INPUT file...")
    input_path = generate_incar(
        config,
        output_path=str(job_folder / "INPUT"),
    )
    print(f"      INPUT file: {input_path}")
    print()

    # 3. Generate the STRU file
    print("[3/3] Generating the STRU file...")
    stru_path = generate_stru(
        config,
        output_path=str(job_folder / "STRU"),
        bond_length=1.89,
        orb_filename=nsw_filename,
    )
    print(f"      STRU file: {stru_path}")
    print()

    # Print the comparison summary
    print("=" * 70)
    print("Generation complete; the following directories can be compared:")
    print("=" * 70)
    print(f"  Reference: {ROOT / 'project' / 'pbe'}")
    print(f"  Generated: {output_root}")
    print()
    print("Comparison commands:")
    print(f"  diff -r {ROOT / 'project' / 'pbe' / 'U-dimer-1.89-9au' / 'INPUT'} {job_folder / 'INPUT'}")
    print(f"  diff -r {ROOT / 'project' / 'pbe' / 'U-dimer-1.89-9au' / 'STRU'} {job_folder / 'STRU'}")
    print(f"  diff {ROOT / 'project' / 'pbe' / 'primitive_jy' / nsw_filename} {nsw_path}")
    print()
    print("Or inspect the generated files directly:")
    print(f"  ls -la {output_root}")
    print(f"  ls -la {job_folder}")
    print(f"  ls -la {primitive_jy_dir}")


if __name__ == "__main__":
    main()
