"""
compare_orb.py: detailed comparison of two .orb files for consistency

Comparison dimensions:
1. MD5 checksum
2. File size
3. Byte-by-byte comparison (cmp)
4. Header information
5. Grid data (positions, chi values)

How to run:
    cd /home/liguozhou/abacus/calculations/orbgen
    python aiida-orbgen/tests/data/pbe/compare_orb.py
"""

import os
import sys
import hashlib
import numpy as np
from pathlib import Path


def md5_of_file(filepath: str) -> str:
    """Compute the MD5 checksum of a file."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def compare_files_bytewise(f1: str, f2: str) -> dict:
    """Compare two files byte by byte."""
    with open(f1, "rb") as fp1, open(f2, "rb") as fp2:
        data1 = fp1.read()
        data2 = fp2.read()

    if data1 == data2:
        return {"identical": True, "diff_bytes": [], "size1": len(data1), "size2": len(data2)}

    # Find the first position where they differ
    size = min(len(data1), len(data2))
    diff_bytes = []
    for i in range(size):
        if data1[i] != data2[i]:
            diff_bytes.append((i, data1[i], data2[i]))
            if len(diff_bytes) > 10:  # record only the first 10 differences
                break

    return {
        "identical": False,
        "diff_bytes": diff_bytes,
        "size1": len(data1),
        "size2": len(data2),
    }


def parse_orb_header(filepath: str) -> dict:
    """Parse the .orb file header."""
    with open(filepath) as f:
        lines = f.readlines()

    header = {}
    for i, line in enumerate(lines[:20]):
        line = line.strip()
        if not line or line.startswith("---"):
            continue
        # Parse the "Key Value" format
        parts = line.split(None, 1)
        if len(parts) == 2:
            key, value = parts
            header[key] = value
        elif len(parts) == 1:
            header[parts[0]] = None

    return header


def parse_orb_grid(filepath: str) -> tuple:
    """Parse the grid data (chi values) of a .orb file."""
    with open(filepath) as f:
        content = f.read()

    # Split the header from the data
    lines = content.split("\n")
    data_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Type") and "L" in line and "N" in line:
            data_start = i + 2  # skip the Type header and the leading 0
            break

    if data_start is None:
        return np.array([]), 0

    # Parse all data
    all_values = []
    for line in lines[data_start:]:
        line = line.strip()
        if not line:
            continue
        tokens = line.split()
        for t in tokens:
            try:
                all_values.append(float(t))
            except ValueError:
                pass

    return np.array(all_values), len(all_values)


def main():
    """Main function: compare two .orb files in detail."""

    f1 = "/home/liguozhou/abacus/calculations/orbgen/project/pbe/primitive_jy/U_gga_9au_100Ry_27s27p26d26f25g.orb"
    f2 = "/home/liguozhou/abacus/calculations/orbgen/project/pbe_interface/primitive_jy/U_gga_9au_100Ry_27s27p26d26f25g.orb"

    print("=" * 70)
    print(" .orb file comparison in detail")
    print("=" * 70)
    print(f"File 1 (reference): {f1}")
    print(f"File 2 (generated): {f2}")
    print()

    # 1. File existence
    print("[1] file existence check")
    print(f"  File 1: {'✓ exists' if os.path.exists(f1) else '✗ missing'}")
    print(f"  File 2: {'✓ exists' if os.path.exists(f2) else '✗ missing'}")
    print()

    if not (os.path.exists(f1) and os.path.exists(f2)):
        print("Error: one or both files do not exist")
        return

    # 2. File size
    print("[2] file size comparison")
    size1 = os.path.getsize(f1)
    size2 = os.path.getsize(f2)
    print(f"  File 1: {size1} bytes")
    print(f"  File 2: {size2} bytes")
    print(f"  Difference: {abs(size1 - size2)} bytes ({abs(size1 - size2) / size1 * 100:.4f}%)")
    print()

    # 3. MD5 checksum
    print("[3] MD5 checksum")
    md5_1 = md5_of_file(f1)
    md5_2 = md5_of_file(f2)
    print(f"  File 1 MD5: {md5_1}")
    print(f"  File 2 MD5: {md5_2}")
    print(f"  MD5 match: {'✓ yes' if md5_1 == md5_2 else '✗ no'}")
    print()

    # 4. Byte-by-byte comparison
    print("[4] byte-by-byte comparison")
    result = compare_files_bytewise(f1, f2)
    if result["identical"]:
        print("  ✓ the two files are identical (byte level)")
    else:
        print(f"  ✗ the two files differ")
        print(f"  Length of file 1: {result['size1']}")
        print(f"  Length of file 2: {result['size2']}")
        print(f"  Number of differing bytes: {len(result['diff_bytes'])}+")
        print(f"  First 10 differing positions:")
        for offset, b1, b2 in result["diff_bytes"][:10]:
            # b1, b2 are ints (0-255)
            print(f"    offset {offset}: 0x{b1:02x} vs 0x{b2:02x} ('{chr(b1) if 32 <= b1 < 127 else '?'}' vs '{chr(b2) if 32 <= b2 < 127 else '?'}')")
    print()

    # 5. Header information comparison
    print("[5] header information comparison")
    h1 = parse_orb_header(f1)
    h2 = parse_orb_header(f2)
    print(f"  {'Key':<30} {'File1':<20} {'File2':<20} {'match':<6}")
    print(f"  {'-'*30} {'-'*20} {'-'*20} {'-'*6}")
    for key in set(list(h1.keys()) + list(h2.keys())):
        v1 = h1.get(key, "(none)")
        v2 = h2.get(key, "(none)")
        match = "✓" if v1 == v2 else "✗"
        print(f"  {key:<30} {str(v1):<20} {str(v2):<20} {match}")
    print()

    # 6. Grid data comparison
    print("[6] grid data (chi values) comparison")
    data1, n1 = parse_orb_grid(f1)
    data2, n2 = parse_orb_grid(f2)
    print(f"  Number of data points in file 1: {n1}")
    print(f"  Number of data points in file 2: {n2}")

    if n1 != n2:
        print(f"  ✗ the numbers of data points differ")
    else:
        # Compute the statistical differences
        diff = data1 - data2
        abs_diff = np.abs(diff)
        max_abs = np.max(abs_diff)
        mean_abs = np.mean(abs_diff)
        max_idx = np.argmax(abs_diff)
        print(f"  Maximum absolute difference: {max_abs:.6e}")
        print(f"  Mean absolute difference: {mean_abs:.6e}")
        print(f"  Position of the maximum difference: index={max_idx}")
        print(f"  At position {max_idx}: file1={data1[max_idx]:.16e}, file2={data2[max_idx]:.16e}")

        # 1 ULP (Unit in Last Place) check
        if max_abs == 0:
            print(f"  ✓ the grid data is identical")
        elif max_abs < 1e-15:
            print(f"  ✓ difference < 1e-15, within floating-point precision (1 ULP level)")
        elif max_abs < 1e-10:
            print(f"  ⚠ difference < 1e-10, possibly caused by a different floating-point summation order")
        else:
            print(f"  ✗ the difference is large and needs investigation")
    print()

    # 7. Summary
    print("=" * 70)
    print(" Summary")
    print("=" * 70)
    if result["identical"]:
        print("✓ the two .orb files are identical (byte level)")
    elif size1 == size2 and md5_1 != md5_2:
        print("✗ the file sizes match but the contents differ (there may be subtle differences)")
    elif size1 != size2:
        print(f"⚠ file sizes differ: {size1} vs {size2}")
    else:
        print("? unknown case")


if __name__ == "__main__":
    main()
