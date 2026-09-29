"""
compare_orb.py: 详细比较两个 .orb 文件是否一致

对比维度:
1. MD5 校验和
2. 文件大小
3. 逐字节比较 (cmp)
4. 头部信息
5. 网格数据 (positions, chi values)

运行方式:
    cd /home/liguozhou/abacus/calculations/orbgen
    python aiida-orbgen/tests/data/pbe/compare_orb.py
"""

import os
import sys
import hashlib
import numpy as np
from pathlib import Path


def md5_of_file(filepath: str) -> str:
    """计算文件 MD5 校验和."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def compare_files_bytewise(f1: str, f2: str) -> dict:
    """逐字节比较两个文件."""
    with open(f1, "rb") as fp1, open(f2, "rb") as fp2:
        data1 = fp1.read()
        data2 = fp2.read()

    if data1 == data2:
        return {"identical": True, "diff_bytes": [], "size1": len(data1), "size2": len(data2)}

    # 找出第一个差异位置
    size = min(len(data1), len(data2))
    diff_bytes = []
    for i in range(size):
        if data1[i] != data2[i]:
            diff_bytes.append((i, data1[i], data2[i]))
            if len(diff_bytes) > 10:  # 只记录前 10 个差异
                break

    return {
        "identical": False,
        "diff_bytes": diff_bytes,
        "size1": len(data1),
        "size2": len(data2),
    }


def parse_orb_header(filepath: str) -> dict:
    """解析 .orb 文件头部."""
    with open(filepath) as f:
        lines = f.readlines()

    header = {}
    for i, line in enumerate(lines[:20]):
        line = line.strip()
        if not line or line.startswith("---"):
            continue
        # 解析 "Key Value" 格式
        parts = line.split(None, 1)
        if len(parts) == 2:
            key, value = parts
            header[key] = value
        elif len(parts) == 1:
            header[parts[0]] = None

    return header


def parse_orb_grid(filepath: str) -> tuple:
    """解析 .orb 文件的网格数据 (chi values)."""
    with open(filepath) as f:
        content = f.read()

    # 分割头部和数据
    lines = content.split("\n")
    data_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Type") and "L" in line and "N" in line:
            data_start = i + 2  # 跳过 Type 标题和第一个 0
            break

    if data_start is None:
        return np.array([]), 0

    # 解析所有数据
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
    """主函数: 详细比较两个 .orb 文件."""

    f1 = "/home/liguozhou/abacus/calculations/orbgen/project/pbe/primitive_jy/U_gga_9au_100Ry_27s27p26d26f25g.orb"
    f2 = "/home/liguozhou/abacus/calculations/orbgen/project/pbe_interface/primitive_jy/U_gga_9au_100Ry_27s27p26d26f25g.orb"

    print("=" * 70)
    print(" .orb 文件详细比较")
    print("=" * 70)
    print(f"文件 1 (参考): {f1}")
    print(f"文件 2 (生成): {f2}")
    print()

    # 1. 文件存在性
    print("[1] 文件存在性检查")
    print(f"  文件 1: {'✓ 存在' if os.path.exists(f1) else '✗ 不存在'}")
    print(f"  文件 2: {'✓ 存在' if os.path.exists(f2) else '✗ 不存在'}")
    print()

    if not (os.path.exists(f1) and os.path.exists(f2)):
        print("错误: 一个或两个文件不存在")
        return

    # 2. 文件大小
    print("[2] 文件大小比较")
    size1 = os.path.getsize(f1)
    size2 = os.path.getsize(f2)
    print(f"  文件 1: {size1} 字节")
    print(f"  文件 2: {size2} 字节")
    print(f"  差异: {abs(size1 - size2)} 字节 ({abs(size1 - size2) / size1 * 100:.4f}%)")
    print()

    # 3. MD5 校验和
    print("[3] MD5 校验和")
    md5_1 = md5_of_file(f1)
    md5_2 = md5_of_file(f2)
    print(f"  文件 1 MD5: {md5_1}")
    print(f"  文件 2 MD5: {md5_2}")
    print(f"  MD5 相同: {'✓ 是' if md5_1 == md5_2 else '✗ 否'}")
    print()

    # 4. 逐字节比较
    print("[4] 逐字节比较")
    result = compare_files_bytewise(f1, f2)
    if result["identical"]:
        print("  ✓ 两个文件完全一致 (字节级)")
    else:
        print(f"  ✗ 两个文件存在差异")
        print(f"  文件 1 长度: {result['size1']}")
        print(f"  文件 2 长度: {result['size2']}")
        print(f"  差异字节数: {len(result['diff_bytes'])}+")
        print(f"  前 10 个差异位置:")
        for offset, b1, b2 in result["diff_bytes"][:10]:
            # b1, b2 是 int (0-255)
            print(f"    偏移 {offset}: 0x{b1:02x} vs 0x{b2:02x} ('{chr(b1) if 32 <= b1 < 127 else '?'}' vs '{chr(b2) if 32 <= b2 < 127 else '?'}')")
    print()

    # 5. 头部信息比较
    print("[5] 头部信息比较")
    h1 = parse_orb_header(f1)
    h2 = parse_orb_header(f2)
    print(f"  {'Key':<30} {'文件1':<20} {'文件2':<20} {'匹配':<6}")
    print(f"  {'-'*30} {'-'*20} {'-'*20} {'-'*6}")
    for key in set(list(h1.keys()) + list(h2.keys())):
        v1 = h1.get(key, "(无)")
        v2 = h2.get(key, "(无)")
        match = "✓" if v1 == v2 else "✗"
        print(f"  {key:<30} {str(v1):<20} {str(v2):<20} {match}")
    print()

    # 6. 网格数据比较
    print("[6] 网格数据 (chi values) 比较")
    data1, n1 = parse_orb_grid(f1)
    data2, n2 = parse_orb_grid(f2)
    print(f"  文件 1 数据点数: {n1}")
    print(f"  文件 2 数据点数: {n2}")

    if n1 != n2:
        print(f"  ✗ 数据点数不同")
    else:
        # 计算统计差异
        diff = data1 - data2
        abs_diff = np.abs(diff)
        max_abs = np.max(abs_diff)
        mean_abs = np.mean(abs_diff)
        max_idx = np.argmax(abs_diff)
        print(f"  最大绝对差异: {max_abs:.6e}")
        print(f"  平均绝对差异: {mean_abs:.6e}")
        print(f"  最大差异位置: index={max_idx}")
        print(f"  位置 {max_idx} 处: 文件1={data1[max_idx]:.16e}, 文件2={data2[max_idx]:.16e}")

        # 1 ULP (Unit in Last Place) 检查
        if max_abs == 0:
            print(f"  ✓ 网格数据完全一致")
        elif max_abs < 1e-15:
            print(f"  ✓ 差异 < 1e-15, 在浮点精度范围内 (1 ULP 级别)")
        elif max_abs < 1e-10:
            print(f"  ⚠ 差异 < 1e-10, 可能是浮点计算顺序差异")
        else:
            print(f"  ✗ 差异较大, 需要检查")
    print()

    # 7. 总结
    print("=" * 70)
    print(" 总结")
    print("=" * 70)
    if result["identical"]:
        print("✓ 两个 .orb 文件完全一致 (字节级)")
    elif size1 == size2 and md5_1 != md5_2:
        print("✗ 文件大小相同, 但内容不同 (可能存在细微差异)")
    elif size1 != size2:
        print(f"⚠ 文件大小不同: {size1} vs {size2}")
    else:
        print("? 未知情况")


if __name__ == "__main__":
    main()
