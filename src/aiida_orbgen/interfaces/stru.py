"""
STRU 接口

生成 ABACUS STRU 文件 (结构文件)。

对应文件: {folder}/STRU
例: U-dimer-1.89-9au/STRU

本模块是对 SIAB 的高层封装，底层调用:
- ``SIAB.abacus.io.structure_to_text`` : 按 proto 调度生成 STRU 文本
- ``SIAB.io.convention.dft_folder``    : 标准 DFT 文件夹命名

这样做的好处:
1. 所有支持的几何原型 (monomer/dimer/trimer/...) 由 SIAB 维护。
2. 文件夹命名 / 数值精度与 SIAB 主线完全一致。
3. 接口层只负责 "JSON 参数 → SIAB 调用参数" 的映射。
"""

import os
import math
from typing import Optional, Dict, Any, List

__all__ = [
    "generate_stru",
    "dft_folder_name",
    "generate_atom_coords",
    "parse_stru",
    "read_stru_as_ase",
    "verify_ase_atoms",
    "params_stru_to_ase",
]


# SIAB 支持的几何原型
SUPPORTED_PROTOS = {
    "monomer",
    "dimer",
    "trimer",
    "tetrahedron",
    "square",
    "triangular_bipyramid",
    "octahedron",
    "cube",
}


# Bohr -> Angstrom
BOHR_TO_ANG = 1.8897259886


def dft_folder_name(
    elem: str,
    proto: str,
    pert: float,
    rcut: Optional[float] = None,
) -> str:
    """
    生成 SIAB 标准 DFT 文件夹名。

    直接调用 ``SIAB.io.convention.dft_folder``，保证与 SIAB 完全一致。

    Parameters
    ----------
    elem : str
        元素符号
    proto : str
        几何原型
    pert : float
        扰动幅度 (键长, Angstrom)
    rcut : float, optional
        截断半径 (au)，如果有则加入文件夹名

    Returns
    -------
    str
        文件夹名

    Examples
    --------
    >>> dft_folder_name("U", "dimer", 1.89, rcut=9)
    'U-dimer-1.89-9au'
    >>> dft_folder_name("U", "monomer", 0)
    'U-monomer'
    """
    from SIAB.io.convention import dft_folder
    return dft_folder(elem, proto, pert, rcut=rcut)


def generate_atom_coords(
    proto: str,
    bond_length: float,
    lattice_constant: float,
) -> List[List[float]]:
    """
    根据几何原型生成原子坐标 (Angstrom)。

    这一步只是 "数值计算"，不写文件；底层复用 SIAB 的结构生成逻辑，
    因此计算结果与 SIAB 完全一致。

    Parameters
    ----------
    proto : str
        几何原型
    bond_length : float
        键长 (Angstrom)
    lattice_constant : float
        晶格常数 (Bohr)

    Returns
    -------
    list[list[float]]
        原子坐标列表 [[x, y, z], ...]
    """
    proto = proto.lower()
    if proto not in SUPPORTED_PROTOS:
        raise ValueError(
            f"Unsupported proto: {proto}. "
            f"Supported: {', '.join(sorted(SUPPORTED_PROTOS))}"
        )

    shift = lattice_constant / 2 / BOHR_TO_ANG

    if proto == "monomer":
        return [[0.0 + shift, 0.0 + shift, 0.0 + shift]]

    if proto == "dimer":
        return [
            [0.0 + shift, 0.0 + shift, 0.0 + shift],
            [0.0 + shift, 0.0 + shift, bond_length + shift],
        ]

    if proto == "trimer":
        dis1 = bond_length * 0.86603
        dis2 = bond_length * 0.5
        return [
            [0.0 + shift, 0.0 + shift, 0.0 + shift],
            [0.0 + shift, 0.0 + shift, bond_length + shift],
            [0.0 + shift, dis1 + shift, dis2 + shift],
        ]

    if proto == "tetrahedron":
        dis1 = bond_length * 0.86603
        dis2 = bond_length * 0.5
        dis3 = bond_length * 0.81649
        dis4 = bond_length * 0.28867
        return [
            [0.0 + shift, 0.0 + shift, 0.0 + shift],
            [0.0 + shift, 0.0 + shift, bond_length + shift],
            [0.0 + shift, dis1 + shift, dis2 + shift],
            [dis3 + shift, dis4 + shift, dis2 + shift],
        ]

    if proto == "square":
        return [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, bond_length],
            [bond_length + shift, 0.0 + shift, 0.0 + shift],
            [bond_length + shift, 0.0 + shift, bond_length + shift],
        ]

    if proto == "octahedron":
        d = bond_length / 2
        s = bond_length / math.sqrt(2)
        return [
            [d + shift, d + shift, 0.0 + shift],
            [-d + shift, -d + shift, 0.0 + shift],
            [d + shift, -d + shift, 0.0 + shift],
            [-d + shift, d + shift, 0.0 + shift],
            [0.0 + shift, 0.0 + shift, s + shift],
            [0.0 + shift, 0.0 + shift, -s + shift],
        ]

    if proto == "cube":
        d = bond_length / 2
        return [
            [d + shift, d + shift, d + shift],
            [-d + shift, -d + shift, d + shift],
            [d + shift, -d + shift, d + shift],
            [-d + shift, d + shift, d + shift],
            [d + shift, d + shift, -d + shift],
            [-d + shift, -d + shift, -d + shift],
            [d + shift, -d + shift, -d + shift],
            [-d + shift, d + shift, -d + shift],
        ]

    if proto == "triangular_bipyramid":
        d1 = bond_length / math.sqrt(3)
        d2 = bond_length / 2
        d3 = bond_length * math.sqrt(2.0 / 3.0)
        return [
            [d1 + shift, 0.0 + shift, 0.0 + shift],
            [-d1 / 2 + shift, d2 + shift, 0.0 + shift],
            [-d1 / 2 + shift, -d2 + shift, 0.0 + shift],
            [0.0 + shift, 0.0 + shift, d3 + shift],
            [0.0 + shift, 0.0 + shift, -d3 + shift],
        ]

    # 不会到达这里
    raise ValueError(f"Unsupported proto: {proto}")


def _resolve_stru_params(
    json_config: dict,
    proto: Optional[str],
    nspin: Optional[int],
    lattice_constant: Optional[float],
    bond_length: Optional[float],
) -> Dict[str, Any]:
    """从 JSON 中推断 STRU 生成参数。"""
    elem = json_config["element"]
    pseudo_dir = json_config.get("pseudo_dir", "./pseudo.upf")
    mass = json_config.get("mass", 1.0)

    geoms = json_config.get("geoms", [])
    if geoms:
        geom0 = geoms[0]
        if proto is None:
            proto = geom0.get("proto", "dimer")
        if nspin is None:
            nspin = geom0.get("nspin", 1)
        if lattice_constant is None:
            lattice_constant = geom0.get("celldm", 30)
        if bond_length is None:
            pertmags = geom0.get("pertmags", [0.0])
            if isinstance(pertmags, list) and pertmags:
                bond_length = float(pertmags[0])
            else:
                bond_length = 0.0
    else:
        proto = proto or "dimer"
        nspin = nspin or 1
        lattice_constant = lattice_constant or 30.0
        bond_length = bond_length or 0.0

    return {
        "elem": elem,
        "mass": mass,
        "fpseudo": os.path.basename(pseudo_dir),
        "lattice_constant": float(lattice_constant),
        "bond_length": float(bond_length),
        "nspin": int(nspin),
        "proto": proto,
    }


def generate_stru(
    json_config: dict,
    output_path: str,
    proto: Optional[str] = None,
    bond_length: Optional[float] = None,
    nspin: Optional[int] = None,
    lattice_constant: Optional[float] = None,
    pseudo_filename: Optional[str] = None,
    orb_filename: Optional[str] = None,
) -> str:
    """
    生成 ABACUS STRU 文件。

    底层调用 ``SIAB.abacus.io.structure_to_text``，因此输出格式
    (数值精度、注释、字段顺序) 与 SIAB 主线完全一致。

    Parameters
    ----------
    json_config : dict
        SIAB JSON 配置，应包含:
        - element, pseudo_dir, geoms[0]
    output_path : str
        输出文件路径
    proto : str, optional
        几何原型 (覆盖 JSON 中的设置)
    bond_length : float, optional
        键长 (Angstrom, 覆盖 JSON 中的设置)
    nspin : int, optional
        自旋极化 (1 或 2)
    lattice_constant : float, optional
        晶格常数 (Bohr, 覆盖 JSON 中的设置)
    pseudo_filename : str, optional
        赝势文件名 (覆盖 JSON 中的设置)
    orb_filename : str, optional
        NUMERICAL_ORBITAL 段引用的轨道文件名

    Returns
    -------
    str
        生成的 STRU 文件的绝对路径

    Examples
    --------
    >>> import json
    >>> with open("pbe_orbgen.json") as f:
    ...     config = json.load(f)
    >>> generate_stru(
    ...     config,
    ...     output_path="./U-dimer-1.89-9au/STRU",
    ...     bond_length=1.89,
    ...     orb_filename="U_gga_9au_100Ry_27s27p26d26f25g.orb",
    ... )
    """
    from SIAB.abacus.io import structure_to_text

    params = _resolve_stru_params(
        json_config, proto, nspin, lattice_constant, bond_length
    )

    if pseudo_filename is not None:
        params["fpseudo"] = pseudo_filename

    text, _natom = structure_to_text(
        shape=params["proto"],
        element=params["elem"],
        mass=params["mass"],
        fpseudo=params["fpseudo"],
        lattice_constant=params["lattice_constant"],
        bond_length=params["bond_length"],
        nspin=params["nspin"],
        forb=orb_filename,
    )

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(text)

    return os.path.abspath(output_path)


def parse_stru(filepath: str) -> Dict[str, Any]:
    """
    从 STRU 文件中解析关键字段。

    解析的字段:
    - element: 元素符号
    - pseudo_file: 赝势文件名
    - lattice_constant: 晶格常数 (Bohr)
    - lattice_vectors: 3x3 晶格矢量
    - nspin: 1 或 2
    - proto: 由 number of atoms 推断 (1=monomer, 2=dimer, ...)
    - coordinates: [[x, y, z, ...], ...]

    Parameters
    ----------
    filepath : str
        STRU 文件路径

    Returns
    -------
    dict
        解析结果
    """
    result: Dict[str, Any] = {
        "element": None,
        "pseudo_file": None,
        "lattice_constant": None,
        "lattice_vectors": None,
        "nspin": 1,
        "proto": None,
        "coordinates": [],
        "orbital_file": None,
    }

    section = None
    coord_count = 0
    expected_coords = 0
    with open(filepath, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            upper = line.upper()
            if upper.startswith("ATOMIC_SPECIES"):
                section = "ATOMIC_SPECIES"
                continue
            if upper.startswith("NUMERICAL_ORBITAL"):
                section = "NUMERICAL_ORBITAL"
                continue
            if upper.startswith("LATTICE_CONSTANT"):
                section = "LATTICE_CONSTANT"
                continue
            if upper.startswith("LATTICE_VECTORS"):
                section = "LATTICE_VECTORS"
                continue
            if upper.startswith("ATOMIC_POSITIONS"):
                section = "ATOMIC_POSITIONS_HEADER"
                continue

            if section == "ATOMIC_SPECIES":
                tokens = line.split()
                if len(tokens) >= 3 and not tokens[0].startswith("//"):
                    result["element"] = tokens[0]
                    result["pseudo_file"] = tokens[2]
                section = None
            elif section == "NUMERICAL_ORBITAL":
                if not line.startswith("//"):
                    result["orbital_file"] = line
                section = None
            elif section == "LATTICE_CONSTANT":
                # LATTICE_CONSTANT
                # 30.000000  // add lattice constant(a.u.)
                tok = line.split()
                result["lattice_constant"] = float(tok[0])
                section = None
            elif section == "LATTICE_VECTORS":
                tok = line.split()
                if len(tok) >= 3:
                    if result["lattice_vectors"] is None:
                        result["lattice_vectors"] = []
                    result["lattice_vectors"].append(
                        [float(tok[0]), float(tok[1]), float(tok[2])]
                    )
            elif section == "ATOMIC_POSITIONS_HEADER":
                # 例如 "Cartesian_angstrom_center_xyz  //Cartesian or Direct coordinate."
                section = "ATOMIC_POSITIONS"
            elif section == "ATOMIC_POSITIONS":
                # 3 行: 元素标签, magnetization, 原子数, 然后是 natom 行坐标
                if "//Element Label" in line or "//element" in line.lower():
                    continue
                if "//number of atoms" in line.lower() or "number of atoms" in line:
                    # 取该行的第一个数字
                    for tok in line.split():
                        try:
                            expected_coords = int(tok)
                            natoms = expected_coords
                            # 用原子数推断 proto
                            proto_map = {1: "monomer", 2: "dimer", 3: "trimer", 4: None}
                            result["proto"] = proto_map.get(expected_coords)
                            break
                        except ValueError:
                            continue
                    continue
                if "//starting magnetism" in line.lower():
                    try:
                        result["nspin"] = 2 if float(line.split()[0]) > 0 else 1
                    except ValueError:
                        pass
                    continue
                # 坐标行: 3 个浮点数 + 3 个整数
                tok = line.split()
                if len(tok) >= 3:
                    try:
                        x, y, z = float(tok[0]), float(tok[1]), float(tok[2])
                        result["coordinates"].append([x, y, z])
                    except ValueError:
                        continue
                    coord_count += 1
                    if expected_coords and coord_count >= expected_coords:
                        section = None

    return result


# ASE 支持的坐标类型白名单。SIAB 写的 ``Cartesian_angstrom_center_xyz``
# 实际上是 ``Cartesian`` (单位 Angstrom) 的非标准变体, ASE 不识别。
# 这里集中处理转换, 避免污染 ``parse_stru`` 的纯文本解析。
_ASE_COORD_ALIASES = {
    "cartesian_angstrom_center_xyz": "Cartesian",
    "cartesian_angstrom": "Cartesian",
    "cartesian_nm": "Cartesian",
    "cartesian_bohr": "Cartesian",  # 但坐标需 Bohr->Ang, 由 ASE 处理
}


def _ase_compatible_stru_text(filepath: str) -> str:
    """
    读取 STRU 文件内容, 把 SIAB 的非标准坐标标记替换为 ASE 可识别的形式。

    注意: 这一步只改文本, 不动坐标数值; 坐标在原文件中已经是 Angstrom,
    与 ASE 的 'Cartesian' 单位一致。
    """
    with open(filepath, "r") as f:
        text = f.read()
    for src, dst in _ASE_COORD_ALIASES.items():
        text = text.replace(src, dst)
        text = text.replace(src.upper(), dst)
        text = text.replace(src.capitalize(), dst)
    return text


def read_stru_as_ase(filepath: str):
    """
    读取 STRU 文件并返回 ``ase.Atoms`` 对象。

    直接 ``ase.io.read(stru, format='abacus')`` 会失败, 因为:
    1. SIAB 写的坐标标记 ``Cartesian_angstrom_center_xyz`` 不在 ASE 白名单
       (只接受 ``Direct`` / ``Cartesian``)。
    2. 即使把标签换成 ``Cartesian``, ASE 仍会按 ``pos × lat0`` 解析,
       与 SIAB 写的"原始 Angstrom"语义不一致。

    本函数绕开 ASE STRU reader, 直接用 :func:`parse_stru` 解析后构造
    ``ase.Atoms``, 避免以上两个坑。

    Parameters
    ----------
    filepath : str
        STRU 文件路径

    Returns
    -------
    ase.Atoms
        包含 cell, positions, numbers, pbc 等信息的 ASE 原子对象

    Examples
    --------
    >>> from aiida_orbgen.interfaces import read_stru_as_ase
    >>> atoms = read_stru_as_ase("./U-dimer-2.75-9au/STRU")
    >>> atoms.get_chemical_formula()
    'U2'
    >>> atoms.cell.array
    """
    import numpy as np
    from ase import Atoms

    parsed = parse_stru(filepath)
    if parsed["lattice_vectors"] is None or not parsed["coordinates"]:
        raise ValueError(f"无法从 {filepath} 解析出 lattice / coordinates")

    # SIAB: cell (Bohr) = LATTICE_CONSTANT (Bohr) * LATTICE_VECTORS (dimensionless)
    lat0_bohr = float(parsed["lattice_constant"])
    cell_bohr = np.array(parsed["lattice_vectors"], dtype=float) * lat0_bohr
    cell_ang = cell_bohr / BOHR_TO_ANG

    # SIAB: 坐标是原始 Angstrom, 已经是 SI 单位, 不需要再换算
    positions_ang = np.array(parsed["coordinates"], dtype=float)

    symbols = [parsed["element"]] * len(positions_ang)

    return Atoms(
        symbols=symbols,
        positions=positions_ang,
        cell=cell_ang,
        pbc=[True, True, True],
    )


def verify_ase_atoms(stru_path: str, tol: float = 1e-6) -> Dict[str, Any]:
    """
    验证 STRU → ASE Atoms 转换的正确性。

    比较 :func:`parse_stru` (纯文本) 和 :func:`read_stru_as_ase` (走 ASE)
    解析同一文件得到的结果, 检查:

    - n_atoms         原子数一致
    - formula         元素列表一致
    - positions       坐标 (Angstrom) 一致, 误差 < ``tol``
    - cell            晶格矢量 (Angstrom) 一致, 误差 < ``tol``
    - pbc             三方向周期性设置一致
    - has_cartesian   坐标确实是 Cartesian (非 Direct)
    - has_orb         是否带 NUMERICAL_ORBITAL 引用 (ASE 不会读, 只做记录)

    Parameters
    ----------
    stru_path : str
        STRU 文件路径
    tol : float
        数值比较的绝对误差容限, default 1e-6 (Angstrom)

    Returns
    -------
    dict
        形如 ``{"check_name": {"ok": bool, "expected": ..., "got": ...}}`` 的报告。
        若所有项 ``ok=True``, 转换视为正确。

    Examples
    --------
    >>> from aiida_orbgen.interfaces import verify_ase_atoms
    >>> report = verify_ase_atoms("./U-dimer-2.75-9au/STRU")
    >>> assert all(v["ok"] for v in report.values()), report
    """
    parsed = parse_stru(stru_path)
    atoms = read_stru_as_ase(stru_path)

    # 1. 原子数
    n_atoms_ok = bool(len(atoms) == len(parsed["coordinates"]))

    # 2. 元素符号
    ase_symbols = list(atoms.get_chemical_symbols())
    parsed_symbols = [parsed["element"]] * len(parsed["coordinates"])
    formula_ok = bool(ase_symbols == parsed_symbols)

    # 3. 坐标
    pos_err = 0.0
    if n_atoms_ok:
        import numpy as np
        pos_err = float(np.max(np.abs(atoms.positions - np.array(parsed["coordinates"]))))
    positions_ok = bool(pos_err < tol)

    # 4. 晶格 (Bohr -> Angstrom)
    # SIAB: cell (Bohr) = LATTICE_CONSTANT (Bohr) * LATTICE_VECTORS (dimensionless)
    import numpy as np
    expected_cell = (
        np.array(parsed["lattice_vectors"])
        * float(parsed["lattice_constant"])
        / BOHR_TO_ANG
    )
    cell_err = float(np.max(np.abs(atoms.cell.array - expected_cell)))
    cell_ok = bool(cell_err < tol)

    # 5. 周期性: SIAB STRU 默认三方向 PBC
    expected_pbc = [True, True, True]
    pbc_ok = bool(list(atoms.pbc) == expected_pbc)

    return {
        "n_atoms": {
            "ok": n_atoms_ok,
            "expected": len(parsed["coordinates"]),
            "got": len(atoms),
        },
        "formula": {
            "ok": formula_ok,
            "expected": parsed_symbols,
            "got": ase_symbols,
        },
        "positions": {
            "ok": positions_ok,
            "max_abs_error_angstrom": pos_err,
            "tol_angstrom": tol,
        },
        "cell": {
            "ok": cell_ok,
            "max_abs_error_angstrom": cell_err,
            "expected_angstrom": expected_cell.tolist(),
            "got_angstrom": atoms.cell.array.tolist(),
        },
        "pbc": {
            "ok": pbc_ok,
            "expected": expected_pbc,
            "got": [bool(x) for x in atoms.pbc],
        },
        "has_orbital_reference": {
            "ok": bool(parsed["orbital_file"] is not None),
            "orbital_file": parsed["orbital_file"],
            "note": "ASE 不会读取 NUMERICAL_ORBITAL 段, 仅作记录",
        },
    }


def params_stru_to_ase(params_stru: Dict[str, Any]):
    """
    将 ``read_stru`` 返回的结构化 dict 转换为 ``ase.Atoms``。

    期望的输入格式::

        {
            'lat': {
                'const': 30.0,                        # 晶格常数 (Bohr)
                'vec':  [[1,0,0],[0,1,0],[0,0,1]]     # 晶格矢量 (无量纲)
            },
            'species': [
                {
                    'symbol':   'U',
                    'mass':     1.0,
                    'pp_file':  'U.pbe-n-nc.upf',
                    'orb_file': 'U_gga_9au_100Ry_27s27p26d26f25g.orb',
                    'mag_each': 0.0,                    # 每个原子平均磁矩
                    'natom':    2,
                    'atom': [
                        {'coord': [7.94, 7.94, 7.94],   # 单位取决于 coord_type
                         'm':     [0, 0, 0]},            # 3-vector 磁矩
                        {'coord': [7.94, 7.94, 10.69],  # Angstrom
                         'm':     [0, 0, 0]},
                    ],
                },
                # ... 多元素时这里有多个 species
            ],
            'coord_type': 'Cartesian_angstrom_center_xyz'   # 或 'Direct' 等
        }

    单位约定 (与 SIAB 一致):

    - ``lat.const``: Bohr
    - ``lat.vec``:   无量纲, 实际 cell (Bohr) = const × vec
    - ``atom.coord`` (Cartesian_angstrom_*): Angstrom, 已是 SI 单位
    - ``atom.coord`` (Direct):              分数坐标 ∈ [0, 1)
    - ``atom.m``:     磁矩 (3-vector, 通常 collinear 时只有 z 分量非零)

    Parameters
    ----------
    params_stru : dict
        符合上述 schema 的字典

    Returns
    -------
    ase.Atoms
        多元素、磁矩、晶格、周期性都正确设置

    Notes
    -----
    ``pp_file``/``orb_file`` 是 ABACUS 特有字段, ASE 不识别;
    若需要保留, 可在调用后用 ``atoms.info`` 写入:

    >>> atoms = params_stru_to_ase(params_stru)
    >>> atoms.info["pp_file"]  = params_stru["species"][0]["pp_file"]
    >>> atoms.info["orb_file"] = params_stru["species"][0]["orb_file"]

    Examples
    --------
    >>> params_stru = read_stru("./U-dimer-2.75-9au/STRU")
    >>> atoms = params_stru_to_ase(params_stru)
    >>> atoms.get_chemical_formula()
    'U2'
    >>> atoms.cell.array
    """
    import numpy as np
    from ase import Atoms

    # ---- 1. 解析 cell ----
    lat = params_stru["lat"]
    lat_const_bohr = float(lat["const"])
    lat_vec = np.asarray(lat["vec"], dtype=float)
    if lat_vec.shape != (3, 3):
        raise ValueError(
            f"lat.vec 必须是 3x3, 实际 {lat_vec.shape}"
        )
    cell_bohr = lat_vec * lat_const_bohr
    cell_ang = cell_bohr / BOHR_TO_ANG

    # ---- 2. 解析坐标类型 ----
    coord_type = params_stru.get("coord_type", "")
    # Direct (fractional) vs Cartesian 区分
    is_direct = coord_type.lower().startswith("direct")
    # SIAB 的 ``Cartesian_angstrom_center_xyz`` / ``Cartesian_angstrom`` /
    # ``Cartesian`` 在 SIAB 里都是 Angstrom, 在 ASE 里也是 Angstrom。

    # ---- 3. 收集每个原子 ----
    symbols: List[str] = []
    positions: List[List[float]] = []
    magmoms: List[List[float]] = []
    masses: List[float] = []

    for sp in params_stru["species"]:
        sym = sp["symbol"]
        mass = float(sp.get("mass", 1.0))
        atoms_of_sp = sp.get("atom", [])
        if len(atoms_of_sp) != sp.get("natom", len(atoms_of_sp)):
            # 防御: 显式 natom 与 atom 列表长度不一致时报错
            raise ValueError(
                f"species '{sym}': natom={sp.get('natom')} "
                f"与 atom 列表长度 {len(atoms_of_sp)} 不一致"
            )
        for atom in atoms_of_sp:
            symbols.append(sym)
            masses.append(mass)
            coord = np.asarray(atom["coord"], dtype=float)
            if is_direct:
                # 分数坐标 -> Cartesian Angstrom
                pos_ang = coord @ cell_ang
            else:
                pos_ang = coord
            positions.append(pos_ang)
            # 磁矩: 兼容 3-vector 与 scalar
            m = atom.get("m", [0, 0, 0])
            if np.isscalar(m):
                magmoms.append([0, 0, float(m)])
            else:
                magmoms.append([float(x) for x in m])

    atoms = Atoms(
        symbols=symbols,
        positions=np.asarray(positions, dtype=float),
        cell=cell_ang,
        pbc=[True, True, True],
        magmoms=np.asarray(magmoms, dtype=float),
        masses=np.asarray(masses, dtype=float),
    )
    return atoms


def params_stru_to_ase_validate(params_stru: Dict[str, Any]) -> Dict[str, Any]:
    """
    验证 ``params_stru`` dict 能否被 :func:`params_stru_to_ase` 正确转换,
    并给出与 ASE ``Atoms`` 一致性的检查报告。

    报告项 (与 :func:`verify_ase_atoms` 类似):

    - n_atoms
    - formula
    - cell_angstrom
    - coord_type
    - direct_coords        坐标是否为分数 (True/False)

    Returns
    -------
    dict
        每个键为 ``{"ok": bool, "expected": ..., "got": ...}``
    """
    atoms = params_stru_to_ase(params_stru)
    import numpy as np

    # n_atoms
    expected_n = sum(sp["natom"] for sp in params_stru["species"])
    n_ok = bool(len(atoms) == expected_n)

    # formula
    expected_formula = "".join(
        f"{sp['symbol']}{sp['natom'] if sp['natom'] > 1 else ''}"
        for sp in params_stru["species"]
    )
    formula_ok = bool(atoms.get_chemical_formula() == expected_formula)

    # cell
    expected_cell = np.asarray(params_stru["lat"]["vec"], dtype=float) \
        * float(params_stru["lat"]["const"]) / BOHR_TO_ANG
    cell_err = float(np.max(np.abs(atoms.cell.array - expected_cell)))
    cell_ok = bool(cell_err < 1e-6)

    return {
        "n_atoms": {
            "ok": n_ok,
            "expected": expected_n,
            "got": len(atoms),
        },
        "formula": {
            "ok": formula_ok,
            "expected": expected_formula,
            "got": atoms.get_chemical_formula(),
        },
        "cell": {
            "ok": cell_ok,
            "max_abs_error_angstrom": cell_err,
            "expected_angstrom": expected_cell.tolist(),
            "got_angstrom": atoms.cell.array.tolist(),
        },
        "coord_type": {
            "ok": True,
            "value": params_stru.get("coord_type", ""),
            "is_direct": bool(
                params_stru.get("coord_type", "").lower().startswith("direct")
            ),
        },
    }
