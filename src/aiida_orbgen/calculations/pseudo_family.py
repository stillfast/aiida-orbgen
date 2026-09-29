"""
aiida_orbgen.calculations.pseudo_family - Pseudo-family utilities for SIAB workflow

集中管理从 SIAB 生成的 UPF + ORB 文件到 AiiDA ``AtomicOrbitalFamily`` 的
所有操作:

- :func:`resolve_paths_from_json`  : 从 SIAB JSON + generate_all 结果解析
  UPF/ORB/family_label 路径
- :func:`resolve_family_label`     : 从 ORB/UPF 文件名推断 family label
- :func:`family_exists`            : 检查 family 是否已存在
- :func:`build_orb_family`         : 创建/取出 family
- :func:`ensure_pseudo_family`     : 检查或创建 (供 WorkChain 使用)

依赖
----
- ``aiida_orbgen.static.defaults``    : DEFAULT_FAMILY_LABEL_TEMPLATE
- ``aiida_orbgen.interfaces``         : generate_all_from_json
- ``aiida-abacus``                    : AtomicOrbitalData / Family / Collection
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

# AiiDA profile 必须先 load (本模块几乎所有函数都用 orm.QueryBuilder)
from aiida import load_profile
load_profile()

from aiida import orm
from aiida_abacus.group.orb_group import AtomicOrbitalFamily

from aiida_orbgen.static.defaults import DEFAULT_FAMILY_LABEL_TEMPLATE


__all__ = [
    "resolve_paths_from_json",
    "resolve_family_label",
    "family_exists",
    "build_orb_family",
    "ensure_pseudo_family",
]


# ===========================================================================
#  Path / family label 解析
# ===========================================================================


def resolve_upf_path(pseudo_dir: str, *bases: "os.PathLike | str") -> str:
    """Resolve ``config["pseudo_dir"]`` against the given base directories.

    SIAB itself interprets a relative ``pseudo_dir`` relative to the JSON file
    it was handed.  In the AiiDA path that JSON is a *copy* written into the run
    directory (``workflows.batch.run_siab_pipeline``), never the file the user
    wrote — so the base has to be chosen explicitly instead of being assumed.

    Absolute paths are returned as-is.  A relative path is tried against each
    base in order, and the first existing candidate wins; if none exists the
    error lists every location that was tried, because the previous behaviour
    (silently returning a path inside a ``/tmp`` tempfile) surfaced much later
    as "pseudo family not registered".
    """
    if not pseudo_dir:
        raise ValueError(
            "the SIAB config has no 'pseudo_dir'; set static.pseudo_path in "
            "input.json or pseudo_dir in the orbgen preset"
        )
    candidate = Path(pseudo_dir).expanduser()
    if candidate.is_absolute():
        return str(candidate)

    tried = []
    for base in bases:
        if not base:
            continue
        resolved = (Path(base).expanduser() / candidate).resolve()
        if resolved.is_file():
            return str(resolved)
        tried.append(str(resolved))
    raise FileNotFoundError(
        f"pseudo_dir={pseudo_dir!r} is relative and was not found under "
        f"{tried or ['<no base directory given>']}. Use an absolute path "
        f"(input.json['static']['pseudo_path'] is the canonical way)."
    )


def resolve_paths_from_json(siab_json_path: str, siab_result: dict) -> dict:
    """从 SIAB JSON + ``generate_all_from_json`` 返回值自动推断 UPF/ORB
    路径和 family label。

    Returns
    -------
    dict
        ``{"upf_path", "orb_path", "family_label"}``
    """
    siab_json_path = os.path.abspath(siab_json_path)
    with open(siab_json_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # The JSON is the config copy in the run directory; the UPF may also sit
    # next to it or in the current working directory (SIAB's own convention for
    # the shipped examples is a relative "./X.upf").
    upf_path = resolve_upf_path(
        config.get("pseudo_dir", ""),
        Path(siab_json_path).parent,
        Path.cwd(),
    )

    orb_path = siab_result["nsw"]

    family_label = resolve_family_label(
        orb_filename=siab_result["nsw_filename"],
        upf_path=upf_path,
    )
    return {
        "upf_path": upf_path,
        "orb_path": orb_path,
        "family_label": family_label,
    }


def _extract_zval_from_upf(upf_path: str) -> str:
    """从 UPF 文件的 PP_HEADER 中提取 z_valence（价电子数）。"""
    try:
        with open(upf_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read(8192)
        match = re.search(r'z_valence="\s*([\d.]+)', content)
        if match:
            zval_float = float(match.group(1))
            return str(int(round(zval_float)))
        match = re.search(r"z_valence\s*=\s*([\d.]+)", content)
        if match:
            zval_float = float(match.group(1))
            return str(int(round(zval_float)))
    except Exception:
        pass
    return "?"


def resolve_family_label(*, orb_filename: str, upf_path: str) -> str:
    """从 ORB / UPF 文件名推断 family label。

    family label 格式 (例):
        siab-u-nr-pbe-z6-nsw-9au-100Ry-g
        |     | |   |  |    |    |     |
        |     | |   |  |    |    |     +-- lmax  (ORB 最高角动量 s/p/d/f/g/...)
        |     | |   |  |    |    +-- ecut (Ry)
        |     | |   |  |    +-- rcut (au)
        |     | |   |  +-- zval  (价电子数, 从 UPF PP_HEADER 提取)
        |     | |   +-- 固定 "nsw"
        |     +-- pp_xc  (UPF XC, pbe/pz/pbesol, 剥离 -n-nc 后缀)
        |     +-- rel  (fr=fully-relativistic, nr=non-relativistic)
        +-- element (小写)
    """
    # ---- ORB 文件名: {element}_{xc}_{rcut}au_{ecut}Ry_{nzeta}.orb ----
    stem = Path(orb_filename).stem
    parts = stem.split("_")
    element = parts[0] if len(parts) >= 1 else ""
    rcut_str = parts[2] if len(parts) >= 3 else ""
    ecut_str = parts[3] if len(parts) >= 4 else ""
    nzeta_str = parts[4] if len(parts) >= 5 else ""

    rcut = rcut_str.rstrip("au") if rcut_str else "?"
    ecut = ecut_str.rstrip("Ry") if ecut_str else "?"
    lmax = nzeta_str[-1] if nzeta_str else "?"

    # ---- UPF 文件名: {elem}.{pp_part}.upf ----
    upf_name = Path(upf_path).name
    upf_stem = upf_name.rsplit(".", 1)[0]  # 去 .upf
    upf_parts = upf_stem.split(".")
    pp_part = upf_parts[1] if len(upf_parts) >= 2 else ""
    if pp_part.startswith("rel-"):
        rel = "fr"
        pp_xc_raw = pp_part[len("rel-"):]
    else:
        rel = "nr"
        pp_xc_raw = pp_part
    # 剥离后缀 -n / -nc / -n-nc ...
    pp_xc = re.sub(r"(-\w+)*$", "", pp_xc_raw)

    zval = _extract_zval_from_upf(upf_path)

    if not element:
        return f"siab-{stem.lower()}"

    return DEFAULT_FAMILY_LABEL_TEMPLATE.format(
        element=element.lower(),
        rel=rel,
        pp_xc=pp_xc,
        zval=zval,
        rcut=rcut,
        ecut=ecut,
        lmax=lmax,
    )


# ===========================================================================
#  Pseudo-family 检查 / 构建
# ===========================================================================


def family_exists(family_label: str) -> Optional[int]:
    """返回 family 的 pk; 不存在则返回 ``None``。"""
    qb = orm.QueryBuilder()
    qb.append(AtomicOrbitalFamily, filters={"label": family_label})
    row = qb.first()
    if row is None:
        return None
    return row[0].pk


def build_orb_family(
    upf_path: str,
    orb_path: str,
    family_label: str,
    description: str = "Built by aiida-orbgen (SIAB workflow)",
    overwrite: bool = False,
) -> str:
    """从 UPF + ORB 文件构建 ``AtomicOrbitalFamily`` (3 步: Data → Collection → Family)。"""
    from aiida_abacus.data.orbital import AtomicOrbitalData

    if not os.path.exists(upf_path):
        raise FileNotFoundError(f"UPF 不存在: {upf_path}")
    if not os.path.exists(orb_path):
        raise FileNotFoundError(f"ORB 不存在: {orb_path}")

    # Step 1: AtomicOrbitalData (UPF+ORB pair, md5 去重)
    print(f"[1/3] AtomicOrbitalData.get_or_create({Path(upf_path).name}, {Path(orb_path).name})")
    orb_data = AtomicOrbitalData.get_or_create(upf_path, orb_path)
    if not orb_data.is_stored:
        orb_data.store()
    print(f"      -> pk={orb_data.pk}, stored={orb_data.is_stored}")

    # Step 2: AtomicOrbitalCollection
    coll_label = family_label + "-collection"
    print(f"[2/3] collection '{coll_label}'")
    coll, _ = orm.Group.collection.get_or_create(label=coll_label)
    coll.description = description
    if orb_data.pk not in [n.pk for n in coll.nodes]:
        coll.add_nodes([orb_data])
        print(f"      -> 节点 {orb_data.pk} 已加入 (共 {coll.count()} 个)")
    else:
        print(f"      -> 节点 {orb_data.pk} 已在 collection 中")

    # Step 3: AtomicOrbitalFamily
    print(f"[3/3] family '{family_label}'")
    existing_pk = family_exists(family_label)
    if existing_pk is not None:
        if overwrite:
            fam_node = orm.load_node(existing_pk)
            fam_node.delete()
        else:
            print(f"      -> family 已存在 (pk={existing_pk}), 跳过创建")
            return family_label

    family = AtomicOrbitalFamily(label=family_label, description=description)
    family.store()
    if orb_data.pk not in [n.pk for n in family.nodes]:
        family.add_nodes([orb_data])
    print(f"      -> 节点 {orb_data.pk} 已加入 (共 {family.count()} 个)")
    print(f"==>  family label: {family_label}")
    return family_label


def ensure_pseudo_family(
    upf_path: str,
    orb_path: str,
    family_label: str,
    *,
    build_if_missing: bool = True,
    description: str = "Built by aiida-orbgen (SIAB workflow)",
) -> str:
    """检查 family 是否存在; 不存在且 ``build_if_missing=True`` 时构建。

    Returns
    -------
    str
        family_label (无论新建还是复用)
    """
    existing_pk = family_exists(family_label)
    if existing_pk is not None:
        print(f"family '{family_label}' 已存在 (pk={existing_pk}), 跳过创建")
        return family_label
    if not build_if_missing:
        raise RuntimeError(
            f"family '{family_label}' 不存在, 且 build_if_missing=False"
        )
    return build_orb_family(
        upf_path, orb_path, family_label, description=description
    )
