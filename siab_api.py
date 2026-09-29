"""
SIAB (ABACUS-CSW-NAO) Python API 接口文档

本文档描述了如何使用Python API调用SIAB库的各个功能模块。
主要涉及两个核心流程：
1. JSON配置 -> NSW生成 -> INPUT文件生成
2. DFT计算结果 + NSW -> Orbgen优化 -> 轨道文件

代码库路径: /home/liguozhou/install/ABACUS-CSW-NAO
"""

# =============================================================================
# 第一部分：JSON配置解析与参数验证
# =============================================================================

"""
模块: SIAB.io.param

核心功能：
- 读取JSON格式的输入配置文件
- 参数校验与分组
- 几何结构与轨道参数关联
"""

from SIAB.io.param import read, ParamAssert, orb_link_geom, group

# 使用示例:
"""
# 读取JSON配置文件
glbparams, dftparams, spillparams, compute, iop = read('input.json')

# 返回值说明:
# - glbparams: 全局参数，包含 element, bessel_nao_rcut
# - dftparams: DFT计算参数 (ABACUS INPUT参数)
# - spillparams: Spillage优化参数 (geoms, orbitals, primitive_type等)
# - compute: 计算环境参数 (abacus_command, mpi_command等)
# - iop: 内部优化参数 (__iop开头)
"""

# =============================================================================
# 第二部分：NSW生成与ABACUS作业构建
# =============================================================================

"""
模块: SIAB.abacus.api

核心功能：
- 构建ABACUS DFT计算作业
- 生成INPUT和STRU输入文件
- 原子结构定义
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

# 使用示例:
"""
# 1. 构建原子种类信息
atomspecies = _build_atomspecies(
    elem='Si',
    pp='Si.upf',
    ecut=40,
    rcut=7,
    lmaxmax=2,
    primitive_type='reduced'
)

# 2. 构建单个DFT计算case
folder = _build_case(
    proto='dimer',
    pertkind='stretch',
    pertmag=2.0,
    atomspecies={'Si': {'pp': 'Si.upf', 'orb': 'Si.orb'}},
    dftshared={'ecutwfc': 60, 'basis_type': 'lcao'},
    dftspecific={'gamma_only': '1'}
)

# 3. 批量构建ABACUS作业
jobs = build_abacus_jobs(
    atomspecies=[{'elem': 'Si', 'ecutjy': 40, 'zval': 0}],
    rcuts=[7],
    dftparams={'ecutwfc': 60, 'basis_type': 'lcao'},
    geoms=[{'proto': 'dimer', 'pertkind': 'stretch', 'pertmags': [1.5, 2.0, 2.5]}],
    spill_guess='atomic'
)

# 4. 计算nzeta (每个角动量的zeta函数数量)
nzeta, ecut, lmaxmax = _cal_nzeta(
    rcut=7,
    ecut=40,
    lmaxmax=2,
    less_dof=0
)

# 5. 生成INPUT文件内容
input_text = dftparam_to_text(dftparam)

# 6. 生成STRU文件内容
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

# 7. 自动设置DFT参数
dftparam = autoset({
    'ecutwfc': 60,
    'basis_type': 'lcao',
    'nspin': 1
})
"""

# =============================================================================
# 第三部分：轨道优化与Orbgen
# =============================================================================

"""
模块: SIAB.orb.api

核心功能：
- 创建轨道级联优化实例
- 管理轨道初始化和优化流程
"""

from SIAB.orb.api import (
    GetOrbCascadeInstance,
    DeriveCascadeInstance,
)

from SIAB.orb.cascade import OrbgenCascade

# 使用示例:
"""
# 创建轨道级联实例
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
            'iorb_frozen': 0,  # 冻结第一个轨道的系数
        }
    ],
    mode='jy',  # 或 'pw'
    optimizer='torch.swats'  # 或 'scipy.bfgs'
)

# 执行优化
cascade, spillage_values = cascade.opt(
    diagnosis=True,
    options={'maxiter': 100},
    nthreads=4
)

# 导出轨道文件
cascade.to_file(outdir='./output')

# 从已有cascade派生新实例 (添加更多轨道)
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
# 第四部分：Spillage优化核心
# =============================================================================

"""
模块: SIAB.spillage.spillage

核心功能：
- 从DFT计算结果提取参考数据
- 生成轨道初始猜测
- 计算spillage值
"""

from SIAB.spillage.spillage import (
    initgen_jy,
    initgen_pw,
    _jy_data_extract,
)

from SIAB.spillage.spilltorch import SpillTorch_jy, SpillTorch_pw

# 使用示例:
"""
# 1. 从JY模式DFT输出提取数据
data = _jy_data_extract('OUT.ABACUS')
# 返回: {natom, nzeta, wk, S, T, C}

# 2. 从原子计算结果生成初始轨道系数
coef = initgen_jy(
    outdir='Si-monomer-0-7au/OUT.ABACUS',
    nzeta=[1, 1, 0],
    ibands='all',
    nbes_gen=None,
    diagnosis=True
)
# 返回: coef[l][zeta][q] 格式的系数

# 3. 从PW模式计算结果生成初始轨道系数
coef = initgen_pw(
    orb_mat='orb_matrix.0.dat',
    nzeta=[1, 1, 0],
    ibands='all'
)

# 4. 使用Torch优化器进行spillage优化
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
# 第五部分：完整Workflow接口
# =============================================================================

"""
模块: SIAB.driver.main

核心功能：
- 完整的ABACUS-ORBGEN工作流程
- 从JSON配置到轨道文件的一站式解决方案
"""

from SIAB.driver.main import (
    init,
    rundft,
    minimize_spillage,
)

# 使用示例:
"""
# 1. 初始化workflow (读取参数)
glbparams, dftparams, spillparams, compute, iop = init('input.json')

# 2. 运行DFT计算
jobs = rundft(
    atomspecies=[{'elem': 'Si', 'ecutjy': 40, 'zval': 0}],
    rcuts=[7],
    dftparam={'ecutwfc': 60, 'basis_type': 'lcao'},
    geoms=spillparams['geoms'],
    spillguess='atomic',
    compparam={'abacus_command': 'mpirun -np 8 abacus'}
)

# 3. 运行spillage优化
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
# 第六部分：工具函数
# =============================================================================

"""
模块: SIAB.io.convention

核心功能：
- 文件和文件夹命名约定
- 轨道参数字符串生成
"""

from SIAB.io.convention import (
    dft_folder,
    orb_folder,
    orb as orb_filename,
    nzeta_string,
)

# 使用示例:
"""
# 1. 生成DFT计算文件夹名
folder = dft_folder('Si', 'dimer', 2.0, 7)
# 返回: 'Si-dimer-2.00-7au'

# 2. 生成轨道文件夹名
folder = orb_folder('Si', [1, 1, 0])
# 返回: 'Si_s1p1'

# 3. 生成轨道文件名
filename = orb_filename('Si', 7, 40, [1, 1, 0])
# 返回: 'Si_gga_7au_40Ry_1s1p.orb'

# 4. 生成nzeta字符串
nz_str = nzeta_string([1, 1, 1, 0])
# 返回: '1s1p1d'
"""

"""
模块: SIAB.spillage.radial

核心功能：
- 球Bessel函数相关计算
- 径向积分
"""

from SIAB.spillage.radial import (
    _nbes,  # 计算给定l, rcut, ecut下的Bessel函数数量
    jl_reduce,  # Bessel函数约化
)

# 使用示例:
"""
# 计算l=0, rcut=7, ecut=40时的Bessel函数数量
n = _nbes(0, 7, 40)
# 返回: Bessel函数数量

# 约化Bessel函数
reduced = jl_reduce(l=0, nbes=21, rcut=7)
"""

# =============================================================================
# 完整API调用示例
# =============================================================================

"""
以下是一个完整的Python脚本示例，展示了如何使用API：

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

# 步骤1: 读取配置
glbparams, dftparams, spillparams, compparam, iop = read('config.json')

# 步骤2: 构建并运行DFT作业
atomspecies = _build_atomspecies(
    elem=glbparams['element'],
    pp=os.path.join(spillparams['pseudo_dir'], f'{glbparams["element"]}.upf'),
    ecut=spillparams.get('ecutjy', dftparams['ecutwfc']),
    rcut=glbparams['bessel_nao_rcut'][0],
    lmaxmax=2,
    primitive_type=spillparams['primitive_type']
)

# 步骤3: 构建ABACUS作业
jobs = build_abacus_jobs(
    atomspecies=[{'elem': glbparams['element'], 
                  'ecutjy': spillparams.get('ecutjy', dftparams['ecutwfc']),
                  'zval': 0}],
    rcuts=glbparams['bessel_nao_rcut'],
    dftparams=dftparams,
    geoms=spillparams['geoms'],
    spill_guess=spillparams.get('spill_guess')
)

# 步骤4: 等待DFT计算完成后，创建轨道优化cascade
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

# 步骤5: 执行优化
cascade, spillages = cascade.opt(diagnosis=True)

# 步骤6: 导出轨道文件
cascade.to_file(outdir='./orbitals')

print(f"轨道优化完成，最终spillage值: {spillages}")
```
"""

# =============================================================================
# JSON输入配置格式说明
# =============================================================================

"""
JSON输入文件示例:

{
    // 计算环境配置
    "environment": "",
    "mpi_command": "mpirun -np 8",
    "abacus_command": "abacus",

    // 元素信息
    "element": "Si",
    "pseudo_dir": "Si.upf",

    // 基底类型: jy (球Bessel) 或 pw (平面波)
    "fit_basis": "jy",

    // 平面波动能截断 (Ry)
    "ecutwfc": 60,

    // 球Bessel动能截断 (Ry)
    "ecutjy": 40,

    // 轨道截断半径 (au)
    "bessel_nao_rcut": [7, 10],

    // 原始基底类型: reduced 或 normalized
    "primitive_type": "reduced",

    // 轨道初始猜测方式
    "spill_guess": "atomic",

    // 优化器配置
    "optimizer": "scipy.bfgs",
    "max_steps": 9000,
    "nthreads_rcut": 4,

    // 几何结构配置
    "geoms": [
        {
            "proto": "dimer",           // 结构原型: dimer, trimer, monomer等
            "pertkind": "stretch",       // 扰动类型: stretch, shear, twist
            "pertmags": [1.62, 1.82, 2.22, 2.72, 3.22],  // 键长列表 (Angstrom)
            "nbands": 20,               // 计算使用的能带数
            "nspin": 1,                 // 自旋极化: 1 或 2
            "lmaxmax": 2,               // 最大角动量
            "celldm": 30                // 晶格常数 (Bohr)
        }
    ],

    // 轨道生成配置
    "orbitals": [
        {
            "nzeta": [1, 1, 0],         // 每个角动量的zeta数 [s, p, d, ...]
            "geoms": [0],               // 引用的几何结构索引
            "nbands": "occ",            // 参与优化的能带数, "occ" 或整数
            "checkpoint": null,          // 冻结的内部轨道索引
            "greedygrow": false         // 是否启用贪心增长算法
        }
    ]
}
"""
