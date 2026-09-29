"""
IncarData: AiiDA 数据节点 - 将 INPUT (ABACUS INCAR) 转换为 Dict

本模块负责将 ABACUS 的 INPUT 文件转换为 AiiDA Dict 节点。

ABACUS INPUT 文件格式:
    INPUT_PARAMETERS
    <key1>     <value1>
    <key2>     <value2>
    ...

数据流:
    INPUT 文件 → parse_incar_to_dict() → Dict (AiiDA 节点)
    Dict 节点可通过 .get_dict() 获取参数字典
"""

import re
import os
from typing import Optional, Dict, Any, Union

from aiida import orm

__all__ = ["IncarData", "parse_incar_to_dict", "incar_to_dict", "read_incar"]


def parse_incar_to_dict(incar_path: str) -> dict:
    """
    解析 ABACUS INPUT 文件, 跳过首行 "INPUT_PARAMETERS"。

    Parameters
    ----------
    incar_path : str
        INPUT 文件路径

    Returns
    -------
    dict
        解析出的参数字典

    Examples
    --------
    >>> params = parse_incar_to_dict("./U-dimer-1.89-9au/INPUT")
    >>> params["ecutwfc"]
    150
    >>> params["basis_type"]
    "lcao"
    """
    if not os.path.isfile(incar_path):
        raise FileNotFoundError(f"INPUT file not found: {incar_path}")

    result = {}
    pattern = re.compile(r"^\s*([\w_]+)\s+([^\#]+)")

    with open(incar_path, "r") as f:
        for line in f:
            line = line.strip()
            # 跳过空行
            if not line:
                continue
            # 跳过首行 "INPUT_PARAMETERS"
            if line == "INPUT_PARAMETERS" or line.startswith("INPUT_PARAMETERS"):
                continue
            # 跳过注释行
            if line.startswith("#") or line.startswith("//"):
                continue

            match = pattern.match(line)
            if match is not None:
                key = match.group(1)
                raw_value = match.group(2).strip()
                # 转换类型: 尝试 int / float / str
                value = _convert_value(raw_value)
                result[key] = value

    return result


def _convert_value(raw_value: str) -> Union[int, float, str]:
    """
    将字符串值转换为合适的 Python 类型。

    规则:
    - 如果是纯整数 (如 "100", "-5"), 转 int
    - 如果是浮点数 (如 "1.0", "1.0e-7"), 转 float
    - 否则保持为 str
    """
    # 尝试 int
    try:
        return int(raw_value)
    except ValueError:
        pass

    # 尝试 float
    try:
        return float(raw_value)
    except ValueError:
        pass

    # 保持为 str
    return raw_value


def incar_to_dict(incar_path: str) -> dict:
    """`parse_incar_to_dict` 的别名."""
    return parse_incar_to_dict(incar_path)


def read_incar(incar_path: str) -> dict:
    """`parse_incar_to_dict` 的别名."""
    return parse_incar_to_dict(incar_path)


class IncarData(orm.Dict):
    """
    AiiDA Dict 节点, 用于存储 ABACUS INPUT 参数。

    继承自 aiida.orm.Dict, 提供:
    - 从 INPUT 文件创建
    - 与 dict 互转
    - AiiDA 数据库存储

    Examples
    --------
    >>> incar = IncarData.from_file("./U-dimer-1.89-9au/INPUT")
    >>> params = incar.get_dict()
    >>> params["ecutwfc"]
    150
    >>> print(incar.pk)  # 存储后才有 pk
    """

    @classmethod
    def from_file(cls, filepath: str) -> "IncarData":
        """
        从 INPUT 文件创建 IncarData。

        Parameters
        ----------
        filepath : str
            INPUT 文件路径

        Returns
        -------
        IncarData
            AiiDA Dict 节点 (未存储)
        """
        params = parse_incar_to_dict(filepath)
        return cls(dict=params)

    @classmethod
    def from_dict(cls, params: dict) -> "IncarData":
        """
        从 dict 创建 IncarData。

        Parameters
        ----------
        params : dict
            INPUT 参数

        Returns
        -------
        IncarData
        """
        return cls(dict=params)

    def to_file(self, filepath: str) -> str:
        """
        序列化为 INPUT 文件文本并写入。

        Parameters
        ----------
        filepath : str
            输出文件路径

        Returns
        -------
        str
            输出文件的绝对路径
        """
        text = self.to_text()
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w") as f:
            f.write(text)
        return os.path.abspath(filepath)

    def to_text(self) -> str:
        """
        序列化为 ABACUS INPUT 文本。

        Returns
        -------
        str
            INPUT 文件内容
        """
        out = "INPUT_PARAMETERS\n"
        for key, value in self.get_dict().items():
            if value is None:
                continue
            if isinstance(value, list):
                value = " ".join(str(v) for v in value)
            out += f"{key:<20} {str(value)}\n"
        return out

    @classmethod
    def from_text(cls, text: str) -> "IncarData":
        """
        从 INPUT 文本创建 IncarData。

        Parameters
        ----------
        text : str
            INPUT 文件内容

        Returns
        -------
        IncarData
        """
        result = {}
        pattern = re.compile(r"^\s*([\w_]+)\s+([^\#]+)")
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line == "INPUT_PARAMETERS":
                continue
            match = pattern.match(line)
            if match is not None:
                key = match.group(1)
                raw_value = match.group(2).strip()
                result[key] = _convert_value(raw_value)
        return cls(dict=result)

    def get(self, key: str, default: Any = None) -> Any:
        """获取参数, 支持默认值."""
        return self.get_dict().get(key, default)
