"""
IncarData: AiiDA data node - converts an INPUT (ABACUS INCAR) file into a Dict

This module converts an ABACUS INPUT file into an AiiDA Dict node.

ABACUS INPUT file format:
    INPUT_PARAMETERS
    <key1>     <value1>
    <key2>     <value2>
    ...

Data flow:
    INPUT file → parse_incar_to_dict() → Dict (AiiDA node)
    The dictionary can be retrieved from the node via .get_dict()
"""

import re
import os
from typing import Any, Union

from aiida import orm

__all__ = ["IncarData", "parse_incar_to_dict", "incar_to_dict", "read_incar"]


def parse_incar_to_dict(incar_path: str) -> dict:
    """
    Parse an ABACUS INPUT file, skipping the leading "INPUT_PARAMETERS" line.

    Parameters
    ----------
    incar_path : str
        Path to the INPUT file

    Returns
    -------
    dict
        The parsed parameter dictionary

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
            # Skip blank lines
            if not line:
                continue
            # Skip the leading "INPUT_PARAMETERS" line
            if line == "INPUT_PARAMETERS" or line.startswith("INPUT_PARAMETERS"):
                continue
            # Skip comment lines
            if line.startswith("#") or line.startswith("//"):
                continue

            match = pattern.match(line)
            if match is not None:
                key = match.group(1)
                raw_value = match.group(2).strip()
                # Convert the type: try int / float / str
                value = _convert_value(raw_value)
                result[key] = value

    return result


def _convert_value(raw_value: str) -> Union[int, float, str]:
    """
    Convert a string value to the appropriate Python type.

    Rules:
    - A plain integer (e.g. "100", "-5") becomes int
    - A float (e.g. "1.0", "1.0e-7") becomes float
    - Anything else stays a str
    """
    # Try int
    try:
        return int(raw_value)
    except ValueError:
        pass

    # Try float
    try:
        return float(raw_value)
    except ValueError:
        pass

    # Keep as str
    return raw_value


def incar_to_dict(incar_path: str) -> dict:
    """Alias for ``parse_incar_to_dict``."""
    return parse_incar_to_dict(incar_path)


def read_incar(incar_path: str) -> dict:
    """Alias for ``parse_incar_to_dict``."""
    return parse_incar_to_dict(incar_path)


class IncarData(orm.Dict):
    """
    AiiDA Dict node used to store ABACUS INPUT parameters.

    Inherits from aiida.orm.Dict and provides:
    - Creation from an INPUT file
    - Conversion to and from a dict
    - Storage in the AiiDA database

    Examples
    --------
    >>> incar = IncarData.from_file("./U-dimer-1.89-9au/INPUT")
    >>> params = incar.get_dict()
    >>> params["ecutwfc"]
    150
    >>> print(incar.pk)  # pk exists only after storing
    """

    @classmethod
    def from_file(cls, filepath: str) -> "IncarData":
        """
        Create an IncarData from an INPUT file.

        Parameters
        ----------
        filepath : str
            Path to the INPUT file

        Returns
        -------
        IncarData
            An AiiDA Dict node (unstored)
        """
        params = parse_incar_to_dict(filepath)
        return cls(dict=params)

    @classmethod
    def from_dict(cls, params: dict) -> "IncarData":
        """
        Create an IncarData from a dict.

        Parameters
        ----------
        params : dict
            INPUT parameters

        Returns
        -------
        IncarData
        """
        return cls(dict=params)

    def to_file(self, filepath: str) -> str:
        """
        Serialise to INPUT file text and write it out.

        Parameters
        ----------
        filepath : str
            Output file path

        Returns
        -------
        str
            Absolute path of the output file
        """
        text = self.to_text()
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w") as f:
            f.write(text)
        return os.path.abspath(filepath)

    def to_text(self) -> str:
        """
        Serialise to ABACUS INPUT text.

        Returns
        -------
        str
            INPUT file content
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
        Create an IncarData from INPUT text.

        Parameters
        ----------
        text : str
            INPUT file content

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
        """Get a parameter, with support for a default value."""
        return self.get_dict().get(key, default)
