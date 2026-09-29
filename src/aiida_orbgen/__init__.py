"""
aiida-orbgen - AiiDA plugin for ABACUS orbital generation

This plugin provides AiiDA support for the ABACUS-CSW-NAO orbital generation workflow.
"""

from aiida import load_profile

load_profile()

__version__ = "0.1.0"

__all__ = ["__version__"]
