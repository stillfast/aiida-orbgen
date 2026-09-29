"""
Pytest configuration for aiida_orbgen tests.
"""

import pytest


@pytest.fixture(scope="session")
def aiida_profile():
    """Load AiiDA test profile."""
    from aiida import load_profile
    load_profile("test_profile")
