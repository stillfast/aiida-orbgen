"""CLI implementations for aiida_orbgen.

The top-level ``aiida-orbgen`` command is registered in ``pyproject.toml``:

.. code-block:: toml

    [project.scripts]
    aiida-orbgen = "aiida_orbgen.cli.run:main"
"""

from aiida_orbgen.cli.run import main

__all__ = ["main"]
