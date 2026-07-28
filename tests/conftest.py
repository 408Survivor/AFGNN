"""Pytest bootstrap for the AFGNN test suite.

The project is not yet an installable package (see engineering_review.md P1-4),
and the scripts are run as ``python src/xxx.py`` with ``src/`` on ``sys.path``.
We mirror that here so tests can ``from utils.builders import ...`` and
``from models.afgnn import ...`` when running ``pytest`` from the repo root.
"""

import os
import sys

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
