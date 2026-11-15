"""pytest setup: run every test from the repository root, with the package importable.

The tests read ``data/`` and ``results/`` by relative paths, as when they are run as scripts
(``PYTHONPATH=. python tests/test_x.py``).
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
