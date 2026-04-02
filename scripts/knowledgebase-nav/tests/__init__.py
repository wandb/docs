# ---------------------------------------------------------------------------
# This file marks the tests/ directory as a Python package.
#
# It must exist (even empty) so that pytest and Python's import system
# can resolve relative imports within the test suite.  Without it,
# running ``pytest scripts/knowledgebase-nav/tests/`` from the repo root
# would fail with ModuleNotFoundError because Python would not recognise
# this directory as a package.
#
# No code is needed here. The file's presence is sufficient.
# ---------------------------------------------------------------------------
