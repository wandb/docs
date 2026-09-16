"""
Pytest configuration for readability tests.

Registers custom markers so ``pytest -m integration`` does not emit
``PytestUnknownMarkWarning``.
"""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: tests that need textstat and the docs-skills submodule.",
    )
