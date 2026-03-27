"""
Pytest configuration for knowledgebase-nav tests.

Registers custom markers so ``pytest -m integration`` does not emit
``PytestUnknownMarkWarning``.
"""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: tests that use the real repository layout (golden-file suite).",
    )
