"""The package imports and declares its version."""

import grounded_analytics


def test_package_imports_with_version() -> None:
    assert grounded_analytics.__version__ == "1.0.0"
