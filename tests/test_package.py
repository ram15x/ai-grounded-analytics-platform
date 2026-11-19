"""Skeleton check only: the package imports and declares its version."""

import verified_analyst


def test_package_imports_with_version() -> None:
    assert verified_analyst.__version__ == "0.0.0"
