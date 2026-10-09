"""The package a reader installs is the package this repository builds.

PyPI's ``gnarl`` is an unrelated project. The distribution here is
``gnarl-client`` and the import name is ``gnarl``, and both halves of that have
to stay true together: a README that says ``pip install gnarl`` sends a reader
to somebody else's code under the name they expected to be ours.
"""

from __future__ import annotations

import re
from importlib import metadata
from pathlib import Path

import gnarl

ROOT = Path(__file__).resolve().parent.parent
DIST = "gnarl-client"


def _pyproject_field(name: str) -> str:
    # tomllib is 3.11+, and the floor is 3.10. The two fields read here are
    # plain `key = "value"` lines under [project].
    text = (ROOT / "pyproject.toml").read_text()
    project = text.split("[project]", 1)[1].split("\n[", 1)[0]
    match = re.search(rf'^{name}\s*=\s*"([^"]+)"', project, re.MULTILINE)
    assert match, f"pyproject.toml has no [project] {name}"
    return match.group(1)


def test_the_distribution_is_gnarl_client():
    assert _pyproject_field("name") == DIST


def test_the_runtime_version_matches_the_build():
    """`gnarl.__version__` is what a bug report quotes; it must be what shipped."""
    assert gnarl.__version__ == _pyproject_field("version")


def test_the_installed_distribution_is_this_one():
    """An editable install of this checkout registers as `gnarl-client`.

    If this fails with PackageNotFoundError, the environment holds an install
    from before the rename: reinstall with ``pip install -e '.[dev]'``.
    """
    assert metadata.version(DIST) == gnarl.__version__


def test_no_document_tells_a_reader_to_install_the_wrong_package():
    """Every `pip install` line names `gnarl-client`, never bare `gnarl`."""
    offenders = []
    for path in [ROOT / "README.md", *ROOT.glob("docs/**/*.md")]:
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if re.search(r"pip install\s+(-U\s+)?['\"]?gnarl(?![-\w])", line):
                offenders.append(f"{path.name}:{n}: {line.strip()}")
    assert not offenders, "installs the unrelated PyPI project:\n" + "\n".join(offenders)
