"""The package a reader installs is the package this repository builds.

PyPI's ``gnarl`` is an unrelated project. The distribution here is
``gnarl-client`` and the import name is ``gnarl``, and both halves of that have
to stay true together: a README that says ``pip install gnarl`` sends a reader
to somebody else's code under the name they expected to be ours.
"""

from __future__ import annotations

import re
import subprocess
import sys
import zipfile
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


def _build_wheel(tmp_path: Path) -> Path:
    out = tmp_path / "dist"
    subprocess.run(
        [sys.executable, "-m", "hatchling", "build", "-t", "wheel", "-d", str(out)],
        cwd=ROOT, check=True, capture_output=True,
    )
    (wheel,) = out.glob("gnarl_client-*.whl")
    return wheel


def test_the_wheel_ships_py_typed(tmp_path):
    """gnarl-client 0.2.0 declared ``Typing :: Typed`` and shipped no
    ``py.typed``, so mypy and pyright treated the whole package as untyped —
    every model and method an ``Any`` — in exactly the projects that asked for
    types. PEP 561 makes the marker file the switch, not the classifier.

    Built, not just checked on disk: the file has to survive the wheel's
    include rules, which is where it would be lost.
    """
    with zipfile.ZipFile(_build_wheel(tmp_path)) as whl:
        names = set(whl.namelist())
    assert "gnarl/py.typed" in names
    # And what claims the types still ships beside it.
    assert {"gnarl/__init__.py", "gnarl/_models.py", "gnarl/openapi.yaml"} <= names
