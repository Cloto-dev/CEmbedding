"""The version is declared in one place, and everything else derives from it.

`cembedding/__init__.py` and `pyproject.toml` each carried a version string until
0.7.0, and they disagreed for four releases: nothing imports `__version__`, and no
gate compared it with the packaging version, so the drift was silent. pyproject now
derives the version (hatch dynamic version) and this test holds that wiring — a
static `version = "..."` reintroduced under `[project]` makes it fail.

Whether the derivation *works* is checked by installing: the CI job's
`pip install -e ".[dev]"` fails outright if hatch cannot resolve the version, and
the release workflow additionally refuses to publish artifacts whose filenames do
not carry the tag version. This file guards the shape those two rely on.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_pyproject_derives_the_version() -> None:
    # Read as lines rather than parsed TOML: tomllib is 3.11+, this package
    # supports 3.10, and the release gate greps these same line shapes.
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert not re.search(r"^version = ", pyproject, re.MULTILINE), "the version belongs in cembedding/__init__.py only"
    assert re.search(r'^dynamic = \["version"\]$', pyproject, re.MULTILINE), (
        "declare version as dynamic so it is derived"
    )
    assert re.search(
        r'^\[tool\.hatch\.version\]\npath = "cembedding/__init__\.py"$',
        pyproject,
        re.MULTILINE,
    ), "point hatch at the file that declares the version"


def test_the_declared_version_is_readable_the_way_the_release_gate_reads_it() -> None:
    """The workflow greps the file rather than importing it; keep that grep true."""
    init = (REPO_ROOT / "cembedding" / "__init__.py").read_text(encoding="utf-8")
    matches = re.findall(r'^__version__ = "(.*)"$', init, re.MULTILINE)

    assert len(matches) == 1, f"expected exactly one __version__ line, found {len(matches)}"
    assert re.fullmatch(r"\d+\.\d+\.\d+([abrc].*)?", matches[0]), matches[0]


def test_the_publish_gate_reads_the_declared_version() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "publish.yml").read_text(encoding="utf-8")

    assert "cembedding/__init__.py" in workflow
    assert not re.search(r"grep[^\n]*\^version = [^\n]*pyproject\.toml", workflow), (
        "the gate reads pyproject for a version that is no longer written there"
    )
