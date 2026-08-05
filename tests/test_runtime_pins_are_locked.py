"""Runtime-installed pins must be declared in pyproject, hence in uv.lock.

``uv audit`` and the OSV-Scanner CI job both scan ``uv.lock``. A package that
some module installs at runtime from a hardcoded string — but that pyproject
never declares — is therefore invisible to every vulnerability scanner we run,
while still landing in real user environments.

That is not hypothetical: ``plugins/platforms/google_chat/oauth.py`` pinned
``google-cloud-pubsub==2.39.0`` in a module-level list and pip-installed it on
setup, yet the package appeared nowhere in pyproject.toml or uv.lock. It was
unscanned for its whole life in the tree.

This test scans first-party source for ``"name==version"`` literals and
requires each one to be a declared dependency. Fixing a failure means adding
the package to the appropriate extra (and re-running ``uv lock``) — not
adding an exemption.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]

# First-party trees whose code can trigger a runtime install.
_SOURCE_DIRS = (
    "agent",
    "cron",
    "gateway",
    "hermes_cli",
    "plugins",
    "scripts",
    "skills",
    "tools",
)

# "pkg==1.2.3" / "pkg[extra]==1.2.3" inside a string literal.
_PINNED_LITERAL = re.compile(
    r"""["']([A-Za-z][A-Za-z0-9._-]*)(?:\[[A-Za-z0-9,._-]+\])?==([0-9][^"']*)["']"""
)

# Literals that look like pins but aren't dependency declarations.
_IGNORE_NAMES = frozenset(
    {
        "python",  # `python==3.11` style constraints
    }
)


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def _declared_packages() -> set[str]:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]
    specs = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        specs.extend(extra)
    for group in data.get("dependency-groups", {}).values():
        specs.extend(s for s in group if isinstance(s, str))
    # [tool.uv] overrides are declarations too.
    specs.extend(data.get("tool", {}).get("uv", {}).get("override-dependencies", []))

    names = set()
    for spec in specs:
        head = spec.split(";", 1)[0].split("@", 1)[0].split("[", 1)[0]
        name = re.split(r"[=<>!~]", head, maxsplit=1)[0].strip()
        if name:
            names.add(_canonical(name))
    return names


def _locked_packages() -> set[str]:
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
    return {_canonical(p["name"]) for p in lock.get("package", [])}


def _iter_source_files():
    for rel in _SOURCE_DIRS:
        root = REPO_ROOT / rel
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            parts = set(path.parts)
            if "__pycache__" in parts or ".venv" in parts or "node_modules" in parts:
                continue
            yield path


def _pinned_literals_in_source() -> dict[str, set[str]]:
    """Map canonical package name -> {"relpath:line", ...} for pinned literals."""
    found: dict[str, set[str]] = {}
    for path in _iter_source_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if "==" not in text:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for match in _PINNED_LITERAL.finditer(line):
                name = _canonical(match.group(1))
                if name in _IGNORE_NAMES:
                    continue
                rel = path.relative_to(REPO_ROOT)
                found.setdefault(name, set()).add(f"{rel}:{lineno}")
    return found


@pytest.fixture(scope="module")
def pinned_literals():
    literals = _pinned_literals_in_source()
    assert literals, "scanner found no pinned literals at all — the regex drifted"
    return literals


def test_runtime_pins_are_declared_in_pyproject(pinned_literals):
    declared = _declared_packages()
    undeclared = {
        name: sorted(locs)
        for name, locs in pinned_literals.items()
        if name not in declared
    }
    assert not undeclared, (
        "these packages are pinned in first-party source but are not declared "
        "in pyproject.toml, so nothing installs them through the normal "
        "dependency path and no scanner sees them:\n"
        + "\n".join(
            f"  {name}\n" + "\n".join(f"      {loc}" for loc in locs)
            for name, locs in sorted(undeclared.items())
        )
        + "\n\nAdd each to the appropriate extra and re-run `uv lock`."
    )


def test_runtime_pins_are_present_in_the_lockfile(pinned_literals):
    """Declared isn't enough — the scanners read uv.lock specifically.

    A dependency behind an environment marker that no supported platform
    satisfies would be declared yet absent from the lock, and still unscanned.
    """
    locked = _locked_packages()
    missing = {
        name: sorted(locs)
        for name, locs in pinned_literals.items()
        if name not in locked
    }
    assert not missing, (
        "these packages are pinned in source but absent from uv.lock, so "
        "`uv audit` and the OSV-Scanner CI job cannot see them:\n"
        + "\n".join(
            f"  {name}\n" + "\n".join(f"      {loc}" for loc in locs)
            for name, locs in sorted(missing.items())
        )
        + "\n\nRun `uv lock` after declaring them in pyproject.toml."
    )


def test_source_pins_agree_with_the_lockfile_version(pinned_literals):
    """A hardcoded pin must not disagree with the version uv.lock resolves.

    Drift here means the runtime install downgrades or churns a package the
    rest of the tree expects at a different version.
    """
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
    locked_versions: dict[str, set[str]] = {}
    for pkg in lock.get("package", []):
        locked_versions.setdefault(_canonical(pkg["name"]), set()).add(pkg["version"])

    drift = []
    for path in _iter_source_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if "==" not in text:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for match in _PINNED_LITERAL.finditer(line):
                name = _canonical(match.group(1))
                if name in _IGNORE_NAMES:
                    continue
                version = match.group(2).split(",", 1)[0].strip()
                locked = locked_versions.get(name)
                if locked and version not in locked:
                    rel = path.relative_to(REPO_ROOT)
                    drift.append(
                        f"  {name}=={version} at {rel}:{lineno} "
                        f"(uv.lock has {sorted(locked)})"
                    )
    assert not drift, (
        "hardcoded pins disagree with the locked version:\n" + "\n".join(sorted(drift))
    )
