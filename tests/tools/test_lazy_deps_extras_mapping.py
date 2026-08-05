"""Every lazy feature must map to a real, resolvable pyproject extra.

``tools/lazy_deps.py`` no longer carries its own copy of each backend's pins;
it maps a feature name to a ``[project.optional-dependencies]`` extra and reads
the specs from pyproject.toml. That removes the drift between the two tables,
but introduces a new failure mode: a feature can point at an extra that doesn't
exist (typo, or an extra renamed/removed without updating the map), which fails
only at install time on whichever backend the user happens to enable.

These are invariants about how the two sides must relate, not snapshots of
their current contents — adding a backend should not require touching them.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

from tools import lazy_deps as ld


REPO_ROOT = Path(__file__).resolve().parents[2]


def _pyproject() -> dict:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _extras() -> dict:
    return _pyproject()["project"]["optional-dependencies"]


class TestFeatureExtraMapping:
    def test_every_feature_maps_to_a_declared_extra(self):
        missing = {
            feature: extra
            for feature, extra in ld.LAZY_FEATURES.items()
            if extra not in _extras()
        }
        assert not missing, (
            "these lazy features map to extras that don't exist in "
            f"pyproject.toml: {missing}"
        )

    def test_every_feature_resolves_to_at_least_one_spec(self):
        """A feature resolving to nothing would 'install' silently and then
        fail on import — the exact failure the mapping is meant to prevent."""
        empty = []
        for feature in ld.LAZY_FEATURES:
            try:
                if not ld.feature_specs(feature):
                    empty.append(feature)
            except ld.FeatureUnavailable:
                empty.append(feature)
        assert not empty, f"features resolving to no packages: {empty}"

    def test_resolved_specs_are_valid_requirements(self):
        """Composition rewrites markers, so the output must still be PEP 508."""
        packaging = pytest.importorskip("packaging.requirements")
        bad = []
        for feature in ld.LAZY_FEATURES:
            for spec in ld.feature_specs(feature):
                try:
                    packaging.Requirement(spec)
                except Exception as e:
                    bad.append((feature, spec, str(e)))
        assert not bad, f"invalid requirement strings after resolution: {bad}"

    def test_resolved_specs_pass_the_allowlist_guard(self):
        """Specs flow into an install command, so each must clear _SAFE_SPEC.

        A composed extra that smuggled in a URL or path would otherwise reach
        the installer.
        """
        unsafe = []
        for feature in ld.LAZY_FEATURES:
            for spec in ld.feature_specs(feature):
                head = spec.split(";", 1)[0].strip()
                if not ld._SAFE_SPEC.match(head):
                    unsafe.append((feature, spec))
        assert not unsafe, f"specs rejected by the safety guard: {unsafe}"

    def test_no_feature_maps_to_an_extra_that_is_entirely_core(self):
        """An extra that only restates core dependencies is dead weight.

        ``tool.vision`` -> ``[vision]`` -> ``Pillow`` was exactly this after
        Pillow was promoted to a core dep: the lazy path could never install
        anything not already present.
        """
        core = {
            _canonical_name(d)
            for d in _pyproject()["project"]["dependencies"]
        }
        redundant = []
        for feature in ld.LAZY_FEATURES:
            names = {_canonical_name(s) for s in ld.feature_specs(feature)}
            if names and names <= core:
                redundant.append((feature, sorted(names)))
        assert not redundant, (
            "these features can only install packages that are already core "
            f"dependencies, so the lazy path is dead code: {redundant}"
        )


def _canonical_name(spec: str) -> str:
    head = spec.split(";", 1)[0].split("@", 1)[0].split("[", 1)[0]
    return re.sub(r"[-_.]+", "-", re.split(r"[=<>!~]", head, maxsplit=1)[0].strip().lower())


class TestExtraComposition:
    def test_self_references_resolve(self):
        """No resolved spec may still be an unexpanded ``hermes-agent[...]``."""
        leaked = []
        for extra in _extras():
            for spec in ld.extra_specs(extra):
                if spec.lower().replace("_", "-").startswith("hermes-agent["):
                    leaked.append((extra, spec))
        assert not leaked, f"unexpanded self-references: {leaked}"

    def test_cycles_terminate(self, monkeypatch):
        """A cyclic composition must return, not recurse forever."""
        table = {
            "cyc-a": ("hermes-agent[cyc-b]", "pkg-a==1.0"),
            "cyc-b": ("hermes-agent[cyc-a]", "pkg-b==1.0"),
        }
        monkeypatch.setattr(ld, "_optional_dependencies", lambda: table)
        got = ld.extra_specs("cyc-a")
        assert "pkg-a==1.0" in got and "pkg-b==1.0" in got

    def test_unknown_extra_resolves_to_nothing(self, monkeypatch):
        monkeypatch.setattr(ld, "_optional_dependencies", lambda: {})
        assert ld.extra_specs("no-such-extra") == ()

    def test_marker_on_self_reference_is_distributed(self, monkeypatch):
        """``hermes-agent[x]; marker`` must apply the marker to x's contents.

        Dropping it would install a platform-gated package everywhere; dropping
        the whole entry would never install it at all.
        """
        table = {
            "base": ("pkg==1.0",),
            "outer": ("hermes-agent[base]; platform_system == 'Darwin'",),
        }
        monkeypatch.setattr(ld, "_optional_dependencies", lambda: table)
        got = ld.extra_specs("outer")
        assert got == ("pkg==1.0; platform_system == 'Darwin'",)

    def test_nested_markers_are_conjoined(self, monkeypatch):
        table = {
            "base": ("pkg==1.0; python_version >= '3.11'",),
            "outer": ("hermes-agent[base]; platform_system == 'Darwin'",),
        }
        monkeypatch.setattr(ld, "_optional_dependencies", lambda: table)
        (got,) = ld.extra_specs("outer")
        packaging = pytest.importorskip("packaging.requirements")
        req = packaging.Requirement(got)
        marker = str(req.marker)
        assert "python_version" in marker and "Darwin" in marker
        assert " and " in marker
