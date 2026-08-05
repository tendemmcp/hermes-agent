"""Lazy installs must not downgrade security-pinned core packages.

``uv pip install`` / ``pip install`` do not read ``[tool.uv]
override-dependencies`` from pyproject.toml. A backend whose transitive deps
cap a security-pinned package below its patched floor therefore *downgrades*
the core venv the first time that backend is enabled.

The measured case: the core venv ships ``cryptography==50.0.0``; enabling
DingTalk pulls ``alibabacloud-dingtalk`` -> ``alibabacloud-tea-openapi==0.4.5``,
which caps ``cryptography<49``, and the install resolves ``cryptography`` back
to 48.0.1 — re-introducing GHSA-m2h6-j472-rp4c, GHSA-jwv3-5hgf-82ww and
CVE-2026-69247.

``tools/lazy_deps.py`` guards this by passing an overrides file to every lazy
install. These tests assert the *contract* — an override exists for anything
pinned in pyproject, and both installer tiers receive it — rather than
snapshotting the current override list.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

from tools import lazy_deps as ld


REPO_ROOT = Path(__file__).resolve().parents[2]


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def _requirement_name(spec: str) -> str:
    head = spec.split(";", 1)[0].split("@", 1)[0].split("[", 1)[0]
    return _canonical(re.split(r"[=<>!~]", head, maxsplit=1)[0])


def _pyproject() -> dict:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


class TestOverridesMirrorPyproject:
    """The lazy overrides must not drift from ``[tool.uv]``."""

    def test_every_uv_override_has_a_lazy_override(self):
        """Anything forced in ``[tool.uv] override-dependencies`` must also be
        forced on the lazy path.

        ``[tool.uv]`` only governs ``uv lock`` / ``uv sync``. An override that
        exists there but not here means the eager install is patched while the
        lazy install silently is not — precisely the DingTalk/cryptography bug.
        """
        uv_overrides = _pyproject().get("tool", {}).get("uv", {}).get(
            "override-dependencies", []
        )
        forced = {_requirement_name(s) for s in ld._SECURITY_OVERRIDES}

        missing = []
        for spec in uv_overrides:
            name = _requirement_name(spec)
            # pynacl's override exists to dodge a stale cap in discord.py's
            # metadata, and discord.py's own lazy specs already pin the fixed
            # version directly, so it needs no lazy override. Only packages
            # that are ALSO exact-pinned as core deps can be downgraded out
            # from under a running install.
            core = {
                _requirement_name(d)
                for d in _pyproject()["project"]["dependencies"]
            }
            if name in core and name not in forced:
                missing.append(name)

        assert not missing, (
            "these packages are overridden in [tool.uv] and pinned as core "
            f"dependencies, but lazy installs can still downgrade them: {missing}. "
            "Add a matching entry to tools/lazy_deps._SECURITY_OVERRIDES."
        )

    def test_overrides_are_valid_requirement_specs(self):
        for spec in ld._SECURITY_OVERRIDES:
            assert _requirement_name(spec), f"unparseable override spec: {spec!r}"
            assert any(op in spec for op in ("==", ">=", "<", ">", "~=")), (
                f"override {spec!r} has no version constraint — it would not "
                "force anything"
            )

    def test_override_floor_is_not_below_the_core_pin(self):
        """An override must never permit a version older than the core pin.

        If pyproject pins ``cryptography==50.0.0`` but the lazy override says
        ``>=48``, the lazy path can still install 48 and undo the pin.
        """
        core = {}
        for dep in _pyproject()["project"]["dependencies"]:
            m = re.match(r"^\s*([A-Za-z0-9._-]+)\s*==\s*([0-9][^\s,;]*)", dep)
            if m:
                core[_canonical(m.group(1))] = m.group(2)

        for spec in ld._SECURITY_OVERRIDES:
            name = _requirement_name(spec)
            pinned = core.get(name)
            if not pinned:
                continue
            lower = re.search(r">=\s*([0-9][^\s,]*)", spec)
            assert lower, (
                f"{name} is exact-pinned to {pinned} in pyproject but its lazy "
                f"override {spec!r} has no lower bound"
            )
            got = tuple(int(x) for x in re.findall(r"\d+", lower.group(1)))
            want = tuple(int(x) for x in re.findall(r"\d+", pinned))
            assert got >= want[: len(got)], (
                f"lazy override for {name} allows {lower.group(1)}, which is "
                f"below the core pin {pinned}"
            )


class TestOverridesReachBothInstallerTiers:
    """Both tiers of the install ladder must receive the floor."""

    @pytest.fixture
    def captured(self, monkeypatch, tmp_path):
        """Run ``_venv_pip_install`` with both tiers stubbed, capturing argv.

        Temp files are read *during* the stubbed call, because
        ``_venv_pip_install`` unlinks them in its ``finally`` block.
        """
        calls: list[list[str]] = []
        contents: dict[str, str] = {}

        def fake_run(cmd, *a, **kw):
            cmd = list(cmd)
            calls.append(cmd)
            for flag in ("--overrides", "--constraint"):
                if flag in cmd:
                    p = Path(cmd[cmd.index(flag) + 1])
                    if p.exists():
                        contents[flag] = p.read_text(encoding="utf-8")

            class R:
                # uv tier fails so the ladder falls through to pip; the pip
                # probe and the pip install itself succeed, so the --no-deps
                # repair pass runs.
                returncode = 1 if (cmd and "uv" in cmd[0]) else 0
                stdout = "pip 24.0"
                stderr = "stubbed"

            return R()

        monkeypatch.setattr(ld.subprocess, "run", fake_run)
        monkeypatch.setattr(ld.shutil, "which", lambda _n: "/usr/bin/uv")
        monkeypatch.delenv(ld._LAZY_TARGET_ENV, raising=False)
        ld._venv_pip_install(("alibabacloud-dingtalk==2.2.42",))
        return calls, contents

    def test_uv_tier_receives_overrides_flag(self, captured):
        calls, contents = captured
        uv_calls = [c for c in calls if "uv" in c[0] and "pip" in c]
        assert uv_calls, f"no uv tier invocation captured: {calls}"
        cmd = uv_calls[0]
        assert "--overrides" in cmd, (
            f"uv tier must pass --overrides so [tool.uv] semantics apply: {cmd}"
        )
        body = contents.get("--overrides", "")
        for spec in ld._SECURITY_OVERRIDES:
            assert spec in body, (
                f"override {spec!r} missing from the file handed to uv: {body!r}"
            )

    def test_pip_tier_reasserts_the_floor_with_no_deps(self, captured):
        """pip has no --overrides; it must re-assert the floor via --no-deps.

        Passing the floor as a --constraint instead would hold the pinned
        package but resolve the *backend* backwards, so the repair pass is the
        behaviour under test.
        """
        calls, _ = captured
        repair = [
            c for c in calls if "install" in c and "--no-deps" in c
        ]
        assert repair, (
            f"pip tier must re-assert security overrides with --no-deps: {calls}"
        )
        cmd = repair[0]
        for spec in ld._SECURITY_OVERRIDES:
            assert spec in cmd, (
                f"override {spec!r} missing from the pip repair pass: {cmd}"
            )

    def test_pip_repair_pass_does_not_reinstall_the_backend(self, captured):
        """The repair pass must touch only the overridden packages.

        Including the backend specs would re-run resolution and undo the point
        of --no-deps.
        """
        calls, _ = captured
        repair = [c for c in calls if "install" in c and "--no-deps" in c]
        assert repair
        assert "alibabacloud-dingtalk==2.2.42" not in repair[0], (
            f"repair pass must not re-install the backend: {repair[0]}"
        )

    def test_temp_files_are_cleaned_up(self, captured):
        calls, _ = captured
        for cmd in calls:
            for flag in ("--overrides", "--constraint"):
                if flag in cmd:
                    leaked = Path(cmd[cmd.index(flag) + 1])
                    assert not leaked.exists(), (
                        f"{flag} temp file leaked after install: {leaked}"
                    )

    def test_specs_still_reach_the_installer(self, captured):
        """The override plumbing must not displace the actual packages."""
        calls, _ = captured
        # Exclude the --no-deps repair pass, which deliberately carries only
        # the overridden packages (see test_pip_repair_pass_does_not_reinstall).
        installs = [
            c for c in calls if "install" in c and "--no-deps" not in c
        ]
        assert installs, f"no install invocation captured: {calls}"
        for cmd in installs:
            assert "alibabacloud-dingtalk==2.2.42" in cmd, (
                f"requested spec missing from install command: {cmd}"
            )
