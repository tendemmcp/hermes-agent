"""
Lazy dependency installer for opt-in Hermes Agent backends.

Many Hermes features (Mistral TTS, ElevenLabs TTS, Honcho memory, Bedrock,
Slack, Matrix, etc.) require Python packages that not every user needs. The
historical approach was to bundle them all under ``pyproject.toml`` extras
(``hermes-agent[all]``) and install them eagerly at setup time. That has
two problems:

1. **Fragility.** When one extra's transitive dependency becomes
   unavailable on PyPI (quarantined for malware, yanked, broken upload),
   the *entire* ``[all]`` resolve fails and fresh installs silently fall
   back to a stripped tier — losing 10+ unrelated extras at once.

2. **Bloat.** A user who only ever talks to one provider pulls hundreds
   of packages they will never import.

The lazy-install pattern fixes both. Backends call :func:`ensure` at the
top of their first-import path. If the deps are missing, ``ensure`` checks
the ``security.allow_lazy_installs`` config flag (default true) and runs
a venv-scoped pip install. If the user has explicitly disabled lazy
installs, ``ensure`` raises :class:`FeatureUnavailable` with a clear
remediation hint pointing at ``hermes tools`` or the manual pip command.

Security model:

* **Venv-scoped by default.** Installs target ``sys.executable`` in the
  active venv. We never touch the system Python.
* **Durable-target mode (immutable images).** When the deployment seals the
  agent's own venv (the Docker image sets ``HERMES_DISABLE_LAZY_INSTALLS=1``
  and makes ``/opt/hermes`` read-only), setting
  ``HERMES_LAZY_INSTALL_TARGET`` redirects lazy installs to a writable
  directory on the durable data volume (e.g. ``/opt/data/lazy-packages``).
  That directory is **appended to the end of ``sys.path``** — never
  prepended, never exported via ``PYTHONPATH`` — so the agent's own
  site-packages wins every name collision. A package installed this way can
  only ADD new importable modules; it can never shadow, downgrade, or break
  a module the core already ships. The worst a bad/incompatible backend
  package can do is fail to import and report itself unavailable — the agent
  core stays healthy. This is the structural guarantee that a lazily
  installed package cannot brick Hermes, which is what made it safe to seal
  the venv in the first place. Compiled-wheel safety across image rebuilds
  is handled by an ABI/Python-version stamp on the target subdir (see
  :func:`_ensure_target_ready`).
* **PyPI by package name only.** Specs may be ``"package>=1.0,<2"`` etc.
  We do NOT support ``--index-url`` overrides, ``git+https://``, file:
  paths, or any other input that could be hijacked by a malicious config.
* **Allowlist.** Only specs that appear in :data:`LAZY_FEATURES` can be
  installed via this path. A typo in feature name doesn't get the user
  install-anything semantics.
* **Opt-out.** Setting ``security.allow_lazy_installs: false`` in
  ``config.yaml`` disables runtime installs in BOTH modes. Users in
  restricted networks or strict security postures can pin themselves to
  whatever was installed at setup time.
* **Offline detection.** If the install fails (offline, mirror down,
  PyPI 404 / quarantine), we surface the failure as
  :class:`FeatureUnavailable` with the actual pip stderr — no silent
  retries, no caching of bad state.

Adding a new backend:

1. Add the packages as an extra in pyproject.toml, then map the feature
   to that extra in :data:`LAZY_FEATURES`.
2. At the top of the backend module's import path, call
   ``ensure("feature.name")`` inside a try/except that converts
   :class:`FeatureUnavailable` to a useful runtime error.
"""

from __future__ import annotations

import functools
import logging
import os
import re
import shutil
import site
import subprocess
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from hermes_cli._subprocess_compat import windows_hide_flags

logger = logging.getLogger(__name__)


# =============================================================================
# Feature -> pyproject extra mapping.
#
# Keys are dot-separated feature names ("namespace.backend"). Values are the
# name of the ``[project.optional-dependencies]`` extra in pyproject.toml that
# carries that backend's packages.
#
# The specs themselves live in pyproject.toml and NOWHERE else. This module
# used to keep its own copy of every pin, which meant two sources of truth for
# the same dependency set and a family of drift-detector tests to keep them
# honest. Worse, that copy could not see ``[tool.uv] override-dependencies``,
# so a backend whose metadata capped a security-pinned package silently
# downgraded it on first use (cryptography 50.0.0 -> 48.0.1 via DingTalk).
# Reading the extra means a lazy install resolves exactly what `uv lock`
# audited.
# =============================================================================


LAZY_FEATURES: dict[str, str] = {
    # ─── Inference providers ───────────────────────────────────────────────
    "provider.anthropic": "anthropic",
    "provider.bedrock": "bedrock",
    "provider.vertex": "vertex",
    "provider.azure_identity": "azure-identity",

    # ─── Web search backends ───────────────────────────────────────────────
    "search.exa": "exa",
    "search.firecrawl": "firecrawl",
    "search.parallel": "parallel-web",

    # ─── Monitoring ────────────────────────────────────────────────────────
    "export.otlp": "otlp",

    # ─── Speech to text ────────────────────────────────────────────────────
    "stt.faster_whisper": "voice",
    "stt.mistral": "mistral",
    "stt.silk": "silk",

    # ─── Text to speech ────────────────────────────────────────────────────
    "tts.edge": "edge-tts",
    "tts.elevenlabs": "tts-premium",
    "tts.mistral": "mistral",

    # ─── Wake word engines ─────────────────────────────────────────────────
    "wake.openwakeword": "wake-openwakeword",
    "wake.openwakeword.tflite": "wake-tflite",
    "wake.sherpa": "wake-sherpa",
    "wake.porcupine": "wake-porcupine",

    # ─── Image generation backends ─────────────────────────────────────────
    "image.fal": "fal",

    # ─── Memory providers ──────────────────────────────────────────────────
    "memory.honcho": "honcho",
    "memory.hindsight": "hindsight",
    "memory.supermemory": "supermemory",
    "memory.mem0": "mem0",

    # ─── Messaging platforms ───────────────────────────────────────────────
    "platform.telegram": "telegram",
    "platform.discord": "discord",
    "platform.slack": "slack",
    "platform.matrix": "matrix",
    "platform.dingtalk": "dingtalk",
    "platform.feishu": "feishu",
    "platform.wecom_callback": "wecom",
    "platform.teams": "teams",

    # ─── Terminal backends ─────────────────────────────────────────────────
    "terminal.modal": "modal",
    "terminal.daytona": "daytona",
    "terminal.vercel": "vercel",

    # ─── Skills ────────────────────────────────────────────────────────────
    "skill.google_workspace": "google",
    "skill.youtube": "youtube",

    # ─── Tools ─────────────────────────────────────────────────────────────
    # NOTE: no "tool.acp" entry. [acp] ships eagerly in [all] and nothing ever
    # called ensure("tool.acp"), so the mapping was dead — and it put [acp] in
    # both [all] and the lazy map, which the lazy-install policy forbids
    # (test_lazy_installable_extras_excluded_from_all). Removed rather than
    # dropping [acp] from [all]: the ACP entry point is a console script, so
    # its dep must be present before the agent loop can lazy-install anything.
    "tool.dashboard": "web",
    "tool.computer_use": "computer-use",
    "tool.trace_upload": "trace-upload",
}


# =============================================================================
# pyproject extra -> specs
# =============================================================================


def _project_root() -> Optional[Path]:
    """Return the checkout root holding pyproject.toml, or None.

    Supported installs are git checkouts (``install.sh`` clones the repo) and
    the Docker image, which copies ``pyproject.toml`` + ``uv.lock`` to its
    WORKDIR. Anything else (a stray site-packages copy) has no project root and
    falls back to the vendored spec table.
    """
    root = Path(__file__).resolve().parent.parent
    return root if (root / "pyproject.toml").is_file() else None


@functools.lru_cache(maxsize=1)
def _optional_dependencies() -> dict[str, tuple[str, ...]]:
    """Parse ``[project.optional-dependencies]`` from pyproject.toml."""
    root = _project_root()
    if root is None:
        return {}
    try:
        import tomllib
    except ImportError:  # pragma: no cover - py<3.11, unsupported
        return {}
    try:
        data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    except Exception as e:
        logger.debug("Could not read pyproject.toml: %s", e)
        return {}
    raw = data.get("project", {}).get("optional-dependencies", {}) or {}
    return {k: tuple(v) for k, v in raw.items()}


_SELF_REF = re.compile(r"^hermes[-_]agent\[([^\]]+)\]$", re.IGNORECASE)


def _split_marker(spec: str) -> tuple[str, str]:
    """Split ``"pkg[extra]; marker"`` into ``("pkg[extra]", "; marker")``."""
    head, sep, tail = spec.partition(";")
    return head.strip(), (f";{tail}" if sep else "")


def extra_specs(extra: str, _seen: Optional[frozenset] = None) -> tuple[str, ...]:
    """Return the concrete specs for ``extra``, resolving ``hermes-agent[...]``.

    Extras compose: ``[messaging]`` is ``hermes-agent[telegram]`` +
    ``hermes-agent[discord]`` + ``hermes-agent[slack]``. Self-references are
    expanded recursively; a cycle (or a reference to an extra that doesn't
    exist) resolves to nothing rather than recursing forever.

    A marker on a self-reference is distributed over the expansion, so
    ``hermes-agent[wake-tflite]; platform_system == 'Darwin'`` yields
    ``ai-edge-litert==2.1.6; platform_system == 'Darwin'`` — the same set pip
    and uv would install.
    """
    seen = _seen or frozenset()
    if extra in seen:
        logger.debug("Cyclic extra reference at %r — stopping", extra)
        return ()
    table = _optional_dependencies()
    if extra not in table:
        return ()
    seen = seen | {extra}
    out: list[str] = []

    def _add(spec: str) -> None:
        if spec not in out:
            out.append(spec)

    for spec in table[extra]:
        head, marker = _split_marker(spec)
        m = _SELF_REF.match(head)
        if m:
            for sub in m.group(1).split(","):
                for nested in extra_specs(sub.strip(), seen):
                    if not marker:
                        _add(nested)
                        continue
                    n_head, n_marker = _split_marker(nested)
                    # Both sides carry a marker: they must BOTH hold.
                    _add(
                        f"{n_head}{n_marker} and{marker[1:]}"
                        if n_marker else f"{n_head}{marker}"
                    )
        else:
            _add(spec)
    return tuple(out)


def feature_extra(feature: str) -> str:
    """Return the pyproject extra backing ``feature``, or raise KeyError."""
    if feature not in LAZY_FEATURES:
        raise KeyError(f"Unknown lazy feature: {feature!r}")
    return LAZY_FEATURES[feature]


# Conservative regex for spec validation — package name plus optional
# version range. Reject anything that looks like a URL, file path, or shell
# metacharacter.
_SAFE_SPEC = re.compile(
    r"^[A-Za-z0-9_][A-Za-z0-9_.\-]*"        # package name
    r"(?:\[[A-Za-z0-9_,\-]+\])?"            # optional [extras]
    r"(?:[<>=!~]=?[A-Za-z0-9_.\-+,*<>=!~]+)?"  # optional version specifier
    r"$"
)


class FeatureUnavailable(RuntimeError):
    """A lazily-installable feature is missing and cannot be made available.

    Either the deps were never installed and the user has disabled lazy
    installs, or the install attempt failed.
    """

    def __init__(self, feature: str, missing: tuple[str, ...], reason: str):
        self.feature = feature
        self.missing = missing
        self.reason = reason
        super().__init__(self._format())

    def _format(self) -> str:
        spec_list = " ".join(repr(s) for s in self.missing)
        return (
            f"Feature {self.feature!r} unavailable: {self.reason}. "
            f"To enable manually: uv pip install {spec_list}  "
            f"(or: pip install {spec_list})."
        )


@dataclass(frozen=True)
class _InstallResult:
    success: bool
    stdout: str
    stderr: str


# =============================================================================
# Internals
# =============================================================================


# Environment variable that redirects lazy installs away from the (sealed)
# agent venv and into a writable directory on a durable volume. Set by the
# Docker image to /opt/data/lazy-packages. This is an internal bridge var,
# not user-facing config: the user-facing knob remains
# security.allow_lazy_installs in config.yaml. When unset, lazy installs go
# into the active venv as before.
_LAZY_TARGET_ENV = "HERMES_LAZY_INSTALL_TARGET"

# Name of the stamp file written into the target dir recording the Python
# X.Y + ABI it was populated for. If a container rebuild bumps the
# interpreter, compiled wheels (.so) in the durable store would be ABI-
# incompatible; we detect the mismatch and wipe the store so packages get
# re-resolved against the new interpreter rather than importing a stale .so.
_TARGET_STAMP_NAME = ".python-abi"


def _python_abi_tag() -> str:
    """A stable token identifying the running interpreter's ABI.

    Combines the X.Y version with the EXT_SUFFIX (which encodes the ABI
    tag and platform, e.g. ``cpython-313-x86_64-linux-gnu``). Two
    interpreters that can share compiled wheels produce the same token.
    """
    ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    ext = sysconfig.get_config_var("EXT_SUFFIX") or ""
    return f"{ver}:{ext}"


def _lazy_install_target() -> Optional[Path]:
    """Return the durable install-target dir, or None for venv-scoped mode.

    Returns a path only when :data:`_LAZY_TARGET_ENV` is set to a non-empty
    value. The directory is created on demand by :func:`_ensure_target_ready`.
    """
    raw = os.environ.get(_LAZY_TARGET_ENV, "").strip()
    if not raw:
        return None
    return Path(raw)


def _ensure_target_ready(target: Path) -> Optional[str]:
    """Create the target dir and validate its ABI stamp.

    If the stamp is missing it is written. If it is present but records a
    different interpreter ABI than the one now running (e.g. the container
    image was rebuilt onto a newer Python), the directory's contents are
    wiped and the stamp rewritten, so stale compiled wheels can't be
    imported against an incompatible interpreter.

    Returns ``None`` on success, or an error string if the directory can't
    be created / written (e.g. read-only mount, permission error).
    """
    want = _python_abi_tag()
    stamp = target / _TARGET_STAMP_NAME
    try:
        if target.exists():
            have = ""
            try:
                have = stamp.read_text(encoding="utf-8").strip()
            except (OSError, FileNotFoundError):
                have = ""
            if have and have != want:
                logger.info(
                    "Lazy install target %s was built for ABI %r but running "
                    "ABI is %r; wiping stale packages.",
                    target, have, want,
                )
                for child in target.iterdir():
                    if child.is_dir() and not child.is_symlink():
                        shutil.rmtree(child, ignore_errors=True)
                    else:
                        try:
                            child.unlink()
                        except OSError:
                            pass
        target.mkdir(parents=True, exist_ok=True)
        stamp.write_text(want, encoding="utf-8")
    except OSError as e:
        return f"lazy install target {target} is not writable: {e}"
    return None


def _activate_target_on_syspath(target: Path) -> None:
    """Append the durable target to ``sys.path`` so its packages import.

    Appended to the END (never prepended) so the agent's own venv
    site-packages takes precedence on every name collision. Idempotent.
    Uses :func:`site.addsitedir` so ``.pth`` files (namespace packages,
    editable installs) inside the target are honoured, then enforces the
    append ordering — ``addsitedir`` would otherwise insert near the front.
    """
    target_str = str(target)
    # Snapshot existing entries so we can restore precedence afterwards.
    before = list(sys.path)
    if target_str not in before:
        site.addsitedir(target_str)
    # site.addsitedir may have inserted target (and any .pth-added dirs) at
    # the front. Move every newly-added entry to the end, preserving the
    # core venv's precedence. New entries are those not present `before`.
    new_entries = [p for p in sys.path if p not in before]
    if new_entries:
        sys.path[:] = [p for p in sys.path if p not in new_entries] + new_entries
    # importlib.metadata caches the path-based distribution finder; clear it
    # so a just-activated dir is visible to version() checks this process.
    try:
        import importlib
        importlib.invalidate_caches()
    except Exception:
        pass


def activate_durable_lazy_target() -> None:
    """Public: wire the durable lazy-install target onto ``sys.path``.

    Safe no-op when :data:`_LAZY_TARGET_ENV` is unset or the directory does
    not yet exist. Called once early in process startup (before backends
    import) so packages installed into the durable store on a previous run
    are importable on this run. Never raises.
    """
    target = _lazy_install_target()
    if target is None:
        return
    try:
        if target.exists():
            _activate_target_on_syspath(target)
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("Failed to activate durable lazy target %s: %s", target, e)


def _allow_lazy_installs() -> bool:
    """Return whether lazy installs are permitted in this environment.

    Resolution order:

    1. ``security.allow_lazy_installs: false`` in config.yaml is an absolute
       opt-out — it disables installs in BOTH venv-scoped and durable-target
       modes. This is the user-facing kill switch.
    2. ``HERMES_DISABLE_LAZY_INSTALLS=1`` seals the *agent venv* (set by the
       immutable Docker image). It blocks venv-scoped installs — UNLESS a
       durable install target is configured, in which case installs are
       redirected there (a path that structurally cannot break the sealed
       venv) and are therefore allowed.

    Defaults to True. If config is unreadable we fail open (allow), because
    refusing to install would lock people out of their own backends; the
    decision to block is an explicit user opt-in.
    """
    # (1) Config kill switch wins in every mode.
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
    except Exception:
        cfg = None
    if cfg is not None:
        sec = cfg.get("security") or {}
        if not bool(sec.get("allow_lazy_installs", True)):
            return False

    # (2) Sealed-venv env var: blocks ONLY when there is no safe durable
    # target to redirect into. With a target set, the install goes to the
    # data volume (append-only on sys.path), so the seal is preserved.
    if os.environ.get("HERMES_DISABLE_LAZY_INSTALLS") == "1":
        return _lazy_install_target() is not None

    return True


def _unsupported_feature_reason(feature: str) -> Optional[str]:
    """Return why a lazy feature cannot work on this host, or ``None``.

    This is a platform capability gate, not a security policy gate. It keeps
    known-impossible installs out of both first-use lazy installation and the
    ``hermes update`` lazy-refresh pass.
    """
    if sys.platform == "win32" and feature == "platform.matrix":
        return (
            "unsupported on Windows: Matrix E2EE depends on python-olm, "
            "which has no Windows wheel and requires make + libolm to build "
            "from sdist. Run Hermes under WSL to use Matrix on Windows."
        )
    return None


def _spec_is_safe(spec: str) -> bool:
    """Reject pip specs that contain URLs, paths, or shell metacharacters."""
    if not spec or len(spec) > 200:
        return False
    if any(ch in spec for ch in (";", "|", "&", "`", "$", "\n", "\r", "\t", "\\")):
        return False
    if spec.startswith(("-", "/", ".")) or "://" in spec or "@" in spec:
        return False
    return bool(_SAFE_SPEC.match(spec))


def _pkg_name_from_spec(spec: str) -> str:
    """Extract the bare package name from a pip spec.

    ``"slack-bolt>=1.18.0,<2"`` → ``"slack-bolt"``
    ``"mautrix[encryption]>=0.20"`` → ``"mautrix"``
    """
    m = re.match(r"^([A-Za-z0-9_][A-Za-z0-9_.\-]*)", spec)
    return m.group(1) if m else spec


def _specifier_from_spec(spec: str) -> str:
    """Extract just the version-specifier portion of a pip spec.

    ``"honcho-ai==2.2.0"`` → ``"==2.2.0"``
    ``"mautrix[encryption]>=0.20,<1"`` → ``">=0.20,<1"``
    ``"package"`` → ``""`` (no version constraint)
    """
    # Strip the package name + optional [extras] block.
    m = re.match(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]*(?:\[[A-Za-z0-9_,\-]+\])?", spec)
    if not m:
        return ""
    return spec[m.end():]


def _is_satisfied(spec: str) -> bool:
    """Is ``spec`` already satisfied in the current env?

    Checks both presence AND version. If the package is installed at a
    version outside the spec's range, returns False so the caller will
    upgrade/downgrade to the pinned version. This is what makes
    ``hermes update`` propagate pin bumps in :data:`LAZY_FEATURES` to already-
    installed backends instead of silently leaving stale versions in place.

    If ``packaging`` is unavailable for any reason (it's a transitive of
    pip so this should never happen), we fall back to a presence-only check
    so we err on the side of "don't churn".
    """
    pkg = _pkg_name_from_spec(spec)
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:
        return False
    try:
        installed = version(pkg)
    except PackageNotFoundError:
        return False
    except Exception:
        return False

    spec_tail = _specifier_from_spec(spec)
    if not spec_tail:
        # Bare ``"package"`` — no version constraint, presence is enough.
        return True

    try:
        from packaging.specifiers import InvalidSpecifier, SpecifierSet
        from packaging.version import InvalidVersion, Version
    except ImportError:
        # packaging unavailable — fall back to "installed counts as satisfied".
        return True

    try:
        return Version(installed) in SpecifierSet(spec_tail)
    except (InvalidSpecifier, InvalidVersion, Exception):
        # Malformed spec or installed version we can't parse — don't churn.
        return True


def _is_present(spec: str) -> bool:
    """Cheap presence-only check (package name installed at any version).

    Used by :func:`active_features` to detect backends the user has
    previously activated, regardless of whether the version pin moved.
    """
    pkg = _pkg_name_from_spec(spec)
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:
        return False
    try:
        version(pkg)
        return True
    except PackageNotFoundError:
        return False
    except Exception:
        return False


def _core_constraints_file() -> Optional[Path]:
    """Write a pip constraints file pinning every package already importable
    in the core environment to its installed version.

    Passed as ``--constraint`` for durable-target installs so the resolver
    pins shared transitive deps (httpx, pydantic, aiohttp, …) to the exact
    versions the core venv already ships, instead of pulling newer copies
    into the durable store. Two payoffs:

    * The durable store stays minimal — only genuinely-new packages land
      there; shared deps resolve to "already satisfied" against core.
    * A backend that *requires* a version conflicting with core fails loudly
      at install time (resolver conflict) rather than silently installing a
      shadowed copy that can never win on sys.path anyway.

    Returns the path to a temp constraints file, or None if enumeration
    failed (in which case the caller installs without constraints — still
    safe, just less tidy).
    """
    try:
        from importlib.metadata import distributions
    except ImportError:
        return None
    try:
        import tempfile
        lines = []
        seen = set()
        for dist in distributions():
            name = dist.metadata["Name"] if dist.metadata else None
            ver = dist.version
            if not name or not ver:
                continue
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"{name}=={ver}")
        if not lines:
            return None
        fd, path = tempfile.mkstemp(prefix="hermes-core-constraints-", suffix=".txt")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(lines)) + "\n")
        return Path(path)
    except Exception as e:
        logger.debug("Could not build core constraints file: %s", e)
        return None


# Overrides forced onto every lazy install, mirroring
# ``[tool.uv] override-dependencies`` in pyproject.toml.
#
# ``uv pip install`` / ``pip install`` do NOT read ``[tool.uv]``, so a
# transitive dep that caps a security-pinned package below its patched floor
# silently DOWNGRADES the core venv on first use of the backend that pulls it.
# Measured with cryptography: the core venv ships 50.0.0, then enabling
# DingTalk (``alibabacloud-dingtalk`` -> ``alibabacloud-tea-openapi==0.4.5``,
# which caps ``cryptography<49``) resolved to::
#
#     + cryptography==48.0.1     # three open advisories, re-introduced
#
# Pinning the floor alongside the specs is NOT a fix: the resolver satisfies
# it by walking ``alibabacloud-tea-openapi`` back to 0.3.16 (a two-year-old
# sdist build) instead, and pinning both is simply unsatisfiable. An overrides
# file is the only mechanism that forces the patched version while keeping the
# backend at its intended version, so it is passed to the uv tier below.
_SECURITY_OVERRIDES: tuple[str, ...] = (
    # alibabacloud-tea-openapi 0.4.5 caps cryptography<49; 48.0.1 carries
    # GHSA-m2h6-j472-rp4c, GHSA-jwv3-5hgf-82ww and CVE-2026-69247. The cap is
    # stale, not a real incompatibility — the package touches cryptography
    # only for RSA/AES request signing. Keep in sync with the matching
    # override in pyproject.toml's [tool.uv].
    "cryptography>=50,<51",
)


def _security_overrides_file() -> Optional[Path]:
    """Write ``_SECURITY_OVERRIDES`` to a temp requirements file for ``--overrides``.

    Returns the path, or None if the file can't be written (in which case the
    caller installs without overrides — same behaviour as before, just with
    the downgrade risk this guards against).
    """
    if not _SECURITY_OVERRIDES:
        return None
    try:
        import tempfile

        fd, path = tempfile.mkstemp(prefix="hermes-lazy-overrides-", suffix=".txt")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("\n".join(_SECURITY_OVERRIDES) + "\n")
        return Path(path)
    except Exception as e:
        logger.debug("Could not build security overrides file: %s", e)
        return None


def _pip_reassert_overrides(
    pip_cmd: list[str],
    target_args: list[str],
    *,
    timeout: int,
):
    """Re-install ``_SECURITY_OVERRIDES`` with ``--no-deps`` after a pip install.

    pip has no ``--overrides``. Passing the floor as a ``--constraint`` does
    hold the pinned package, but pip satisfies the constraint by resolving the
    *backend* backwards instead (alibabacloud-tea-openapi 0.4.5 -> 0.3.16, a
    two-year-old sdist). A ``--no-deps`` second pass avoids that entirely: it
    rewrites only the overridden distribution and leaves everything pip already
    resolved in place.

    Returns the failing ``CompletedProcess`` if the repair pass errored, or
    None when there was nothing to do / it succeeded (caller keeps its own
    result). A repair failure is surfaced because silently leaving a
    downgraded security package installed is the bug this exists to prevent.
    """
    if not _SECURITY_OVERRIDES:
        return None
    try:
        r = subprocess.run(
            pip_cmd + ["install", "--no-deps", *target_args, *_SECURITY_OVERRIDES],
            capture_output=True, text=True, encoding='utf-8', errors='replace',
            timeout=timeout,
            stdin=subprocess.DEVNULL,
            creationflags=windows_hide_flags(),
        )
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("pip override re-assert failed to run: %s", e)
        return None
    if r.returncode != 0:
        logger.warning(
            "pip override re-assert failed (rc=%d); a security-pinned package "
            "may have been downgraded by this install: %s",
            r.returncode, (r.stderr or "").strip()[:400],
        )
        return r
    return None


def _uv_sync_extra(feature: str) -> Optional[_InstallResult]:
    """Install ``feature``'s extra with ``uv sync``, or None if not applicable.

    ``uv sync`` is the preferred installer because it is the only one that
    resolves against ``uv.lock`` and applies ``[tool.uv]
    override-dependencies`` — i.e. it installs exactly the versions CI audited,
    including security overrides that ``uv pip`` / ``pip`` cannot see.

    Returns None (caller falls back to the pip ladder) when:

    * durable-target mode is active — that mode deliberately installs to a
      separate dir so it cannot mutate the sealed venv, and ``uv sync``
      manages a venv wholesale with no ``--target`` equivalent;
    * there is no project root with a ``uv.lock`` beside ``pyproject.toml``;
    * uv isn't available;
    * the feature's extra isn't declared in pyproject.

    ``--inexact`` is required: a bare ``uv sync`` prunes everything not in the
    synced extra set, which would uninstall every other lazy backend the user
    has enabled. ``--no-install-project`` keeps it from reinstalling Hermes
    itself over an editable checkout.
    """
    if _lazy_install_target() is not None:
        return None
    root = _project_root()
    if root is None or not (root / "uv.lock").is_file():
        return None
    try:
        extra = feature_extra(feature)
    except KeyError:
        return None
    if extra not in _optional_dependencies():
        return None

    try:
        from hermes_cli.managed_uv import resolve_uv

        uv_bin = resolve_uv() or shutil.which("uv")
    except Exception:
        uv_bin = shutil.which("uv")
    if not uv_bin:
        return None

    try:
        from tools.environments.local import hermes_subprocess_env

        env = hermes_subprocess_env(inherit_credentials=False)
    except Exception:
        env = dict(os.environ)
    # uv sync targets the project environment; point it at the running venv so
    # a lazy install lands where the agent will import from.
    env["UV_PROJECT_ENVIRONMENT"] = str(Path(sys.executable).parent.parent)
    # --locked needs [tool.uv] visible; UV_NO_CONFIG would drop exclude-newer.
    env.pop("UV_NO_CONFIG", None)

    cmd = [
        uv_bin, "sync",
        "--extra", extra,
        "--inexact",
        "--locked",
        "--no-install-project",
        "--python", sys.executable,
    ]
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=600, env=env,
            stdin=subprocess.DEVNULL,
            creationflags=windows_hide_flags(),
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        logger.debug("uv sync unavailable (%s) — falling back to pip ladder", e)
        return None
    if r.returncode == 0:
        logger.info("Installed extra [%s] for feature %r via uv sync", extra, feature)
        return _InstallResult(True, r.stdout or "", r.stderr or "")
    # A stale lockfile (--locked refuses) or any other sync failure falls back
    # rather than hard-failing: the pip ladder can still install the specs.
    logger.debug(
        "uv sync --extra %s failed (rc=%d), falling back: %s",
        extra, r.returncode, (r.stderr or "").strip()[:300],
    )
    return None


def _venv_pip_install(specs: tuple[str, ...], *, timeout: int = 300) -> _InstallResult:
    """Install ``specs`` using the uv → pip → ensurepip ladder.

    Two modes:

    * **Venv-scoped (default).** Installs into the active venv
      (``sys.executable``). Used on normal installs.
    * **Durable-target.** When :data:`_LAZY_TARGET_ENV` is set, installs into
      that directory via ``--target`` and constrains shared deps to the
      core venv's versions (see :func:`_core_constraints_file`). The target
      is append-only on ``sys.path`` so it can never shadow core. Used by
      the immutable Docker image to keep lazy installs off the sealed venv.

    Mirrors the strategy in ``hermes_cli.tools_config._pip_install`` but
    kept independent here so this module has no CLI dependency.
    """
    if not specs:
        return _InstallResult(True, "", "")

    target = _lazy_install_target()
    constraints: Optional[Path] = None

    if target is not None:
        err = _ensure_target_ready(target)
        if err:
            return _InstallResult(False, "", err)
        constraints = _core_constraints_file()

    overrides = _security_overrides_file()

    target_args: list[str] = []
    if target is not None:
        # --target tells both uv and pip to install into an arbitrary dir.
        target_args = ["--target", str(target)]
    constraint_args: list[str] = []
    if constraints is not None:
        constraint_args = ["--constraint", str(constraints)]
    # uv-only: pip has no --overrides. See _SECURITY_OVERRIDES.
    override_args: list[str] = []
    if overrides is not None:
        override_args = ["--overrides", str(overrides)]

    try:
        venv_root = Path(sys.executable).parent.parent
        from tools.environments.local import hermes_subprocess_env
        uv_env = hermes_subprocess_env(inherit_credentials=False)
        uv_env["VIRTUAL_ENV"] = str(venv_root)

        # Tier 1: uv (preferred — fast, doesn't need pip in the venv)
        # Managed uv first: $HERMES_HOME/bin is never on PATH, so a bare
        # which() misses the uv Hermes installed and falls through to the
        # slower pip tier. Deliberately a lookup and not ensure_uv(): this runs
        # mid-turn to install an optional dependency, and downloading uv +
        # migrating the Python runtime as a side effect of that is a far bigger
        # action than the caller asked for. Tier 2 pip covers the no-uv case.
        try:
            from hermes_cli.managed_uv import resolve_uv

            uv_bin = resolve_uv() or shutil.which("uv")
        except Exception:
            uv_bin = shutil.which("uv")
        if uv_bin:
            try:
                r = subprocess.run(
                    [uv_bin, "pip", "install", *target_args, *constraint_args, *override_args, *specs],
                    capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=timeout, env=uv_env,
                    stdin=subprocess.DEVNULL,
                    creationflags=windows_hide_flags(),
                )
                if r.returncode == 0:
                    if target is not None:
                        _activate_target_on_syspath(target)
                    return _InstallResult(True, r.stdout or "", r.stderr or "")
                logger.debug("uv pip install failed: %s", r.stderr)
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                logger.debug("uv invocation failed: %s", e)

        # Tier 2: python -m pip (with ensurepip bootstrap if needed)
        pip_cmd = [sys.executable, "-m", "pip"]
        try:
            probe = subprocess.run(
                pip_cmd + ["--version"],
                capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=15,
                stdin=subprocess.DEVNULL,
                creationflags=windows_hide_flags(),
            )
            if probe.returncode != 0:
                raise FileNotFoundError("pip not in venv")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            try:
                subprocess.run(
                    [sys.executable, "-m", "ensurepip", "--upgrade", "--default-pip"],
                    capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=120, check=True,
                    stdin=subprocess.DEVNULL,
                    creationflags=windows_hide_flags(),
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                return _InstallResult(False, "",
                                      f"pip not available and ensurepip failed: {e}")

        try:
            r = subprocess.run(
                pip_cmd + ["install", *target_args, *constraint_args, *specs],
                capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=timeout,
                stdin=subprocess.DEVNULL,
                creationflags=windows_hide_flags(),
            )
            if r.returncode == 0:
                # pip has no --overrides, so a backend whose metadata caps a
                # security-pinned package below its floor has just downgraded
                # it. Re-assert the floor with --no-deps, which rewrites only
                # the overridden package and leaves the backend at the version
                # pip resolved. Measured on the DingTalk case: this yields
                # cryptography 50.0.0 AND alibabacloud-tea-openapi 0.4.5 —
                # identical to uv's --overrides. (`pip check` will report the
                # violated cap afterwards; that is what an override IS, and uv
                # produces the same end state without the diagnostic.)
                repair = _pip_reassert_overrides(pip_cmd, target_args, timeout=timeout)
                if repair is not None:
                    r = repair
            if r.returncode == 0 and target is not None:
                _activate_target_on_syspath(target)
            return _InstallResult(r.returncode == 0, r.stdout or "", r.stderr or "")
        except subprocess.TimeoutExpired as e:
            return _InstallResult(False, "", f"pip install timed out: {e}")
        except Exception as e:
            return _InstallResult(False, "", f"pip install failed: {e}")
    finally:
        for tmp in (constraints, overrides):
            if tmp is not None:
                try:
                    tmp.unlink()
                except OSError:
                    pass


# =============================================================================
# Public API
# =============================================================================


def feature_specs(feature: str) -> tuple[str, ...]:
    """Return the specs for ``feature``, read from its pyproject extra.

    Raises KeyError for an unknown feature, and FeatureUnavailable if the
    feature maps to an extra that pyproject doesn't define (a mapping typo, or
    a stripped install with no pyproject) — failing loudly beats installing
    nothing and reporting success.
    """
    extra = feature_extra(feature)
    specs = extra_specs(extra)
    if not specs:
        raise FeatureUnavailable(
            feature,
            (),
            f"feature {feature!r} maps to extra [{extra}], which resolved to no "
            f"packages. Either pyproject.toml is unreadable from "
            f"{_project_root()!r}, or [{extra}] does not exist.",
        )
    return specs


def feature_missing(feature: str) -> tuple[str, ...]:
    """Return the subset of specs for ``feature`` not currently installed."""
    return tuple(s for s in feature_specs(feature) if not _is_satisfied(s))


def ensure(feature: str, *, prompt: bool = True) -> None:
    """Make sure all packages for ``feature`` are importable.

    If they're missing, attempts to install them in the active venv. Raises
    :class:`FeatureUnavailable` if the user has disabled lazy installs or
    if the install attempt fails.

    ``prompt``: when True (default) and stdin is a TTY, asks the user to
    confirm before installing. Non-interactive callers (gateway, cron,
    batch) get prompt=False and skip the confirmation — config flag is
    the gate in that case.
    """
    if feature not in LAZY_FEATURES:
        raise FeatureUnavailable(
            feature, (), f"feature {feature!r} not in LAZY_FEATURES allowlist"
        )

    missing = feature_missing(feature)
    if not missing:
        return

    unsupported = _unsupported_feature_reason(feature)
    if unsupported:
        raise FeatureUnavailable(feature, missing, unsupported)

    # Package-manager installs (NixOS, and any other distro that ships Hermes
    # from a read-only store) cannot receive lazy pip installs: the venv's
    # site-packages lives in the store, so the uv -> pip -> ensurepip ladder
    # below burns ~15s bootstrapping ensurepip only to fail on a read-only
    # target. Fail fast with an actionable message instead.
    #
    # Skipped when a durable install target is configured: the container
    # deployment sets HERMES_MANAGED=true *and* HERMES_LAZY_INSTALL_TARGET
    # (a writable volume), where lazy installs legitimately work.
    #
    # The reason string starts with "unsupported " on purpose:
    # refresh_active_features classifies FeatureUnavailable by that prefix and
    # reports anything else as a hard failure rather than a skip.
    if _lazy_install_target() is None:
        try:
            from hermes_cli.config import get_managed_system

            managed_by = get_managed_system()
        except Exception:
            managed_by = ""  # config unreadable — proceed with the install
        if managed_by:
            raise FeatureUnavailable(
                feature, missing,
                f"unsupported on {managed_by}-managed installs: this build's "
                f"packages come from {managed_by}, so Hermes cannot install "
                f"them at runtime. Add the dependencies for {feature!r} via "
                f"{managed_by} (or run a pip/uv install of Hermes instead)."
            )

    # Validate every spec against the allowlist + safety regex. Belt and
    # braces — the keys-in-LAZY_FEATURES check above already constrains this.
    for spec in missing:
        if not _spec_is_safe(spec):
            raise FeatureUnavailable(
                feature, missing,
                f"refusing to install unsafe spec {spec!r}"
            )

    if not _allow_lazy_installs():
        raise FeatureUnavailable(
            feature, missing,
            "lazy installs disabled (security.allow_lazy_installs=false)"
        )

    # Only show the interactive confirmation when we own a TTY and
    # prompt_toolkit isn't running.  A bare input() deadlocks when a
    # prompt_toolkit app owns the terminal because keystrokes route to
    # its event loop rather than stdin, so the prompt blocks forever.
    # Under the TUI we skip the prompt and proceed — lazy installs are
    # gated by security.allow_lazy_installs, so reaching here is
    # already user opt-in.
    _pt_active = False
    if "prompt_toolkit.application.current" in sys.modules:
        try:
            from prompt_toolkit.application.current import get_app_or_none
            _app = get_app_or_none()
            _pt_active = _app is not None and getattr(_app, "is_running", False)
        except Exception:
            _pt_active = False

    if prompt and not _pt_active and sys.stdin.isatty() and sys.stdout.isatty():
        spec_list = ", ".join(missing)
        try:
            answer = input(
                f"\nFeature {feature!r} requires: {spec_list}\n"
                f"Install into the active venv now? [Y/n] "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"
        if answer and answer not in {"y", "yes"}:
            raise FeatureUnavailable(
                feature, missing, "user declined install at prompt"
            )

    logger.info("Lazy-installing %s for feature %r", " ".join(missing), feature)
    # Tier 0: `uv sync --extra <name>`, which resolves against uv.lock and
    # honours [tool.uv] override-dependencies. This is the only installer that
    # reproduces exactly what CI audited, so it is tried before the
    # pip-compatible ladder. Needs a project root + lockfile, and cannot serve
    # durable-target mode (it manages a venv wholesale, and the sealed-venv
    # image redirects installs to a separate dir on purpose).
    result = _uv_sync_extra(feature)
    if result is None:
        result = _venv_pip_install(missing)
    if not result.success:
        # Surface the actual pip error so the user can debug PyPI-side
        # issues (404 quarantine, network down, etc.).
        snippet = (result.stderr or result.stdout or "").strip()
        if snippet:
            # Clip to a readable size — pip can dump pages of resolution traces.
            snippet = snippet[-2000:]
        raise FeatureUnavailable(
            feature, missing,
            f"pip install failed: {snippet or 'no error output'}"
        )

    # Verify post-install. importlib.metadata caches per-process, so if we
    # just installed something the cache may not see it without a refresh.
    try:
        import importlib.metadata as _md
        if hasattr(_md, "_cache_clear"):
            _md._cache_clear()  # type: ignore[attr-defined]
    except Exception:
        pass

    still_missing = feature_missing(feature)
    if still_missing:
        raise FeatureUnavailable(
            feature, still_missing,
            "install reported success but packages still not importable "
            "(may require Python restart)"
        )

    logger.info("Lazy install complete for feature %r", feature)


def is_available(feature: str) -> bool:
    """Return True if the feature's deps are already satisfied."""
    if feature not in LAZY_FEATURES:
        return False
    return not feature_missing(feature)


def feature_install_command(feature: str) -> Optional[str]:
    """Return the ``pip install`` command a user could run manually, or None."""
    if feature not in LAZY_FEATURES:
        return None
    specs = feature_specs(feature)
    return "uv pip install " + " ".join(repr(s) for s in specs)


@dataclass
class InstallSpecsResult:
    """Outcome of :func:`install_specs` for one batch of pip specs.

    ``ok``       — install succeeded (or nothing was missing).
    ``blocked``  — installs are gated off (config kill switch, sealed venv
                   without a durable target) or a spec failed validation;
                   nothing was executed. ``reason`` explains why.
    ``command``  — human-readable description of what ran (for UIs/logs).
    """
    ok: bool
    blocked: bool = False
    reason: str = ""
    command: str = ""
    stdout: str = ""
    stderr: str = ""


def install_specs(specs: list[str] | tuple[str, ...], *, timeout: int = 300) -> InstallSpecsResult:
    """Install arbitrary (validated) pip specs through the lazy-install pipeline.

    This is the environment-aware install path for callers whose package
    lists come from data (e.g. memory-provider plugin manifests declaring
    ``pip_dependencies``) rather than the static :data:`LAZY_FEATURES` allowlist.
    It applies the exact same environment routing as :func:`ensure`:

    * **Venv-scoped by default** — installs into ``sys.executable``'s venv.
    * **Durable-target on immutable images** — when the deployment seals the
      agent venv (``HERMES_DISABLE_LAZY_INSTALLS=1``) and sets
      ``HERMES_LAZY_INSTALL_TARGET``, installs are redirected to the writable
      data-volume dir (``--target`` + core-venv constraints), then activated
      on ``sys.path`` so the packages import in this process immediately.
    * **Gated** — honors ``security.allow_lazy_installs`` and refuses to run
      when the venv is sealed with no durable target (never attempts a write
      to a read-only tree; reports *why* instead of surfacing EROFS/EACCES).

    Every spec must pass :func:`_spec_is_safe` (no URLs, paths, or shell
    metacharacters). Unlike :func:`ensure`, unknown packages are permitted —
    the caller owns manifest trust; this function owns spec hygiene and
    environment routing.

    Never raises; inspect the returned :class:`InstallSpecsResult`.
    """
    cleaned = tuple(str(s).strip() for s in specs if str(s).strip())
    if not cleaned:
        return InstallSpecsResult(ok=True, command="")

    for spec in cleaned:
        if not _spec_is_safe(spec):
            return InstallSpecsResult(
                ok=False, blocked=True,
                reason=f"refusing to install unsafe spec {spec!r}",
            )

    if not _allow_lazy_installs():
        target = _lazy_install_target()
        if os.environ.get("HERMES_DISABLE_LAZY_INSTALLS") == "1" and target is None:
            reason = (
                "runtime installs are disabled on this deployment: the agent "
                "environment is immutable and no writable install target is "
                "configured (HERMES_LAZY_INSTALL_TARGET)"
            )
        else:
            reason = "runtime installs disabled (security.allow_lazy_installs=false)"
        return InstallSpecsResult(ok=False, blocked=True, reason=reason)

    target = _lazy_install_target()
    display = "uv pip install " + (
        f"--target {target} " if target is not None else ""
    ) + " ".join(cleaned)

    logger.info("Installing pip specs %s (target=%s)", " ".join(cleaned), target or "venv")
    try:
        result = _venv_pip_install(cleaned, timeout=timeout)
    except Exception as exc:
        logger.warning("install_specs failed unexpectedly: %s", exc)
        return InstallSpecsResult(
            ok=False, command=display, stderr=f"install failed: {exc}"
        )

    # Freshly-installed dists must be visible to importers and metadata
    # checks in this same process (dashboard rechecks availability inline).
    try:
        import importlib
        importlib.invalidate_caches()
        import importlib.metadata as _md
        if hasattr(_md, "_cache_clear"):
            _md._cache_clear()  # type: ignore[attr-defined]
    except Exception:
        pass

    return InstallSpecsResult(
        ok=result.success,
        command=display,
        stdout=result.stdout,
        stderr=result.stderr,
    )


def active_features() -> list[str]:
    """Return the list of features the user has ever lazy-installed.

    A feature counts as "active" if its anchor package (the first declared
    spec) is currently installed in the venv (presence check, ignoring
    version). We intentionally do NOT treat shared helper packages as proof
    that a backend was enabled: for example ``platform.matrix`` depends on
    generic packages like ``asyncpg``/``aiosqlite`` that can be installed for
    unrelated reasons, while the actual Matrix adapter anchor is ``mautrix``.
    Features the user has never enabled stay quiet.

    Used by ``hermes update`` to figure out which lazy backends need a
    refresh pass when pins move in :data:`LAZY_FEATURES`.
    """
    active = []
    for feature in LAZY_FEATURES:
        try:
            specs = feature_specs(feature)
        except FeatureUnavailable:
            # Extra missing from pyproject (stripped install) — the feature
            # can't be refreshed, and it certainly isn't active.
            continue
        if specs and _is_present(specs[0]):
            active.append(feature)
    return active


def refresh_active_features(*, prompt: bool = False) -> dict[str, str]:
    """Re-run ``ensure`` for every feature the user has previously activated.

    Returns a ``{feature: status}`` map where status is one of:
        ``"current"``  — pins already satisfied, no install run
        ``"refreshed"`` — pins were stale, reinstall succeeded
        ``"failed: <reason>"`` — install attempt failed; caller decides
                                  whether to surface it (we don't raise)
        ``"skipped: <reason>"`` — gated off (config flag, user decline)

    Intended for ``hermes update``. Never raises; lazy-install failures
    here must not block the rest of the update flow.
    """
    results: dict[str, str] = {}
    for feature in active_features():
        missing = feature_missing(feature)
        if not missing:
            results[feature] = "current"
            continue

        unsupported = _unsupported_feature_reason(feature)
        if unsupported:
            results[feature] = f"skipped: {unsupported}"
            continue

        try:
            ensure(feature, prompt=prompt)
            results[feature] = "refreshed"
        except FeatureUnavailable as e:
            # Distinguish "user opted out" or platform-incompatible features
            # from install failures so the update command can render the
            # right non-error message.
            if (
                "lazy installs disabled" in str(e)
                or "declined" in str(e)
                or e.reason.startswith("unsupported ")
            ):
                results[feature] = f"skipped: {e.reason}"
            else:
                results[feature] = f"failed: {e.reason}"
        except Exception as e:
            results[feature] = f"failed: {e}"
    return results


def ensure_and_bind(
    feature: str,
    importer: Callable[[], dict[str, Any]],
    target_globals: dict,
    *,
    prompt: bool = False,
) -> bool:
    """Ensure a feature is installed, then rebind names into the caller's globals.

    Combines :func:`ensure` with a post-install import step that rebinds
    module-level names.  This eliminates the error-prone pattern of manually
    listing every global that needs updating after lazy-install.

    ``importer`` is a zero-arg callable that returns a dict of
    ``{name: value}`` for all symbols the caller needs rebound.  It is called
    only after :func:`ensure` succeeds (or if the packages are already
    installed).

    Returns True on success, False if deps couldn't be installed or imported.

    Example usage in a platform adapter::

        def check_slack_requirements() -> bool:
            if SLACK_AVAILABLE:
                return True
            def _import():
                from slack_bolt.async_app import AsyncApp
                from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
                from slack_sdk.web.async_client import AsyncWebClient
                import aiohttp
                return {
                    "AsyncApp": AsyncApp,
                    "AsyncSocketModeHandler": AsyncSocketModeHandler,
                    "AsyncWebClient": AsyncWebClient,
                    "aiohttp": aiohttp,
                    "SLACK_AVAILABLE": True,
                }
            return ensure_and_bind("platform.slack", _import, globals(), prompt=False)
    """
    try:
        ensure(feature, prompt=prompt)
    except (FeatureUnavailable, Exception):
        return False

    try:
        bindings = importer()
    except ImportError:
        return False

    target_globals.update(bindings)
    return True
