"""The adaptation surface.

This is the only module that names a repository, a path, or a token. Everything
else in this package is a pure function over the shapes defined here.

Pointing the detector at a different repo means editing this file and the
checkout steps in the workflow. Nothing else. That is the whole portability
story -- see ADAPTING.md for what actually varies between repos and what
surprised us about wandb/core.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceRepo:
    """The repo being watched for user-facing change."""

    owner_repo: str
    local_path_env: str
    local_path_default: str
    default_head: str
    token_env: str
    # Only diffs under these roots are fetched. This is what keeps a
    # 2,600-line commit affordable: commit_diff() takes them as a pathspec.
    ui_roots: tuple[str, ...]
    ui_exts: tuple[str, ...]
    exclude_globs: tuple[str, ...]
    # The hand-maintained union of frontend-reachable gate names. Diffing this
    # file across a commit is the flag-lifecycle signal; static presence of a
    # flag is deliberately NOT consulted, because engineers leave gates at 100%
    # forever and the signal has decayed to noise.
    flag_registry: str
    flag_hooks: str
    codeowners: tuple[str, ...]

    @property
    def path(self) -> Path:
        return Path(os.environ.get(self.local_path_env) or self.local_path_default).expanduser()


@dataclass(frozen=True)
class DocsRepo:
    """The docs corpus searched for occurrences of a changed label."""

    local_path_env: str
    local_path_default: str
    content_exts: tuple[str, ...]
    primary_locale: str
    # Indexed and counted for blast radius, but never the find-and-replace target.
    mirror_locales: tuple[str, ...]
    exclude_dirs: tuple[str, ...]
    # Published history. Occurrences here are reported for awareness and never
    # proposed as edits -- rewriting a changelog is falsifying a record.
    immutable_globs: tuple[str, ...]
    nav_manifest: str

    @property
    def path(self) -> Path:
        return Path(os.environ.get(self.local_path_env) or self.local_path_default).expanduser()


SOURCE = SourceRepo(
    owner_repo="wandb/core",
    local_path_env="CORE_REPO",
    local_path_default="~/core",
    default_head="origin/master",
    token_env="WANDB_CORE_TOKEN",
    ui_roots=("frontends/app/src",),
    ui_exts=(".tsx", ".jsx"),
    exclude_globs=(
        "*.test.tsx", "*.test.ts", "*.spec.tsx",
        "*.stories.tsx", "*/__mocks__/*", "*/__tests__/*",
        "*/wandb-admin/*",  # internal admin UI, not a customer surface
    ),
    flag_registry="frontends/app/src/util/useRampFlag.ts",
    flag_hooks="frontends/app/src/util/rampFeatureFlags/rampFeatureFlags.ts",
    codeowners=(".github/CODEOWNERS", "CODEOWNERS"),
)

DOCS = DocsRepo(
    local_path_env="DOCS_REPO",
    # The docs corpus is native -- this package lives at <repo>/scripts/uidrift,
    # so the repo root is two levels up. Resolved from the module's own location
    # rather than the cwd, so the detector works the same from a test runner, a
    # subdirectory, and a CI step.
    local_path_default=str(Path(__file__).resolve().parents[2]),
    content_exts=(".mdx",),
    primary_locale="en",
    mirror_locales=("ja", "ko", "fr"),
    # NB: snippets/ is deliberately NOT excluded. Reusable fragments carry real
    # UI prose ("go to the **Service Accounts** tab") and render into many
    # pages, so a label there has wider blast radius than one in a single page.
    # .claude is excluded because it holds git worktrees -- indexing it would
    # double-count every occurrence.
    # uidrift is excluded to break a feedback loop: a report quotes the very
    # labels it is reporting on, so indexing our own output would make every
    # finding look documented. Reports are .md and only .mdx is indexed today,
    # which makes this insurance rather than a fix.
    exclude_dirs=(
        ".git", "node_modules", ".claude", "docengine-site", "docengine",
        "scripts", "images", "assets", "media", "css", "icons", "layouts",
        "uidrift",
    ),
    immutable_globs=("release-notes/*",),
    nav_manifest="docs.json",
)

# Where state and output land, relative to the host repo root.
LEDGER_PATH = Path("uidrift/ledger.json")
REPORT_DIR = Path("uidrift/reports")

# --- Tunables -------------------------------------------------------------
# A string that has not changed for this long, and has not been re-changed, is
# safe to act on. Observed re-churn interval in wandb/core is 7 days: a label
# renamed on 2026-06-03 was renamed again on 2026-06-10. Filing docs work on
# day 2 wastes a PR and teaches the group the detector generates churn.
SETTLED_DAYS = 7

# A literal appearing on more pages than this is too generic to be a UI label.
# Calibration: naive substring matching puts "search" on 215 pages.
MAX_DOCS_PAGES = 15

# There is deliberately no confidence threshold here. Agent eligibility is
# decided by `triage()` from structural facts -- literal kind, docs coverage,
# whether every occurrence is emphasized -- not by a score. A
# MIN_AGENT_CONFIDENCE tunable used to sit here and was never read by anything,
# which advertised a safety floor that did not exist. If the model pass later
# produces a real confidence value, wire it into triage() in the same change
# that introduces it.
