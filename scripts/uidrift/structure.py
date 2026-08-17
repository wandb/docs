"""Structural signals: what kind of change is this, and is anyone seeing it yet?

Pure functions over a diff and the deltas extracted from it. No I/O, no model,
no network -- everything here is readable off the patch, which is what makes it
testable against frozen fixtures and cheap enough to run on every candidate.

The one exception is `resolve_gate_key`, which reads the ramp registry out of
the watched repo. It is separate, optional, and degrades to None so the rest of
the module stays hermetic.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

from . import config
from ._vendor.diff_signals import _iter_file_diffs
from .extract import LabelDelta

# Two strings pair as a rename above this casefolded similarity. Chosen so that
# a pure case change (ratio 1.0 after casefold) and a light reword both pair,
# while two unrelated column headers do not.
RENAME_RATIO = 0.6

# How far back to look for the conditional that governs a line.
GATE_LOOKBACK = 40

_HUNK = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")

# `if (shouldShowX) {`, `} else if (canY) {`
_IF_COND = re.compile(r"\bif\s*\(\s*([A-Za-z_$][\w$]*)\s*\)")
# `{showX && (`  -- the JSX short-circuit form
_JSX_COND = re.compile(r"\{\s*([A-Za-z_$][\w$]*)\s*&&")
# `const shouldShowX = useStatsigGateFoo(orgName);`
_GATE_ASSIGN = re.compile(
    r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*"
    r"((?:use)[A-Za-z_$][\w$]*(?:Gate|RampFlag|Flag)[\w$]*)\s*\("
)
# Only the assigned form above is matched. A gate hook called inline -- no
# intermediate variable -- is not detected: measured against core's UI tree,
# _GATE_ASSIGN finds 261 call sites and an inline matcher would add ~110 more,
# but ~19 of those are `useGatedValue`, an unrelated Weave utility that any
# `use*Gate*` pattern also matches. Widening this needs a name filter and its
# own tests, so it is a follow-up rather than a regex.

# The hand-maintained union of frontend-reachable gate names in useRampFlag.ts.
_RAMP_KEY_LINE = re.compile(r"^\s*\|\s*'([a-z0-9_.-]+)'\s*$")

# Hook definition in rampFeatureFlags.ts, for resolving a hook to its Statsig key.
_HOOK_TO_KEY = r"{hook}\s*=[\s\S]{{0,400}}?['\"]([a-z0-9_.-]+)['\"]"

_TESTID = re.compile(r"\b(data-test|data-testid|data-dd-action-name)\s*=")
_SLUG = re.compile(r"^\s*slug\s*:\s*['\"`]([^'\"`]*)['\"`]")

# Where a "new setting" plausibly lives.
_SETTINGS_PATH = re.compile(r"(Settings|Privacy|Preferences|Profile)", re.I)


@dataclass(frozen=True)
class DiffLine:
    sign: str  # "+" | "-" | " "
    old_no: int
    new_no: int
    text: str

    @property
    def indent(self) -> int:
        return len(self.text) - len(self.text.lstrip())


@dataclass(frozen=True)
class RenamePair:
    old: LabelDelta
    new: LabelDelta
    ratio: float
    ambiguous: bool

    @property
    def case_only(self) -> bool:
        return self.old.norm != self.new.norm and self.old.norm.casefold() == self.new.norm.casefold()


@dataclass(frozen=True)
class GateScope:
    """The conditional governing a changed line, if there is one."""

    variable: str
    hook: Optional[str]
    key: Optional[str]  # the Statsig key, when resolvable
    conditional_added: bool  # the `if` itself is a + line in this commit

    @property
    def name(self) -> str:
        return self.key or self.hook or self.variable


def parse_streams(diff: str) -> dict[str, list[DiffLine]]:
    """Turn a unified diff into per-file positioned line streams."""
    streams: dict[str, list[DiffLine]] = {}
    for path, body in _iter_file_diffs(diff):
        lines: list[DiffLine] = []
        old_no = new_no = 0
        for raw in body.splitlines():
            m = _HUNK.match(raw)
            if m:
                old_no, new_no = int(m.group(1)), int(m.group(2))
                continue
            if raw.startswith("+++") or raw.startswith("---"):
                continue
            if raw.startswith("+"):
                lines.append(DiffLine("+", 0, new_no, raw[1:]))
                new_no += 1
            elif raw.startswith("-"):
                lines.append(DiffLine("-", old_no, 0, raw[1:]))
                old_no += 1
            elif raw.startswith(" ") or not raw:
                lines.append(DiffLine(" ", old_no, new_no, raw[1:] if raw else ""))
                old_no += 1
                new_no += 1
        streams[path] = lines
    return streams


# --- rename pairing -------------------------------------------------------


def _change_blocks(lines: Sequence[DiffLine]) -> list[tuple[list[DiffLine], list[DiffLine]]]:
    """Contiguous runs of changed lines, split into their removed and added halves."""
    blocks: list[tuple[list[DiffLine], list[DiffLine]]] = []
    rem: list[DiffLine] = []
    add: list[DiffLine] = []
    for ln in lines:
        if ln.sign == "-":
            rem.append(ln)
        elif ln.sign == "+":
            add.append(ln)
        else:
            if rem or add:
                blocks.append((rem, add))
                rem, add = [], []
    if rem or add:
        blocks.append((rem, add))
    return blocks


def _positional_pairs(
    streams: dict[str, list[DiffLine]],
    added: Sequence[LabelDelta],
    removed: Sequence[LabelDelta],
) -> list[tuple[int, int]]:
    """Pair by position in the diff, before considering similarity at all.

    A complete reword shares almost no characters -- 'Hide manually hidden runs'
    and 'List only visible runs' score 0.55, under any threshold loose enough to
    be safe elsewhere. But git presents an in-place edit as a `-` line and the
    `+` line that replaced it at the same offset in the same change block, which
    is far stronger evidence than string similarity ever is.

    Lowering the similarity threshold to catch these would manufacture false
    pairs between unrelated labels. Position does not have that failure mode.
    """
    rem_at: dict[tuple[str, int], list[int]] = {}
    add_at: dict[tuple[str, int], list[int]] = {}
    for i, d in enumerate(removed):
        rem_at.setdefault((d.path, d.line_no), []).append(i)
    for j, d in enumerate(added):
        add_at.setdefault((d.path, d.line_no), []).append(j)

    out: list[tuple[int, int]] = []
    for path, lines in streams.items():
        for rem_lines, add_lines in _change_blocks(lines):
            for rl, al in zip(rem_lines, add_lines):
                r_cands = rem_at.get((path, rl.old_no), [])
                a_cands = add_at.get((path, al.new_no), [])
                for i in r_cands:
                    for j in a_cands:
                        if removed[i].kind == added[j].kind and removed[i].key == added[j].key:
                            out.append((i, j))
                            break
    return out


def pair_renames(
    added: Sequence[LabelDelta],
    removed: Sequence[LabelDelta],
    streams: Optional[dict[str, list[DiffLine]]] = None,
) -> tuple[list[RenamePair], list[LabelDelta], list[LabelDelta]]:
    """Match removed strings to the strings that replaced them.

    Two passes, strongest evidence first:

    1. Position -- a `-`/`+` at the same offset in one change block is an
       in-place edit, whatever the strings look like.
    2. Similarity -- grouped by (path, kind) rather than (path, kind, key),
       because a rename can move between fields: `header: 'WEAVE ACCESS'` became
       `name: 'Weave Access'` in the same commit, and keying on the field name
       would miss it. Greedy on descending ratio, so a file renaming eight
       column headers at once pairs all eight instead of collapsing into one
       ambiguous blob.
    """
    pairs: list[RenamePair] = []
    used_add: set[int] = set()
    used_rem: set[int] = set()

    if streams:
        for i, j in _positional_pairs(streams, added, removed):
            if i in used_rem or j in used_add:
                continue
            used_rem.add(i)
            used_add.add(j)
            ratio = difflib.SequenceMatcher(
                None, removed[i].norm.casefold(), added[j].norm.casefold()
            ).ratio()
            pairs.append(RenamePair(removed[i], added[j], ratio, ambiguous=False))

    groups: dict[tuple[str, str], tuple[list[int], list[int]]] = {}
    for i, d in enumerate(removed):
        if i not in used_rem:
            groups.setdefault((d.path, d.kind), ([], []))[0].append(i)
    for j, d in enumerate(added):
        if j not in used_add:
            groups.setdefault((d.path, d.kind), ([], []))[1].append(j)

    for (_path, _kind), (rem_idx, add_idx) in groups.items():
        if not rem_idx or not add_idx:
            continue

        scored: list[tuple[float, int, int]] = []
        for i in rem_idx:
            for j in add_idx:
                ratio = difflib.SequenceMatcher(
                    None, removed[i].norm.casefold(), added[j].norm.casefold()
                ).ratio()
                if ratio >= RENAME_RATIO:
                    scored.append((ratio, i, j))
        scored.sort(key=lambda t: (-t[0], t[1], t[2]))

        # A removal with several equally-good candidates cannot be resolved from
        # the diff alone, so it is paired but flagged -- never agent-eligible.
        best: dict[int, float] = {}
        for ratio, i, _j in scored:
            best[i] = max(best.get(i, 0.0), ratio)
        tie_count: dict[int, int] = {}
        for ratio, i, _j in scored:
            if ratio == best[i]:
                tie_count[i] = tie_count.get(i, 0) + 1

        for ratio, i, j in scored:
            if i in used_rem or j in used_add:
                continue
            used_rem.add(i)
            used_add.add(j)
            pairs.append(
                RenamePair(removed[i], added[j], ratio, ambiguous=tie_count.get(i, 1) > 1)
            )

    return (
        pairs,
        [d for j, d in enumerate(added) if j not in used_add],
        [d for i, d in enumerate(removed) if i not in used_rem],
    )


# --- gating ---------------------------------------------------------------


def gate_scope(
    streams: dict[str, list[DiffLine]], delta: LabelDelta
) -> Optional[GateScope]:
    """Find the conditional that governs this line, if any.

    Per-surface, not per-feature. One gate can govern three surfaces in a single
    component with a different answer for each, so the question is always
    "is *this string* behind a conditional", answered by reading the enclosing
    block -- never by matching a flag name against the commit message.
    """
    lines = streams.get(delta.path)
    if not lines:
        return None

    anchor = None
    for idx, ln in enumerate(lines):
        pos = ln.new_no if delta.sign == "+" else ln.old_no
        if ln.sign == delta.sign and pos == delta.line_no and delta.raw in ln.text:
            anchor = idx
            break
    if anchor is None:
        return None

    target_indent = lines[anchor].indent
    for idx in range(anchor - 1, max(-1, anchor - GATE_LOOKBACK) - 1, -1):
        ln = lines[idx]
        if ln.sign == "-":
            continue
        if ln.indent >= target_indent:
            continue

        m = _IF_COND.search(ln.text) or _JSX_COND.search(ln.text)
        if not m:
            # Dedented past the enclosing block without finding a conditional.
            if ln.text.strip() and ln.indent == 0:
                break
            continue

        variable = m.group(1)
        hook = _resolve_hook(lines, variable)
        if hook is None:
            # An ordinary conditional, not a gate. `hideManuallyHidden` is UI
            # state; treating every `if` as gating would mark most of the app
            # "not yet visible". No resolvable hook means no claim.
            return None
        return GateScope(
            variable=variable,
            hook=hook,
            key=None,
            conditional_added=ln.sign == "+",
        )
    return None


def _resolve_hook(lines: Iterable[DiffLine], variable: str) -> Optional[str]:
    """Walk from the conditional variable back to the gate hook that set it."""
    for ln in lines:
        if ln.sign == "-":
            continue
        m = _GATE_ASSIGN.search(ln.text)
        if m and m.group(1) == variable:
            return m.group(2)
    return None


def resolve_gate_key(
    hook: str, core: Optional[Path] = None, cfg: config.SourceRepo = config.SOURCE
) -> Optional[str]:
    """Map a gate hook to its Statsig key by reading the watched repo.

    Optional enrichment: the hook definition usually lives outside the commit
    being examined, so this is the one function here that touches the repo.
    Returns None on any failure -- a missing key degrades the finding's
    confidence, it does not break the scan.
    """
    import subprocess

    root = core or cfg.path
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "show", f"{cfg.default_head}:{cfg.flag_hooks}"],
            capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    m = re.search(_HOOK_TO_KEY.format(hook=re.escape(hook)), out.stdout)
    return m.group(1) if m else None


def flag_lifecycle(diff: str, cfg: config.SourceRepo = config.SOURCE) -> dict[str, str]:
    """Gate names added or removed from the frontend ramp registry by this commit.

    This is the whole flag signal. Static presence is deliberately not consulted:
    engineers leave a gate in place after ramping it to 100% because removing it
    is riskier than keeping it, so "there is a flag" has decayed to noise. The
    lifecycle events have not:

        added in the same commit as the copy -> not visible yet
        removal commit                       -> GA
        present, added long ago              -> no signal at all
    """
    out: dict[str, str] = {}
    for path, body in _iter_file_diffs(diff):
        if path != cfg.flag_registry:
            continue
        for raw in body.splitlines():
            if raw.startswith("+++") or raw.startswith("---"):
                continue
            if not raw or raw[0] not in "+-":
                continue
            m = _RAMP_KEY_LINE.match(raw[1:])
            if m:
                out[m.group(1)] = "added" if raw[0] == "+" else "removed"
    return out


# --- corroborating signals ------------------------------------------------


def testid_corroboration(
    streams: dict[str, list[DiffLine]], delta: LabelDelta, window: int = 6
) -> bool:
    """Is there an unchanged test id next to this string?

    A `data-test` that stayed put while the adjacent text changed is close to
    proof of a pure rename: the element is the same, only its copy moved. Purely
    a confidence booster -- absence means nothing, since only a minority of
    files carry test ids at all.
    """
    lines = streams.get(delta.path)
    if not lines:
        return False

    for idx, ln in enumerate(lines):
        pos = ln.new_no if delta.sign == "+" else ln.old_no
        if ln.sign == delta.sign and pos == delta.line_no and delta.raw in ln.text:
            lo, hi = max(0, idx - window), min(len(lines), idx + window + 1)
            return any(
                _TESTID.search(n.text) and n.sign == " " for n in lines[lo:hi]
            )
    return False


def slug_stability(streams: dict[str, list[DiffLine]], pair: RenamePair) -> str:
    """Did the URL survive the rename?

    `name:` changed while a sibling `slug:` was left alone means the tab was
    renamed but its route did not move: docs prose is stale, docs links are
    fine. A changed slug is a different and worse problem, because anchors and
    deep links break too.
    """
    if pair.new.kind != "obj":
        return "n/a"
    lines = streams.get(pair.new.path)
    if not lines:
        return "n/a"

    changed = [ln for ln in lines if ln.sign in "+-" and _SLUG.match(ln.text)]
    if not changed:
        return "url_stable" if any(_SLUG.match(ln.text) for ln in lines) else "n/a"

    added = {_SLUG.match(ln.text).group(1) for ln in changed if ln.sign == "+"}
    removed = {_SLUG.match(ln.text).group(1) for ln in changed if ln.sign == "-"}
    return "url_stable" if added == removed else "url_changed"


def is_new_setting(added: Sequence[LabelDelta], path: str) -> bool:
    """A newly-added object literal carrying both a title and a description.

    The shape a settings row takes in this codebase, and the reason a new
    setting is always a human finding: there is no old string to swap, so
    somebody has to write prose.
    """
    if not _SETTINGS_PATH.search(path):
        return False
    keys = {d.key for d in added if d.path == path and d.kind == "obj"}
    return "title" in keys and "description" in keys
