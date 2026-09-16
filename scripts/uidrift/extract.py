"""Stage 1: a unified diff in, label deltas out.

Deterministic, no model, no network, no token. This is the module that must
produce identical output for identical input forever -- the ledger's dedupe and
`settled` logic both assume a given commit always yields the same finding id.

Two rules here were learned the expensive way and are load-bearing:

1. Normalize whitespace, NEVER case. Lowercasing collapses `MODELS SEAT` into
   `Models Seat`, which silently destroys the single richest real finding in the
   corpus -- an eight-header title-casing that has been wrong in published docs
   for six weeks.

2. Never filter on conventional-commit type. `refactor(app):` and `chore(ui):`
   carried real user-visible label changes in every sampled window. The commit
   type is recorded as metadata and used by nothing.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from typing import Iterator, Sequence

from . import config
from ._vendor.diff_signals import _iter_file_diffs

# --- Extractors -----------------------------------------------------------
# Four shapes, because wandb/core has no i18n catalog. Strings are inline in
# JSX, so there is no single file to watch and no key to diff.

# Attribute names that carry copy are open-ended -- a component library invents
# `saveLabel`, `cancelLabel`, `emptyText`, `confirmText` as it grows. Enumerating
# them guarantees silent misses: the secret-drawer consolidation moved its copy
# into `saveLabel=` and an enumerated matcher scored zero on it. Match on suffix.
_ATTR_SUFFIX = r"[Ll]abel|[Tt]ext|[Tt]itle|[Hh]eader|[Pp]laceholder|[Tt]ooltip|[Hh]eading"
ATTR = re.compile(
    r'\b((?:[a-zA-Z][a-zA-Z0-9]*)?(?:' + _ATTR_SUFFIX + r')|aria-label|name)'
    r'\s*=\s*"([^"]{2,})"'
)

# `name=` is genuinely ambiguous: `<Hotkey name="List only visible runs" />` is
# copy, `<Icon name="info" />` is an identifier. Reject all-lowercase slug-shaped
# values for these keys only, so unambiguous keys keep values like
# `placeholder="my-reference-bucket"`.
_AMBIGUOUS_ATTR_KEYS = frozenset({"name", "text"})
_IDENTIFIER_VALUE = re.compile(r"^[a-z][a-z0-9_-]*$")

# A label reached through an expression rather than a plain string, e.g.
#   saveLabel={drawerMode === 'edit' ? 'Replace secret' : 'Add secret'}
# Common wherever one component serves two modes. Each literal is emitted
# separately; `wrapped` marks them, since replacing one blind would be wrong.
ATTR_EXPR = re.compile(
    r'\b((?:[a-zA-Z][a-zA-Z0-9]*)?(?:' + _ATTR_SUFFIX + r'))\s*=\s*\{([^}]{2,200})\}'
)
_EXPR_LITERAL = re.compile(r"['\"]([A-Z][^'\"]{1,80})['\"]")

_OBJ_KEYS = (
    "name|title|slug|label|description|header|Header|tooltip"
    "|placeholder|text|subtitle"
)
# Plain literal value: safe for find-and-replace.
OBJ = re.compile(
    r"^\s*(" + _OBJ_KEYS + r")\s*:\s*['\"`]([^'\"`$]{2,})['\"`]\s*,?\s*$"
)
# Template literal carrying interpolation, e.g. `Allow ${AGENT_NAME} to ...`.
# Captured so a new setting is still detected, but marked wrapped=True so it can
# never become an unattended find-and-replace target.
OBJ_INTERP = re.compile(
    r"^\s*(" + _OBJ_KEYS + r")\s*:\s*`([^`]*\$\{[^`]*)`\s*,?\s*$"
)

# Text between tags on one line: <span className="pr-10">List only visible runs</span>
# Requiring a real closing `</` rather than a bare `<` is what kills TypeScript
# generics (`Promise<void>` would otherwise capture `Promise`).
JSX_INLINE = re.compile(r">\s*([A-Z][A-Za-z0-9 ,'’.\-?!:%()/]{1,80}?)\s*</")

# Prettier split the text onto its own line.
JSX_OWNLINE = re.compile(r"^([A-Z][A-Za-z0-9 ,'’.\-?!:%()/]{2,80})$")

_PASCAL_IDENT = re.compile(r"^[A-Z][A-Za-z0-9]*$")
_SIMPLE_WORD = re.compile(r"^[A-Z][a-z]+$")
_HUNK = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")


@dataclass(frozen=True)
class LabelDelta:
    """One user-facing string added or removed by a diff."""

    sign: str  # "+" | "-"
    path: str
    kind: str  # "attr" | "obj" | "jsx"
    key: str  # the attribute/field name; "_" for bare JSX text
    raw: str
    norm: str  # whitespace-collapsed; CASE PRESERVED
    line_no: int
    # True only when the captured string is NOT a complete, self-contained
    # literal -- template interpolation (`Allow ${AGENT_NAME} to ...`) or one
    # branch of a ternary. Those can never be blind find-and-replace targets.
    #
    # Prettier reflowing text onto its own line does NOT set this: the literal
    # is captured exactly, only the surrounding markup moved. Conflating the two
    # would disqualify `MODELS SEAT` -- the best real finding in the corpus --
    # from the agent lane for no reason.
    wrapped: bool

    @property
    def ident(self) -> tuple[str, str, str]:
        """Strict identity, for detecting reflow within a file.

        Prettier re-indentation preserves the expression form exactly, so kind
        and key must participate: only an identical string in an identical
        position is reflow.
        """
        return (self.kind, self.key, self.norm)

    @property
    def moved_ident(self) -> str:
        """Loose identity, for detecting a string relocating across the commit.

        Deliberately just the string. A label can move from JSX text into a
        prop -- the secret-drawer consolidation moved `Add secret` from
        `<span>Add secret</span>` into `saveLabel="Add secret"`. The user still
        sees it, so it is a move, not a removal. Keying on (kind, key) here
        would report 23 phantom removals for that one commit.
        """
        return self.norm


def normalize(raw: str) -> str:
    """Collapse whitespace. Case is preserved deliberately -- see module docstring."""
    return re.sub(r"\s+", " ", raw).strip()


def path_is_ui(path: str, cfg: config.SourceRepo = config.SOURCE) -> bool:
    if not any(path.startswith(root) for root in cfg.ui_roots):
        return False
    if not any(path.endswith(ext) for ext in cfg.ui_exts):
        return False
    return not any(fnmatch.fnmatch(path, pat) for pat in cfg.exclude_globs)


def _reject_ownline(text: str) -> bool:
    # `  Avatar,` is an import specifier, not a label.
    if text.endswith(","):
        return True
    # A single word with no space is overwhelmingly an identifier.
    return " " not in text


def _reject_inline(text: str) -> bool:
    if not _PASCAL_IDENT.match(text):
        return False
    # Keep `Save`, `Cancel`, `Delete`; drop `Promise`, `ReactNode`, `WBTable`.
    return not _SIMPLE_WORD.match(text)


def _scan_line(line: str, sign: str, path: str, line_no: int, in_import: bool) -> Iterator[LabelDelta]:
    body = line[1:]

    for m in ATTR.finditer(body):
        key, raw = m.group(1), m.group(2)
        if key in _AMBIGUOUS_ATTR_KEYS and _IDENTIFIER_VALUE.match(raw):
            continue
        yield LabelDelta(sign, path, "attr", key, raw, normalize(raw), line_no, False)

    for m in ATTR_EXPR.finditer(body):
        key = m.group(1)
        for lit in _EXPR_LITERAL.finditer(m.group(2)):
            raw = lit.group(1)
            yield LabelDelta(sign, path, "attr", key, raw, normalize(raw), line_no, True)

    m = OBJ.match(body)
    if m:
        raw = m.group(2)
        yield LabelDelta(sign, path, "obj", m.group(1), raw, normalize(raw), line_no, False)
        return

    m = OBJ_INTERP.match(body)
    if m:
        raw = m.group(2)
        yield LabelDelta(sign, path, "obj", m.group(1), raw, normalize(raw), line_no, True)
        return

    for m in JSX_INLINE.finditer(body):
        raw = m.group(1)
        if _reject_inline(raw):
            continue
        yield LabelDelta(sign, path, "jsx", "_", raw, normalize(raw), line_no, False)

    stripped = body.strip()
    if not in_import and JSX_OWNLINE.match(stripped) and not _reject_ownline(stripped):
        # Prettier reflowed this out of its element. The literal itself is
        # complete, so it stays a valid replace target.
        yield LabelDelta(sign, path, "jsx", "_", stripped, normalize(stripped), line_no, False)


def extract_deltas(
    diff: str, cfg: config.SourceRepo = config.SOURCE
) -> list[LabelDelta]:
    """Walk a unified diff and return every added/removed user-facing string."""
    out: list[LabelDelta] = []
    if not diff:
        return out

    for path, body in _iter_file_diffs(diff):
        if not path_is_ui(path, cfg):
            continue

        old_ln = new_ln = 0
        in_import = False

        for line in body.splitlines():
            m = _HUNK.match(line)
            if m:
                old_ln, new_ln = int(m.group(1)), int(m.group(2))
                in_import = False
                continue
            if line.startswith("+++") or line.startswith("---"):
                continue

            payload = line[1:] if line and line[0] in "+- " else line
            if re.match(r"^\s*import\s", payload) and " from " not in payload:
                in_import = True
            elif in_import and ("from " in payload or payload.rstrip().endswith(";")):
                in_import = False

            if line.startswith("+"):
                out.extend(_scan_line(line, "+", path, new_ln, in_import))
                new_ln += 1
            elif line.startswith("-"):
                out.extend(_scan_line(line, "-", path, old_ln, in_import))
                old_ln += 1
            else:
                old_ln += 1
                new_ln += 1

    return out


# --- Set arithmetic -------------------------------------------------------
# Generalized from diff_signals.graphql_contract_change, which uses the same
# trick to decide whether a .graphql change is client-visible.


def file_has_net_change(deltas: Sequence[LabelDelta]) -> bool:
    """Did this file's copy actually change?

    Prettier reflow shows up as `-aria-label="X"` / `+  aria-label="X"`: the same
    string on both sides. Set-equality kills it deterministically, with no model
    and no heuristic. This is the dominant false positive.
    """
    added = sorted(d.ident for d in deltas if d.sign == "+")
    removed = sorted(d.ident for d in deltas if d.sign == "-")
    return added != removed


def commit_net_change(
    deltas: Sequence[LabelDelta],
) -> tuple[list[LabelDelta], list[LabelDelta], list[LabelDelta]]:
    """Split commit-wide deltas into (added, removed, moved).

    A string removed from file X and added in file Y cancels globally: the user
    still sees it, it just lives somewhere else now. Without this, a drawer
    consolidation that relocates 23 strings files 23 false "removed" findings.
    """
    added = [d for d in deltas if d.sign == "+"]
    removed = [d for d in deltas if d.sign == "-"]
    both = {d.moved_ident for d in added} & {d.moved_ident for d in removed}

    moved = [d for d in added if d.moved_ident in both]
    return (
        [d for d in added if d.moved_ident not in both],
        [d for d in removed if d.moved_ident not in both],
        moved,
    )


def surviving_deltas(deltas: Sequence[LabelDelta]) -> list[LabelDelta]:
    """Drop files whose copy did not net-change, then return what remains."""
    by_path: dict[str, list[LabelDelta]] = {}
    for d in deltas:
        by_path.setdefault(d.path, []).append(d)
    out: list[LabelDelta] = []
    for path_deltas in by_path.values():
        if file_has_net_change(path_deltas):
            out.extend(path_deltas)
    return out
