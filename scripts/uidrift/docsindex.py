"""The docs corpus, indexed for one question: where does this UI label appear?

DIRECTION DISCIPLINE. This module can raise confidence that a finding is real.
It can never lower it. There is deliberately no function here that returns a
score, a penalty, or a "not documented" verdict that a caller could subtract.

The reason is a feedback loop, not style. If missing docs were allowed to
suppress a finding, then: a surface looks unreleased -> we suppress the row ->
nobody writes the page -> there are still no docs -> it is still suppressed.
Forever. And "available but undocumented" is precisely the gap this whole
project exists to find. So absence of docs is routed to a human as a coverage
gap; it is never evidence of anything.

The asymmetry is enforced by what this module refuses to expose, so a future
caller cannot reintroduce the loop by accident.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import config

# Context in which a literal appears on a page. Everything except `prose` is a
# UI reference -- the writer marked it up as a thing on screen.
CTX_BOLD = "bold"
CTX_CODE = "code"
CTX_QUOTED = "quoted"
CTX_DEICTIC = "deictic"
CTX_PROSE = "prose"

UI_CONTEXTS = frozenset({CTX_BOLD, CTX_CODE, CTX_QUOTED, CTX_DEICTIC})

# Nouns that mark "the X <noun>" as a reference to a control rather than prose.
_UI_NOUNS = (
    "button|tab|toggle|field|menu|dropdown|drop-down|column|setting|settings"
    "|page|panel|section|checkbox|dialog|modal|drawer|link|icon|header|option"
)

_TOKEN = re.compile(r"[a-z0-9]+")
_FRONTMATTER = re.compile(r"\A---\n.*?\n---\n", re.S)


@dataclass(frozen=True)
class DocsOccurrence:
    page: str  # repo-relative
    line: int
    context: str
    locale: str
    immutable: bool  # published history; report, never propose an edit

    @property
    def is_ui_reference(self) -> bool:
        return self.context in UI_CONTEXTS


@dataclass
class DocsLookup:
    """Result of looking one literal up in the corpus.

    Carries no score. `eligible=False` means the literal was too generic to
    search for meaningfully -- which is NOT the same as "this surface is
    undocumented", and callers must not conflate them.
    """

    literal: str
    eligible: bool
    reason: str
    occurrences: list[DocsOccurrence] = field(default_factory=list)
    translations_affected: dict[str, int] = field(default_factory=dict)

    @property
    def ui_occurrences(self) -> list[DocsOccurrence]:
        return [o for o in self.occurrences if o.is_ui_reference]

    @property
    def pages(self) -> list[str]:
        seen, out = set(), []
        for o in self.ui_occurrences:
            if o.page not in seen:
                seen.add(o.page)
                out.append(o.page)
        return out

    @property
    def corpus_frequency(self) -> int:
        """Pages the literal appears on at all, emphasized or not."""
        return len({o.page for o in self.occurrences})

    @property
    def all_occurrences_emphasized(self) -> bool:
        """Every appearance is inside UI markup.

        The predicate that decides whether a find-and-replace is safe.
        `**MODELS SEAT**` inside a numbered step is a token substitution with no
        grammatical consequence. The same words in "users with a models seat
        can..." is prose that a substitution would mangle.
        """
        return bool(self.occurrences) and all(o.is_ui_reference for o in self.occurrences)

    @property
    def touches_immutable(self) -> bool:
        return any(o.immutable for o in self.occurrences)

    @property
    def match_confidence(self) -> str:
        """How sure are we that these occurrences are references to the control?

        Deliberately three coarse buckets, not a taxonomy. The docs corpus has
        more context shapes than are worth classifying -- SDK output, MDX
        component props, headings, CSV enum values -- and chasing each one buys
        less than reporting the edge case at low confidence and letting a human
        glance at it.

        The rule this encodes: text a writer marked up as a control must match
        the UI exactly, so a rename is real drift. Text in a run of prose is
        governed by the style guide, not by the UI, so a rename usually means
        nothing there.

        NB: this grades MATCH QUALITY, not coverage. Absence of docs still
        cannot lower anything -- see the module docstring.
        """
        if not self.occurrences:
            return "low"
        if not self.ui_occurrences:
            return "low"  # prose only: probably style-governed, not a control
        if all(o.context == CTX_CODE for o in self.ui_occurrences):
            # A backticked string is as often an API value or identifier as a
            # UI label -- `"Models Seat"` in org_dashboard.mdx is a CSV column
            # value, not a button.
            return "medium"
        if self.all_occurrences_emphasized:
            return "high"
        return "medium"

    @property
    def replace_targets(self) -> list[DocsOccurrence]:
        """The occurrences a fix may touch: marked-up, mutable ones only.

        Prose is never a target. If the UI renames `MODELS SEAT` to
        `Models Seat`, the bold reference must follow, but "anyone with a models
        seat can write runs" should stay lowercase per the style guide. Leaving
        the page mixed is correct, not inconsistent.
        """
        return [o for o in self.ui_occurrences if not o.immutable]


@dataclass
class DocsIndex:
    root: Path
    primary_locale: str
    pages: list[str]
    text: list[str]
    lines: list[list[str]]
    immutable: list[bool]
    token_pages: dict[str, set[int]]
    # locale -> page -> raw text, kept apart so mirrors never become
    # find-and-replace targets; they are blast-radius reporting only.
    mirrors: dict[str, dict[str, str]]

    def __len__(self) -> int:
        return len(self.pages)


def _is_excluded(rel: Path, cfg: config.DocsRepo) -> bool:
    return any(part in cfg.exclude_dirs for part in rel.parts)


def _locale_of(rel: Path, cfg: config.DocsRepo) -> str:
    head = rel.parts[0] if rel.parts else ""
    return head if head in cfg.mirror_locales else cfg.primary_locale


def _strip_frontmatter(text: str) -> str:
    """Blank out YAML frontmatter, preserving the line count.

    Deleting it outright shifts every line number in the file, so a reported
    `page:line` no longer resolves to what a reader sees. Replacing it with the
    same number of newlines keeps citations exact while stopping frontmatter
    keywords from matching as prose.
    """
    m = _FRONTMATTER.match(text)
    if not m:
        return text
    return "\n" * m.group(0).count("\n") + text[m.end():]


def build_index(cfg: config.DocsRepo = config.DOCS) -> DocsIndex:
    """Load the primary-locale corpus into memory, plus mirror text for counts."""
    root = cfg.path.resolve()
    pages: list[str] = []
    text: list[str] = []
    lines: list[list[str]] = []
    immutable: list[bool] = []
    token_pages: dict[str, set[int]] = {}
    mirrors: dict[str, dict[str, str]] = {loc: {} for loc in cfg.mirror_locales}

    for ext in cfg.content_exts:
        for path in root.rglob(f"*{ext}"):
            rel = path.relative_to(root)
            if _is_excluded(rel, cfg):
                continue
            try:
                body = _strip_frontmatter(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError):
                continue

            locale = _locale_of(rel, cfg)
            relstr = rel.as_posix()
            if locale != cfg.primary_locale:
                mirrors[locale][relstr] = body
                continue

            idx = len(pages)
            pages.append(relstr)
            text.append(body)
            lines.append(body.splitlines())
            immutable.append(
                any(fnmatch.fnmatch(relstr, g) for g in cfg.immutable_globs)
            )
            for tok in set(_TOKEN.findall(body.lower())):
                token_pages.setdefault(tok, set()).add(idx)

    return DocsIndex(
        root, cfg.primary_locale, pages, text, lines, immutable, token_pages, mirrors
    )


def is_specific_enough(literal: str) -> tuple[bool, str]:
    """Is this literal distinctive enough to look up?

    Calibration: naive substring matching puts `search` on 215 pages and
    `delete` on 140. A single Title-Case token in backticks in docs is
    overwhelmingly a code identifier, not a UI label. Requiring two words OR
    all-caps keeps `MODELS SEAT` and `Add reference bucket` while dropping
    `Inference`, `Threshold`, `download`.
    """
    s = literal.strip()
    if len(s) < 3:
        return False, "literal too short"
    if len(s.split()) >= 2:
        return True, ""
    if s.isupper():
        return True, ""
    return False, "single token and not all-caps: too generic to attribute"


def _literal_regex(literal: str) -> re.Pattern[str]:
    """Match the literal as a whole term, not as a substring.

    `Add panel` occurs inside `Add panels`, and without boundaries that one
    plural inflates the literal from 2 pages to 16 -- enough to trip the
    too-generic cap and suppress a real finding -- while also misfiling every
    bold `**Add panels**` as unemphasized prose, because the bold matcher then
    fails on the trailing `s`.

    Lookarounds rather than \\b so literals that begin or end with punctuation
    (`+ Add panel`, `Save...`) still match.
    """
    return re.compile(r"(?<![A-Za-z0-9])" + re.escape(literal) + r"(?![A-Za-z0-9])")


def _context_regexes(literal: str) -> list[tuple[str, re.Pattern[str]]]:
    """Build the UI-emphasis matchers for one literal.

    The literal itself is matched CASE-SENSITIVELY, on purpose. The whole
    case-only rename class depends on it: if docs say `MODELS SEAT` and the code
    now says `Models Seat`, that is real drift. If docs already say
    `Models Seat`, there is nothing to fix. A case-insensitive match cannot tell
    those apart and would report drift that has already been fixed.

    Surrounding words use a scoped (?i:...) so `The`/`the` both work.
    """
    e = r"(?<![A-Za-z0-9])" + re.escape(literal) + r"(?![A-Za-z0-9])"
    return [
        (CTX_BOLD, re.compile(r"\*\*\s*" + e + r"\s*\*\*")),
        (CTX_CODE, re.compile(r"`" + e + r"`")),
        (CTX_QUOTED, re.compile(r"[\"“”]" + e + r"[\"“”]")),
        (CTX_DEICTIC, re.compile(r"(?i:\bthe)\s+" + e + r"\s+(?i:" + _UI_NOUNS + r")\b")),
    ]


def _candidate_pages(index: DocsIndex, literal: str) -> set[int]:
    """Narrow to pages that could contain the literal, using the token index."""
    tokens = set(_TOKEN.findall(literal.lower()))
    if not tokens:
        return set()
    sets = [index.token_pages.get(t, set()) for t in tokens]
    if any(not s for s in sets):
        return set()
    return set.intersection(*sets)


def find(index: DocsIndex, literal: str) -> DocsLookup:
    """Locate every appearance of `literal` in the primary-locale corpus."""
    eligible, reason = is_specific_enough(literal)
    if not eligible:
        return DocsLookup(literal, False, reason)

    matchers = _context_regexes(literal)
    term = _literal_regex(literal)
    occurrences: list[DocsOccurrence] = []

    for page_idx in _candidate_pages(index, literal):
        if not term.search(index.text[page_idx]):
            continue
        for line_no, line in enumerate(index.lines[page_idx], start=1):
            if not term.search(line):
                continue
            context = CTX_PROSE
            for name, rx in matchers:
                if rx.search(line):
                    context = name
                    break
            occurrences.append(
                DocsOccurrence(
                    page=index.pages[page_idx],
                    line=line_no,
                    context=context,
                    locale=index.primary_locale,
                    immutable=index.immutable[page_idx],
                )
            )

    lookup = DocsLookup(literal, True, "", occurrences)
    lookup.translations_affected = {
        loc: sum(1 for body in pages.values() if term.search(body))
        for loc, pages in index.mirrors.items()
    }
    lookup.translations_affected = {
        k: v for k, v in lookup.translations_affected.items() if v
    }
    return lookup


def coverage(lookup: DocsLookup, cfg_max_pages: Optional[int] = None) -> str:
    """`covered` or `none`.

    Returning `none` is a FINDING, not a failure and not a penalty: it means
    this surface exists in the product and nothing documents it. It routes to a
    human as a coverage gap. It must never be read as "this change does not
    matter" -- see the module docstring.
    """
    from .finding import COVERAGE_COVERED, COVERAGE_NONE

    max_pages = config.MAX_DOCS_PAGES if cfg_max_pages is None else cfg_max_pages
    if not lookup.eligible:
        return COVERAGE_NONE
    if not lookup.ui_occurrences:
        return COVERAGE_NONE
    if len({o.page for o in lookup.ui_occurrences}) > max_pages:
        # Too widespread to be a specific control. Not a coverage claim either
        # way; the triage rule routes this to a pair for a human read.
        return COVERAGE_COVERED
    return COVERAGE_COVERED
