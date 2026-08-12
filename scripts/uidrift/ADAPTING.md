# Adapting this detector to another repo

This is a worked example, not a framework. There is no plugin interface to
implement and no abstract base class to subclass — those would require guessing
the second implementation's shape before anyone has built one. Instead, this
file states what is generic, what is `wandb/core`-specific, and what surprised
us, so that adapting it is a reading exercise rather than an archaeology one.

If you are an agent being told *"do this, but watch `coreweave/sunk` for
releases"* — read this file first, then `config.py`, then `extract.py`. The
other modules follow from those.

## The four-part anatomy

Every repo-watching doc-drift detector has the same four parts. Only the second
column changes.

| Part | Generic — reuse as-is | Repo-specific — expect to rewrite |
|---|---|---|
| **1. Event detector** | commit iteration, diff splitting, set-equality dedupe | which paths are user-facing; which literal patterns are labels |
| **2. Evidence** | structural signals, gate scope, ownership | gate registry location and mechanism; CODEOWNERS shape |
| **3. Triage** | the agent/pair/human decision procedure | thresholds; which paths are immutable |
| **4. Sink** | markdown table renderer, ledger, `[skip ci]` commit-back | report location; JIRA project and component |

Part 1 is where nearly all the adaptation cost lives. Parts 3 and 4 usually
transfer unchanged.

## Step zero for a new repo: is there an i18n catalog?

**Ask this before anything else.** It determines whether the job takes two days
or two weeks.

- **A catalog exists** (`en.json`, `messages.po`, `.ftl`, an i18next/Lingui/
  react-intl setup): you are diffing structured key-value pairs. Keys give you
  free stable identity across renames, and you can skip most of `extract.py`.
  This is the easy case.
- **No catalog** — `wandb/core`'s situation: strings are inline in JSX and you
  are parsing source. Everything below applies.

How to check, quickly:

```bash
git -C <repo> grep -lE 'useTranslation|defineMessages|FormattedMessage|i18next|@lingui' <ref> | head
git -C <repo> ls-tree -r --name-only <ref> | grep -iE 'locales?/|translations?/|/en\.json$'
```

For `wandb/core` at `origin/master` both return nothing. There is no i18n
toolkit in the repo at all. (There is a Locadex/gt-react localization pilot, but
it runs against a *fork* — `wandb/mattcore` — and does not affect `wandb/core`.)

## What surprised us on wandb/core

These are the findings that cost real time. They are the reason this file exists.

### 1. Enumerating attribute names guarantees silent misses

The first extractor listed the attributes that carry copy: `aria-label`,
`placeholder`, `title`, `tooltip`. It scored **zero** on a drawer-consolidation
commit, because that component library takes its copy as `saveLabel=`,
`cancelLabel=`, `isPendingAriaLabel=`. A design system invents new label props
as it grows, and the list is open-ended.

**Match on suffix, not on membership.** See `_ATTR_SUFFIX` in `extract.py`.
The cost is that `name=` becomes ambiguous (`<Icon name="info" />` is an
identifier, `<Hotkey name="List only visible runs" />` is copy), handled by
rejecting slug-shaped values for ambiguous keys only.

### 2. Prettier reflow is the dominant false positive

Re-indentation shows up as `-aria-label="X"` / `+  aria-label="X"` — a removal
and an addition of the same string. Killed deterministically by per-file
set-equality, with no model and no heuristic. Roughly 25 of 233 label-touching
commits in a 60-day window are pure reflow.

Generalized from `diff_signals.graphql_contract_change`, which uses the same
trick to decide whether a `.graphql` change is client-visible.

### 3. Refactor-titled commits are the dominant false negative

**Never filter on conventional-commit type.** The single richest real finding in
the corpus — eight table headers title-cased, still wrong in published docs six
weeks later — arrived half under `feat(app): migrate ... to Table` and half
under `refactor(app): migrate OrgDashboard UsersTable`. Neither subject line
suggests user-visible copy changed. Both changed it.

The commit type is recorded as metadata and used by nothing.

### 4. Never normalize case

`normalize()` collapses whitespace and stops there. Lowercasing would make
`MODELS SEAT` and `Models Seat` identical, so the case-only rename would cancel
itself out in the set arithmetic and vanish without a trace. There is a test
pinning this (`test_case_is_never_normalized`) precisely because the failure is
silent.

### 5. Move detection needs a *looser* identity than reflow detection

These are two different questions and they want two different keys:

- *Did this file's copy change?* → strict identity `(kind, key, string)`.
  Reflow preserves the expression form exactly, so form must participate.
- *Did this string leave the product?* → the string alone.

A drawer consolidation moved `Add secret` out of `<span>Add secret</span>` and
into `saveLabel="Add secret"`. Same string, different form, still on screen.
Keyed strictly, that commit reports 23 phantom removals — 23 false rows in the
very first report anyone reads. See `LabelDelta.ident` vs `.moved_ident`.

### 6. `wrapped` means "not a complete literal", not "Prettier moved it"

Easy to conflate, and conflating it disqualifies good findings from the agent
lane for no reason. Interpolation (`` `Allow ${AGENT_NAME} to ...` ``) and
ternary branches are genuinely unsafe to find-and-replace. Text that Prettier
pushed onto its own line is captured exactly and is perfectly safe.

### 7. Merged ≠ visible, and flag *presence* is a decayed signal

New UI ships behind Statsig ramp flags. But engineers rarely remove a flag once
it reaches 100% — leaving it is safer — so a gate being present tells you almost
nothing. Do not suppress on it.

The **lifecycle events** are the signal:

| Event | Meaning |
|---|---|
| Flag added in the same commit as the copy | not visible yet |
| Flag removal commit | GA |
| Flag merely present, added long ago | no signal — ignore |

Gating is also **per-surface, not per-feature**: one gate governs three surfaces
in `APIKeysTabContent.tsx`, with different answers for each. The diff shows
whether the changed element sits inside the conditional; use that, not the flag
name.

Deployment semantics for `wandb/core` specifically are involved enough to live
in their own skill — see `beta-deployment-availability` in `coreweave/docs-skills`.
Do not re-derive them here.

### 8. The docs oracle runs in one direction only

Docs presence **raises** confidence that a surface is live, so drift on it is
real. Docs absence must **never** lower it — "available but undocumented" is
precisely the gap being hunted, and using absence to suppress closes a loop the
detector never escapes: looks unreleased → suppress → nobody writes docs → still
no docs → still suppressed.

This is enforced structurally rather than by convention: `docsindex` exposes no
function that returns a negative score, so the loop is unrepresentable.

Also: naive substring matching is useless. `search` appears on 215 docs pages.
Require UI-emphasis context (`**bold**`, backticks, quotes, or "the X button"),
a ≥2-token-or-ALL-CAPS specificity gate, and a page-count cap.

### 9. Match the literal case-sensitively, or you report already-fixed drift

Non-obvious and easy to get backwards. The lookup asks "does the OLD string
still appear in docs?" If docs say `MODELS SEAT` and the code now says
`Models Seat`, that is drift. If docs already say `Models Seat`, there is
nothing to do. A case-insensitive match cannot tell those apart, so it reports
the fixed page as broken — and the case-only rename is exactly the class where
this matters most.

Surrounding words (`the`, the noun) can be case-insensitive via a scoped
`(?i:...)`. The literal itself must not be.

### 10. Blank frontmatter; do not delete it

Deleting YAML frontmatter shifts every line number after it, so a reported
`page:line` stops resolving to what a reader sees — off by five, in our corpus.
Replace it with an equal number of newlines instead. Cheap, and it keeps
citations exact while still preventing frontmatter keys from matching as prose.

### 11. Published release notes are immutable, and they are a big share of hits

Roughly half the docs hits in a 60-day window land in `release-notes/**`. Those
are a historical record of what shipped under the name it shipped under.
Rewriting them would be falsifying a changelog. Report them for awareness, never
propose an edit, and never count them toward agent eligibility.

### 12. Include reusable fragments; exclude worktrees

Two corpus-selection mistakes with opposite signs:

- **`snippets/`** carries real UI prose (`go to the **Service Accounts** tab`)
  and renders into many pages, so a label there has *wider* blast radius than
  one in a single page. Excluding it creates blind spots.
- **`.claude/`** contains git worktrees — full copies of the tree. Indexing it
  double-counts every occurrence and silently inflates page counts, which then
  trips the too-generic cap and suppresses real findings.

### 13. Pair renames by position before you consider similarity

The obvious approach — match a removed string to the added string it most
resembles — fails on the most important case. A label that was genuinely
reworded shares almost no characters with its replacement:

| Old | New | Similarity |
|---|---|---|
| `Hide manually hidden runs` | `List only visible runs` | 0.55 |
| `Only show visualized` | `Hide manually hidden runs` | 0.22 |

No threshold catches those and still refuses to pair two unrelated column
headers. But git already answers the question: an in-place edit appears as a `-`
line and the `+` line that replaced it at the same offset in one change block.
Position is stronger evidence than string similarity, and it has no false-pair
failure mode.

Similarity is still worth a second pass, for renames that are *not* in-place —
`header: 'WEAVE ACCESS'` became `name: 'Weave Access'` on a different line and a
different field. Group by (path, kind), not (path, kind, key), or that one is
invisible.

### 14. Not every conditional is a feature gate

Walking up from a changed line to the enclosing `if` finds plenty of blocks that
have nothing to do with visibility. `if (hideManuallyHidden)` is UI state.
Reporting it as a gate would mark half the app "not yet visible" and destroy
trust in the one signal that should mean something.

Require the conditional's variable to resolve to a gate hook —
`const shouldShowX = useStatsigGateX(orgName)` — and report nothing when it does
not. The chain is fully readable inside a single diff. The Statsig key itself
usually is not; it lives in the ramp registry, so treat it as optional
enrichment rather than a precondition.

### 15. Freeze real diffs as fixtures, immediately

Six frozen `git show` outputs in `tests/fixtures/` are the entire regression
surface, and they caught three bugs that survived design review: the inline-JSX
miss, the enumerated-attribute miss, and the move-identity bug. None of these
were visible in the plan. All three were obvious within a minute of running
against real diffs.

When someone reports a miss, add it as a fixture before fixing it.

## Volume expectations

Calibrate before building. For `wandb/core` over 60 days:

| Stage | Count |
|---|---|
| Commits on `origin/master` | 2,990 |
| Touching `frontends/app/src/**/*.tsx` | 592 |
| **Stage-1 candidates** | **170** (~20/week) |
| …with a published-docs occurrence | **12** (~1.5/week) |
| Reduction | 71% at stage 1; 93% after the docs join |

The docs join is the real filter, and it is deterministic. Do not reach for a
model until after it: judging 170 commits costs an order of magnitude more than
judging the 12 that actually touch published copy.

Full scan runs in ~27 seconds with no network and no token, because
`gitsource.commit_diff` takes a pathspec and never fetches diffs outside the UI
roots. Keep that property: without it, a 2,600-line commit is unaffordable.

If your stage-1 count exceeds ~250/60d, tighten before adding stage 2 — a model
pass over the whole `.tsx` stream is mostly waste.

## The vendored modules

`_vendor/` holds copies of `gitsource.py`, `diff_signals.py`, and
`commit_text.py` from `wandb/release-note-genie`, each with a provenance header
naming the source commit. They are copies rather than imports on purpose: this
detector must run inside `wandb/docs` with no dependency on another repo being
checked out, and all three are stdlib-only and stable.

Re-vendor deliberately, never automatically.
