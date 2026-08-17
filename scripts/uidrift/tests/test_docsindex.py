"""Docs-corpus index tests.

Mostly hermetic, against a synthetic corpus in a temp dir. One integration test
runs against the real wandb/docs tree and is skipped when it is absent.
"""

from __future__ import annotations

import dataclasses
import tempfile
import unittest
from pathlib import Path

from .. import config, docsindex

CORPUS = {
    "platform/members.mdx": (
        "---\n"
        "title: Manage members\n"
        "description: long frontmatter that shifts line numbers\n"
        "---\n"
        "\n"
        "# Manage members\n"
        "\n"
        "3. From the **MODELS SEAT** dropdown, select a level.\n"
        "4. Open the `Billing Admin` panel.\n"
        '5. Click the "Weave Access" control.\n'
        "6. Go to the Service Accounts tab to continue.\n"
        "7. You can add a panel to any section at any time.\n"
    ),
    "platform/other.mdx": (
        "---\ntitle: Other\n---\n"
        "Prose mentioning MODELS SEAT without any markup at all.\n"
    ),
    "release-notes/server-releases.mdx": (
        "---\ntitle: Releases\n---\n"
        "Renamed the **Hide sidebar** button.\n"
    ),
    "ja/platform/members.mdx": (
        "---\ntitle: メンバー\n---\n"
        "3. **MODELS SEAT** ドロップダウンから選択します。\n"
    ),
    "ko/platform/members.mdx": (
        "---\ntitle: 멤버\n---\n"
        "3. **MODELS SEAT** 드롭다운에서 선택합니다.\n"
    ),
    ".claude/worktrees/copy/platform/members.mdx": (
        "---\ntitle: Manage members\n---\n"
        "3. From the **MODELS SEAT** dropdown, select a level.\n"
    ),
}


def build_temp_index(tmpdir: str) -> docsindex.DocsIndex:
    root = Path(tmpdir)
    for rel, body in CORPUS.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    cfg = dataclasses.replace(config.DOCS, local_path_default=str(root))
    return docsindex.build_index(cfg)


class DocsIndexTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.index = build_temp_index(self._tmp.name)


class TestCorpusSelection(DocsIndexTestCase):

    def test_mirror_locales_are_not_in_the_primary_index(self):
        self.assertNotIn("ja/platform/members.mdx", self.index.pages)
        self.assertNotIn("ko/platform/members.mdx", self.index.pages)

    def test_worktrees_under_dot_claude_are_excluded(self):
        # A git worktree inside .claude would double-count every occurrence and
        # silently inflate blast radius.
        self.assertFalse([p for p in self.index.pages if p.startswith(".claude")])

    def test_release_notes_are_marked_immutable(self):
        idx = self.index.pages.index("release-notes/server-releases.mdx")
        self.assertTrue(self.index.immutable[idx])

    def test_ordinary_pages_are_not_immutable(self):
        idx = self.index.pages.index("platform/members.mdx")
        self.assertFalse(self.index.immutable[idx])


class TestCaseSensitivity(DocsIndexTestCase):
    """The case-only rename class depends entirely on this."""

    def test_old_casing_is_found(self):
        self.assertTrue(docsindex.find(self.index, "MODELS SEAT").ui_occurrences)

    def test_new_casing_is_not_found_in_place_of_the_old(self):
        # If this ever matches, the detector reports drift that is already fixed.
        self.assertFalse(docsindex.find(self.index, "Models Seat").ui_occurrences)

    def test_prose_mention_is_not_a_ui_reference(self):
        # "You can add a panel to any section" is a verb phrase, not a control.
        lookup = docsindex.find(self.index, "add a panel")
        self.assertFalse(lookup.ui_occurrences)


class TestContextClassification(DocsIndexTestCase):

    def _context_for(self, literal: str) -> set[str]:
        return {o.context for o in docsindex.find(self.index, literal).occurrences}

    def test_bold(self):
        self.assertIn(docsindex.CTX_BOLD, self._context_for("MODELS SEAT"))

    def test_code(self):
        self.assertIn(docsindex.CTX_CODE, self._context_for("Billing Admin"))

    def test_quoted(self):
        self.assertIn(docsindex.CTX_QUOTED, self._context_for("Weave Access"))

    def test_deictic(self):
        # "Go to the Service Accounts tab" -- no markup, but unambiguous.
        self.assertIn(docsindex.CTX_DEICTIC, self._context_for("Service Accounts"))

    def test_bare_prose_is_classified_as_prose(self):
        contexts = self._context_for("MODELS SEAT")
        self.assertIn(docsindex.CTX_PROSE, contexts)

    def test_mixed_contexts_block_the_all_emphasized_predicate(self):
        # MODELS SEAT appears bold on one page and bare on another, so a blind
        # find-and-replace is unsafe.
        self.assertFalse(
            docsindex.find(self.index, "MODELS SEAT").all_occurrences_emphasized
        )

    def test_uniformly_emphasized_literal_passes_the_predicate(self):
        self.assertTrue(
            docsindex.find(self.index, "Billing Admin").all_occurrences_emphasized
        )


class TestLineCitations(DocsIndexTestCase):

    def test_line_number_survives_frontmatter(self):
        # Frontmatter is blanked, not deleted, so citations stay exact.
        occ = docsindex.find(self.index, "MODELS SEAT").ui_occurrences[0]
        source = (Path(self._tmp.name) / occ.page).read_text().splitlines()
        self.assertIn("MODELS SEAT", source[occ.line - 1])

    def test_frontmatter_text_does_not_match(self):
        # The description key lives only in frontmatter, so it must not register
        # as page content. (`title` is a bad probe here: "Manage members" is also
        # a real H1 further down the page.)
        self.assertFalse(
            docsindex.find(self.index, "long frontmatter that shifts line numbers").occurrences
        )


class TestSpecificityGate(unittest.TestCase):

    def test_single_lowercase_token_is_ineligible(self):
        for literal in ("search", "delete", "Inference", "Threshold"):
            with self.subTest(literal=literal):
                ok, reason = docsindex.is_specific_enough(literal)
                self.assertFalse(ok)
                self.assertTrue(reason)

    def test_all_caps_single_token_is_eligible(self):
        self.assertTrue(docsindex.is_specific_enough("TEAMS")[0])

    def test_multi_word_is_eligible(self):
        self.assertTrue(docsindex.is_specific_enough("Add reference bucket")[0])

    def test_ineligible_lookup_is_not_a_coverage_claim(self):
        lookup = docsindex.DocsLookup("search", False, "too generic")
        self.assertFalse(lookup.eligible)
        self.assertEqual(lookup.occurrences, [])


class TestTranslations(DocsIndexTestCase):

    def test_mirrors_are_counted_but_kept_separate(self):
        lookup = docsindex.find(self.index, "MODELS SEAT")
        self.assertEqual(lookup.translations_affected, {"ja": 1, "ko": 1})
        # ...and never appear as replace targets.
        self.assertFalse([p for p in lookup.pages if p.startswith(("ja/", "ko/"))])

    def test_absent_locales_are_omitted_rather_than_zeroed(self):
        self.assertNotIn("fr", docsindex.find(self.index, "MODELS SEAT").translations_affected)


class TestDirectionDiscipline(unittest.TestCase):
    """Absence of docs must never be expressible as a penalty."""

    def test_no_public_function_returns_a_negative_number(self):
        import inspect

        for name, obj in vars(docsindex).items():
            if name.startswith("_") or not inspect.isfunction(obj):
                continue
            sig = inspect.signature(obj)
            ann = str(sig.return_annotation)
            self.assertNotIn("float", ann, f"{name} returns a score; docs absence must not be scoreable")

    def test_empty_lookup_reports_a_gap_not_a_penalty(self):
        lookup = docsindex.DocsLookup("Enable project memory", True, "")
        self.assertEqual(docsindex.coverage(lookup), "none")
        self.assertFalse(lookup.all_occurrences_emphasized)
        self.assertEqual(lookup.corpus_frequency, 0)


class TestLiveCorpus(unittest.TestCase):
    """Integration: the drift this project was built to find."""

    @classmethod
    def setUpClass(cls):
        target = config.DOCS.path / "platform/hosting/iam/access-management"
        if not target.exists():
            raise unittest.SkipTest("wandb/docs corpus not present")
        cls.index = docsindex.build_index()

    def test_flagship_drift_is_located_precisely(self):
        lookup = docsindex.find(self.index, "MODELS SEAT")
        occ = lookup.ui_occurrences
        self.assertTrue(occ, "MODELS SEAT should still be present in published docs")
        self.assertEqual(
            occ[0].page, "platform/hosting/iam/access-management/manage-organization.mdx"
        )
        self.assertEqual(occ[0].context, docsindex.CTX_BOLD)

    def test_flagship_is_agent_shaped(self):
        lookup = docsindex.find(self.index, "MODELS SEAT")
        self.assertTrue(lookup.all_occurrences_emphasized)
        self.assertLessEqual(lookup.corpus_frequency, config.MAX_DOCS_PAGES)
        self.assertFalse(lookup.touches_immutable)

    def test_generic_literals_are_rejected_before_search(self):
        self.assertFalse(docsindex.find(self.index, "search").eligible)


if __name__ == "__main__":
    unittest.main()


class TestMatchConfidence(DocsIndexTestCase):
    """Edge cases are reported at low confidence rather than classified."""

    def test_uniformly_emphasized_is_high(self):
        # Quoted, single occurrence, no bare prose anywhere.
        self.assertEqual(
            docsindex.find(self.index, "Weave Access").match_confidence, "high"
        )

    def test_code_context_only_is_medium(self):
        # A backticked string is as often an API value or CSV enum as a control,
        # so `Billing Admin` alone does not earn high confidence.
        self.assertEqual(
            docsindex.find(self.index, "Billing Admin").match_confidence, "medium"
        )

    def test_mixed_bold_and_prose_is_medium_not_blocked(self):
        # MODELS SEAT is bold on one page, bare prose on another. That is the
        # normal case, not a problem: the bold reference must track the UI, the
        # prose one follows the style guide.
        lookup = docsindex.find(self.index, "MODELS SEAT")
        self.assertEqual(lookup.match_confidence, "medium")
        self.assertTrue(lookup.replace_targets)

    def test_prose_only_is_low(self):
        lookup = docsindex.find(self.index, "add a panel")
        self.assertEqual(lookup.match_confidence, "low")
        self.assertEqual(lookup.replace_targets, [])

    def test_prose_is_never_a_replace_target(self):
        for occ in docsindex.find(self.index, "MODELS SEAT").replace_targets:
            self.assertNotEqual(occ.context, docsindex.CTX_PROSE)

    def test_immutable_pages_are_never_replace_targets(self):
        lookup = docsindex.find(self.index, "Hide sidebar")
        self.assertTrue(lookup.ui_occurrences, "should still be reported")
        self.assertEqual(lookup.replace_targets, [], "release notes are history")


class TestWholeTermMatching(DocsIndexTestCase):

    def test_plural_does_not_match_singular(self):
        # `Add panel` inside `Add panels` inflated one literal from 2 pages to
        # 16 and misfiled every bold occurrence as prose.
        page = Path(self._tmp.name) / "platform/panels.mdx"
        page.write_text(
            "---\ntitle: Panels\n---\nClick **Add panels** in the control bar.\n",
            encoding="utf-8",
        )
        cfg = dataclasses.replace(config.DOCS, local_path_default=str(self._tmp.name))
        index = docsindex.build_index(cfg)
        self.assertFalse(docsindex.find(index, "Add panel").occurrences)
        self.assertTrue(docsindex.find(index, "Add panels").ui_occurrences)
