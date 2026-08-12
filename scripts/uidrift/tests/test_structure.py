"""Structural signal tests: pairing, gating, flag lifecycle, corroboration."""

from __future__ import annotations

import unittest
from pathlib import Path

from .. import extract, structure

FIXTURES = Path(__file__).parent / "fixtures"


def analyze(stem: str):
    diff = (FIXTURES / f"{stem}.diff").read_text()
    surviving = extract.surviving_deltas(extract.extract_deltas(diff))
    added, removed, moved = extract.commit_net_change(surviving)
    streams = structure.parse_streams(diff)
    pairs, unpaired_add, unpaired_rem = structure.pair_renames(added, removed, streams)
    return diff, streams, added, removed, moved, pairs, unpaired_add, unpaired_rem


def synth(*lines: str) -> str:
    head = (
        "diff --git a/frontends/app/src/A.tsx b/frontends/app/src/A.tsx\n"
        "--- a/frontends/app/src/A.tsx\n"
        "+++ b/frontends/app/src/A.tsx\n"
        "@@ -1,20 +1,20 @@\n"
    )
    return head + "".join(l + "\n" for l in lines)


class TestPositionalPairing(unittest.TestCase):
    """Position beats similarity, and it has to -- a full reword scores 0.55."""

    def test_complete_reword_pairs_despite_low_similarity(self):
        *_, pairs, ua, ur = analyze("ccd66e2")
        self.assertTrue(pairs)
        self.assertEqual((ua, ur), ([], []), "nothing should be left unpaired")
        p = pairs[0]
        self.assertEqual(p.old.norm, "Hide manually hidden runs")
        self.assertEqual(p.new.norm, "List only visible runs")
        self.assertLess(p.ratio, structure.RENAME_RATIO,
                        "if this ever exceeds the threshold the test has lost its point")

    def test_pairs_below_any_safe_similarity_threshold(self):
        # 'Only show visualized' -> 'Hide manually hidden runs' scores 0.22.
        *_, pairs, _ua, _ur = analyze("c99e959")
        self.assertTrue(pairs)
        self.assertLess(min(p.ratio for p in pairs), 0.3)

    def test_positional_pairs_are_never_flagged_ambiguous(self):
        *_, pairs, _ua, _ur = analyze("ccd66e2")
        self.assertFalse(any(p.ambiguous for p in pairs))


class TestSimilarityPairing(unittest.TestCase):

    def test_case_only_renames_pair(self):
        *_, pairs, _ua, _ur = analyze("f4861ad")
        by_old = {p.old.norm: p for p in pairs}
        for old, new in (("MODELS SEAT", "Models Seat"),
                         ("WEAVE ACCESS", "Weave Access"),
                         ("LAST ACTIVE", "Last Active"),
                         ("BILLING ADMIN", "Billing Admin")):
            with self.subTest(old=old):
                self.assertIn(old, by_old)
                self.assertEqual(by_old[old].new.norm, new)
                self.assertTrue(by_old[old].case_only)

    def test_pairing_survives_a_changed_field_name(self):
        # `header: 'WEAVE ACCESS'` became `name: 'Weave Access'`. Grouping on
        # (path, kind, key) would miss this entirely.
        *_, pairs, _ua, _ur = analyze("f4861ad")
        weave = next(p for p in pairs if p.old.norm == "WEAVE ACCESS")
        self.assertNotEqual(weave.old.key, weave.new.key)

    def test_multiple_candidates_are_flagged_ambiguous(self):
        *_, pairs, _ua, _ur = analyze("f4861ad")
        invited = [p for p in pairs if p.old.norm == "Invited User"]
        self.assertTrue(invited)
        self.assertTrue(invited[0].ambiguous, "two 'Invited' additions -- cannot resolve")

    def test_unmatched_removal_stays_unpaired(self):
        # ORG ROLE's replacement landed in a different commit; inventing a pair
        # here would be worse than reporting it as a removal.
        *_, pairs, _ua, unpaired_rem = analyze("f4861ad")
        self.assertIn("ORG ROLE", {d.norm for d in unpaired_rem})
        self.assertNotIn("ORG ROLE", {p.old.norm for p in pairs})


class TestGateScope(unittest.TestCase):

    def test_resolves_conditional_to_its_gate_hook(self):
        _diff, streams, added, *_ = analyze("e1bc1e6")
        setting = next(d for d in added if d.norm == "Enable project memory")
        scope = structure.gate_scope(streams, setting)
        self.assertIsNotNone(scope)
        self.assertEqual(scope.variable, "shouldShowARIAProjectMemory")
        self.assertEqual(scope.hook, "useStatsigGateARIAProjectMemory")

    def test_newly_added_conditional_is_flagged(self):
        # The gate pre-existed; the *conditional* is new. That is the visibility
        # evidence, and it is available without any flag-lifecycle signal.
        _diff, streams, added, *_ = analyze("e1bc1e6")
        setting = next(d for d in added if d.norm == "Enable project memory")
        self.assertTrue(structure.gate_scope(streams, setting).conditional_added)

    def test_plain_conditional_is_not_a_gate(self):
        # `hideManuallyHidden` is UI state. Treating every `if` as gating would
        # mark most of the app "not yet visible".
        _diff, streams, added, *_ = analyze("ccd66e2")
        self.assertTrue(added)
        self.assertTrue(all(structure.gate_scope(streams, d) is None for d in added))

    def test_ungated_change_reports_no_scope(self):
        _diff, streams, added, *_ = analyze("f4861ad")
        self.assertFalse(any(structure.gate_scope(streams, d) for d in added))


class TestFlagLifecycle(unittest.TestCase):

    def test_flag_added_in_the_same_commit(self):
        diff, *_ = analyze("cb100df")
        self.assertEqual(
            structure.flag_lifecycle(diff), {"reference_bucket_byob": "added"}
        )

    def test_flag_removal_is_detected(self):
        diff = (
            "diff --git a/frontends/app/src/util/useRampFlag.ts "
            "b/frontends/app/src/util/useRampFlag.ts\n"
            "--- a/frontends/app/src/util/useRampFlag.ts\n"
            "+++ b/frontends/app/src/util/useRampFlag.ts\n"
            "@@ -1,3 +1,2 @@\n"
            "   | 'kept_gate'\n"
            "-  | 'disable_fuzzy_search'\n"
        )
        self.assertEqual(
            structure.flag_lifecycle(diff), {"disable_fuzzy_search": "removed"}
        )

    def test_static_flag_presence_produces_no_signal(self):
        # A commit that merely *uses* an existing gate says nothing about
        # visibility -- engineers leave gates at 100% indefinitely.
        diff = synth("+  const x = useStatsigGateSomething(orgId);")
        self.assertEqual(structure.flag_lifecycle(diff), {})

    def test_unrelated_files_are_ignored(self):
        diff = synth("+  | 'looks_like_a_gate'")
        self.assertEqual(structure.flag_lifecycle(diff), {})


class TestCorroboration(unittest.TestCase):

    def test_unchanged_testid_beside_changed_text_corroborates(self):
        diff = synth(
            '   <Button data-test="save-btn"',
            '-    aria-label="Save changes"',
            '+    aria-label="Save edits"',
            "   />",
        )
        deltas = extract.extract_deltas(diff)
        streams = structure.parse_streams(diff)
        removed = next(d for d in deltas if d.sign == "-")
        self.assertTrue(structure.testid_corroboration(streams, removed))

    def test_absent_testid_is_not_evidence_against(self):
        diff = synth(
            '-    aria-label="Save changes"',
            '+    aria-label="Save edits"',
        )
        deltas = extract.extract_deltas(diff)
        streams = structure.parse_streams(diff)
        removed = next(d for d in deltas if d.sign == "-")
        self.assertFalse(structure.testid_corroboration(streams, removed))


class TestSlugStability(unittest.TestCase):

    def _pair_from(self, diff: str) -> structure.RenamePair:
        deltas = extract.surviving_deltas(extract.extract_deltas(diff))
        added, removed, _ = extract.commit_net_change(deltas)
        streams = structure.parse_streams(diff)
        pairs, _, _ = structure.pair_renames(added, removed, streams)
        return streams, pairs[0]

    def test_renamed_tab_with_untouched_slug_keeps_urls(self):
        diff = synth(
            "-  name: 'Runs',",
            "+  name: 'Run list',",
            "   slug: 'table',",
        )
        streams, pair = self._pair_from(diff)
        self.assertEqual(structure.slug_stability(streams, pair), "url_stable")

    def test_changed_slug_breaks_urls(self):
        diff = synth(
            "-  name: 'Runs',",
            "+  name: 'Run list',",
            "-  slug: 'table',",
            "+  slug: 'run-list',",
        )
        streams, pair = self._pair_from(diff)
        self.assertEqual(structure.slug_stability(streams, pair), "url_changed")


class TestNewSetting(unittest.TestCase):

    def test_title_plus_description_in_a_settings_path(self):
        _diff, _streams, added, *_ = analyze("e1bc1e6")
        paths = {d.path for d in added}
        self.assertTrue(any(structure.is_new_setting(added, p) for p in paths))

    def test_a_table_migration_is_not_a_new_setting(self):
        _diff, _streams, added, *_ = analyze("f4861ad")
        paths = {d.path for d in added}
        self.assertFalse(any(structure.is_new_setting(added, p) for p in paths))


class TestStreamParsing(unittest.TestCase):

    def test_line_numbers_track_both_sides(self):
        diff = synth(
            "   context",
            "-  removed line",
            "+  added line",
            "   more context",
        )
        lines = structure.parse_streams(diff)["frontends/app/src/A.tsx"]
        removed = next(l for l in lines if l.sign == "-")
        added = next(l for l in lines if l.sign == "+")
        self.assertEqual(removed.old_no, 2)
        self.assertEqual(added.new_no, 2)

    def test_change_blocks_split_on_context(self):
        diff = synth(
            "-  a", "+  b",
            "   ctx",
            "-  c", "+  d",
        )
        lines = structure.parse_streams(diff)["frontends/app/src/A.tsx"]
        blocks = structure._change_blocks(lines)
        self.assertEqual(len(blocks), 2)


if __name__ == "__main__":
    unittest.main()
