"""Triage, finding assembly, and report rendering."""

from __future__ import annotations

import dataclasses
import tempfile
import unittest
from datetime import date
from pathlib import Path

from .. import build, config, docsindex, extract, finding, report
from .test_docsindex import build_temp_index

FIXTURES = Path(__file__).parent / "fixtures"
TODAY = date(2026, 8, 12)


def make(**kw) -> finding.Finding:
    base = dict(
        kind=finding.KIND_RENAME,
        surface="Members table",
        old_string="MODELS SEAT",
        new_string="Models Seat",
        literal_kind="jsx",
        literal_key="_",
        settled=True,
        docs={
            "coverage": finding.COVERAGE_COVERED,
            "replace_targets": [{"page": "a.mdx", "line": 290, "context": "bold"}],
            "corpus_frequency": 1,
            "code_context_only": False,
        },
    )
    base.update(kw)
    return finding.Finding(**base)


class TestTriageLanes(unittest.TestCase):

    def test_clean_rename_reaches_the_agent_lane(self):
        self.assertEqual(finding.triage(make())[0], finding.TRIAGE_AGENT)

    def test_new_setting_needs_a_human(self):
        f = make(kind=finding.KIND_NEW_SETTING, old_string="")
        self.assertEqual(finding.triage(f)[0], finding.TRIAGE_HUMAN)

    def test_removal_of_a_documented_control_needs_a_human(self):
        self.assertEqual(
            finding.triage(make(kind=finding.KIND_REMOVED, new_string=""))[0],
            finding.TRIAGE_HUMAN,
        )

    def test_gated_change_pairs_rather_than_shipping(self):
        # Docs must not describe a control users cannot see yet.
        self.assertEqual(finding.triage(make(not_yet_visible=True))[0], finding.TRIAGE_PAIR)

    def test_unsettled_change_pairs(self):
        self.assertEqual(finding.triage(make(settled=False))[0], finding.TRIAGE_PAIR)

    def test_moved_string_pairs(self):
        self.assertEqual(finding.triage(make(kind=finding.KIND_MOVED))[0], finding.TRIAGE_PAIR)

    def test_ambiguous_pairing_pairs(self):
        self.assertEqual(
            finding.triage(make(signals=["ambiguous_pairing"]))[0], finding.TRIAGE_PAIR
        )

    def test_changed_slug_pairs(self):
        self.assertEqual(finding.triage(make(signals=["url_changed"]))[0], finding.TRIAGE_PAIR)

    def test_release_notes_only_pairs(self):
        f = make(docs={**make().docs, "replace_targets": []})
        lane, reason = finding.triage(f)
        self.assertEqual(lane, finding.TRIAGE_PAIR)
        self.assertIn("release notes", reason)

    def test_code_context_only_pairs(self):
        # A backticked string may be an API value rather than a control.
        f = make(docs={**make().docs, "code_context_only": True})
        self.assertEqual(finding.triage(f)[0], finding.TRIAGE_PAIR)

    def test_too_many_pages_pairs(self):
        f = make(docs={**make().docs, "corpus_frequency": config.MAX_DOCS_PAGES + 1})
        self.assertEqual(finding.triage(f)[0], finding.TRIAGE_PAIR)

    def test_degraded_evidence_never_reaches_agent(self):
        f = make(degradations=["literal is interpolated or a ternary branch"])
        self.assertEqual(finding.triage(f)[0], finding.TRIAGE_HUMAN)

    def test_mixed_bold_and_prose_still_reaches_agent(self):
        # Bold references must track the UI; prose answers to the style guide.
        # A page ending up mixed is correct, so it must not block the lane.
        f = make(docs={**make().docs, "all_occurrences_emphasized": False,
                       "match_confidence": "medium"})
        self.assertEqual(finding.triage(f)[0], finding.TRIAGE_AGENT)

    def test_every_lane_gives_a_reason(self):
        for f in (make(), make(not_yet_visible=True), make(kind=finding.KIND_NEW_SETTING)):
            self.assertTrue(finding.triage(f)[1].strip())


class TestFindingIdentity(unittest.TestCase):

    def test_same_rename_on_different_surfaces_is_one_finding(self):
        # Three member tables render the same column. The docs page says it
        # once, so it is one edit.
        a = make(surface="Organization members table")
        b = make(surface="Team members table")
        self.assertEqual(a.id, b.id)

    def test_different_renames_are_different_findings(self):
        self.assertNotEqual(make().id, make(old_string="WEAVE ACCESS").id)


class BuildTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.index = build_temp_index(self._tmp.name)

    def run_fixture(self, stem: str, when: str):
        diff = (FIXTURES / f"{stem}.diff").read_text()
        surviving = extract.surviving_deltas(extract.extract_deltas(diff))
        added, removed, moved = extract.commit_net_change(surviving)
        commit = {
            "sha": stem + "0" * 33,
            "commit": {"message": f"test: {stem}", "author": {"date": when, "name": "tester"}},
        }
        return build.build_findings(
            commit, added, removed, moved, diff, self.index,
            today=TODAY, resolve_owners=False,
        )


class TestBuildSuppressesNonDrift(BuildTestCase):

    def test_undocumented_rename_is_counted_not_listed(self):
        # Renaming a label no page mentions makes nothing wrong. Listing each
        # one buried the real findings under twenty rows of empty-state copy.
        findings, gaps = self.run_fixture("f4861ad", "2026-07-01T13:52:32-07:00")
        listed = {f.old_string for f in findings}
        self.assertNotIn("PROFILE", listed)
        self.assertNotIn("EMAIL", listed)
        self.assertIn("PROFILE", set(gaps))

    def test_documented_rename_is_listed(self):
        findings, _gaps = self.run_fixture("f4861ad", "2026-07-01T13:52:32-07:00")
        self.assertIn("MODELS SEAT", {f.old_string for f in findings})

    def test_incidental_new_copy_is_not_a_finding(self):
        findings, gaps = self.run_fixture("f4861ad", "2026-07-01T13:52:32-07:00")
        self.assertNotIn("Loading members", {f.new_string for f in findings})
        self.assertIn("Loading members", set(gaps))

    def test_new_settings_panel_is_one_row_not_many(self):
        findings, _gaps = self.run_fixture("e1bc1e6", "2026-08-10T10:00:00-07:00")
        settings = [f for f in findings if f.kind == finding.KIND_NEW_SETTING]
        self.assertEqual(len(settings), 1, "a new settings panel is one docs task")
        self.assertEqual(settings[0].new_string, "Enable project memory")


class TestBuildSignals(BuildTestCase):

    def test_gated_new_setting_is_flagged_not_visible(self):
        findings, _ = self.run_fixture("e1bc1e6", "2026-08-10T10:00:00-07:00")
        f = next(f for f in findings if f.kind == finding.KIND_NEW_SETTING)
        self.assertTrue(f.not_yet_visible)
        self.assertIsNotNone(f.gate)

    def test_recent_change_is_unsettled(self):
        findings, _ = self.run_fixture("e1bc1e6", "2026-08-10T10:00:00-07:00")
        self.assertFalse(any(f.settled for f in findings))

    def test_old_change_is_settled(self):
        findings, _ = self.run_fixture("f4861ad", "2026-07-01T13:52:32-07:00")
        self.assertTrue(all(f.settled for f in findings))


class TestSurfaceNaming(unittest.TestCase):

    def test_acronyms_survive(self):
        self.assertEqual(
            build.surface_from_path("a/LLMAsAJudgeScorerForm.tsx"),
            "LLM As A Judge Scorer Form",
        )

    def test_camel_case_is_split(self):
        self.assertEqual(
            build.surface_from_path("a/OrganizationMembersTable.tsx"),
            "Organization Members Table",
        )


class TestReport(unittest.TestCase):

    def _render(self, findings, gaps=0):
        return report.render(
            findings, scanned_range="a..b", today=TODAY,
            commits=1, ui_commits=1, candidates=1, docs_pages=10, gaps=gaps,
        )

    def test_empty_report_says_so_explicitly(self):
        out = self._render([])
        self.assertIn("No drift found", out)
        self.assertIn("real result", out, "an empty table must not read as a broken run")

    def test_agent_rows_are_listed_first(self):
        f = make()
        f.triage, f.triage_reason = finding.triage(f)
        out = self._render([f])
        self.assertLess(out.index("Agent can fix unattended"), out.index("MODELS SEAT"))

    def test_pipes_in_strings_do_not_break_the_table(self):
        f = make(old_string="a | b", new_string="c")
        f.triage, f.triage_reason = finding.triage(f)
        row = [l for l in self._render([f]).splitlines() if "`a \\| b`" in l]
        self.assertTrue(row, "pipe must be escaped or the markdown table collapses")

    def test_merged_surfaces_are_disclosed(self):
        f = make()
        f.surfaces = ["Users table", "Team members table", "Organization members table"]
        f.triage, f.triage_reason = finding.triage(f)
        self.assertIn("(+2 more)", self._render([f]))

    def test_gap_count_is_reported_without_rows(self):
        f = make()
        f.triage, f.triage_reason = finding.triage(f)
        out = self._render([f], gaps=890)
        self.assertIn("890", out)
        self.assertIn("Undocumented surfaces", out)


if __name__ == "__main__":
    unittest.main()
