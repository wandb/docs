"""Cross-commit merge, decision persistence, and the reopen rule."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from .. import ledger, report
from ..finding import (
    COVERAGE_COVERED,
    COVERAGE_NONE,
    KIND_ADDED,
    KIND_RENAME,
    TRIAGE_AGENT,
    TRIAGE_PAIR,
    CommitRef,
    Finding,
    action_for,
    triage,
)

TODAY = date(2026, 8, 13)
LONG_AGO = "2026-06-01T00:00:00+00:00"
ALSO_LONG_AGO = "2026-06-08T00:00:00+00:00"
YESTERDAY = "2026-08-12T00:00:00+00:00"


def docs(*pages: str, coverage: str = COVERAGE_COVERED, targets: bool = True) -> dict:
    return {
        "coverage": coverage,
        "pages": [{"page": p, "line": 10, "context": "prose", "immutable": False} for p in pages],
        "replace_targets": (
            [{"page": p, "line": 10, "context": "prose"} for p in pages] if targets else []
        ),
        "translations_affected": {},
        "corpus_frequency": len(pages),
        "match_confidence": 1.0,
        "code_context_only": False,
        "touches_immutable": False,
    }


def finding(
    *,
    old: str = "ORG ROLE",
    new: str = "Org role",
    sha: str = "a" * 40,
    when: str = LONG_AGO,
    surface: str = "Members table",
    kind: str = KIND_RENAME,
    pages: tuple[str, ...] = ("manage-organization.mdx",),
    coverage: str = COVERAGE_COVERED,
    targets: bool = True,
    signals: tuple[str, ...] = (),
    degradations: tuple[str, ...] = (),
) -> Finding:
    f = Finding(
        kind=kind,
        surface=surface,
        old_string=old,
        new_string=new,
        literal_kind="jsx",
        literal_key="label",
        commits=[CommitRef(sha=sha, date=when, subject="x", author="Ada",
                           file="frontends/app/src/A.tsx", line=1)],
        first_seen_date=when,
        last_changed_date=when,
        settled=True,
        docs=docs(*pages, coverage=coverage, targets=targets),
        signals=list(signals),
        degradations=list(degradations),
    )
    f.surfaces = [surface]
    f.triage, f.triage_reason = triage(f)
    f.action = action_for(f.triage, f.kind)
    return f


class TestMergeAcrossCommits(unittest.TestCase):
    def test_same_change_in_two_commits_is_one_row(self):
        merged = ledger.merge_findings(
            [finding(sha="a" * 40, when=LONG_AGO),
             finding(sha="b" * 40, when=ALSO_LONG_AGO)],
            today=TODAY,
        ).findings
        self.assertEqual(len(merged), 1)
        self.assertEqual(len(merged[0].commits), 2)

    def test_dates_span_all_commits(self):
        merged = ledger.merge_findings(
            [finding(sha="a" * 40, when=LONG_AGO),
             finding(sha="b" * 40, when=ALSO_LONG_AGO)],
            today=TODAY,
        ).findings
        self.assertEqual(merged[0].first_seen_date, LONG_AGO)
        self.assertEqual(merged[0].last_changed_date, ALSO_LONG_AGO)

    def test_settledness_comes_from_the_last_change_not_the_first(self):
        # This is the bug the merge fixes. An old first commit and a fresh second
        # one is still moving, however old the first one is.
        merged = ledger.merge_findings(
            [finding(sha="a" * 40, when=LONG_AGO),
             finding(sha="b" * 40, when=YESTERDAY)],
            today=TODAY,
        ).findings
        self.assertFalse(merged[0].settled)
        self.assertEqual(merged[0].triage, TRIAGE_PAIR)

    def test_the_org_role_cluster_settles_once_both_commits_are_old(self):
        merged = ledger.merge_findings(
            [finding(sha="a" * 40, when=LONG_AGO),
             finding(sha="b" * 40, when=ALSO_LONG_AGO)],
            today=TODAY,
        ).findings
        self.assertTrue(merged[0].settled)
        self.assertEqual(merged[0].triage, TRIAGE_AGENT)

    def test_surfaces_accumulate(self):
        merged = ledger.merge_findings(
            [finding(sha="a" * 40, surface="Members table"),
             finding(sha="b" * 40, surface="Team settings")],
            today=TODAY,
        ).findings
        self.assertEqual(merged[0].surfaces, ["Members table", "Team settings"])

    def test_degradations_from_any_commit_route_the_merged_row_down(self):
        merged = ledger.merge_findings(
            [finding(sha="a" * 40),
             finding(sha="b" * 40, degradations=("literal is interpolated",))],
            today=TODAY,
        ).findings
        self.assertEqual(merged[0].triage, "human")

    def test_distinct_changes_stay_distinct(self):
        merged = ledger.merge_findings(
            [finding(old="ORG ROLE", new="Org role"),
             finding(old="MODELS SEAT", new="Models Seat")],
            today=TODAY,
        ).findings
        self.assertEqual(len(merged), 2)

    def test_duplicate_sha_is_not_counted_twice(self):
        merged = ledger.merge_findings(
            [finding(sha="a" * 40), finding(sha="a" * 40)], today=TODAY,
        ).findings
        self.assertEqual(len(merged[0].commits), 1)

    def test_signals_are_deduped_and_ordered(self):
        merged = ledger.merge_findings(
            [finding(sha="a" * 40, signals=("url_stable", "case_only_change")),
             finding(sha="b" * 40, signals=("case_only_change", "testid_unchanged"))],
            today=TODAY,
        ).findings
        self.assertEqual(
            merged[0].signals, ["url_stable", "case_only_change", "testid_unchanged"],
        )

    def test_empty_input(self):
        self.assertEqual(ledger.merge_findings([], today=TODAY).findings, [])

    def test_inputs_are_not_mutated(self):
        one = finding(sha="a" * 40, surface="Members table")
        ledger.merge_findings([one, finding(sha="b" * 40, surface="Other")], today=TODAY)
        self.assertEqual(one.surfaces, ["Members table"])
        self.assertEqual(len(one.commits), 1)


class TestRenameChains(unittest.TestCase):
    """A->B->C is one docs task, and its target is C.

    Reporting A->B because that was the commit carrying docs evidence would tell
    a writer to publish a label the product had already stopped using.

    Synthetic rather than fixture-driven, deliberately: the closest pair in the
    corpus (`c99e959` then `ccd66e2`) turns out to ADD a gated toggle and then
    rename it, which emits one rename, not two. A real two-rename chain has not
    been captured as a fixture yet, so these cases are constructed.
    """

    def chain(self, *links, today=TODAY):
        findings = [
            finding(old=old, new=new, sha=chr(ord("a") + i) * 40, when=when)
            for i, (old, new, when) in enumerate(links)
        ]
        return ledger.merge_findings(findings, today=today)

    def test_two_step_chain_collapses_to_one_row(self):
        merged = self.chain(
            ("Hide manually hidden runs", "List only visible runs", LONG_AGO),
            ("List only visible runs", "Only visible runs", ALSO_LONG_AGO),
        ).findings
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].old_string, "Hide manually hidden runs")
        self.assertEqual(merged[0].new_string, "Only visible runs")

    def test_the_intermediate_label_is_never_the_target(self):
        merged = self.chain(
            ("A", "B", LONG_AGO), ("B", "C", ALSO_LONG_AGO),
        ).findings
        self.assertNotIn("B", [f.new_string for f in merged])

    def test_chain_keeps_every_commit(self):
        merged = self.chain(
            ("A", "B", LONG_AGO), ("B", "C", ALSO_LONG_AGO),
        ).findings
        self.assertEqual(len(merged[0].commits), 2)
        self.assertIn("rename_chain:2", merged[0].signals)

    def test_three_step_chain(self):
        merged = self.chain(
            ("A", "B", "2026-06-01T00:00:00+00:00"),
            ("B", "C", "2026-06-08T00:00:00+00:00"),
            ("C", "D", "2026-06-15T00:00:00+00:00"),
        ).findings
        self.assertEqual(len(merged), 1)
        self.assertEqual((merged[0].old_string, merged[0].new_string), ("A", "D"))

    def test_a_revert_is_not_reported_but_is_counted(self):
        # A->B->A. Docs still say A, and A is still correct.
        result = self.chain(("A", "B", LONG_AGO), ("B", "A", ALSO_LONG_AGO))
        self.assertEqual(result.findings, [])
        self.assertEqual(len(result.reverted), 1)

    def test_settledness_of_a_chain_comes_from_the_last_link(self):
        merged = self.chain(
            ("A", "B", LONG_AGO), ("B", "C", YESTERDAY),
        ).findings
        self.assertFalse(merged[0].settled)

    def test_a_fork_is_not_chained_and_cannot_reach_the_agent_lane(self):
        # A->B, then B->C and B->D. Which one is current is unknowable here.
        merged = self.chain(
            ("A", "B", LONG_AGO),
            ("B", "C", ALSO_LONG_AGO),
            ("B", "D", ALSO_LONG_AGO),
        ).findings
        head = [f for f in merged if f.old_string == "A"][0]
        self.assertEqual(head.new_string, "B")
        self.assertIn("ambiguous_chain", head.signals)
        self.assertEqual(head.triage, TRIAGE_PAIR)

    def test_a_join_is_not_chained(self):
        # A->C and B->C. Chaining either into C's successor would be a guess.
        merged = self.chain(
            ("A", "C", LONG_AGO),
            ("B", "C", LONG_AGO),
            ("C", "D", ALSO_LONG_AGO),
        ).findings
        self.assertEqual(len(merged), 3)

    def test_unrelated_renames_are_left_alone(self):
        merged = self.chain(
            ("A", "B", LONG_AGO), ("C", "D", LONG_AGO),
        ).findings
        self.assertEqual(len(merged), 2)

    def test_a_later_rename_does_not_chain_backwards_in_time(self):
        # B->C landed BEFORE A->B, so it is not this rename's successor.
        merged = self.chain(
            ("A", "B", ALSO_LONG_AGO), ("B", "C", LONG_AGO),
        ).findings
        self.assertEqual(len(merged), 2)

    def test_a_cycle_does_not_hang_or_lose_findings(self):
        merged = self.chain(
            ("A", "B", LONG_AGO),
            ("B", "C", ALSO_LONG_AGO),
            ("C", "A", "2026-06-15T00:00:00+00:00"),
        )
        self.assertEqual(len(merged.findings) + len(merged.reverted), 1)

    def test_non_renames_are_passed_through(self):
        added = finding(old="", new="New panel", kind=KIND_ADDED)
        merged = ledger.merge_findings([added], today=TODAY).findings
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].kind, KIND_ADDED)


class TestApplyDecisions(unittest.TestCase):
    def test_a_finding_with_no_decision_is_reported(self):
        f = finding()
        applied = ledger.apply_decisions([f], {})
        self.assertEqual(applied.findings, [f])
        self.assertEqual(applied.suppressed, [])

    def test_dismissed_is_suppressed(self):
        f = finding()
        decision = ledger.record(f, ledger.STATUS_DISMISSED, today=TODAY, decided_by="matt")
        applied = ledger.apply_decisions([f], {f.id: decision})
        self.assertEqual(applied.findings, [])
        self.assertEqual([x.id for x in applied.suppressed], [f.id])

    def test_accepted_is_still_reported_and_annotated(self):
        f = finding()
        decision = ledger.record(
            f, ledger.STATUS_ACCEPTED, today=TODAY,
            decided_by="matt", jira_key="DOCS-1234", assignee="matt",
        )
        applied = ledger.apply_decisions([f], {f.id: decision})
        self.assertEqual(len(applied.findings), 1)
        self.assertEqual(applied.findings[0].jira_key, "DOCS-1234")
        self.assertEqual(applied.findings[0].status, ledger.STATUS_ACCEPTED)

    def test_fixed_but_still_detected_is_surfaced_not_suppressed(self):
        f = finding()
        decision = ledger.record(f, ledger.STATUS_FIXED, today=TODAY, docs_pr=99)
        applied = ledger.apply_decisions([f], {f.id: decision})
        self.assertEqual([x.id for x in applied.unresolved], [f.id])
        self.assertIn("marked_fixed_still_detected", applied.findings[0].signals)

    def test_decision_for_an_absent_finding_is_an_orphan(self):
        f = finding()
        decision = ledger.record(f, ledger.STATUS_DISMISSED, today=TODAY)
        applied = ledger.apply_decisions([], {f.id: decision})
        self.assertEqual(applied.orphans, [f.id])

    def test_decisions_do_not_leak_between_findings(self):
        kept, dismissed = finding(old="A", new="B"), finding(old="C", new="D")
        decision = ledger.record(dismissed, ledger.STATUS_DISMISSED, today=TODAY)
        applied = ledger.apply_decisions([kept, dismissed], {dismissed.id: decision})
        self.assertEqual([x.id for x in applied.findings], [kept.id])

    def test_prunable_lists_only_finished_orphans(self):
        done, queued = finding(old="A", new="B"), finding(old="C", new="D")
        decisions = {
            done.id: ledger.record(done, ledger.STATUS_FIXED, today=TODAY),
            queued.id: ledger.record(queued, ledger.STATUS_ACCEPTED, today=TODAY),
        }
        applied = ledger.apply_decisions([], decisions)
        self.assertEqual(ledger.prunable(applied, decisions), [done.id])


class TestReopenRule(unittest.TestCase):
    """A stored decision must never hide a finding that has since become real."""

    def test_a_new_documented_page_reopens_a_dismissal(self):
        before = finding(pages=("manage-organization.mdx",))
        decision = ledger.record(before, ledger.STATUS_DISMISSED, today=TODAY)

        after = finding(pages=("manage-organization.mdx", "teams.mdx"))
        applied = ledger.apply_decisions([after], {after.id: decision})

        self.assertEqual(applied.suppressed, [])
        self.assertEqual([x.id for x in applied.reopened], [after.id])
        self.assertEqual(len(applied.findings), 1)

    def test_coverage_appearing_reopens_a_dismissal(self):
        before = finding(pages=(), coverage=COVERAGE_NONE, targets=False)
        decision = ledger.record(before, ledger.STATUS_DISMISSED, today=TODAY)

        after = finding(pages=("manage-organization.mdx",), coverage=COVERAGE_COVERED)
        applied = ledger.apply_decisions([after], {after.id: decision})
        self.assertEqual([x.id for x in applied.reopened], [after.id])

    def test_a_new_editable_occurrence_reopens_a_dismissal(self):
        before = finding(pages=("manage-organization.mdx",), targets=False)
        decision = ledger.record(before, ledger.STATUS_DISMISSED, today=TODAY)

        after = finding(pages=("manage-organization.mdx",), targets=True)
        applied = ledger.apply_decisions([after], {after.id: decision})
        self.assertEqual([x.id for x in applied.reopened], [after.id])

    def test_a_second_occurrence_on_a_known_page_reopens_a_dismissal(self):
        # The page was already in the evidence, so a set of page names could not
        # see this: the docs grew a second editable occurrence and the decision
        # stayed suppressed. Counts per page catch it; line numbers still do not
        # enter the fingerprint.
        before = finding(pages=("manage-organization.mdx",))
        decision = ledger.record(before, ledger.STATUS_DISMISSED, today=TODAY)

        after = finding(pages=("manage-organization.mdx",))
        after.docs["replace_targets"].append(
            {"page": "manage-organization.mdx", "line": 42, "context": "prose"}
        )
        applied = ledger.apply_decisions([after], {after.id: decision})
        self.assertEqual([x.id for x in applied.reopened], [after.id])

    def test_fewer_occurrences_on_a_known_page_is_not_a_reopen(self):
        before = finding(pages=("manage-organization.mdx",))
        before.docs["replace_targets"].append(
            {"page": "manage-organization.mdx", "line": 42, "context": "prose"}
        )
        decision = ledger.record(before, ledger.STATUS_DISMISSED, today=TODAY)

        after = finding(pages=("manage-organization.mdx",))
        applied = ledger.apply_decisions([after], {after.id: decision})
        self.assertEqual(applied.reopened, [])

    def test_a_decision_stored_in_the_older_shape_does_not_reopen_itself(self):
        # Decisions written before targets carried counts stored a bare list of
        # page names. Reading that as one-occurrence-per-page keeps an unchanged
        # corpus comparing equal -- otherwise upgrading the shape would reopen
        # every stored decision at once, which is the fastest way to make a
        # writer stop trusting the ledger.
        f = finding()
        decision = ledger.record(f, ledger.STATUS_DISMISSED, today=TODAY)
        decision.evidence["targets"] = ["manage-organization.mdx"]
        applied = ledger.apply_decisions([finding()], {f.id: decision})
        self.assertEqual(applied.reopened, [])
        self.assertEqual(len(applied.suppressed), 1)

    def test_shrinking_evidence_is_not_a_reopen(self):
        # Someone did the work on one of two pages. That is progress, not a
        # reason to reopen a decision.
        before = finding(pages=("manage-organization.mdx", "teams.mdx"))
        decision = ledger.record(before, ledger.STATUS_DISMISSED, today=TODAY)

        after = finding(pages=("manage-organization.mdx",))
        applied = ledger.apply_decisions([after], {after.id: decision})
        self.assertEqual(applied.reopened, [])
        self.assertEqual([x.id for x in applied.suppressed], [after.id])

    def test_identical_evidence_is_not_a_reopen(self):
        f = finding()
        decision = ledger.record(f, ledger.STATUS_DISMISSED, today=TODAY)
        applied = ledger.apply_decisions([finding()], {f.id: decision})
        self.assertEqual(applied.reopened, [])
        self.assertEqual(len(applied.suppressed), 1)

    def test_line_number_churn_is_not_a_reopen(self):
        f = finding()
        decision = ledger.record(f, ledger.STATUS_DISMISSED, today=TODAY)
        moved = finding()
        for page in moved.docs["pages"]:
            page["line"] = 999
        applied = ledger.apply_decisions([moved], {f.id: decision})
        self.assertEqual(applied.reopened, [])

    def test_a_hand_written_decision_without_evidence_is_honored(self):
        # A writer dismissing something by editing the JSON will not write an
        # evidence block, and must not be overruled for it.
        f = finding()
        applied = ledger.apply_decisions(
            [f], {f.id: ledger.Decision(status=ledger.STATUS_DISMISSED)},
        )
        self.assertEqual(applied.reopened, [])
        self.assertEqual(len(applied.suppressed), 1)

    def test_reopened_finding_carries_the_prior_status(self):
        before = finding(pages=("a.mdx",))
        decision = ledger.record(before, ledger.STATUS_DISMISSED, today=TODAY)
        after = finding(pages=("a.mdx", "b.mdx"))
        applied = ledger.apply_decisions([after], {after.id: decision})
        self.assertIn("reopened:dismissed", applied.findings[0].signals)


class TestPersistence(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "uidrift" / "ledger.json"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_missing_file_is_an_empty_ledger_not_an_error(self):
        self.assertEqual(ledger.load(self.path), {})

    def test_round_trip(self):
        f = finding()
        decisions = {
            f.id: ledger.record(
                f, ledger.STATUS_ACCEPTED, today=TODAY,
                decided_by="matt", jira_key="DOCS-1234", note="waiting on eng",
            )
        }
        ledger.save(decisions, self.path)
        loaded = ledger.load(self.path)
        self.assertEqual(loaded[f.id].jira_key, "DOCS-1234")
        self.assertEqual(loaded[f.id].note, "waiting on eng")
        self.assertEqual(loaded[f.id].evidence, decisions[f.id].evidence)

    def test_save_creates_the_directory(self):
        ledger.save({}, self.path)
        self.assertTrue(self.path.exists())

    def test_saved_file_is_stable_across_writes(self):
        f, g = finding(old="A", new="B"), finding(old="C", new="D")
        first = {f.id: ledger.record(f, ledger.STATUS_DISMISSED, today=TODAY),
                 g.id: ledger.record(g, ledger.STATUS_DISMISSED, today=TODAY)}
        ledger.save(first, self.path)
        once = self.path.read_text()
        ledger.save(ledger.load(self.path), self.path)
        self.assertEqual(once, self.path.read_text())

    def test_malformed_json_raises_rather_than_resetting(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text("{not json")
        with self.assertRaises(ledger.LedgerError):
            ledger.load(self.path)

    def test_wrong_version_raises(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text(json.dumps({"version": 99, "decisions": {}}))
        with self.assertRaises(ledger.LedgerError):
            ledger.load(self.path)

    def test_unknown_status_raises(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text(json.dumps(
            {"version": 1, "decisions": {"abc": {"status": "dismissed!"}}}
        ))
        with self.assertRaises(ledger.LedgerError):
            ledger.load(self.path)

    def test_typoed_field_raises_rather_than_being_ignored(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text(json.dumps(
            {"version": 1, "decisions": {"abc": {"status": "dismissed", "assignnee": "matt"}}}
        ))
        with self.assertRaises(ledger.LedgerError) as caught:
            ledger.load(self.path)
        self.assertIn("assignnee", str(caught.exception))

    def test_record_rejects_an_unknown_status(self):
        with self.assertRaises(ledger.LedgerError):
            ledger.record(finding(), "wontfix", today=TODAY)

    def test_a_minimal_hand_written_ledger_loads(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text(json.dumps(
            {"version": 1, "decisions": {"abc123def456": {"status": "dismissed"}}}
        ))
        loaded = ledger.load(self.path)
        self.assertEqual(loaded["abc123def456"].status, "dismissed")
        self.assertEqual(loaded["abc123def456"].evidence, {})


class TestReportSurfacesLedgerState(unittest.TestCase):
    def render(self, findings, **kw):
        return report.render(
            findings, scanned_range="x..y", today=TODAY, commits=1,
            ui_commits=1, candidates=1, docs_pages=10, **kw,
        )

    def test_suppressed_findings_are_accounted_for_not_silently_dropped(self):
        f = finding()
        f.status, f.decided_by, f.decided_at = "dismissed", "matt", "2026-08-01"
        out = self.render([], suppressed=[f])
        self.assertIn("Held back by earlier decisions (1)", out)
        self.assertIn("previously decided finding(s) held back", out)
        self.assertIn(f.id, out)

    def test_reopened_is_called_out_at_the_top(self):
        f = finding(signals=("reopened:dismissed",))
        out = self.render([f], reopened=[f])
        self.assertIn("reopened", out.lower())
        self.assertIn("Reopened decisions (1)", out)
        self.assertIn("was `dismissed`", out)

    def test_unresolved_gets_its_own_section(self):
        f = finding(signals=("marked_fixed_still_detected",))
        out = self.render([f], unresolved=[f])
        self.assertIn("Marked fixed, still detected (1)", out)

    def test_orphans_are_listed_and_nothing_claims_to_have_deleted_them(self):
        out = self.render([], orphans=["abc123def456"])
        self.assertIn("abc123def456", out)
        self.assertIn("nothing was deleted", out)

    def test_status_and_ticket_ride_along_in_the_row(self):
        f = finding()
        f.status, f.jira_key, f.docs_pr = "accepted", "DOCS-1234", 77
        out = self.render([f])
        self.assertIn("DOCS-1234", out)
        self.assertIn("docs#77", out)

    def test_multi_commit_findings_say_so(self):
        f = finding()
        f.commits.append(CommitRef(sha="b" * 40, date=ALSO_LONG_AGO, subject="y",
                                   author="Ada", file="x.tsx", line=2))
        out = self.render([f])
        self.assertIn("changed 2×", out)

    def test_a_clean_run_with_no_ledger_state_is_unchanged(self):
        out = self.render([])
        self.assertIn("No drift to act on in this window.", out)
        self.assertNotIn("Held back", out)
        self.assertNotIn("Reopened", out)

    def test_reverted_renames_are_counted(self):
        out = self.render([], reverted=[finding()])
        self.assertIn("Renamed and renamed back (1)", out)

    def test_ledger_state_shows_even_when_there_are_no_findings(self):
        # Otherwise a run where everything was already dismissed is
        # indistinguishable from a run that found nothing.
        f = finding()
        f.status, f.decided_at = "dismissed", "2026-08-01"
        out = self.render([], suppressed=[f], orphans=["deadbeef1234"])
        self.assertIn("Held back by earlier decisions (1)", out)
        self.assertIn("deadbeef1234", out)
        # And the empty-state prose must not claim nothing was found.
        self.assertIn("was already decided on", out)

    def test_a_finding_with_an_unrecognized_lane_is_still_shown(self):
        # The header count and the table have to agree. A row that is tallied
        # but never printed is the one failure a reader cannot detect.
        f = finding()
        f.triage = "somethingelse"
        out = self.render([f])
        self.assertIn("**1 findings**", out)
        self.assertIn(f.id, out)
        self.assertIn("Unclassified (somethingelse)", out)

    def test_a_finding_with_no_lane_at_all_is_still_shown(self):
        f = finding()
        f.triage = ""
        out = self.render([f])
        self.assertIn(f.id, out)
        self.assertIn("no lane", out)


if __name__ == "__main__":
    unittest.main()
