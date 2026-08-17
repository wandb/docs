"""Entrypoint logic: range resolution, the incremental watermark, and decide.

The pipeline itself is covered by the other test modules; what is unique here is
how a run picks its commit range and how `decide` guards the ledger.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
import unittest.mock
from datetime import date
from pathlib import Path

from .. import ledger, scan
from ..finding import CommitRef, Finding


def _touch_report(root: Path, name: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    p = root / name
    p.write_text("# report\n", encoding="utf-8")
    return p


def _finding(fid_old: str = "ORG ROLE") -> Finding:
    f = Finding(
        kind="rename", surface="Members table", old_string=fid_old,
        new_string="Org role", literal_kind="jsx", literal_key="label",
        commits=[CommitRef(sha="a" * 40, date="2026-06-01T00:00:00+00:00",
                           subject="x", author="Ada", file="A.tsx", line=1)],
        docs={"coverage": "covered", "pages": [], "replace_targets": []},
    )
    f.surfaces = [f.surface]
    return f


class TestReportWatermark(unittest.TestCase):
    """--incremental reads its base from the newest report, not a state file."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.reports = Path(self._tmp.name) / "reports"

    def test_no_reports_yet(self):
        self.reports.mkdir(parents=True)
        self.assertIsNone(scan._last_report(self.reports))

    def test_missing_directory_is_not_an_error(self):
        self.assertIsNone(scan._last_report(self.reports))

    def test_picks_the_newest_by_date(self):
        _touch_report(self.reports, "2026-08-01-aaaaaaaaaaaa.md")
        _touch_report(self.reports, "2026-08-13-bbbbbbbbbbbb.md")
        _touch_report(self.reports, "2026-07-02-cccccccccccc.md")
        when, sha, _ = scan._last_report(self.reports)
        self.assertEqual(when, date(2026, 8, 13))
        self.assertEqual(sha, "bbbbbbbbbbbb")

    def test_ignores_files_that_are_not_reports(self):
        _touch_report(self.reports, "2026-08-01-aaaaaaaaaaaa.md")
        _touch_report(self.reports, "README.md")
        _touch_report(self.reports, "notes-2026-08-20.md")
        _touch_report(self.reports, "2026-13-99-bbbbbbbbbbbb.md")
        when, sha, _ = scan._last_report(self.reports)
        self.assertEqual((when, sha), (date(2026, 8, 1), "aaaaaaaaaaaa"))

    def test_accepts_short_and_full_shas(self):
        _touch_report(self.reports, "2026-08-01-abcdef1.md")
        _, sha, _ = scan._last_report(self.reports)
        self.assertEqual(sha, "abcdef1")


class TestRangeResolution(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        repo = Path(cls._tmp.name)
        cls.repo = repo
        env = {"GIT_AUTHOR_NAME": "Ada", "GIT_AUTHOR_EMAIL": "a@e.com",
               "GIT_COMMITTER_NAME": "Ada", "GIT_COMMITTER_EMAIL": "a@e.com",
               "GIT_AUTHOR_DATE": "2026-06-01T00:00:00Z",
               "GIT_COMMITTER_DATE": "2026-06-01T00:00:00Z",
               "PATH": "/usr/bin:/bin", "HOME": str(repo)}
        for args in (["init", "-q", "-b", "master"],):
            subprocess.run(["git", "-C", str(repo), *args], check=True,
                           capture_output=True, env=env)
        (repo / "a.txt").write_text("one")
        for args in (["add", "a.txt"], ["commit", "-m", "one"]):
            subprocess.run(["git", "-C", str(repo), *args], check=True,
                           capture_output=True, env=env)
        cls.head = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_explicit_base_is_resolved_to_a_sha(self):
        base, desc = scan._resolve_base(
            self.repo, since=None, base=self.head, head="HEAD",
        )
        self.assertEqual(base, self.head)
        self.assertIn(self.head[:12], desc)

    def test_unresolvable_base_is_a_scan_error(self):
        with self.assertRaises(scan.ScanError):
            scan._resolve_base(self.repo, since=None, base="nope123", head="HEAD")

    def test_since_with_no_commit_in_range_is_a_scan_error(self):
        # Every commit is dated 2026-06-01, so nothing precedes 1999.
        with self.assertRaises(scan.ScanError):
            scan._resolve_base(self.repo, since="1999-01-01", base=None, head="HEAD")

    def test_base_takes_precedence_over_since(self):
        base, _ = scan._resolve_base(
            self.repo, since="1999-01-01", base=self.head, head="HEAD",
        )
        self.assertEqual(base, self.head)

    def test_incremental_without_a_prior_report_explains_itself(self):
        empty = Path(self._tmp.name) / "no-reports-here"
        with self.assertRaises(scan.ScanError) as caught:
            scan.scan(incremental=True, core=self.repo, report_dir=empty)
        # Must name the fix, not just the failure.
        self.assertIn("--since", str(caught.exception))

    def test_a_missing_checkout_is_checked_before_anything_else(self):
        # Cheapest failure first: no point resolving a watermark against a repo
        # that is not there.
        with self.assertRaises(scan.ScanError) as caught:
            scan.scan(incremental=True, core=Path("/nonexistent/core"))
        self.assertIn("not a git checkout", str(caught.exception))

    def test_a_missing_checkout_names_the_env_var_to_set(self):
        with self.assertRaises(scan.ScanError) as caught:
            scan.scan(core=Path("/nonexistent/core"))
        self.assertIn("CORE_REPO", str(caught.exception))


class TestScanResultLookup(unittest.TestCase):
    def _result(self, **kw) -> scan.ScanResult:
        return scan.ScanResult(
            markdown="", stats={},
            applied=ledger.Applied(**{k: v for k, v in kw.items() if k != "reverted"}),
            merged=ledger.Merged(reverted=kw.get("reverted", [])),
        )

    def test_finds_a_reported_finding(self):
        f = _finding()
        self.assertIs(self._result(findings=[f]).by_id(f.id), f)

    def test_finds_a_suppressed_finding(self):
        # Reversing a dismissal is the main reason to run decide twice, so a
        # suppressed row has to stay addressable.
        f = _finding()
        self.assertIs(self._result(suppressed=[f]).by_id(f.id), f)

    def test_finds_a_reverted_finding(self):
        f = _finding()
        self.assertIs(self._result(reverted=[f]).by_id(f.id), f)

    def test_unknown_id_is_none(self):
        self.assertIsNone(self._result(findings=[_finding()]).by_id("deadbeef"))


class TestSummaryJson(unittest.TestCase):
    """The counts a CI step reads, so nothing has to parse the prose report."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out = Path(self._tmp.name) / "summary.json"

        self._stats = {"head": "b" * 40, "findings": 1, "reopened": 0,
                       "lanes": {"agent": 1}}
        result = scan.ScanResult(
            markdown="# report\n", stats=self._stats,
            applied=ledger.Applied(findings=[_finding()]),
            merged=ledger.Merged(),
        )
        patcher = unittest.mock.patch.object(scan, "scan", return_value=result)
        self.scan = patcher.start()
        self.addCleanup(patcher.stop)

    def _run(self, *extra: str) -> dict:
        code = scan.main(["scan", "--stdout", "--quiet",
                          "--summary-json", str(self.out), *extra])
        self.assertEqual(code, 0)
        return json.loads(self.out.read_text(encoding="utf-8"))

    def test_counts_are_written(self):
        self.assertEqual(self._run()["findings"], 1)

    def test_lane_counts_survive_serialization(self):
        # The PR body reports lanes; a dict of Finding objects would not encode.
        self.assertEqual(self._run()["lanes"], {"agent": 1})

    def test_stdout_mode_reports_no_report_path(self):
        # Nothing was written, so claiming a path would send a caller looking
        # for a file that does not exist.
        self.assertNotIn("report", self._run())

    def test_the_scan_stats_are_not_mutated(self):
        # _cmd_scan adds "report" for the caller's benefit; doing that in place
        # would leave a library caller holding a path it never asked for.
        self._run()
        self.assertNotIn("report", self._stats)

    def test_reopened_still_sets_the_exit_code(self):
        # The summary is written before the exit code is chosen, so a run that
        # fails a CI step still leaves the counts explaining why.
        self._stats["reopened"] = 2
        code = scan.main(["scan", "--stdout", "--quiet",
                          "--summary-json", str(self.out)])
        self.assertEqual(code, 3)
        self.assertEqual(json.loads(self.out.read_text())["reopened"], 2)

    def test_no_summary_is_written_unless_asked(self):
        self.assertEqual(scan.main(["scan", "--stdout", "--quiet"]), 0)
        self.assertFalse(self.out.exists())


class TestCli(unittest.TestCase):
    def test_no_subcommand_prints_help_and_fails(self):
        self.assertEqual(scan.main([]), 2)

    def test_decide_rejects_an_unknown_status(self):
        with self.assertRaises(SystemExit):
            scan.main(["decide", "abc123", "--status", "wontfix"])

    def test_decide_accepts_every_ledger_status(self):
        # Guards against STATUSES and the CLI choices drifting apart.
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--status", choices=ledger.STATUSES)
        for status in ledger.STATUSES:
            self.assertEqual(parser.parse_args(["--status", status]).status, status)


if __name__ == "__main__":
    unittest.main()
