"""Ownership resolution: author ranking, CODEOWNERS matching, and per-run caching.

These run against a real throwaway git repo rather than mocks. The thing most
likely to break here is the batched `git log --name-only` parse, and a mock of
git's output would just be a restatement of the parser's assumptions.
"""

from __future__ import annotations

import subprocess
import tempfile
import unittest
import unittest.mock as mock
from pathlib import Path

from .. import config, ownership

UI = config.SOURCE.ui_roots[0]

CODEOWNERS = """\
# Coarse default, then progressively narrower overrides.
*                                   @wandb/docs-platform
/frontends/app/                     @wandb/frontend-reviewers
/frontends/app/src/weave/           @wandb/weave-team
/frontends/app/**/*ramp*            @wandb/growth
"""


def _git(repo: Path, *args: str, author: str = "Tester") -> None:
    env = {
        "GIT_AUTHOR_NAME": author,
        "GIT_AUTHOR_EMAIL": f"{author.replace(' ', '.').lower()}@example.com",
        "GIT_COMMITTER_NAME": author,
        "GIT_COMMITTER_EMAIL": f"{author.replace(' ', '.').lower()}@example.com",
        "GIT_AUTHOR_DATE": "2026-06-01T00:00:00Z",
        "GIT_COMMITTER_DATE": "2026-06-01T00:00:00Z",
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "HOME": str(repo),
    }
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True, capture_output=True, text=True, env=env,
    )


def _commit(repo: Path, rel: str, body: str, author: str) -> None:
    target = repo / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body)
    _git(repo, "add", rel, author=author)
    _git(repo, "commit", "-m", f"touch {rel}", author=author)


class OwnershipTestCase(unittest.TestCase):
    """A repo whose history is rigged so author ranking has a known answer."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        repo = Path(cls._tmp.name)
        cls.repo = repo
        _git(repo, "init", "-q", "-b", "master")

        # members.tsx: Ada 3, Grace 1, plus a bot that must not be ranked.
        for i in range(3):
            _commit(repo, f"{UI}/members.tsx", f"ada {i}", "Ada Lovelace")
        _commit(repo, f"{UI}/members.tsx", "grace", "Grace Hopper")
        _commit(repo, f"{UI}/members.tsx", "bot", "renovate[bot]")

        # thin.tsx: a single drive-by, below MIN_RECENT_AUTHORS.
        _commit(repo, f"{UI}/thin.tsx", "solo", "Solo Contributor")

        # A path CODEOWNERS routes to the weave override.
        _commit(repo, f"{UI}/weave/panel.tsx", "weave", "Ada Lovelace")

        (repo / ".github").mkdir(parents=True, exist_ok=True)
        _commit(repo, ".github/CODEOWNERS", CODEOWNERS, "Ada Lovelace")

        # config.SOURCE.default_head is origin/master, which a fresh repo lacks.
        _git(repo, "update-ref", f"refs/remotes/{config.SOURCE.default_head}", "HEAD")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def setUp(self) -> None:
        ownership.reset_caches()

    def tearDown(self) -> None:
        ownership.reset_caches()


class TestAuthorRanking(OwnershipTestCase):
    def test_ranks_by_commit_count(self):
        authors = ownership._git_authors(self.repo, f"{UI}/members.tsx", None)
        self.assertEqual(authors[:2], ["Ada Lovelace", "Grace Hopper"])

    def test_excludes_bots(self):
        authors = ownership._git_authors(self.repo, f"{UI}/members.tsx", None)
        self.assertNotIn("renovate[bot]", authors)

    def test_unknown_path_is_empty_not_an_error(self):
        self.assertEqual(ownership._git_authors(self.repo, f"{UI}/nope.tsx", None), [])

    def test_index_separates_paths(self):
        index = ownership._author_index(self.repo, None)
        self.assertIn(f"{UI}/members.tsx", index)
        self.assertIn(f"{UI}/thin.tsx", index)
        self.assertNotIn("Solo Contributor", index[f"{UI}/members.tsx"])


class TestSuggestReviewers(OwnershipTestCase):
    def test_widens_when_the_recent_window_is_empty(self):
        # Every commit is dated 2026-06-01, so this window sees nothing at all
        # and only the all-time fallback can answer.
        reviewers, source = ownership.suggest_reviewers(
            f"{UI}/members.tsx", core=self.repo, since="2026-07-01",
        )
        self.assertEqual(source, "all-time")
        self.assertEqual(reviewers[0], "Ada Lovelace")

    def test_widens_when_the_recent_window_is_merely_thin(self):
        # The window covers every commit, but two ranked authors is below
        # MIN_RECENT_AUTHORS, so it still widens rather than crowning a
        # drive-by contributor.
        self.assertLess(2, ownership.MIN_RECENT_AUTHORS)
        _, source = ownership.suggest_reviewers(
            f"{UI}/members.tsx", core=self.repo, since="2026-05-31",
        )
        self.assertEqual(source, "all-time")

    def test_thin_path_still_reports_its_one_author(self):
        reviewers, source = ownership.suggest_reviewers(
            f"{UI}/thin.tsx", core=self.repo, since="2026-05-31",
        )
        self.assertEqual(reviewers, ["Solo Contributor"])
        self.assertEqual(source, "all-time")

    def test_caps_at_top(self):
        reviewers, _ = ownership.suggest_reviewers(
            f"{UI}/members.tsx", core=self.repo, since="2026-05-31", top=1,
        )
        self.assertEqual(reviewers, ["Ada Lovelace"])


class TestOwningTeam(OwnershipTestCase):
    def test_last_match_wins(self):
        team = ownership.owning_team(f"{UI}/weave/panel.tsx", core=self.repo)
        self.assertEqual(team, "@wandb/weave-team")

    def test_falls_back_to_the_broader_rule(self):
        team = ownership.owning_team(f"{UI}/members.tsx", core=self.repo)
        self.assertEqual(team, "@wandb/frontend-reviewers")

    def test_double_star_crosses_segments(self):
        team = ownership.owning_team(f"{UI}/billing/rampBanner.tsx", core=self.repo)
        self.assertEqual(team, "@wandb/growth")

    def test_unmatched_path_takes_the_catch_all(self):
        self.assertEqual(ownership.owning_team("README.md", core=self.repo),
                         "@wandb/docs-platform")


class TestResolve(OwnershipTestCase):
    def test_commit_author_leads_and_is_not_duplicated(self):
        own = ownership.resolve(
            f"{UI}/members.tsx", core=self.repo, commit_author="Grace Hopper",
        )
        self.assertEqual(own.reviewers[0], "Grace Hopper")
        self.assertEqual(own.reviewers.count("Grace Hopper"), 1)

    def test_bot_commit_author_is_not_promoted(self):
        own = ownership.resolve(
            f"{UI}/members.tsx", core=self.repo, commit_author="renovate[bot]",
        )
        self.assertNotIn("renovate[bot]", own.reviewers)

    def test_reports_team_alongside_humans(self):
        own = ownership.resolve(f"{UI}/weave/panel.tsx", core=self.repo)
        self.assertEqual(own.team, "@wandb/weave-team")
        self.assertTrue(own.reviewers)


class TestPerRunCaching(OwnershipTestCase):
    """The point of the rewrite: cost is per run, not per finding."""

    def _count_git_calls(self, fn):
        real = subprocess.run
        calls: list[list[str]] = []

        def counting(cmd, *a, **kw):
            calls.append(list(cmd))
            return real(cmd, *a, **kw)

        with mock.patch.object(ownership.subprocess, "run", counting):
            fn()
        return calls

    def test_codeowners_is_read_once_for_many_paths(self):
        paths = [f"{UI}/members.tsx", f"{UI}/weave/panel.tsx", f"{UI}/thin.tsx"]
        calls = self._count_git_calls(
            lambda: [ownership.owning_team(p, core=self.repo) for p in paths]
        )
        shows = [c for c in calls if "show" in c]
        self.assertEqual(len(shows), 1, f"expected one git show, got {len(shows)}")

    def test_author_history_is_read_once_for_many_paths(self):
        paths = [f"{UI}/members.tsx", f"{UI}/thin.tsx", f"{UI}/weave/panel.tsx"]
        calls = self._count_git_calls(
            lambda: [ownership._git_authors(self.repo, p, None) for p in paths]
        )
        logs = [c for c in calls if "log" in c]
        self.assertEqual(len(logs), 1, f"expected one git log, got {len(logs)}")

    def test_resolving_many_findings_is_a_bounded_number_of_subprocesses(self):
        # Three paths, each needing reviewers + team. The naive version cost
        # ~3 subprocesses per path; this is the regression guard on that.
        paths = [f"{UI}/members.tsx", f"{UI}/thin.tsx", f"{UI}/weave/panel.tsx"]
        calls = self._count_git_calls(
            lambda: [ownership.resolve(p, core=self.repo) for p in paths]
        )
        # One recent-window log, one all-time log, one CODEOWNERS show.
        self.assertLessEqual(len(calls), 3, f"too many subprocesses: {calls}")

    def test_reset_caches_forces_a_reread(self):
        first = self._count_git_calls(
            lambda: ownership.owning_team(f"{UI}/members.tsx", core=self.repo)
        )
        ownership.reset_caches()
        second = self._count_git_calls(
            lambda: ownership.owning_team(f"{UI}/members.tsx", core=self.repo)
        )
        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)


class TestCodeownersGlobs(unittest.TestCase):
    """Pure glob translation; no repo needed."""

    def test_anchored_directory_does_not_match_a_sibling(self):
        pattern = ownership._codeowners_regex("/frontends/app/")
        self.assertTrue(pattern.match("frontends/app/src/A.tsx"))
        self.assertFalse(pattern.match("frontends/appx/src/A.tsx"))

    def test_unanchored_pattern_matches_at_any_depth(self):
        pattern = ownership._codeowners_regex("docs/")
        self.assertTrue(pattern.match("a/b/docs/x.md"))
        self.assertTrue(pattern.match("docs/x.md"))

    def test_single_star_stays_within_a_segment(self):
        pattern = ownership._codeowners_regex("/src/*.tsx")
        self.assertTrue(pattern.match("src/A.tsx"))
        self.assertFalse(pattern.match("src/nested/A.tsx"))

    def test_catch_all(self):
        self.assertTrue(ownership._codeowners_regex("*").match("anything/at/all.ts"))


if __name__ == "__main__":
    unittest.main()
