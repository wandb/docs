"""Stage-1 regression tests, frozen against six real wandb/core commits.

No network, no clone. Each fixture is a `git show --format= -M <sha> --
frontends/app/src` output checked into the repo.

These exist because design review does not catch extractor bugs. The first
draft of this extractor scored ZERO on both pure-rename fixtures -- the highest
value class there is -- and looked completely reasonable while doing it.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from .. import extract

FIXTURES = Path(__file__).parent / "fixtures"


def load(stem: str) -> list[extract.LabelDelta]:
    diff = (FIXTURES / f"{stem}.diff").read_text()
    return extract.surviving_deltas(extract.extract_deltas(diff))


def strings(deltas, sign: str) -> set[str]:
    return {d.norm for d in deltas if d.sign == sign}


class TestAllFixturesProduceSignal(unittest.TestCase):
    """The assertion that matters most: none of the six may score zero."""

    def test_every_fixture_yields_deltas(self):
        for stem in ("ccd66e2", "c99e959", "e1bc1e6", "cb100df", "f4861ad", "9573f30"):
            with self.subTest(fixture=stem):
                self.assertTrue(load(stem), f"{stem} produced no label deltas")

    def test_every_delta_is_signed(self):
        for stem in ("ccd66e2", "c99e959", "e1bc1e6", "cb100df", "f4861ad", "9573f30"):
            for d in load(stem):
                self.assertIn(d.sign, ("+", "-"), f"{stem}: unsigned delta {d!r}")


class TestPureRename(unittest.TestCase):
    """ccd66e2 -- 'Hide manually hidden runs' -> 'List only visible runs'.

    Guards the inline-JSX regression. Prettier does not always break text onto
    its own line, so a bare-line heuristic alone has ~0% recall here.
    """

    def setUp(self):
        self.deltas = load("ccd66e2")

    def test_both_sides_captured(self):
        self.assertIn("Hide manually hidden runs", strings(self.deltas, "-"))
        self.assertIn("List only visible runs", strings(self.deltas, "+"))

    def test_captured_as_both_jsx_text_and_attribute(self):
        # <span>Hide manually hidden runs</span> and <Hotkey name="..." />
        kinds = {(d.kind, d.key) for d in self.deltas if d.norm == "Hide manually hidden runs"}
        self.assertIn(("jsx", "_"), kinds)
        self.assertIn(("attr", "name"), kinds)

    def test_not_classified_as_moved(self):
        _, _, moved = extract.commit_net_change(self.deltas)
        self.assertEqual(moved, [], "a rename is not a move")

    def test_deltas_carry_line_citations(self):
        for d in self.deltas:
            self.assertGreater(d.line_no, 0, f"no line number on {d!r}")


class TestCaseOnlyRename(unittest.TestCase):
    """f4861ad -- the flagship. Titled 'migrate ... to Table', quietly title-cased
    a family of column headers that is still wrong in published docs."""

    def setUp(self):
        self.deltas = load("f4861ad")

    def test_case_is_never_normalized(self):
        removed, added = strings(self.deltas, "-"), strings(self.deltas, "+")
        self.assertIn("MODELS SEAT", removed)
        self.assertIn("Models Seat", added)

    def test_the_whole_cluster_is_present(self):
        removed = strings(self.deltas, "-")
        for header in ("MODELS SEAT", "WEAVE ACCESS", "ORG ROLE",
                       "LAST ACTIVE", "BILLING ADMIN"):
            self.assertIn(header, removed)

    def test_case_change_does_not_cancel_as_a_move(self):
        # If normalize() ever lowercased, MODELS SEAT and Models Seat would
        # cancel and the richest finding in the corpus would vanish silently.
        _, _, moved = extract.commit_net_change(self.deltas)
        self.assertNotIn("Models Seat", {d.norm for d in moved})


class TestNewSetting(unittest.TestCase):
    """e1bc1e6 -- a new ARIA setting behind a Statsig gate. Additions only."""

    def setUp(self):
        self.deltas = load("e1bc1e6")

    def test_title_captured(self):
        self.assertIn("Enable project memory", strings(self.deltas, "+"))

    def test_no_removals(self):
        self.assertEqual(strings(self.deltas, "-"), set())

    def test_interpolated_description_is_marked_wrapped(self):
        # `Allow ${AGENT_NAME} to remember ...` must never be a blind
        # find-and-replace target.
        interp = [d for d in self.deltas if d.key == "description"]
        self.assertTrue(interp, "description literal not captured")
        self.assertTrue(all(d.wrapped for d in interp))


class TestLargeGatedAddition(unittest.TestCase):
    """cb100df -- new settings section, 797 insertions, flag added same commit."""

    def setUp(self):
        self.deltas = load("cb100df")

    def test_new_copy_captured(self):
        added = strings(self.deltas, "+")
        self.assertIn("Bucket name", added)
        self.assertIn("Add reference bucket", added)

    def test_cost_stays_bounded(self):
        # Pathspec limiting plus per-file set-equality must keep a large commit
        # from exploding the stage-2 payload.
        self.assertLess(len(self.deltas), 60)


class TestMoveNotRemoval(unittest.TestCase):
    """9573f30 -- drawer consolidation. Strings relocate; some change form.

    Without loose move-identity this reports ~23 phantom removals, which would
    be 23 false rows in the first report anyone ever reads.
    """

    def setUp(self):
        self.deltas = load("9573f30")
        self.added, self.removed, self.moved = extract.commit_net_change(self.deltas)

    def test_relocated_strings_classified_as_moved(self):
        moved = {d.norm for d in self.moved}
        self.assertIn("Close drawer", moved)
        self.assertIn("Ex. OPENAI_API_KEY", moved)

    def test_move_survives_a_change_of_expression_form(self):
        # `<span>Add secret</span>` became `saveLabel="Add secret"`. The user
        # still sees it, so it is a move.
        self.assertIn("Add secret", {d.norm for d in self.moved})

    def test_removals_do_not_include_relocated_strings(self):
        self.assertNotIn("Close drawer", {d.norm for d in self.removed})


class TestFalsePositiveDefenses(unittest.TestCase):

    def test_prettier_reflow_is_killed_by_set_equality(self):
        diff = (
            "diff --git a/frontends/app/src/A.tsx b/frontends/app/src/A.tsx\n"
            "--- a/frontends/app/src/A.tsx\n"
            "+++ b/frontends/app/src/A.tsx\n"
            "@@ -1,2 +1,2 @@\n"
            '-<Button aria-label="Save changes" />\n'
            '+  <Button aria-label="Save changes" />\n'
        )
        deltas = extract.extract_deltas(diff)
        self.assertTrue(deltas, "reflow should still be extracted")
        self.assertEqual(extract.surviving_deltas(deltas), [],
                         "reflow must not survive set-equality")

    def test_import_specifiers_are_not_labels(self):
        diff = (
            "diff --git a/frontends/app/src/A.tsx b/frontends/app/src/A.tsx\n"
            "--- a/frontends/app/src/A.tsx\n"
            "+++ b/frontends/app/src/A.tsx\n"
            "@@ -1,3 +1,3 @@\n"
            "+import {\n"
            "+  Avatar,\n"
            "+  SearchField,\n"
        )
        self.assertEqual(extract.extract_deltas(diff), [])

    def test_typescript_generics_are_not_labels(self):
        diff = (
            "diff --git a/frontends/app/src/A.tsx b/frontends/app/src/A.tsx\n"
            "--- a/frontends/app/src/A.tsx\n"
            "+++ b/frontends/app/src/A.tsx\n"
            "@@ -1,1 +1,1 @@\n"
            "+const x: Promise<void> = load<ReactNode>();\n"
        )
        self.assertEqual(extract.extract_deltas(diff), [])

    def test_icon_identifiers_are_not_labels(self):
        diff = (
            "diff --git a/frontends/app/src/A.tsx b/frontends/app/src/A.tsx\n"
            "--- a/frontends/app/src/A.tsx\n"
            "+++ b/frontends/app/src/A.tsx\n"
            "@@ -1,1 +1,1 @@\n"
            '+<Icon name="info" />\n'
        )
        self.assertEqual(extract.extract_deltas(diff), [])

    def test_test_files_are_excluded(self):
        diff = (
            "diff --git a/frontends/app/src/A.test.tsx b/frontends/app/src/A.test.tsx\n"
            "--- a/frontends/app/src/A.test.tsx\n"
            "+++ b/frontends/app/src/A.test.tsx\n"
            "@@ -1,1 +1,1 @@\n"
            '+<Button aria-label="Save changes" />\n'
        )
        self.assertEqual(extract.extract_deltas(diff), [])

    def test_non_ui_paths_are_excluded(self):
        diff = (
            "diff --git a/services/gorilla/x.go b/services/gorilla/x.go\n"
            "--- a/services/gorilla/x.go\n"
            "+++ b/services/gorilla/x.go\n"
            "@@ -1,1 +1,1 @@\n"
            '+label = "Save changes"\n'
        )
        self.assertEqual(extract.extract_deltas(diff), [])


class TestSuffixMatching(unittest.TestCase):
    """Attribute names carrying copy are open-ended; an enumerated list misses."""

    def test_unenumerated_label_attribute_is_matched(self):
        diff = (
            "diff --git a/frontends/app/src/A.tsx b/frontends/app/src/A.tsx\n"
            "--- a/frontends/app/src/A.tsx\n"
            "+++ b/frontends/app/src/A.tsx\n"
            "@@ -1,1 +1,1 @@\n"
            '+<Drawer saveLabel="Add secret" isPendingAriaLabel="Saving secret" />\n'
        )
        norms = {d.norm for d in extract.extract_deltas(diff)}
        self.assertIn("Add secret", norms)
        self.assertIn("Saving secret", norms)

    def test_classname_is_not_treated_as_a_label(self):
        diff = (
            "diff --git a/frontends/app/src/A.tsx b/frontends/app/src/A.tsx\n"
            "--- a/frontends/app/src/A.tsx\n"
            "+++ b/frontends/app/src/A.tsx\n"
            "@@ -1,1 +1,1 @@\n"
            '+<div className="Flex Row" />\n'
        )
        self.assertEqual(extract.extract_deltas(diff), [])


if __name__ == "__main__":
    unittest.main()


class TestWrappedSemantics(unittest.TestCase):
    """`wrapped` means "not a complete literal", not "Prettier moved it"."""

    def test_reflowed_jsx_text_is_a_valid_replace_target(self):
        # MODELS SEAT sits on its own line. If this ever flips to wrapped, the
        # flagship finding silently loses agent eligibility.
        models = [d for d in load("f4861ad") if d.norm in ("MODELS SEAT", "Models Seat")]
        self.assertTrue(models)
        self.assertFalse(any(d.wrapped for d in models))

    def test_interpolated_literal_is_wrapped(self):
        desc = [d for d in load("e1bc1e6") if d.key == "description"]
        self.assertTrue(desc)
        self.assertTrue(all(d.wrapped for d in desc))

    def test_ternary_branch_is_wrapped(self):
        diff = (
            "diff --git a/frontends/app/src/A.tsx b/frontends/app/src/A.tsx\n"
            "--- a/frontends/app/src/A.tsx\n"
            "+++ b/frontends/app/src/A.tsx\n"
            "@@ -1,1 +1,1 @@\n"
            "+<D saveLabel={mode === 'edit' ? 'Replace secret' : 'Add secret'} />\n"
        )
        deltas = extract.extract_deltas(diff)
        self.assertTrue(deltas)
        self.assertTrue(all(d.wrapped for d in deltas))
