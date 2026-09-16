#!/usr/bin/env bash
#
# Checks that a pull request's DocEngine pin points at a published release, and
# fails if it does not.
#
#
# WHAT A "PIN" IS
#
# CoreWeave's documentation tooling, DocEngine, lives in its own repository and
# is attached to this one as the `docengine` git submodule. A submodule records
# one exact commit of the other repository - that recorded commit is the pin.
# Changing it decides which version of the tooling this repo runs.
#
# DocEngine publishes releases as version tags such as v3.5.0. The rule this
# script enforces is that only a published release may be merged here, so the
# tooling is always a known, released version rather than whatever happened to
# be on DocEngine's main branch that day.
#
#
# WHY THE CHECK EXISTS WHEN THE UPDATER ALREADY PICKS RELEASES
#
# Automated dependency-update pull requests bump the pin to the newest
# published release on their own, so they pass this check without help, and it
# is tempting to conclude the check is pointless. It is not aimed at them. It
# is aimed at every other way the pin can move:
#
#   - By accident, riding along in a pull request about something else. Every
#     working copy of this repository has the submodule populated, so whenever
#     someone's local copy of it drifts from the recorded pin, git reports
#     "modified: docengine (new commits)" and a habitual `git add -A` or
#     `git commit -a` quietly stages that drift into an unrelated docs change.
#     This is the most common way submodule pins move in any repository. The
#     damage can also surface late: a pin recording a commit that exists only
#     on someone's machine, or on a since-deleted branch, merges cleanly and
#     breaks OTHER people's builds afterwards, when the pinned commit cannot
#     be fetched. This check turns both into an immediate red X on the pull
#     request that caused them.
#
#   - By a botched merge, which can silently regress the pin to an older
#     commit, flatten the submodule into a plain directory, or repoint
#     .gitmodules at a different repository. The workflow that runs this
#     script checks for each of those and names the real cause.
#
#   - On purpose, in a draft. Pinning a prerelease is the supported way to
#     test unreleased tooling in real CI (see below). This check is what makes
#     such a draft safe to leave open: it stays red, so it cannot merge by
#     accident.
#
# Two further reasons watchfulness cannot replace it:
#
#   - A reviewer cannot check this by eye. A pin bump diffs as two bare commit
#     IDs, and telling a release commit from an arbitrary one takes exactly
#     the tag lookup this script automates.
#
#   - The updater preferring release tags over newer untagged commits is
#     observed behavior, confirmed by experiment, not a documented guarantee.
#     If it ever changes, this check is what stands between an untagged
#     commit and main.
#
#
# WHY IT CHECKS TAGS INSTEAD OF "IS THIS COMMIT ON MAIN"
#
# DocEngine squash-merges its pull requests. That does not rewrite main, which
# only gains commits. It means the individual commits on a pull request branch
# never join main at all: the merge creates one new commit standing in for the
# whole branch. Once that branch is deleted, the original commits are not
# reachable from any branch, so a pin pointing at one refers to a commit nobody
# can find from main. A tag avoids that, because the tag is itself a reference
# and keeps its commit reachable for good. "Does this commit carry a release
# tag" is therefore both a stricter rule and a more stable one.
#
#
# WHY PRERELEASE (rc) TAGS ARE REJECTED
#
# DocEngine also publishes prerelease tags such as v3.6.0-rc.1, so that a draft
# pull request here can test unreleased tooling in real CI before it ships.
# Those are for draft pull requests only. A pull request pinned to one is
# SUPPOSED to fail this check, and stays red until it is repinned to a real
# release. That is the workflow behaving correctly, not a problem to fix.
#
# The full rules:
# https://github.com/coreweave/docengine/blob/main/docs/adr/0009-semver-releases-and-tag-propagation.md
#
#
# WHY THIS SCRIPT LIVES IN THIS REPOSITORY
#
# DocEngine publishes shared CI actions that this repo uses, and normally logic
# like this would live there. But, this script cannot live in DocEngine, because
# it checks the very pin that would be used to fetch those shared actions.
# Getting it from DocEngine would be circular. It stays here on purpose.
#
#
# RUN IT LOCALLY
#
#   PINNED_SHA=<commit-sha> READ_TOKEN=<token> .github/scripts/docengine-pin-guard.sh
#
# Environment:
#   PINNED_SHA    (required) the DocEngine commit this pull request pins
#   READ_TOKEN    (optional) a token that can read tags on the DocEngine repo
#   ENGINE_REPO   (optional) defaults to coreweave/docengine
#   REQUIRE_TOKEN (optional) "true" to fail when READ_TOKEN is missing instead
#                 of skipping. See the note in docengine-pin-guard-pr.yaml.

set -euo pipefail

ENGINE_REPO="${ENGINE_REPO:-coreweave/docengine}"
PINNED_SHA="${PINNED_SHA:-}"
READ_TOKEN="${READ_TOKEN:-}"
REQUIRE_TOKEN="${REQUIRE_TOKEN:-false}"

DOC_URL="https://github.com/coreweave/docengine/blob/main/docs/adr/0009-semver-releases-and-tag-propagation.md"

summary() {
  if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
    printf '%s\n' "$*" >> "${GITHUB_STEP_SUMMARY}"
  fi
}

if [ -z "${PINNED_SHA}" ]; then
  echo "::error title=DocEngine pin guard::No docengine submodule entry was found. Expected one at the path 'docengine'."
  exit 1
fi

if [ -z "${READ_TOKEN}" ]; then
  if [ "${REQUIRE_TOKEN}" = "true" ]; then
    echo "::error title=DocEngine pin guard::No read token was available, so the pin could not be checked against ${ENGINE_REPO} tags."
    exit 1
  fi
  # Skip instead of blocking. Automated dependency-update pull requests are by
  # far the most common thing this check sees, and they do not always have
  # access to the secret. Failing here would turn every one of them red and
  # stop the pin ever moving forward. Ordinary pull requests from branches in
  # this repository always have the secret, so this is a rare path.
  echo "::warning title=DocEngine pin guard::No read token was available, so the release check was skipped for pin ${PINNED_SHA}."
  summary "### DocEngine pin guard: SKIPPED"
  summary ""
  summary "No token was available to read tags from \`${ENGINE_REPO}\`, so the pin \`${PINNED_SHA}\` was **not** checked against the published releases."
  exit 0
fi

echo "::add-mask::${READ_TOKEN}"

# In CI this script runs inside a checkout made by actions/checkout, which
# writes an "http.https://github.com/.extraheader" Authorization header (the
# host repository's own GITHUB_TOKEN) into that checkout's local git config.
# A custom Authorization header OVERRIDES the credential embedded in the URL
# below, so without the reset git authenticates every request as the host's
# GITHUB_TOKEN - which cannot read the private engine repository - and this
# lookup fails no matter how good READ_TOKEN is. Setting the extraheader to
# an empty value on the command line clears the accumulated list, so the URL
# credential is the one that gets sent.
ls_remote_err="$(mktemp)"
if ! raw_tags="$(git -c "http.https://github.com/.extraheader=" ls-remote --tags "https://x-access-token:${READ_TOKEN}@github.com/${ENGINE_REPO}.git" 'refs/tags/v*' 2>"${ls_remote_err}")"; then
  echo "::error title=DocEngine pin guard::Could not read the tags on ${ENGINE_REPO}. Check that the token being used can read that repository. git said: $(tr '\n' ' ' <"${ls_remote_err}")"
  rm -f "${ls_remote_err}"
  exit 1
fi
rm -f "${ls_remote_err}"

# Work out which commit each tag actually points at.
#
# Git has two kinds of tag. A lightweight tag points straight at a commit. An
# annotated tag points at a small tag object which in turn points at the commit,
# and `git ls-remote` reports it as two lines: the tag object, then a second
# line ending in '^{}' holding the real commit. So when a '^{}' line exists it
# is the one to trust. Getting this wrong would compare against tag object IDs
# and never match anything.
tag_commits="$(
  printf '%s\n' "${raw_tags}" | awk -F'\t' '
    $2 != "" {
      ref = $2
      if (ref ~ /\^\{\}$/) {
        sub(/\^\{\}$/, "", ref)
        peeled[ref] = $1
      } else {
        direct[ref] = $1
      }
    }
    END {
      for (r in direct) {
        sha = (r in peeled) ? peeled[r] : direct[r]
        name = r
        sub(/^refs\/tags\//, "", name)
        print sha "\t" name
      }
    }
  '
)"

if [ -z "${tag_commits}" ]; then
  echo "::error title=DocEngine pin guard::${ENGINE_REPO} has no version tags at all, so no pin can be valid yet. A release needs to be published first."
  exit 1
fi

# Releases only: v<major>.<minor>.<patch> with nothing after the patch number.
# Anything with a suffix, such as v3.6.0-rc.1, is a prerelease and is handled
# separately below so the error message can explain itself.
matched_release="$(printf '%s\n' "${tag_commits}" | awk -F'\t' -v sha="${PINNED_SHA}" '$1 == sha && $2 ~ /^v[0-9]+\.[0-9]+\.[0-9]+$/ { print $2 }' | head -n 1)"

if [ -n "${matched_release}" ]; then
  echo "DocEngine pin guard: ${PINNED_SHA} is release ${matched_release}."
  summary "### DocEngine pin guard: PASS"
  summary ""
  summary "The pinned commit \`${PINNED_SHA}\` is release **${matched_release}**."
  exit 0
fi

matched_prerelease="$(printf '%s\n' "${tag_commits}" | awk -F'\t' -v sha="${PINNED_SHA}" '$1 == sha && $2 ~ /^v[0-9]+\.[0-9]+\.[0-9]+-/ { print $2 }' | head -n 1)"

latest_release="$(printf '%s\n' "${tag_commits}" | awk -F'\t' '$2 ~ /^v[0-9]+\.[0-9]+\.[0-9]+$/ { print $2 }' | sort -V | tail -n 1)"

# Reaching here with no release at all is possible: the repository can have v*
# tags (so the earlier "no tags" check passed) while every one of them is a
# prerelease. Without this, the messages below would end with a blank value and
# read as though something had gone wrong with the check itself.
if [ -n "${latest_release}" ]; then
  latest_hint="The latest release is ${latest_release}."
  latest_hint_md="The latest release is \`${latest_release}\`."
else
  latest_hint="No release has been published yet, only prereleases."
  latest_hint_md="No release has been published yet, only prereleases."
fi

summary "### DocEngine pin guard: FAIL"
summary ""
summary "The pinned commit \`${PINNED_SHA}\` is not a published DocEngine release."

if [ -n "${matched_prerelease}" ]; then
  echo "::error title=DocEngine pin guard::This pull request pins DocEngine to the prerelease ${matched_prerelease}. Prereleases are for testing in draft pull requests and cannot be merged, so this check stays red until you repin to a release. ${latest_hint} Details: ${DOC_URL}"
  summary "It is the prerelease **${matched_prerelease}**. Prereleases let a draft pull request test unreleased tooling in real CI, but they must never be merged, so this check stays red by design. Repin to a release before merging. ${latest_hint_md}"
  summary ""
  summary "[Why this rule exists](${DOC_URL})"
  exit 1
fi

echo "::error title=DocEngine pin guard::This pull request pins DocEngine to ${PINNED_SHA}, which is not a published release. This repository can only use released versions of the tooling, not arbitrary commits. ${latest_hint} Details: ${DOC_URL}"
summary "It carries no version tag on \`${ENGINE_REPO}\`. This repository can only use published releases of the tooling, not arbitrary commits. ${latest_hint_md}"
summary ""
summary "[Why this rule exists](${DOC_URL})"
exit 1
