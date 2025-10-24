#!/bin/bash

# Get list of changed MDX files in a PR
# This script is used by the GitHub Action to identify which files need validation

# Get the base and head commits for the PR
BASE_SHA="${1:-origin/main}"
HEAD_SHA="${2:-HEAD}"

# Get list of changed files that are MDX files
git diff --name-only --diff-filter=ACMRT "$BASE_SHA" "$HEAD_SHA" | grep '\.mdx$' || true
