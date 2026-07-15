#!/bin/bash
set -e

# Match .github/workflows/validate-mdx.yml: pages (.mdx), Mintlify config / OpenAPI (.json, .yaml),
# and common doc assets. OpenAPI and docs.json drive generated MDX at build time.
MINTLIFY_RELEVANT_EXT_REGEX='\.(mdx|json|ya?ml|png|jpe?g|webp)$'
CHECK=${1:-all}

case "$CHECK" in
  all|validate|broken-links) ;;
  *)
    echo "Usage: $0 [all|validate|broken-links]"
    exit 2
    ;;
esac

# In CI pull_request runs, prefer explicit base/head SHAs (reliable; origin/$GITHUB_BASE_REF
# may not exist after checkout). PR_BASE_SHA / PR_HEAD_SHA are set from the workflow.
if [ -n "${PR_BASE_SHA:-}" ] && [ -n "${PR_HEAD_SHA:-}" ]; then
  echo "Checking for Mintlify-relevant files in PR (${PR_BASE_SHA:0:7}...${PR_HEAD_SHA:0:7})..."
  CHANGED_RELEVANT=$(git diff --name-only --no-renames "${PR_BASE_SHA}...${PR_HEAD_SHA}" | grep -E "$MINTLIFY_RELEVANT_EXT_REGEX" || true)
elif [ -n "${GITHUB_BASE_REF:-}" ]; then
  # Local or legacy CI: compare against remote base ref
  echo "Checking for Mintlify-relevant files in PR..."
  CHANGED_RELEVANT=$(git diff --name-only --no-renames "origin/${GITHUB_BASE_REF}...HEAD" | grep -E "$MINTLIFY_RELEVANT_EXT_REGEX" || true)
else
  CHANGED_RELEVANT=""
fi

if { [ -n "${PR_BASE_SHA:-}" ] && [ -n "${PR_HEAD_SHA:-}" ]; } || [ -n "${GITHUB_BASE_REF:-}" ]; then
  if [ -z "$CHANGED_RELEVANT" ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "NO MINTLIFY-RELEVANT FILES CHANGED"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Nothing in this PR affects Mintlify content or config. Skipping validation."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    exit 0
  fi
  echo "Found changed files (Mintlify-relevant):"
  while IFS= read -r changed_file; do
    echo "  $changed_file"
  done <<< "$CHANGED_RELEVANT"
  echo ""
fi

run_validate() {
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "VALIDATING DOCUMENTATION BUILD"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Running: mint validate"
  echo ""

  # mint validate exits with non-zero if there are any errors or warnings.
  if mint validate; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✅ MINTLIFY VALIDATION PASSED"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "No errors or warnings detected"
  else
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "❌ MINTLIFY VALIDATION FAILED"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Errors or warnings were detected. Please fix them or"
    echo "file an issue if you believe they are incorrect:"
    echo "https://github.com/wandb/docs/issues/new"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    exit 1
  fi
}

run_broken_links() {
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "CHECKING FOR BROKEN LINKS"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Running: mint broken-links"
  echo ""

  # mint broken-links exits with non-zero if broken links are found.
  if mint broken-links; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✅ NO BROKEN LINKS FOUND"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  else
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "❌ BROKEN LINKS DETECTED"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Please fix the broken links reported above"
    exit 1
  fi
}

case "$CHECK" in
  validate)
    run_validate
    ;;
  broken-links)
    run_broken_links
    ;;
  all)
    run_validate
    echo ""
    run_broken_links
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✅ ALL VALIDATION CHECKS PASSED"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "- No validation errors or warnings"
    echo "- No broken links"
    ;;
esac

exit 0
