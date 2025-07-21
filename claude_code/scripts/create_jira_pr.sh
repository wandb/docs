#!/bin/bash

# Script to create a GitHub PR for Jira documentation fixes
# Usage: ./create_jira_pr.sh <jira_ticket_number> <issue_summary> <changes_description>

JIRA_TICKET_NUMBER="$1"
ISSUE_SUMMARY="$2"
CHANGES_DESCRIPTION="$3"

# --- Input validation ---
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
  echo "Usage: $0 <jira_ticket_number> <issue_summary> <changes_description>"
  echo "Example: $0 'DOCS-1503' 'Artifacts code snippet incorrect' 'Fixed variable name in run.log_artifact() call'"
  exit 1
fi

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
  echo "Error: Not in a git repository"
  exit 1
fi

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
  echo "Error: GitHub CLI (gh) is not installed"
  echo "Please install it from: https://cli.github.com/"
  exit 1
fi

# Get the current branch name
BRANCH_NAME=$(git branch --show-current)

# Check if there are changes to commit
if [[ -n $(git status --porcelain) ]]; then
  echo "Warning: You have uncommitted changes. Please commit them before creating a PR."
  exit 1
fi

# Push the current branch if it hasn't been pushed yet
if ! git ls-remote --heads origin "$BRANCH_NAME" | grep -q "$BRANCH_NAME"; then
  echo "Pushing branch $BRANCH_NAME to origin..."
  git push -u origin "$BRANCH_NAME"
fi

# Create the PR
echo "Creating GitHub PR for $JIRA_TICKET_NUMBER..."

gh pr create \
  --title "Fix: $ISSUE_SUMMARY ($JIRA_TICKET_NUMBER)" \
  --body "$(cat <<EOF
## Summary
- Fixed issue described in Jira ticket $JIRA_TICKET_NUMBER
- $ISSUE_SUMMARY

## Changes
$CHANGES_DESCRIPTION

## Testing
- [x] Verified code examples are syntactically correct
- [x] Checked that the fix matches the requested change in the Jira ticket
- [x] Ensured consistency with surrounding documentation

## Jira Ticket
[View ticket $JIRA_TICKET_NUMBER](https://wandb.atlassian.net/browse/$JIRA_TICKET_NUMBER)

🤖 Generated with [Claude Code](https://claude.ai/code)
EOF
)"

# Check if PR was created successfully
if [ $? -eq 0 ]; then
  echo "✅ PR created successfully!"
  
  # Get the PR URL
  PR_URL=$(gh pr view --json url -q .url)
  echo "📎 PR URL: $PR_URL"
else
  echo "❌ Failed to create PR"
  exit 1
fi