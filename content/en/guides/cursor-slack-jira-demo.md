---
# Cursor Slack→JIRA Background Agent Demo

> This file is an independent demo note. It does not modify existing docs.
> Replace placeholders as needed during a live demo.

## Goal
Demonstrate how a Cursor background agent can:
- Read a Slack channel message/thread
- Detect and follow a referenced JIRA ticket
- Start an autonomous background task to address it
- Produce an artifact (this Markdown) as the result

## Demo Setup
- **Slack channel**: `<#C09G3LN2WDS>`
- **JIRA issue**: Mentioned in the Slack thread (replace with real link during demo)
- **Agent**: Cursor background agent

## How it works (conceptual)
1. The agent listens to Slack for mentions or trigger phrases.
2. When a message references a JIRA issue URL/key, the agent fetches the ticket details.
3. The agent plans steps to resolve or address the task, then executes them autonomously.
4. The agent produces an output artifact and posts progress back to Slack.

## What happened in this run
- Slack API access was not available in this environment (received `not_authed`).
- No JIRA credentials were detected in environment variables.
- As a demo, the agent created this self-contained Markdown artifact instead of editing docs.

## Placeholders for a live, fully wired demo
- **SLACK_BOT_TOKEN**: `xoxb-***` with `channels:history` or `conversations:history` scope
- **SLACK_CHANNEL_ID**: `C09G3LN2WDS`
- **JIRA_BASE_URL**: `https://your-domain.atlassian.net`
- **JIRA_EMAIL**: `user@company.com`
- **JIRA_API_TOKEN**: `****` (from Atlassian account)

## Example commands (for local testing)
```bash
# Slack: get channel history
curl -sS -H "Authorization: Bearer $SLACK_BOT_TOKEN" \
  "https://slack.com/api/conversations.history?channel=$SLACK_CHANNEL_ID" | jq

# Extract the first JIRA URL from the latest message text (toy example)
LATEST_TEXT=$(curl -sS -H "Authorization: Bearer $SLACK_BOT_TOKEN" \
  "https://slack.com/api/conversations.history?channel=$SLACK_CHANNEL_ID&limit=1" | jq -r '.messages[0].text')
JIRA_URL=$(grep -oE 'https?://[^ ]+atlassian\.net/browse/[A-Z]+-[0-9]+' <<<"$LATEST_TEXT" | head -n1)

# JIRA: fetch issue details
curl -sS -u "$JIRA_EMAIL:$JIRA_API_TOKEN" \
  -H 'Accept: application/json' \
  "$JIRA_BASE_URL/rest/api/3/issue/${JIRA_URL##*/}" | jq
```

## Sample agent behavior
- Parse Slack thread for context and acceptance criteria
- Read JIRA summary, description, labels, priority
- Define a minimal plan and deliverable
- Create/update a branch and produce a Markdown artifact
- Report completion back in Slack with a link to the artifact

## This artifact
You are reading the demo artifact the agent produced. In a real run, the content would:
- Quote the Slack thread’s key requirements
- Summarize the JIRA ticket and acceptance criteria
- Provide a proposed solution and next steps

## Notes
- This demo intentionally avoids touching existing documentation.
- Safe to delete after the demo or keep as a reference.