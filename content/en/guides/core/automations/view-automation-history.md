---
menu:
  default:
    identifier: view-automation-history
    parent: automations
title: View an automation's history
weight: 3
---
{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-cloud-only.md" >}}
{{% /pageinfo %}}

This page describes how to view and understand the execution history of your W&B [automations]({{< relref "/guides/core/automations/">}}). The automation history feature allows you to track when automations were triggered, what actions were taken, and whether they succeeded or failed.

## Overview

Automation history provides a comprehensive log of all automation executions, including:
- **Execution timestamp**: When the automation was triggered
- **Trigger event**: The specific event that caused the automation to run
- **Status**: Whether the automation succeeded, failed, or is still running
- **Action details**: Information about what action was performed (Slack notification sent, webhook called, etc.)
- **Error messages**: Detailed error information if the automation failed

### Execution statuses

Automations can have one of the following statuses:

| Status | Icon | Description |
|--------|------|-------------|
| **Success** | ✅ Green checkmark | The automation completed successfully and the action was performed |
| **Failed** | ❌ Red X | The automation encountered an error and could not complete |
| **In Progress** | 🔄 Spinning icon | The automation is currently executing |
| **Cancelled** | ⏹️ Gray square | The automation was manually stopped before completion |
| **Skipped** | ⏭️ Gray forward | The automation was triggered but skipped due to conditions not being met |

## Access automation history

You can view automation history from multiple locations in the W&B interface:

### From the Automations tab

{{< tabpane text=true >}}
{{% tab "Registry" %}}
1. Navigate to your registry by clicking on **Model Registry** in the left sidebar
2. Select your registry from the list
3. Click the **Automations** tab in the registry view
4. In the automations list, locate the automation you want to investigate
5. Click on the automation name (displayed as a blue link) to open its details page
6. Select the **History** tab from the automation details page navigation
7. The history view will display all past executions in reverse chronological order (most recent first)

{{% /tab %}}
{{% tab "Project" %}}
1. Navigate to your project from the W&B home page or by using the project selector
2. Click the **Automations** tab in the project navigation bar (located alongside Overview, Workspace, Runs, etc.)
3. In the automations list, find the automation you want to investigate
   - Use the search bar to filter by automation name
   - Sort by last triggered date to find recently active automations
4. Click on the automation name (displayed as a blue link) to open its details page
5. Select the **History** tab from the tabs displayed (Configuration, History, etc.)
6. The history view will show all executions for this specific automation

{{% /tab %}}
{{< /tabpane >}}

### From the automation details page

When viewing an individual automation:

<!-- TODO: Add screenshot of automation history tab -->

1. The **History** tab displays a chronological list of all executions
2. Each execution entry in the list shows:
   - **Timestamp**: Exact date and time (e.g., "Dec 15, 2024 3:45:22 PM UTC")
   - **Status icon**: 
     - ✅ Green checkmark for successful executions
     - ❌ Red X for failed executions
     - 🔄 Spinning icon for in-progress executions
   - **Trigger details**: Brief description (e.g., "Artifact version 'v1.2.3' added to collection")
   - **Duration**: Execution time (e.g., "2.3s" or "1m 45s")
   - **Action type**: Icon indicating Slack notification or webhook

3. The list includes pagination controls at the bottom for navigating through large histories
4. The history updates in real-time - new executions appear automatically without needing to refresh the page

## Understanding execution details

Click on any execution entry in the history list to open a detailed view panel:

### Successful executions
For successful executions, the details panel displays:
- **Trigger information**: 
  - Event type (e.g., "Artifact alias added")
  - Source details (artifact name, version, user who triggered)
  - Exact timestamp with timezone
- **Payload sent**: 
  - For Slack: The formatted message content
  - For webhooks: The complete JSON payload (with sensitive values masked)
- **Delivery confirmation**: 
  - HTTP status code (e.g., "200 OK")
  - Response time in milliseconds
  - For Slack: Channel and thread information
- **Response data** (webhook automations only):
  - Response headers
  - Response body (truncated if large)
  - Any returned job IDs or reference numbers

### Failed executions
For failed executions, the error details panel includes:
- **Error summary**: High-level description (e.g., "Connection timeout", "Authentication failed")
- **Detailed error message**: 
  ```text
  Error: Failed to connect to webhook endpoint
  URL: https://api.example.com/webhook
  Status: 502 Bad Gateway
  Response: "upstream server temporarily unavailable"
  ```
- **Failure stage**: Where in the process it failed:
  - "Pre-validation" - Failed before sending
  - "Connection" - Network or DNS issues
  - "Authentication" - Invalid credentials or tokens
  - "Processing" - Remote server rejected the request
- **Debugging information**:
  - Request headers sent
  - Curl command equivalent for testing
  - Suggested fixes based on error type
- **Retry options**:
  - "Retry Now" button (if automation is still valid)
  - "Edit and Retry" to modify payload before retrying

## Filter and search history

The automation history interface provides powerful filtering and search capabilities located at the top of the history list:

### Status filter dropdown
Click the **Status** dropdown to filter executions:
- **All statuses** (default): Shows every execution
- **Successful**: Shows only executions with green checkmarks
- **Failed**: Shows only executions with red X marks
- **In Progress**: Shows currently running executions
- **Cancelled**: Shows manually stopped executions

The filter updates the list in real-time, and the count badge shows matching results (e.g., "Failed (23)")

### Date range picker
Click the **calendar icon** to open the date range selector:
- **Quick ranges** (buttons at the top):
  - Last 24 hours
  - Last 7 days
  - Last 30 days
  - Last 90 days
- **Custom range**: 
  - Select start and end dates from the calendar
  - Time selection available for precision
  - Timezone selector (defaults to browser timezone)

### Search bar
The search bar supports multiple search patterns:
- **Basic search**: Type any text to search across all execution data
- **Advanced search operators**:
  - `status:failed` - Find failed executions
  - `trigger:"artifact alias"` - Find specific trigger types
  - `error:timeout` - Search within error messages
  - `artifact:model-v2` - Find executions related to specific artifacts
  - `user:jane@company.com` - Find executions triggered by specific users

**Search examples**:
- `status:failed error:401` - Failed executions with authentication errors
- `trigger:"run metric" metric:loss` - Metric-triggered automations for loss values
- `artifact:"production-model" last 7 days` - Recent executions for production model
- `webhook:https://api.example.com` - Executions calling a specific webhook endpoint
- `duration:>10s` - Automations taking longer than 10 seconds

## Common use cases

### Debugging failed automations
1. Filter the history to show only failed executions using the status dropdown
2. Click on a failed execution to open the error details panel
3. Review the error information to identify the issue:
   
   **Common webhook endpoint issues**:
   - **404 Not Found**: Verify the webhook URL is correct
   - **500 Internal Server Error**: Check with the webhook service provider
   - **SSL Certificate Error**: Ensure valid HTTPS certificates
   
   **Authentication problems**:
   - **401 Unauthorized**: 
     - Navigate to Team Settings > Secrets
     - Update the secret value used by the automation
     - Test with the "Test webhook" button
   - **403 Forbidden**: Check API permissions and scope
   
   **Network connectivity**:
   - **Connection timeout**: 
     - Verify the endpoint is accessible
     - Check firewall rules if using private endpoints
     - Consider increasing timeout in webhook configuration (Edit automation > Advanced settings > Request timeout)
   
   **Payload formatting**:
   - **400 Bad Request**: 
     - Review the JSON syntax in the payload template
     - Ensure all required fields are included
     - Check data types match the endpoint's expectations

4. After fixing the issue:
   - Use "Retry Now" to test the fix immediately
   - Monitor the next scheduled execution

### Verifying automation triggers
1. Check the history to confirm an automation was triggered by a specific event
2. Verify the timing and frequency of executions
3. Ensure automations are not triggering too frequently or missing expected events

### Auditing automation activity
1. Export automation history for compliance or reporting
2. Track which users' actions triggered automations
3. Monitor the overall health and reliability of your automation workflows

## History retention and data export

### Retention policy
- **Standard retention**: 90 days of execution history
- **Extended retention**: Available for Enterprise plans (configurable up to 365 days)
- **Failed execution details**: Retained with full error logs and request/response data
- **Successful execution summaries**: Retained with essential details (payload details may be truncated after 30 days)

### Exporting history data
To export automation history for compliance or analysis:

1. Click the **Export** button (download icon) at the top of the history list
2. Select export format:
   - **CSV**: Tabular format with key fields
   - **JSON**: Complete execution details including payloads
   - **PDF**: Formatted report for documentation
3. Choose the date range to export
4. Click **Generate Export**
5. The export will be downloaded to your browser's default download location

**CSV export includes**:
- Execution ID (e.g., `exec_1234567890`)
- Timestamp (UTC) (e.g., `2024-01-15T14:30:00Z`)
- Status (e.g., `Success`, `Failed`, `Cancelled`)
- Trigger type and details (e.g., `artifact_alias_added: model-v2`)
- Duration (e.g., `2.3s`)
- Error message (if applicable) (e.g., `Connection timeout after 30s`)
- User who triggered (for manual triggers) (e.g., `user@example.com`)

## Troubleshooting

### Automation not appearing in history
If an expected automation execution doesn't appear:

1. **Verify the trigger event occurred**:
   - For artifact events: Check the artifact's version history
   - For run metrics: Confirm the run logged the expected metric values
   - For aliases/tags: Verify they were actually applied

2. **Check automation status**:
   - Look for a "Disabled" badge on the automation list
   - Disabled automations show a gray toggle switch
   - Re-enable via the automation's configuration page

3. **Review filter criteria**:
   - Open the automation's configuration
   - Check the "Filters" section:
     - Artifact name patterns (regex)
     - Collection restrictions
     - User filters
   - Test your event against the filter using the "Test filters" tool

4. **Inspect conditional logic**:
   - Advanced automations may have "Only if" conditions
   - Example: "Only trigger if artifact size > 100MB"
   - Check if your event met all conditions

5. **Timing considerations**:
   - History may have a 1-2 minute delay
   - Refresh the page if viewing immediately after trigger
   - Check the "Last checked" timestamp at the top of the history

### Missing execution details
Some execution details may be limited if:
- The automation was created before history tracking was enabled
- Network issues prevented complete logging
- The automation was deleted and recreated

## Best practices

1. **Regular monitoring**: 
   - Set a weekly reminder to review automation histories
   - Focus on automations critical to your workflow
   - Look for patterns in execution times and success rates

2. **Set up alerts**: 
   - Configure email notifications for automation failures in Team Settings
   - Create a dedicated Slack channel for automation alerts
   - Use webhook automations to trigger PagerDuty for critical failures

3. **Document patterns**: 
   - Keep a runbook of common errors and their solutions
   - Document which external services each webhook depends on
   - Note any time-based patterns (e.g., failures during maintenance windows)

4. **Test automations**: 
   - Use test artifacts/events before enabling for production
   - Verify the first few executions after creation
   - Test webhook endpoints independently using the provided curl commands

5. **Performance optimization**:
   - Monitor execution duration trends
   - Investigate automations taking longer than 30 seconds
   - Consider breaking complex automations into smaller, focused ones

## Next steps
- Learn about [automation events]({{< relref "/guides/core/automations/automation-events.md" >}}) that can trigger automations
- [Create a Slack automation]({{< relref "/guides/core/automations/create-automations/slack.md" >}})
- [Create a webhook automation]({{< relref "/guides/core/automations/create-automations/webhook.md" >}})