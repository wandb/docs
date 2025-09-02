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

This page describes how to view and understand the execution history of your W&B [automations]({{< relref "/guides/core/automations/">}}) and track what triggered an automation, what actions were taken, and whether they succeeded or failed.

## Overview

Each executed automation generates a record that includes:
- **Execution timestamp**: When the automation was triggered.
- **Triggering event**: The specific event triggered the automation.
- **Status**: The execution's status. See [Execution status](#execution-status).
- **Action details**: Information about what action was performed, such as notifying a Slack channel or running a webhook.
- **Result details**: Additional information, if any, about the final outcome of executing the automation, including the error for a failed execution.

## View automation history

You can view automation history from multiple locations in the W&B interface:

### From the Automations tab

{{< tabpane text=true >}}
{{% tab "Registry" %}}
1. Navigate to your registry by clicking on **Registry** in the left sidebar.
1. Select your registry from the list.
1. Click **Automations** tab to view the registry's automations. In each row, the **Last execution** column shows when the automation last executed.
1. In the **Automations history** tab, view all executions of the registry's automations in reverse chronological order, starting with the most recent execution. Each execution's metadata displays, including the event, action, and status.

{{% /tab %}}
{{% tab "Project" %}}
1. Navigate to your project from the W&B home page or by using the project selector.
1. Click the **Automations** tab in the project navigation bar (located alongside Overview, Workspace, Runs, etc.). The project's automations display.

    - Find the automation you want to investigate. You can use the search bar to filter by automation name, and you can sort by the last triggered date to find recently executed automations.

    - Click an automation name to open its details page.
1. In the **History** tab, view all executions of the project's automations in reverse chronological order, starting with the most recent execution. Each execution's metadata displays, including the event, action, and status.

{{% /tab %}}
{{< /tabpane >}}

### From the automation details page

When viewing an individual automation:

1. Navigate to the automation details page by clicking an automation name from the Automations tab.
1. Click the **History** tab to view a chronological list of all executions.
1. Each entry shows:
   - Execution date and time.
   - Triggering event details.
   - Status indicator (success, failure, or in progress).
   - Duration of execution.

## Understanding execution details

Click any automation execution entry to view detailed information. The details shown depend on the execution's status. See [Execution status](#execution-status) and the following sections.

### Execution status

| Status | Icon | Description |
|--------|------|-------------|
| **Finished** | ✅ | A green checkmark indicates that the automation completed successfully. |
| **Failed** | ❌ | A red X indicates that the automation terminated unsuccessfully for any reason. |
| **Pending** | 🔄 Spinning icon | A spinning arrow icon indicates that the automation is running. |

#### Successful executions
A successful execution shows:
- **Trigger information**:
  - Event type (e.g., "Artifact alias added")
  - Source details (artifact name, version, user who triggered)
  - Exact timestamp with timezone
- **Payload sent**:
  - For Slack: The formatted message content
  - For webhooks: The complete JSON payload. No values are masked.
- **Delivery confirmation**:
  - HTTP status code (e.g., "200 OK")
  - For Slack: Channel and thread information
- **Response data** (webhook automations only):
  - Full response body.
Request and response headers are omitted.

#### Failed executions
A failed execution shows:
- **Error summary**: High-level description (e.g., "Connection timeout", "Authentication failed")
- **Detailed error message**:
  ```text
  Error: Failed to connect to webhook endpoint
  URL: https://api.example.com/webhook
  Status: 502 Bad Gateway
  Response: "upstream server temporarily unavailable"
  ```

## Filter and search automation history
If you have a large number of automations or executions:
- Use the search bar to filter or search for automations by name.
- Click a column name to sort by that column. Click a second time to reverse the sort order.


## Common use cases

### Debug failed automations
1. Filter the history to show only failed executions using the status dropdown.
1. Click a failed execution to open the error details panel.
1. Review the error information to identify the issue:

   **Common webhook endpoint issues**:
   - **404 Not Found**: Verify the webhook URL is correct.
   - **500 Internal Server Error**: Check with the webhook service provider.
   - **SSL Certificate Error**: Ensure valid HTTPS certificates.

   **Authentication problems**:
   - **401 Unauthorized**:
     - Navigate to Team Settings > Secrets.
     - Update the secret value used by the automation.
     - Test with the **Test webhook** button.
   - **403 Forbidden**: Check API permissions and scope.

   **Network connectivity**:
   - **Connection timeout**:
     - Verify the endpoint is accessible.
     - Check firewall rules if using private endpoints.
     - Consider increasing timeout in webhook configuration (Edit automation > Advanced settings > Request timeout).

   **Payload formatting**:
   - **400 Bad Request**:
     - Review the JSON syntax in the payload template.
     - Ensure all required fields are included.
     - Check data types match the endpoint's expectations.

1. After fixing the issue:
   - Click **Retry Now** to test the fix immediately.
   - Monitor the next scheduled execution.

### Verify automation triggers
1. Check the history to confirm an automation was triggered by a specific event.
1. Verify the timing and frequency of executions.
1. If necessary, adjust automations that are triggering too frequently or missing expected events or conditions.

### Audit automation activity
1. Export automation history for compliance or reporting.
1. Track which user and action triggered a given automation.
1. Monitor the overall health and reliability of your automation workflows.



## Troubleshooting

### Automation not appearing in history
If an expected automation execution doesn't appear:

1. **Verify the trigger event occurred**:
   - For artifact events: Check the artifact's version history.
   - For run metrics: Confirm the run logged the expected metric values.
   - For aliases/tags: Verify they were actually applied.

1. **Check automation status**:
   - Look for a **Disabled** badge on the automation list.
   - Click the automation's name to open its configuration.
   - Turn the automation back on using the toggle.

1. **Review filter criteria**:
   - Click the automation's name to open its configuration.
   - Check the **Filters** section for:
     - Artifact name patterns (regex).
     - Collection restrictions.
     - User filters.
   - Test your event against the filter using the **Test filters** tool.

1. **Inspect conditional logic**:
   - Advanced automations may have "Only if" conditions. For example, "Only trigger if artifact size > 100MB".
   - Check if your event met all conditions.

1. **Timing considerations**:
   - History may have a 1-2 minute delay to update after an automation runs.
   - Refresh the page after a few minutes, then check the "Last checked" timestamp at the top of the history.

### Missing execution details
Some execution details may be limited if:
- The automation was created before history tracking was turned on.
- Network issues prevented complete logging.
- The automation was deleted and recreated with the same name.

## Recommendations

1. **Monitor automations**:
   - Set a regular reminder to review automation histories.
   - Focus on automations critical to your workflow, and look for patterns in execution times and success rates.

1. **Set up alerts**:
   - Configure email notifications for automation failures in your team settings.
   - Send automtion alerts to a dedicated Slack channel.
   - Use webhook automations to trigger PagerDuty for critical failures.

1. **Document patterns**:
   - Keep a runbook of common errors and their solutions.
   - Document which external services each webhook depends on.
   - Note any time-based patterns to expect, such as transient failures during maintenance.

1. **Test automations**:
   - Use test artifacts or events while developing an automation and before turning it on in production.
   - Verify the first few executions for a new automation.
   - Test webhook endpoints independently using tools or scripts outside W&B.

1. **Performance optimization**:
   - Monitor execution duration trends.
   - Investigate automations that unexpectedly take longer than 30 seconds.
   - To improve performance, consider breaking complex automations into smaller, focused ones.

## Next steps
- Learn about [automation events]({{< relref "/guides/core/automations/automation-events.md" >}}) that can trigger automations
- [Create a Slack automation]({{< relref "/guides/core/automations/create-automations/slack.md" >}})
- [Create a webhook automation]({{< relref "/guides/core/automations/create-automations/webhook.md" >}})
