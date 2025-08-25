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

## Access automation history

You can view automation history from multiple locations:

### From the Automations tab

{{< tabpane text=true >}}
{{% tab "Registry" %}}
1. Navigate to your registry and click the **Automations** tab
2. Find the automation you want to investigate
3. Click on the automation name to view its details
4. Select the **History** tab to see all past executions

{{% /tab %}}
{{% tab "Project" %}}
1. Navigate to your project and click the **Automations** tab
2. Find the automation you want to investigate
3. Click on the automation name to view its details
4. Select the **History** tab to see all past executions

{{% /tab %}}
{{< /tabpane >}}

### From the automation details page

When viewing an individual automation:
1. The **History** tab displays a chronological list of all executions
2. Each entry shows:
   - Execution date and time
   - Triggering event details
   - Status indicator (success, failure, or in progress)
   - Duration of execution

## Understanding execution details

Click on any execution entry to view detailed information:

### Successful executions
For successful executions, you'll see:
- The exact trigger that initiated the automation
- The payload or message that was sent
- Confirmation of successful delivery
- Response details (for webhook automations)

### Failed executions
For failed executions, the details include:
- The error message explaining why the automation failed
- The stage at which the failure occurred
- Debugging information to help resolve the issue
- Options to retry the automation (if applicable)

## Filter and search history

The automation history interface provides several ways to find specific executions:

### Filter by status
- **All**: View all executions
- **Successful**: Show only successful executions
- **Failed**: Show only failed executions
- **In Progress**: Show currently running automations

### Filter by date range
- Use the date picker to select a specific time period
- Options include: Last 24 hours, Last 7 days, Last 30 days, or custom range

### Search functionality
- Search by trigger event details
- Search by error messages
- Search by artifact names or aliases

## Common use cases

### Debugging failed automations
1. Filter the history to show only failed executions
2. Click on a failed execution to see the error details
3. Use the error information to:
   - Fix webhook endpoint issues
   - Update authentication credentials
   - Resolve network connectivity problems
   - Correct payload formatting errors

### Verifying automation triggers
1. Check the history to confirm an automation was triggered by a specific event
2. Verify the timing and frequency of executions
3. Ensure automations are not triggering too frequently or missing expected events

### Auditing automation activity
1. Export automation history for compliance or reporting
2. Track which users' actions triggered automations
3. Monitor the overall health and reliability of your automation workflows

## History retention

- Automation history is retained for 90 days by default
- Failed executions may include additional diagnostic information for troubleshooting
- History data can be exported for long-term storage if needed

## Troubleshooting

### Automation not appearing in history
If an expected automation execution doesn't appear:
- Verify the trigger event actually occurred
- Check that the automation is enabled
- Ensure the event matches the automation's filter criteria
- Review any conditional logic in the automation configuration

### Missing execution details
Some execution details may be limited if:
- The automation was created before history tracking was enabled
- Network issues prevented complete logging
- The automation was deleted and recreated

## Best practices

1. **Regular monitoring**: Check automation history periodically to ensure workflows are running as expected
2. **Set up alerts**: Configure notifications for failed automations to address issues quickly
3. **Document patterns**: Note common failure patterns to improve automation reliability
4. **Test automations**: Use the history to verify new automations are working correctly before relying on them for critical workflows

## Next steps
- Learn about [automation events]({{< relref "/guides/core/automations/automation-events.md" >}}) that can trigger automations
- [Create a Slack automation]({{< relref "/guides/core/automations/create-automations/slack.md" >}})
- [Create a webhook automation]({{< relref "/guides/core/automations/create-automations/webhook.md" >}})