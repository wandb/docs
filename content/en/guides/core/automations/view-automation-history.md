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
   - Status indicator (finished, failed, or pending).

## Understanding execution details

Click any automation execution entry to view detailed information. The details shown depend on the execution's status.

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

## Troubleshooting

### Automation not appearing in history
If an expected automation execution doesn't appear:

- After an automation runs, it may take a few minutes to appear in the history. Refresh the page, then check the **Last checked** column to verify that the automation ran.
- Confirm that the automation is turned on. If the automation has a **Disabled** badge, click the automation's name, then turn the automation on using the toggle.
- Verify that the triggering event occurred or the triggering conditions were met. For example:
   - For an automation triggered by artifact events, such as when an artifact has an alias applied, check the artifact's version history.
   - For an automation triggered by a run metric, confirm that the run logged the expected metric values.
   - For an automation with a filter, verify the details. For example, for a regex filter, verify the regex.

### Missing execution details
Some execution details may be limited if:
- The automation was created before history tracking was turned on.
- Network issues prevented complete logging.
- The automation was deleted and recreated with the same name.

## Next steps
- Learn about [automation events]({{< relref "/guides/core/automations/automation-events.md" >}}) that can trigger automations
- [Create a Slack automation]({{< relref "/guides/core/automations/create-automations/slack.md" >}})
- [Create a webhook automation]({{< relref "/guides/core/automations/create-automations/webhook.md" >}})
