---
menu:
  default:
    identifier: org_dashboard
    parent: monitoring-and-usage
title: View organization activity
---

This page describes how to view activity in your W&B organization. Select your deployment type to continue.

## View user status and activity 

{{< tabpane text=true >}}
{{% tab header="Dedicated Cloud / Self-Managed" value="dedicated" %}}
1. Navigate to the **Organization Dashboard** at `https://<org-name>.io/org/dashboard/`. Replace `<org-name>` with your organization name. The **Users** tab opens by default and lists every user in the organization.
2. To sort the list by user status, click the **Last Active** column header. Each user's status is one of the following:
   * **Invite pending**: An invitation was sent but not yet accepted.
   * **A timestamp**: The user accepted the invitation and has signed in at least once. The timestamp indicates the most recent activity.
   * **Deactivated**: An admin revoked the user's access.
   * **No status**: Indicated by a hyphen. The user was previously active but has not been active in the last six months.
3. Hover over a user's **Last Active** field to see the date the user was added and their total active days.

A user is _active_ if they:
- sign in to W&B.
- open any page in the W&B App.
- log runs.
- use the SDK to track an experiment.
- interact with the W&B server in any way.
{{% /tab %}}

{{% tab header="Multi-tenant Cloud" value="saas" %}}
1. Open the [**Members** page](https://wandb.ai/account-settings/wandb/members/). The table lists every user in your organization.
2. Click the **Last Active** column header to sort by user status. Each user's status is one of the following:
   * **A timestamp**: The user has signed in at least once. The timestamp indicates the most recent activity.
   * **No status**: Indicated by a hyphen. The user has not yet been active within the organization.

A user is _active_ if they perform any auditable action scoped to the organization after May 8 2025. For a full list, see [Actions]({{< relref "/guides/hosting/monitoring-usage/audit-logging.md#actions" >}}) in **Audit logging**.
{{% /tab %}}
{{< /tabpane >}}

## View activity over time  {#view-activity-over-time-dedicated}
{{< tabpane text=true >}}
{{% tab header="Dedicated Cloud / Self-Managed" value="dedicated" %}}
Use the **Activity** tab to see how many users have been active during a given period.

1. Open the **Organization Dashboard** (`https://<org-name>.io/org/dashboard/`).
2. Click **Activity**.
3. Review the following plots:
   * **Total active users**: unique active users during the selected period (defaults to 3 months).
   * **Users active over time**: fluctuation of active users over the period (defaults to 6 months). Hover over a point to see the exact count on that date.

To change the period, use the drop-down above a plot. Options are Last 30 days, Last 3 months, Last 6 months, Last 12 months, and All time.
{{% /tab %}}
{{% tab header="Multi-tenant Cloud" value="saas" %}}
Use the **Activity Dashboard** to view aggregate activity.

1. Click your user icon in the upper-right corner of the W&B App.
2. Under **Account**, click **Users**.
3. Above the table of users, review the Activity Panel:
   * **Active user count**: unique active users during the selected period (defaults to 3 months).
   * **Weekly active users**: users active per week.
   * **Most active user**: top-10 users ranked by active days and last-active date.
4. To change the date range (7, 30, or 90 days; default is 30 days), click the date picker in the upper-right corner. All plots update automatically.
{{% /tab %}}
{{< /tabpane >}}

## Export user details

From the **Users** tab you can download a CSV that lists each user's details (user name, email address, last-active time, roles, and more).

{{< tabpane text=true >}}
{{% tab header="Dedicated Cloud / Self-Managed" value="dedicated" %}}

1. In the **Users** tab, click the **…** actions menu next to **Invite new user**.
2. Click **Export as CSV**.
{{% /tab %}}

{{% tab header="Multi-tenant Cloud" value="saas" %}}

1. In the **Users** tab, click the **…** actions menu in the upper-right corner.
2. Select **Export as CSV** to download the file.
{{% /tab %}}
{{< /tabpane >}}

The CSV export uses the comma (`,`) as the separator, encloses strings in double quotes, and includes a header row that defines these columns:

- `"Name"`
- `"Username"`
- `"Last active"`
- `"Role"`
- `"Email"`
- `"Teams"`
- `"Status"`
- `"Number of Reports"`
- `"Number of Runs"`
- `"Number of active days"`
- `"Models Seat"`
