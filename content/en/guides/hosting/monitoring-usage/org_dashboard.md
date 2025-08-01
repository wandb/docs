---
menu:
  default:
    identifier: org_dashboard
    parent: monitoring-and-usage
title: View organization activity
---

This page shows various ways to view activity within your W&B organization.

## View user status and activity

{{< tabpane text=true >}}
{{% tab header="Dedicated / Self-managed" value="dedicated" %}}
1. To access the **Organization Dashboard**, navigate to `https://<org-name>.io/org/dashboard/`. Replace `<org-name>` with your organization name. The **Users** tab opens by default. It lists all users, along with data about each user.
1. To sort the list by user status, click the **Last Active** column label. Each user's status is one of the following:

    * **Invite pending**: Admin has sent invite but user has not accepted invitation. 
    * **Active**: User has accepted the invite and created an account.
    * **-**: The user was previously active but has not been active in the last 6 months.
    * **Deactivated**: Admin has revoked access of the user.
1. To see details about a user's last activity, hover your mouse over the **Last Active** field for the user.  A tooltip appears that shows when the user was added and how many total days the user has been active.

    A user is _active_ if they:
    - log in to W&B.
    - view any page in the W&B App.
    - log runs.
    - use the SDK to track an experiment.
    - interact with the W&B Server in any way.
{{% /tab %}}

{{% tab header="Multi-tenant Cloud" value="saas" %}}
1. Navigate to the [**Members** page](https://wandb.ai/account-settings/wandb/members/). This page lists all users, along with data about each user.
1. To sort the list by user status, click the **Last Active** column label. Each user's status is one of the following:

    * **Invite pending**: Admin has sent invite but user has not accepted invitation. 
    * **Active**: User has accepted the invite and created an account.
    * `-`: A hyphen indicates that the user has not yet been active within the organization.

    A user is _active_ if they perform any auditable action scoped to the organization. For a full list, refer to [Actions]({{< relref "/guides/hosting/monitoring-usage/audit-logging.md#actions" >}}) in the Audit Logging page.

{{% /tab %}}
{{< /tabpane >}}

## Export user details

{{< tabpane text=true >}}
{{% tab header="Dedicated or Self-managed" value="dedicated" %}}
From the **Users** tab, you can export details about how your organization uses W&B in CSV format.

1. Navigate to the **Organization Dashboard** at `https://<org-name>.io/org/dashboard/`. Replace `<org-name>` with your organization name. The **Users** tab opens by default.
1. Click the action `...` menu next to the **Invite new user user** button.
1. Click **Export as CSV**. The downloaded CSV file lists details about each user of an organization, such as their user name and email address, the time they were last active, their roles, and more.
{{% /tab %}}

{{% tab header="Multi-tenant Cloud" value="saas" %}}
Exporting users is not available for Multi-tenant Cloud.
{{% /tab %}}
{{< /tabpane >}}

## View activity over time
This section shows how to get an aggregate view of activity over time.

{{< tabpane text=true >}}
{{% tab header="Dedicated or Self-managed" value="dedicated" %}}

Use the plots in the **Activity** tab to get an aggregate view of how many users have been active over time.

1. To access the **Organization Dashboard**, navigate to `https://<org-name>.io/org/dashboard/`. Replace `<org-name>` with your organization name.
1. Click the **Activity** tab.
1. The **Total active users** plot shows how many unique users have been active in a period of time (defaults to 3 months).
1. The **Users active over time** plot shows the fluctuation of active users over a period of time (defaults to 6 months). Hover your mouse over a pointo to see the number of users on that date.

To change the period of time for a plot, use the drop-down. You can select:
- Last 30 days
- Last 3 months
- Last 6 months
- Last 12 months
- All time

{{% /tab %}}
{{% tab header="Multi-tenant Cloud" value="saas" %}}

Use the plots in the **Activity Dashboard** to get an aggregate view of activity over time:

1. Click the user profile icon at the top right.
1. Under **Account**, click **Users**.
1. View the Activity Panel above the list of users. It shows:

  - The **Active user count** badge shows how many unique users have been active in a period of time (defaults to 3 months). A user is _active_ if they perform any auditable action scoped to the organization. For a full list, refer to [Actions]({{< relref "/guides/hosting/monitoring-usage/audit-logging.md#actions" >}}) in the Audit Logging page.
  - The **Weekly active users** plot shows the number of users active per week.
  - The **Most active user** leaderboard ranks the top ten most active users by how many days they were active over the period of time, as well as when they were most recently active.

1. To adjust the span of time the plots show, click the date picker in the top right. You can choose 7, 30, or 90 days. The default date range is 30 days. All of the plots share the same time range and update automatically.

{{% /tab %}}
{{< /tabpane >}}