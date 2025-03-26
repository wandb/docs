---
menu:
  default:
    identifier: org_dashboard
    parent: monitoring-and-usage
title: View organization dashboard
---

{{% alert color="secondary" %}}
Organization dashboard is available only with [Dedicated Cloud]({{< relref "/guides/hosting/hosting-options/dedicated_cloud.md" >}}) and [Self-managed instances]({{< relref "/guides/hosting/hosting-options/self-managed.md" >}}).
{{% /alert %}}

## View organization usage of W&B
Use the organization dashboard to get a holistic view of your organization's usage of W&B. The dashboard is organized by tab.

- **Users**: lists details about each user, including their name, email, teams, roles, and last activity.
- **Service accounts**: Lists details about service accounts and allows you to create service accounts.
- **Activity**: lists details about each user's activity.
- **Teams**: lists details about each team, including the number of users and tracked hours, and allows an admin to join a team.
- **Billing**: ssummarizes your organization's charges, allows you to run and export billing reports, and shows details about your license, such as when it expires.
- **Settings**: allows you to configure custom roles and settings related to privacy and authentication.

## View the status of a user
The **Users** tab lists all users, along with data about each user. The **Last Active** column shows whether a user has accepted an invitation and the user's current status:

* **Invite pending**: Admin has sent invite but user has not accepted invitation. 
* **Active**: User has accepted the invite and created an account.
* **-**: The user was previously active but has not been active in the last 6 months.
* **Deactivated**: Admin has revoked access of the user.

To sort the list of users by activity, click the **Last Active** column heading.

## View and share how your organization uses W&B
From the **Users** tab, view details about how your organization uses W&B in CSV format.

1. Click the action `...` menu next to the **Invite new user user** button.
2. Click **Export as CSV**. The CSV files that downloads lists details about each user of an organization, such as their user name and email address, the time they were last active, their roles, and more.

## View user activity
In the **Users** tab, use the **Last Active** column to get an **Activity summary** of an individual user. 

1. To sort the list of users by **Last Active**, click the column name.
1. To see details about a user's last activity, hover your mouse over the **Last Active** field for the user.  A tooltip appears that shows when the user was added and how many total days the user has been active.

A user is _active_ if they:
- log in to W&B.
- view any page in the W&B App.
- log runs.
- use the SDK to track an experiment.
- interact with the W&B Server in any way.

## View active users over time
Use the plots in the **Activity** tab to get an aggregate view of how many users have been active over time

1. Click the **Activity** tab.
1. The **Total active users** plot shows how many users have been active in a period of time (defaults to 3 months).
1. The **Users active over time** plot shows the fluctuation of active users over a period of time (defaults to 6 months). Hover your mouse over a pointo to see the number of users on that date.

To change the period of time for a plot, use the drop-down. You can select:
- Last 30 days
- Last 3 months
- Last 6 months
- Last 12 months
- All time
