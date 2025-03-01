---
menu:
  default:
    identifier: org_dashboard
    parent: monitoring-and-usage
title: View organization dashboard
---

{{% alert color="secondary" %}}
Organization dashboard is only available with [Dedicated Cloud]({{< relref "/guides/hosting/hosting-options/dedicated_cloud.md" >}}) and [Self-managed instances]({{< relref "/guides/hosting/hosting-options/self-managed.md" >}}).
{{% /alert %}}


## View organization usage of W&B
Use the organization dashboard to get a holistic view of users that belong to your organization, how users of your organization use W&B, along with properties such as:

* **Name**: The name of the user and their W&B username.
* **Last active**: The time the user last used W&B. This includes any activity that requires authentication, including viewing pages in the product, logging runs or taking any other action, or logging in.
* **Role**: The role of the user. 
* **Email**: The email of the user.
* **Team**: The names of teams the user belongs to.

### View the status of a user
The **Last Active** column shows if a user is pending an invitation or an active user. A user is one of three states:

* **Invite pending**: Admin has sent invite but user has not accepted invitation. 
* **Active**: User has accepted the invite and created an account.
* **Deactivated**: Admin has revoked access of the user.

{{< img src="/images/hosting/view_status_of_user.png" alt="" >}}

### View and share how your organization uses W&B
View how your organization uses W&B in CSV format.

1. Select the three dots next to the **Add user** button.
2. From the dropdown, select **Export as CSV**.

    {{< img src="/images/hosting/export_org_usage.png" alt="" >}}

This exports a CSV file that lists all users of an organization along with details about the user, such as their user name, time stamp of when they were last active, roles, email, and more.

### View user activity
Use the **Last Active** column to get an **Activity summary** of an individual user. 

1. Hover your mouse over the **Last Active** entry for a user. 
2. A tooltip appears and provides a summary of information about the user's activity.


{{< img src="/images/hosting/activity_tooltip.png" alt="" >}}

A user is _active_ if they:
- log in to W&B.
- view any page in the W&B App.
- log runs.
- use the SDK to track an experiment.
- interact with the W&B Server in any way.

### View active users over time
Use the **Users active over time**  plot in the Organization dashboard to get an aggregate overview of how many users are active over time (right most plot in image below). 

{{< img src="/images/hosting/dashboard_summary.png" alt="" >}}

You can use the dropdown menu to filter results based on days, months, or all time.