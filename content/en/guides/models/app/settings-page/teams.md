---
description: Collaborate with your colleagues, share results, and track all the experiments across your team. Manage team settings including members, alerts, and privacy.
menu:
  default:
    identifier: teams
    parent: settings
title: Team settings
weight: 30
---

Use W&B Teams as a central workspace for your ML team to build better models faster. Organization admins and team admins can view and edit a team's settings.

* **Track all the experiments** your team has tried so you never duplicate work.
* **Save and reproduce** previously trained models.
* **Share progress** and results with your boss and collaborators.
* **Catch regressions** and immediately get alerted when performance drops.
* **Benchmark model performance** and compare model versions.

{{< img src="/images/app_ui/teams_overview.webp" alt="Teams workspace overview" >}}

{{% alert %}}
Only Administration account types can change team settings or remove a member from a team.
{{% /alert %}}

## Create a collaborative team

1. [Sign up or log in](https://app.wandb.ai/login?signup=true) to your free W&B account.
2. Click **Invite Team** in the navigation bar.
3. Create your team and invite collaborators.
4. Configure your team using the settings described below.

{{% alert %}}
**Note**: Only the admin of an organization can create a new team.
{{% /alert %}}

## Team configuration
You can configure various settings for your team, including its membership, alerts, and privacy settings.

To manage your team's settings, navigate to the **Teams** section in the left-menu and click the team you want to change settings for. You can update the following settings:
### Members

The Members section shows a list of all pending invitations and the members that have either accepted the invitation to join the team. Each member listed displays a member's name, username, email, team role, as well as their access privileges to Models and W&B Weave, which is inherited by from the Organization. You can choose from the standard team roles **Admin**, **Member**, and **View-only**. If your organization has created [custom roles]({{< relref "/guides/hosting/iam/access-management/manage-organization.md#create-custom-roles" >}}), you can assign a custom role instead.

See [Add and Manage teams]({{< relref "/guides/hosting/iam/access-management/manage-organization.md#add-and-manage-teams" >}}) for information on how to create a team, manage teams, and manage team membership and roles. To configure who can invite new members and configure other privacy settings for the team, refer to [Privacy]({{< relref "#privacy" >}}).

Team admins can open the team settings page and click the delete button next to the departing member's name. Any runs logged to the team remain after a user leaves.

### Avatar

Set an avatar by navigating to the **Avatar** section and uploading an image.

1. Select the **Update Avatar** to prompt a file dialog to appear.
2. Hover over the team's default avatar image in the upper-left corner of the page and click the **Upload photo** button. This opens a file prompt.
3. From the file prompt, select the image you want to use and then click **Open**.  This uploads the photo to your team and sets it as your team's avatar.

### Alerts

You can set up alerts to notify your team when runs crash and finish. Alerts can be sent through either email or Slack, and you can customize them to meet your needs.

In the **Team alerts** section, toggle the switch next to the event type you want to receive alerts from. Weights and Biases provides the following event type options:

* **Runs finished**: whether a Weights and Biases run successfully finished.
* **Run crashed**: if a run has failed to finish.
* **wandb.alert()**: A custom scriptable alert. See [Send alerts with `wandb.Run.alert()`]({{< relref "/guides/models/track/runs/alert.md" >}}) for more information.


### Slack notifications

Configure Slack destinations where your team's [automations]({{< relref "/guides/core/automations/" >}}) can send notifications when an event occurs in a registry or a project, such as when a new artifact is created or when a run metric meets a defined threshold. Refer to [Create a Slack automation]({{< relref "/guides/core/automations/create-automations/slack.md" >}}).

{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-only.md" >}}
{{% /pageinfo %}}

### Webhooks

Configure webhooks that your team's [automations]({{< relref "/guides/core/automations/" >}}) can run when an event occurs in a registry or a project, such as when a new artifact is created or when a run metric meets a defined threshold. Refer to [Create a webhook automation]({{< relref "/guides/core/automations/create-automations/webhook.md" >}}).

{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-only.md" >}}
{{% /pageinfo %}}

### Privacy

Team privacy settings allow you to manage visibility and sharing abilities on your team. Only organization admins can modify privacy settings.

You can change the following privacy settings:

- Hide this team from all non-members.
- Make all future team projects private (public sharing not allowed).
- Allow any team member to invite other team members (not just admins).
- Disable public sharing to outside of team for reports in private projects. This disables existing magic links.
- Automatically recommend new users with matching email domains join this team upon signup.
- Enable code saving by default.

You can see the privacy settings of all team projects on the team settings page: `app.wandb.ai/teams/your-team-name`

### Usage

The **Usage** section describes the total memory usage the team has consumed on the Weights and Biases servers. The default storage plan is 100GB. For more information about storage and pricing, see the [Pricing](https://wandb.ai/site/pricing) page.

### Storage

The **Storage** section describes the cloud storage bucket configuration that is being used for the team's data. For more information, see [Secure Storage Connector]({{< relref "#secure-storage-connector" >}}) or check out our [W&B Server]({{< relref "/guides/hosting/data-security/secure-storage-connector.md" >}}) docs if you are self-hosting.

## Create a team profile

You can customize your team's profile page to show an introduction and showcase reports and projects that are visible to the public or team members. Present reports, projects, and external links.

* **Highlight your best research** to visitors by showcasing your best public reports
* **Showcase the most active projects** to make it easier for teammates to find them
* **Find collaborators** by adding external links to your company or research lab's website and any papers you've published

<!-- To do: show team profiles -->

## Team roles and permissions
Select a team role when you invite colleagues to join a team. There are following team role options:

- **Admin**: Team admins can add and remove other admins or team members. They have permissions to modify all projects and full deletion permissions. This includes, but is not limited to, deleting runs, projects, artifacts, and sweeps.
- **Member**: A regular member of the team. By default, only an admin can invite a team member. To change this behavior, refer to [Privacy settings]({{< relref "#privacy" >}}).
- **View-Only (Enterprise-only feature)**: View-Only members can view assets within the team such as runs, reports, and workspaces. They can follow and comment on reports, but they can not create, edit, or delete project overview, reports, or runs.
- **Custom roles (Enterprise-only feature)**: Custom roles allow organization admins to compose new roles based on either of the **View-Only** or **Member** roles, together with additional permissions to achieve fine-grained access control. Team admins can then assign any of those custom roles to users in their respective teams. Refer to [Introducing Custom Roles for W&B Teams](https://wandb.ai/wandb_fc/announcements/reports/Introducing-Custom-Roles-for-W-B-Teams--Vmlldzo2MTMxMjQ3) for details.

A team member can delete only runs they created. Suppose you have two members A and B. Member B moves a run from team B's project to a different project owned by Member A. Member A cannot delete the run Member B moved to Member A's project. An admin can manage runs and sweep runs created by any team member.

### Service accounts

In addition to user roles, teams can also use **service accounts** for automation. Service accounts are not users, but rather non-human identities used for automated workflows. Refer to [Use service accounts to automate workflows]({{< relref "/guides/hosting/iam/authentication/service-accounts.md" >}}) for detailed information.

{{% alert %}}
W&B recommends assigning more than one admin in a team to ensure that admin operations can continue when the primary admin is not available.
{{% /alert %}}

### Team settings
Team settings allow you to manage the settings for your team and its members. With these privileges, you can effectively oversee and organize your team within W&B.

| Permissions         | View-Only | Team Member | Team Admin | 
| ------------------- | --------- | ----------- | ---------- |
| Add team members    |           |             |     X      |
| Remove team members |           |             |     X      |
| Manage team settings|           |             |     X      |

### Reports
Report permissions grant access to create, view, and edit reports. The proceeding table lists permissions that apply to all reports across a given team.

| Permissions   | View-Only | Team Member                                     | Team Admin | 
| -----------   | --------- | ----------------------------------------------- | ---------- |
|View reports   | X         | X                                               | X          |
|Create reports |           | X                                               | X          |
|Edit reports   |           | X (team members can only edit their own reports)| X          |
|Delete reports |           | X (team members can only edit their own reports)| X          |

### Experiments
The proceeding table lists permissions that apply to all experiments across a given team.

| Permissions | View-Only | Team Member | Team Admin | 
| ------------------------------------------------------------------------------------ | --------- | ----------- | ---------- |
| View experiment metadata (includes history metrics, system metrics, files, and logs) | X         | X           | X          |
| Edit experiment panels and workspaces                                                |           | X           | X          |
| Log experiments                                                                      |           | X           | X          |
| Delete experiments                                                                   |           | X (team members can only delete experiments they created) |  X  |
|Stop experiments                                                                      |           | X (team members can only stop experiments they created)   |  X  |

### Artifacts
The proceeding table lists permissions that apply to all artifacts across a given team.

| Permissions      | View-Only | Team Member | Team Admin | 
| ---------------- | --------- | ----------- | ---------- |
| View artifacts   | X         | X           | X          |
| Create artifacts |           | X           | X          |
| Delete artifacts |           | X           | X          |
| Edit metadata    |           | X           | X          |
| Edit aliases     |           | X           | X          |
| Delete aliases   |           | X           | X          |
| Download artifact|           | X           | X          |

### System settings (W&B Server only)
Use system permissions to create and manage teams and their members and to adjust system settings. These privileges enable you to effectively administer and maintain the W&B instance.

| Permissions              | View-Only | Team Member | Team Admin | System Admin | 
| ------------------------ | --------- | ----------- | ---------- | ------------ |
| Configure system settings|           |             |            | X            |
| Create/delete teams      |           |             |            | X            |

### Team service account behavior

* When you configure a team in your training environment, you can use a service account from that team to log runs in either of private or public projects within that team. Additionally, you can attribute those runs to a user if **WANDB_USERNAME** or **WANDB_USER_EMAIL** variable exists in your environment and the referenced user is part of that team.
* When you **do not** configure a team in your training environment and use a service account, the runs log to the named project within that service account's parent team. In this case as well, you can attribute the runs to a user if **WANDB_USERNAME** or **WANDB_USER_EMAIL** variable exists in your environment and the referenced user is part of the service account's parent team.
* A service account can not log runs to a private project in a team different from its parent team. A service account can log to runs to project only if the project is set to `Open` project visibility.

## Team trials

See the [pricing page](https://wandb.ai/site/pricing) for more information on W&B plans. You can download all your data at any time, either using the dashboard UI or the [Export API]({{< relref "/ref/python/public-api/index.md" >}}).

## Advanced configuration

### Secure storage connector

The team-level secure storage connector allows teams to use their own cloud storage bucket with W&B. This provides greater data access control and data isolation for teams with highly sensitive data or strict compliance requirements. Refer to [Secure Storage Connector]({{< relref "/guides/hosting/data-security/secure-storage-connector.md" >}}) for more information.
