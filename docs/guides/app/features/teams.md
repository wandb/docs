---
description: >-
  Collaborate with your colleagues, share results, and track all the experiments
  across your team
displayed_sidebar: default
---

# Teams

Use W&B Teams as a central workspace for your ML team to build better models faster.

* **Track all the experiments** your team has tried so you never duplicate work.
* **Save and reproduce** previously trained models.
* **Share progress** and results with your boss and collaborators.
* **Catch regressions** and immediately get alerted when performance drops.
* **Benchmark model performance** and compare model versions.

![](/images/app_ui/teams_overview.webp)

## Create a collaborative team

1. [**Sign up or log in**](https://app.wandb.ai/login?signup=true) to your free W&B account.
2. Click **Invite Team** in the navigation bar.
3. Create your team and invite collaborators.

:::info
**Note**: Only the admin of an organization can create a new team.
:::

## Create a Team Profile

You can customize your team's profile page to show an introduction and showcase reports and projects that are visible to the public or team members. Present reports, projects, and external links.

* **Highlight your best research** to visitors by showcasing your best public reports
* **Showcase the most active projects** to make it easier for teammates to find them
* **Find collaborators** by adding external links to your company or research lab's website and any papers you've published

<!-- To do: show team profiles -->

<!-- To do: show how to remove team members -->

## Remove Team members

Team admins can open the team settings page and click the delete button next to the departing member's name. Any runs that they logged to the team will remain after a user is removed.


## Team Roles and Permissions
Select a team role when you invite colleagues to join a team. There are four team role options:

- **Admin**: Team admins can add and remove other admins or team members. They have permissions to modify all projects and full deletion permissions. This includes, but is not limited to, deleting runs, projects, artifacts, and sweeps.
- **Member**: A regular member of the team. A team member is invited by email by the team admin. A team member cannot invite other members. Team members can only delete runs and sweep runs created by that member. Suppose you have two members A and B. Member B moves a Run from team B's project to a different project owned by Member A. Member A can not delete the Run Member B moved to Member A's project. Only the member that creates the Run, or the team admin, can delete the run.
- **Service (Enterprise-only feature)**: A service worker, an API key useful for using W&B with your run automation tools. If you use the API key from a service account for your team, make sure to set the environment variable **WANDB_USERNAME** to attribute runs to the correct user.
- **View-Only (Enterprise-only feature)**: View-Only members can view assets within the team such as runs, reports, and workspaces. They can follow and comment on reports, but they can not create, edit, or delete project overview, reports, or runs. View-Only members do not have an API key.

### Team Settings
Team settings allow you to manage the settings for your team and its members. With these privileges, you can effectively oversee and organize your team within W&B.

| Permissions         | View-Only | Team Member | Team Admin | 
| ------------------- | --------- | ----------- | ---------- |
| Add team members    |           |             |     X      |
| Remove team members |           |             |     X      |
| Manage team settings|           |             |     X      |

### Model Registry
The proceeding table lists permissions that apply to all projects across a given team.

| Permissions                | View-Only | Team Member | Model Registry Admin | Team Admin | 
| ---------------------------| --------- | ----------- | -------------- | ---------- |
| Add aliases                |           | X           | X              | X |
| Add models to the registry |           | X           | X              | X |
| View models in the registry| X         | X           | X              | X |
|Download models             |           | X           | X              | X |
|Add/Remove Registry Admins  |           |             | X              | X | 
|Add/Remove Protected Aliases|           |             | X              |   | 

See the [Model Registry](../../model_registry/access_controls.md) chapter for more information about protected aliases.

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

### System Settings (W&B Server Only)
With system permissions, you have the ability to manage members, create and modify teams, adjust system settings, and view user activity. These privileges enable you to effectively administer and maintain the W&B instance.

| Permissions              | View-Only | Team Member | Team Admin | System Admin | 
| ------------------------ | --------- | ----------- | ---------- | ------------ |
| Configure system settings|           |             |            | X            |
| Create/delete teams      |           |             |            | X            |
| View activity dashboard  |           |             |            | X            |

### Team Service Account Behavior

* When you configure a team in your training environment, you can use a service account from that team to log runs in either of private or public projects within that team. Additionally, you can attribute those runs to a user if **WANDB_USERNAME** or **WANDB_USER_EMAIL** variable is configured in your environment and the referenced user is part of that team.
* When you **do not** configure a team in your training environment and use a service account, the runs would be logged to the named project within that service account's parent team. In this case as well, you can attribute the runs to a user if **WANDB_USERNAME** or **WANDB_USER_EMAIL** variable is configured in your environment and the referenced user is part of the service account's parent team.
* A service account can not be used to log runs to a private project in a team different from its parent team, but it can be used to log runs to public projects in other teams.

#### Add Social Badges to your Intro

In your Intro, type `/` and choose Markdown and paste the markdown snippet that will render your badge. Once you convert it to WYSIWYG, you can resize it.

 [![Twitter: @weights_biases](https://img.shields.io/twitter/follow/weights\_biases?style=social)](https://twitter.com/intent/follow?screen\_name=weights\_biases)

For example, to add a Twitter follow badge, add `[![Twitter: @weights_biase](https://img.shields.io/twitter/follow/weights_biases?style=social)](https://twitter.com/intent/follow?screen_name=weights_biases` replacing `weights_biases` with your Twitter username.

## Team Trials

W&B offers free trials for business teams, **no credit card required**. During the trial, you and your colleagues will have access to all the features in W&B. Once the trial is over, you can upgrade your plan to continue using a W&B Team to collaborate. Your personal W&B account will always remain free, and if you're a student or teacher you can enroll in an academic plan.

See the [pricing page](https://wandb.ai/site/pricing) for more information on our plans. You can download all your data at any time, either using the dashboard UI or via our [Export API](../../../ref/python/public-api/README.md).

## Privacy Settings

You can see the privacy settings of all team projects on the team settings page:
`app.wandb.ai/teams/your-team-name`

## Advanced Configuration

### Secure Storage Connector

:::caution
W&B does not currently support migrating buckets. More specifically W&B does not support:
* Migrating from one team-level bucket to another team-level bucket
* Migrating from a W&B SaaS bucket to a team-level bucket and vice versa.
:::

The team-level secure storage connector allows teams to use their own cloud storage bucket with W&B. This provides greater data access control and data isolation for teams with highly sensitive data or strict compliance requirements.

:::info
This feature is only available for Google Cloud Storage buckets and Amazon S3 buckets. Only enterprise teams can use this feature. To learn more about enterprise plans, please contact us at  support@wandb.com
:::

To provision a cloud storage bucket, we recommend using our secure storage connector terraform module for [AWS](https://github.com/wandb/terraform-aws-wandb/tree/main/modules/secure_storage_connector) or [GCP](https://github.com/wandb/terraform-google-wandb/tree/main/modules/secure_storage_connector).

A cloud storage bucket can be configured only once for a team at the time of team creation. Select **External Storage** when you create a team to configure a cloud storage bucket. Select your provider and fill out your bucket name and storage encryption key ID, if applicable, and select **Create Team**.

An error or warning will appear at the bottom of the page if there are issues accessing the bucket or the bucket has invalid settings.

![](/images/hosting/saas_setup_secure_storage.png)

Only organization administrators have the permissions to configure the secure storage connector. The same cloud storage bucket can be used amongst multiple teams by selecting an existing cloud storage bucket from the dropdown.
