---
description: Manage a team's members, avatar, alerts, and privacy settings with the
  Team Settings page.
menu:
  default:
    identifier: team-settings
    parent: settings
title: Manage team settings
weight: 30
---

# Team settings

Change your team's settings, including members, avatar, alerts, privacy, and usage. Organization admins and team admins can view and edit a team's settings.

{{% alert %}}
Only Administration account types can change team settings or remove a member from a team.
{{% /alert %}}


## Members
The Members section shows a list of all pending invitations and the members that have either accepted the invitation to join the team. Each member listed displays a member’s name, username, email, team role, as well as their access privileges to Models and Weave, which is inherited by from the Organization. You can choose from the standard team roles **Admin**, **Member**, and **View-only**. If your organization has created [custom roles]({{< relref "/guides/hosting/iam/access-management/manage-organization.md#create-custom-roles" >}}), you can assign a custom role instead.

See [Add and Manage teams]({{< relref "/guides/hosting/iam/access-management/manage-organization.md#add-and-manage-teams" >}}) for information on how to create a team, manage teams, and manage team membership and roles. To configure who can invite new members and configure other privacy settings for the team, refer to [Privacy]({{< relref "#privacy" >}}).

## Avatar

Set an avatar by navigating to the **Avatar** section and uploading an image.

1. Select the **Update Avatar** to prompt a file dialog to appear.
2. From the file dialog, choose the image you want to use.

## Alerts

Notify your team when runs crash, finish, or set custom alerts. Your team can receive alerts either through email or Slack.

Toggle the switch next to the event type you want to receive alerts from. Weights and Biases provides the following event type options be default:

* **Runs finished**: whether a Weights and Biases run successfully finished.
* **Run crashed**: if a run has failed to finish.

For more information about how to set up and manage alerts, see [Send alerts with wandb.alert]({{< relref "/guides/models/track/runs/alert.md" >}}).

## Slack notifications
Configure Slack destinations where your team's [automations]({{< relref "/guides/core/automations/" >}}) can send notifications when an event occurs in a Registry or a project, such as when a new artifact is created or when a run metric meets a defined threshold. Refer to [Create a Slack automation]({{< relref "/guides/core/automations/create-automations/slack.md" >}}).

{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-only.md" >}}
{{% /pageinfo %}}

## Webhooks
Configure webhooks that your team's [automations]({{< relref "/guides/core/automations/" >}}) can run when an event occurs in a Registry or a project, such as when a new artifact is created or when a run metric meets a defined threshold. Refer to [Create a webhook automation]({{< relref "/guides/core/automations/create-automations/slack.md" >}}).

{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-only.md" >}}
{{% /pageinfo %}}

## Privacy

Navigate to the **Privacy** section to change privacy settings. Only organization admins can modify privacy setting.

- Turn off the ability to make future projects public or to share reports publicly.
- Allow any team member to invite other members, rather than only team admins.
- Manage whether code saving is turned on by default.

## Usage

The **Usage** section describes the total memory usage the team has consumed on the Weights and Biases servers. The default storage plan is 100GB. For more information about storage and pricing, see the [Pricing](https://wandb.ai/site/pricing) page.

## Storage

The **Storage** section describes the cloud storage bucket configuration that is being used for the team's data. For more information, see [Secure Storage Connector]({{< relref "teams.md#secure-storage-connector" >}}) or check out our [W&B Server]({{< relref "/guides/hosting/data-security/secure-storage-connector.md" >}}) docs if you are self-hosting.