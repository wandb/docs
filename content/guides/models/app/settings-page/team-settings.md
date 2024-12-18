---
description: Manage a team's members, avatar, alerts, and privacy settings with the
  Team Settings page.
menu:
  default:
    identifier: team-settings
    parent: settings
title: Manage team settings
---

# Team settings

Change your team's settings, including members, service accounts, avatar, alerts, privacy, and usage. Only team administrators can view and edit a team's settings.

{{% alert %}}
Only Administration account types can change team settings or remove a member from a team.
{{% /alert %}}

## Members

The Members section shows a list of all pending invitations and the members that have either accepted the invitation to join the team. Each member listed displays a member’s name, username, email, team role, as well as their access privileges to Models and Weave, which is inherited by from the Organization. There are three standard team roles: Administrator (Admin), Member, and View-only.

See [Add and Manage teams](../../hosting/iam/manage-organization.md#add-and-manage-teams) for information on how to create a tea, invite users to a team, remove users from a team, and change a user's role.

## Service accounts

Refer to [Use service accounts to automate workflows](../../hosting/iam/service-accounts.md).

## Avatar

Set an avatar by navigating to the **Avatar** section and uploading an image.

1. Select the **Update Avatar** to prompt a file dialog to appear.
2. From the file dialog, choose the image you want to use.

## Alerts

Notify your team when runs crash, finish, or set custom alerts. Your team can receive alerts either through email or Slack.

Toggle the switch next to the event type you want to receive alerts from. Weights and Biases provides the following event type options be default:

* **Runs finished**: whether a Weights and Biases run successfully finished.
* **Run crashed**: if a run has failed to finish.

For more information about how to set up and manage alerts, see [Send alerts with wandb.alert](../../runs/alert.md).

## Privacy

Navigate to the **Privacy** section to change privacy settings. Refer to [Configure privacy settings](../../hosting/privacy-settings.md).

## Usage

The **Usage** section describes the total memory usage the team has consumed on the Weights and Biases servers. The default storage plan is 100GB. For more information about storage and pricing, see the [Pricing](https://wandb.ai/site/pricing) page.

## Storage

The **Storage** section describes the cloud storage bucket configuration that is being used for the team's data. For more information, see [Secure Storage Connector](../features/teams.md#secure-storage-connector) or check out our [W&B Server](../../hosting/data-security/secure-storage-connector.md) docs if you are self-hosting.