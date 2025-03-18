---
menu:
  default:
    identifier: create-slack-automations
    parent: create-automations
title: Create a Slack automation
weight: 1
---

This page shows how to create a Slack [automation]({{< relref "/guides/core/automations/" >}}> ). To create a webhook automation, refer to [Create a webhook automation]({{< relref "/guides/core/automations/create-automations/webhook.md" >}}) instead.

At a high level, to create a Slack automation, you take these steps:
1. [Add a Slack integration]({{< relref "#add-a-slack-channel" >}}), which authorizes W&B to post to the Slack instance and channel.
1. [Create the Slack automation]({{< relref "#create-slack-automation" >}}), which defines the [event]({{< relref "/guides/core/automations/automation-events.md" >}}) to watch for and the channel to post to.

## Add a Slack integration
1. Log in to W&B and go to Team Settings page.
1. To integrate with a new Slack workspace and channel, click **Connect Slack**.

    To integrate with a new Slack channel in a workspace that is already set up, click **New integration**.

    If necessary, sign in to your Slack instance. A browser window appears, asking that you grant Weights and Biases permission to post to the Slack channel you select. Read the page, then click **Search for a channel** and begin typing the channel name. Select the channel from the list, then click **Allow**.

1. In Slack, go to the channel you selected. If you see a post like `[Your Slack handle] added an integration to this channel: Weights & Biases`, the integration is configured correctly.

Now you can [create a Slack automation]({{< relref "#create-a-slack-automation" >}}).

## View and manage Slack integration
1. Log in to W&B and go to **Team Settings**.
1. The **Slack channel integrations** section lists each Slack destination.
1. Delete a destination by clicking its trash icon.

## Create a Slack automation
After you [configure a Slack integration]({{< relref "#add-a-slack-integration" >}}), follow these steps to create a Slack automation that uses it.

1. Log in to W&B and go to the project page.
1. In the sidebar, click **Automations**.
1. Click **Create automation**.
1. Choose the [**Event**]({{< relref "/guides/core/automations/automation-events.md" >}}) to watch for. Fill in any additional fields that appear, which depend upon the event. Click **Next step**.
1. Select the team that owns the [Slack integration]({{< relref "#add-a-slack-integration" >}}).
1. Set **Action type** to **Slack notification**.
1. Select the Slack channel, then click **Next step**.
1. Provide a name for the automation. Optionally, provide a description.
1. Click **Create automation**.

## View and manage automations
View and manage a project's automations from the project's **Automations** tab.

- To view an automation's details, click its name.
- To edit an automation, click its action `...` menu, then click **Edit automation**.
- To delete an automation, click its action `...` menu, then click **Delete automation**. Confiruation is required.