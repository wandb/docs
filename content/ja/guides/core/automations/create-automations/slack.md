---
menu:
  default:
    identifier: ja-guides-core-automations-create-automations-slack
    parent: create-automations
title: Create a Slack automation
weight: 1
---

{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-cloud-only.md" >}}
{{% /pageinfo %}}

This page shows how to create a Slack [automation]({{< relref path="/guides/core/automations/" lang="ja" >}}> ). To create a webhook automation, refer to [Create a webhook automation]({{< relref path="/guides/core/automations/create-automations/webhook.md" lang="ja" >}}) instead.

At a high level, to create a Slack automation, you take these steps:
1. [Add a Slack integration]({{< relref path="#add-a-slack-integration" lang="ja" >}}), which authorizes W&B to post to the Slack instance and channel.
1. [Create the automation]({{< relref path="#create-an-automation" lang="ja" >}}), which defines the [event]({{< relref path="/guides/core/automations/automation-events.md" lang="ja" >}}) to watch for and the channel to notify.

## Add a Slack integration
A team admin can add a Slack integration to the team.

1. Log in to W&B and go to **Team Settings**.
1. In the **Slack channel integrations** section, click **Connect Slack** to add a new Slack instance. To add a channel for an existing Slack instance, click **New integration**.

    ![Screenshot showing two Slack integrations in a Team](/images/automations/slack_integrations.png)
1. If necessary, sign in to Slack in your browser. When prompted, grant W&B permission to post to the Slack channel you select. Read the page, then click **Search for a channel** and begin typing the channel name. Select the channel from the list, then click **Allow**.
1. In Slack, go to the channel you selected. If you see a post like `[Your Slack handle] added an integration to this channel: Weights & Biases`, the integration is configured correctly.

Now you can [create an automation]({{< relref path="#create-an-automation" lang="ja" >}}) that notifies the Slack channel you configured.

## View and manage Slack integrations
A team admin can view and manage the team's Slack instances and channels.

1. Log in to W&B and go to **Team Settings**.
1. View each Slack destination in the **Slack channel integrations** section.
1. Delete a destination by clicking its trash icon.

## Create an automation
After you [add a Slack integration]({{< relref path="#add-a-slack-integreation" lang="ja" >}}), select **Registry** or **Project**, then follow these steps to create an automation that notifies the Slack channel.

{{< tabpane text=true >}}
{{% tab "Registry" %}}
A Registry admin can create automations in that registry.

1. Log in to W&B.
1. Click the name of a registry to view its details, 
1. To create an automation scoped to the registry, click the **Automations** tab, then click **Create automation**. An automation that is scoped to a registry is automatically applied to all of its collections (including those created in the future).

    To create an automation scoped only to a specific collection in the registry, click the collection's action `...` menu, then click **Create automation**. Alternatively, while viewing a collection, create an automation for it using the **Create automation** button in the **Automations** section of the collection's details page.
1. Choose the [event]({{< relref path="/guides/core/automations/automation-events.md" lang="ja" >}}) to watch for.

    Fill in any additional fields that appear, which depend upon the event. For example, if you select **An artifact alias is added**, you must specify the **Alias regex**.

    Click **Next step**.
1. Select the team that owns the [Slack integration]({{< relref path="#add-a-slack-integration" lang="ja" >}}).
1. Set **Action type** to **Slack notification**. Select the Slack channel, then click **Next step**.
1. Provide a name for the automation. Optionally, provide a description.
1. Click **Create automation**.

{{% /tab %}}
{{% tab "Project" %}}
A W&B admin can create automations in a project.

1. Log in to W&B.
1. Go the project page and click the **Automations** tab, then click **Create automation**.

    Or, from a line plot in the workspace, you can quickly create a [run metric automation]({{< relref path="/guides/core/automations/automation-events.md#run-events" lang="ja" >}}) for the metric it shows. Hover over the panel, then click the bell icon at the top of the panel.
    {{< img src="/images/automations/run_metric_automation_from_panel.png" alt="Automation bell icon location" >}}
1. Choose the [event]({{< relref path="/guides/core/automations/automation-events.md" lang="ja" >}}) to watch for.

    Fill in any additional fields that appear, which depend upon the event. For example, if you select **An artifact alias is added**, you must specify the **Alias regex**.

    Click **Next step**.
1. Select the team that owns the [Slack integration]({{< relref path="#add-a-slack-integration" lang="ja" >}}).
1. Set **Action type** to **Slack notification**. Select the Slack channel, then click **Next step**.
1. Provide a name for the automation. Optionally, provide a description.
1. Click **Create automation**.

{{% /tab %}}
{{< /tabpane >}}

## View and manage automations

{{< tabpane text=true >}}
{{% tab "Registry" %}}

- Manage the registry's automations from the registry's **Automations** tab.
- Mamage a collection's automations from the **Automations** section of the collection's details page.

From either of these pages, a Registry admin can manage existing automations:
- To view an automation's details, click its name.
- To edit an automation, click its action `...` menu, then click **Edit automation**.
- To delete an automation, click its action `...` menu, then click **Delete automation**. Confirmation is required.


{{% /tab %}}
{{% tab "Project" %}}
A W&B admin can view and manage a project's automations from the project's **Automations** tab.

- To view an automation's details, click its name.
- To edit an automation, click its action `...` menu, then click **Edit automation**.
- To delete an automation, click its action `...` menu, then click **Delete automation**. Confirmation is required.
{{% /tab %}}
{{< /tabpane >}}