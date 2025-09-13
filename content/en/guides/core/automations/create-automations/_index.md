---
menu:
  default:
    identifier: create-automations
    parent: automations
title: Create an automation
weight: 1
url: guides/automations/create-automations
cascade:
- url: guides/automations/create-automations/:filename
---
{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-cloud-only.md" >}}
{{% /pageinfo %}}

This page gives an overview of creating and managing W&B [automations]({{< relref "/guides/core/automations/">}}). For more detailed instructions, refer to [Create a Slack automation]({{< relref "/guides/core/automations/create-automations/slack.md" >}}) or [Create a webhook automation]({{< relref "/guides/core/automations/create-automations/webhook.md" >}}).

{{% alert %}}
Looking for companion tutorials for automations?
- [Learn to automatically triggers a Github Action for model evaluation and deployment](https://wandb.ai/wandb/wandb-model-cicd/reports/Model-CI-CD-with-W-B--Vmlldzo0OTcwNDQw).
- [Watch a video demonstrating automatically deploying a model to a Sagemaker endpoint](https://www.youtube.com/watch?v=s5CMj_w3DaQ).
- [Watch a video series introducing automations](https://youtube.com/playlist?list=PLD80i8An1OEGECFPgY-HPCNjXgGu-qGO6&feature=shared).
{{% /alert %}}

## Requirements
- A team admin can create and manage automations for the team's projects, as well as components of their automations, such as webhooks, secrets, and Slack integrations. Refer to [Team settings]({{< relref "/guides/models/app/settings-page/team-settings/" >}}).
- To create a registry automation, you must have access to the registry. Refer to [Configure Registry access]({{< relref "/guides/core/registry/configure_registry.md#registry-roles" >}}).
- To create a Slack automation, you must have permission to post to the Slack instance and channel you select.

## Create an automation
Create an automation from the project or registry's **Automations** tab. At a high level, to create an automation, follow these steps:

1. If necessary, [create a W&B secret]({{< relref "/guides/core/secrets.md" >}}) for each sensitive string required by the automation, such as an access token, password, or SSH key. Secrets are defined in your **Team Settings**. Secrets are most commonly used in webhook automations.
1. Configure the webhook or Slack integration to authorize W&B to post to Slack or run the webhook on your behalf. A single webhook or Slack integration can be used by multiple automations. These actions are defined in your **Team Settings**.
1. In the project or registry, create the automation, which specifies the event to watch for and the action to take (such as posting to Slack or running a webhook). When you create a webhook automation, you configure the payload it sends.

Or, from a line plot in the workspace, you can quickly create a [run metric automation]({{< relref "/guides/core/automations/automation-events.md#run-events" >}}) for the metric it shows:

1. Hover over the panel, then click the bell icon at the top of the panel.

    {{< img src="/images/automations/run_metric_automation_from_panel.png" alt="Automation bell icon location" >}}
1. Configure the automation using the basic or advanced configuration controls. For example, apply a run filter to limit the scope of the automation, or configure an absolute threshold.

For details, refer to:

- [Create a Slack automation]({{< relref "slack.md" >}})
- [Create a webhook automation]({{< relref "webhook.md" >}})

## View and manage automations
View and manage automations from a project or registry's **Automations** tab.

- To view an automation's details, click its name.
- To view an automation's execution history, click its name and select the **History** tab. See [View an automation's history]({{< relref "/guides/core/automations/view-automation-history.md" >}}) for details.
- To edit an automation, click its action `...` menu, then click **Edit automation**.
- To delete an automation, click its action `...` menu, then click **Delete automation**.

## Next steps
- Learn more about [automation events and scopes]({{< relref "/guides/core/automations/automation-events.md" >}})
- [Create a Slack automation]({{< relref "slack.md" >}}).
- [Create a webhook automation]({{< relref "webhook.md" >}}).
- [Create a secret]({{< relref "/guides/core/secrets.md" >}}).
