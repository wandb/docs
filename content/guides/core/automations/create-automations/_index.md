---
menu:
  default:
    identifier: create-automations
    parent: automations
title: Create an automation
weight: 1
---

This page gives an overview of creating and managing W&B [automations]({{< relref "/guides/core/automations/">}}). For more detailed instructions, refer to [Create a Slack automation]({{< relref "/guides/core/automations/create-automations/slack.md" >}}) or [Create a webhook automation]({{< relref "/guides/core/automations/create-automations/webhook.md" >}}).

{{% alert %}}
Looking for companion tutorials for automations? 
- [Learn to automatically triggers a Github Action for model evaluation and deployment](https://wandb.ai/wandb/wandb-model-cicd/reports/Model-CI-CD-with-W-B--Vmlldzo0OTcwNDQw).
- [Watch a video demonstrating automatically deploying a model to a Sagemaker endpoint](https://www.youtube.com/watch?v=s5CMj_w3DaQ).
- [Watch a video series introducing automations](https://youtube.com/playlist?list=PLD80i8An1OEGECFPgY-HPCNjXgGu-qGO6&feature=shared).
{{% /alert %}}

## Requirements
- A team admin can create and manage automations for the team's projects, as well as components of their automations, such as webhooks, secrets, or Slack instance integrations. Refer to [Team settings]({{< relref "/guides/models/app/settings-page/team-settings/" >}}).
- To create a registry automation, you must have access to the registry. Refer to [Configure Registry access]({{< relref "/guides/core/registry/configure_registry.md#registry-roles" >}}).
- To create a Slack automation, you must have permission to post to the Slack instance and channel you select.

## Create an automation
Create an automation from the project's **Automations** tab. At a high level, to create an automation, follow these steps:

1. If necessary, [create a W&B secret]({{< relref "/guides/core/secrets.md" >}}) for each sensitive string required by the automation, such as an access token, password, or SSH key. Secrets are defined in your team settings. Secrets are most commonly used in webhook automation.
1. Configure the webhook or Slack integration to authorize W&B to post to Slack or run the webhook on your behalf. A single integration can be used by multiple automations. When you create the integration, you grant it access to any secrets it requires for authentication. Automation integrations are defined in your team settings.
1. Create the Slack or webhook automation. When you create a webhook automation, you configure the payload it sends and grant it access to any secrets it requires for authorization or to compose the payload.

## View and manage automations
View and manage a project's automations from the project's **Automations** tab.

- To view an automation's details, click its name.
- To edit an automation, click its action `...` menu, then click **Edit automation**.
- To delete an automation, click its action `...` menu, then click **Delete automation**. Confiruation is required.

## Next steps
- [Create a Slack automation]({{< relref "slack.md" >}}).
- [Create a webhook automation]({{< relref "webhook.md" >}}).
- [Create a secret]({{< relref "/guides/core/secrets.md" >}}).


