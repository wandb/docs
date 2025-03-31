---
menu:
  default:
    identifier: ko-guides-core-automations-create-automations-_index
    parent: automations
title: Create an automation
weight: 1
---

{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-cloud-only.md" >}}
{{% /pageinfo %}}

This page gives an overview of creating and managing W&B [automations]({{< relref path="/guides/core/automations/" lang="ko" >}}). For more detailed instructions, refer to [Create a Slack automation]({{< relref path="/guides/core/automations/create-automations/slack.md" lang="ko" >}}) or [Create a webhook automation]({{< relref path="/guides/core/automations/create-automations/webhook.md" lang="ko" >}}).

{{% alert %}}
Looking for companion tutorials for automations? 
- [Learn to automatically triggers a Github Action for model evaluation and deployment](https://wandb.ai/wandb/wandb-model-cicd/reports/Model-CI-CD-with-W-B--Vmlldzo0OTcwNDQw).
- [Watch a video demonstrating automatically deploying a model to a Sagemaker endpoint](https://www.youtube.com/watch?v=s5CMj_w3DaQ).
- [Watch a video series introducing automations](https://youtube.com/playlist?list=PLD80i8An1OEGECFPgY-HPCNjXgGu-qGO6&feature=shared).
{{% /alert %}}

## Requirements
- A team admin can create and manage automations for the team's projects, as well as components of their automations, such as webhooks, secrets, or Slack connections. Refer to [Team settings]({{< relref path="/guides/models/app/settings-page/team-settings/" lang="ko" >}}).
- To create a registry automation, you must have access to the registry. Refer to [Configure Registry access]({{< relref path="/guides/core/registry/configure_registry.md#registry-roles" lang="ko" >}}).
- To create a Slack automation, you must have permission to post to the Slack instance and channel you select.

## Create an automation
Create an automation from the project or registry's **Automations** tab. At a high level, to create an automation, follow these steps:

1. If necessary, [create a W&B secret]({{< relref path="/guides/core/secrets.md" lang="ko" >}}) for each sensitive string required by the automation, such as an access token, password, or SSH key. Secrets are defined in your **Team Settings**. Secrets are most commonly used in webhook automations.
1. Configure the webhook or Slack notification to authorize W&B to post to Slack or run the webhook on your behalf. A single automation action (webhook or Slack notification) can be used by multiple automations. These actions are defined in your **Team Settings**. 
1. In the project or registry, create the automation, which specifies the event to watch for and the action to take (such as posting to Slack or running a webhook). When you create a webhook automation, you configure the payload it send.

For details, refer to:

- [Create a Slack automation]({{< relref path="slack.md" lang="ko" >}})
- [Create a webhook automation]({{< relref path="webhook.md" lang="ko" >}})

## View and manage automations
View and manage automations from a project or registry's **Automations** tab.

- To view an automation's details, click its name.
- To edit an automation, click its action `...` menu, then click **Edit automation**.
- To delete an automation, click its action `...` menu, then click **Delete automation**.

## Next steps
- [Create a Slack automation]({{< relref path="slack.md" lang="ko" >}}).
- [Create a webhook automation]({{< relref path="webhook.md" lang="ko" >}}).
- [Create a secret]({{< relref path="/guides/core/secrets.md" lang="ko" >}}).