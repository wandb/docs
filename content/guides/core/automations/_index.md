---
menu:
  default:
    identifier: automations
    parent: core
title: Automations
weight: 4
---
{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-cloud-only.md" >}}
{{% /pageinfo %}}

This page describes _automations_ in W&B. [Create an automation]({{< relref "create-automations/" >}}) to trigger workflow steps, such as automated model testing and deployment, based on an event in W&B, such as when an [artifact]({{< relref "/guides/core/artifacts" >}}) artifact version is created.

For example, an automation can post to a Slack channel when a new version is created, or can run a webhook to trigger automated testing when the `production` alias is added to an artifact.

## Overview
An automation can run when a specific [event]({{< relref "automation-events.md" >}}) occurs in a registry or project.

For an artifact in a [Registry]({{< relref "/guides/core/registry/">}}), you can configure an automation to run:
- When a new artifact version is linked to a collection. For example, trigger testing and validation workflows for new candidate models.
- When an alias is added to an artifact version. For example, trigger a deployment workflow when an alias is added to a model version.

For an artifact in a project, you can configure an automation to run:
- When a new version of an artifact is created. For example, start a training job when a new version of a dataset artifact is created.
- When a new artifact is linked to a project or collection. For example, trigger testing and validation workflows for new models.
- When an alias is added to a new version of an artifact. For example, trigger a deployment workflow when the `deploy` alias is added to a model version.
- When a metric for a run or set of runs meets or exceeds a configured threshold.
- When a metric for a run or set of runs changes by a configured threshold.

For more details, refer to [Automation events and scopes]({{< relref "automation-events.md" >}}).

To [create an automation]({{< relref "create-automations/" >}}), you:

1. If required, configure [secrets]({{< relref "/guides/core/secrets.md" >}}) for sensitive strings the automation requires, such as access tokens, passwords, or sensitive configuration details. Secrets are defined in your **Team Settings**. Secrets are most commonly used in webhook automations to securely pass credentials or tokens to the webhook's external service without exposing it in plain text or hard-coding it in the webhook's payload.
1. Configure the webhook or Slack notification to authorize W&B to post to Slack or run the webhook on your behalf. A single automation action (webhook or Slack notification) can be used by multiple automations. These actions are defined in your **Team Settings**.
1. In the project or registry, create the automation:
  1. Grant it access to any secrets you created for it.
  1. Define the [event]({{< relref "#automation-events" >}}) to watch for, such as when a new artifact or version is created or linked, or when a run metric changes by a configured threshold.
  1. Define the action to take when the event occurs (posting to a Slack channel or running a webhook) and the payload to send.

## Next steps
- [Create an automation]({{< relref "create-automations/" >}}).
- Learn about [Automation events and scopes]({{< relref "automation-events.md" >}}).
- [Create a secret]({{< relref "/guides/core/secrets.md" >}}).
