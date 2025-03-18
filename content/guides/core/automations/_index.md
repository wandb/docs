---
menu:
  default:
    identifier: automations
    parent: core
title: Automations
weight: 4
---

This page describes _automations_ in W&B. [Create an automation]({{< relref "create-automations/" >}}) to trigger workflow steps, such as automated model testing and deployment, based on an event in W&B, such as when an [artifact]({{< relref "/guides/core/artifacts" >}}) is created or changed or a [registered model is changed]({{< relref "/guides/core/registry/" >}}).

For example, an automation can post to a Slack channel when a new artifact is created, or run a webhook to trigger automated testing when a model has the `test` label applied.

## Overview
An automation can run when a specific [event]({{< relref "automation-events.md" >}}) occurs in a registry or project.

For an artifact in [Registry]({{< relref "/guides/core/registry/">}}), you can configure an automation to run:
- When a new artifact is linked to a collection. For example, trigger testing and validation workflows for new models.
- When an alias is added to a new version of an artifact. For example, trigger a deployment workflow when the `deploy` alias is added to a model version.

For an artifact in a project, you can configure an automation to run:
- When a new version of an artifact is created. For example, start a training job when a new version of a dataset artifact is created.
- When a new artifact is linked to a project or collection. For example, trigger testing and validation workflows for new models.
- When an alias is added to a new version of an artifact. For example, trigger a deployment workflow when the `deploy` alias is added to a model version.

For more details, refer to [Automation events]({{< relref "automation-events.md" >}}).

To [create an automation]({{< relref "create-automations/" >}}), you:

1. If required, configure [secrets]({{< relref "/guides/core/secrets.md" >}}) for sensitive strings the automation requires, such as access tokens, passwords, or sensitive configuration details.
1. Authorize W&B to post to Slack or run the webhook on your behalf. This is required once per Slack channel or webhook.
1. Create the automation:
  1. Grant it access to any secrets you created for it.
  1. Define the [event]({{< relref "#automation-events" >}}) to watch for at a given scope, such as in a registry, a project, or a collection.
  1. Define the action to take when the event occurs (posting to a Slack channel or running a webhook) and the payload to send.

## Next steps
- [Create an automation]({{< relref "create-automations/" >}}).
- Learn about [Automation events and scopes]({{< relref "automation-events.md" >}}).
- [Create a secret]({{< relref "/guides/core/secrets.md" >}}).
