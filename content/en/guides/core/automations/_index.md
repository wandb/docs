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

This page describes _automations_ in W&B. [Create an automation]({{< relref "create-automations/" >}}) to trigger workflow steps, such as automated model testing and deployment, based on an event in W&B, such as when an [artifact]({{< relref "/guides/core/artifacts" >}}) artifact version is created or when a [run metric]({{< relref "/guides/models/track/runs.md" >}}) meets or changes by a threshold.

For example, an automation can notify a Slack channel when a new version is created, trigger an automated testing webhook when the `production` alias is added to an artifact, or start a validation job only when a run's `loss` is within acceptable bounds.

## Overview
An automation can start when a specific [event]({{< relref "automation-events.md" >}}) occurs in a registry or project.

In a [Registry]({{< relref "/guides/core/registry/">}}), an automation can start:
- When a new artifact version is linked to a collection. For example, trigger testing and validation workflows for new candidate models.
- When an alias is added to an artifact version. For example, trigger a deployment workflow when an alias is added to a model version.

In a [project]({{< relref "/guides/models/track/project-page.md" >}}), an automation can start:
- When a new version is added to an artifact. For example, start a training job when a new version of a dataset artifact is added to a given collection.
- When an alias is added to an artifact version. For example, trigger a PII redaction workflow when the alias "redaction" is added to a dataset artifact.
- When a metric for a run meets or exceeds a configured threshold. Optionally filter runs by user or run name.
- When a metric for a run changes by a configured threshold. Optionally filter runs by user or run name.
- When a run's status changes to **Running**, **Failed** (including jobs that were interactively killed), or **Finished**. Optionally filter runs by user or run name.

This diagram shows the relationship between automation events and actions.

{{< img src="/images/automations/automation_events_actions.png" alt="Diagram showing the relationship between automation events and actions" >}}

For more details, refer to [Automation events and scopes]({{< relref "automation-events.md" >}}).

To [create an automation]({{< relref "create-automations/" >}}), you:

1. If required, configure [secrets]({{< relref "/guides/core/secrets.md" >}}) for sensitive strings the automation requires, such as access tokens, passwords, or sensitive configuration details. Secrets are defined in your **Team Settings**. Secrets are most commonly used in webhook automations to securely pass credentials or tokens to the webhook's external service without exposing it in plain text or hard-coding it in the webhook's payload.
1. Configure the webhook or Slack notification to authorize W&B to post to Slack or run the webhook on your behalf. A single automation action (webhook or Slack notification) can be used by multiple automations. These actions are defined in your **Team Settings**.
1. In the project or registry, create the automation:
    1. Define the [event]({{< relref "#automation-events" >}}) to watch for, such as when a new artifact version is added. 
    1. Define the action to take when the event occurs (posting to a Slack channel or running a webhook). For a webhook, specify a secret to use for the access token and/or a secret to send with the payload, if required.

## Limitations
[Run metric automations]({{< relref "automation-events.md#run-metrics-events">}}) are currently supported only in [W&B Multi-tenant Cloud]({{< relref "/guides/hosting/#wb-multi-tenant-cloud" >}}).

## Next steps
- [Create an automation]({{< relref "create-automations/" >}}).
- Learn about [Automation events and scopes]({{< relref "automation-events.md" >}}).
- [Create a secret]({{< relref "/guides/core/secrets.md" >}}).
