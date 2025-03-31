---
menu:
  default:
    identifier: ko-guides-core-automations-_index
    parent: core
title: Automations
weight: 4
---

{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-cloud-only.md" >}}
{{% /pageinfo %}}

This page describes _automations_ in W&B. [Create an automation]({{< relref path="create-automations/" lang="ko" >}}) to trigger workflow steps, such as automated model testing and deployment, based on an event in W&B, such as when an [artifact]({{< relref path="/guides/core/artifacts" lang="ko" >}}) artifact version is created.

For example, an automation can post to a Slack channel when a new version is created, or can run a webhook to trigger automated testing when the `production` alias is added to an artifact.

## Overview
An automation can run when a specific [event]({{< relref path="automation-events.md" lang="ko" >}}) occurs in a registry or project.

For an artifact in a [Registry]({{< relref path="/guides/core/registry/" lang="ko" >}}), you can configure an automation to run:
- When a new artifact version is linked to a collection. For example, trigger testing and validation workflows for new candidate models.
- When an alias is added to an artifact version. For example, trigger a deployment workflow when an alias is added to a model version.

For an artifact in a [project]({{< relref path="/guides/models/track/project-page.md" lang="ko" >}}), you can configure an automation to run:
- When a new version is added to an artifact. For example, start a training job when a new version of a dataset artifact is added to a given collection.
- When an alias is added to an artifact version. For example, trigger a PII redaction workflow when the alias "redaction" is added to a dataset artifact.

For more details, refer to [Automation events and scopes]({{< relref path="automation-events.md" lang="ko" >}}).

To [create an automation]({{< relref path="create-automations/" lang="ko" >}}), you:

1. If required, configure [secrets]({{< relref path="/guides/core/secrets.md" lang="ko" >}}) for sensitive strings the automation requires, such as access tokens, passwords, or sensitive configuration details. Secrets are defined in your **Team Settings**. Secrets are most commonly used in webhook automations to securely pass credentials or tokens to the webhook's external service without exposing it in plain text or hard-coding it in the webhook's payload.
1. Configure the webhook or Slack notification to authorize W&B to post to Slack or run the webhook on your behalf. A single automation action (webhook or Slack notification) can be used by multiple automations. These actions are defined in your **Team Settings**.
1. In the project or registry, create the automation:
    1. Define the [event]({{< relref path="#automation-events" lang="ko" >}}) to watch for, such as when a new artifact version is added. 
    1. Define the action to take when the event occurs (posting to a Slack channel or running a webhook). For a webhook, specify a secret to use for the access token and/or a secret to send with the payload, if required.

## Next steps
- [Create an automation]({{< relref path="create-automations/" lang="ko" >}}).
- Learn about [Automation events and scopes]({{< relref path="automation-events.md" lang="ko" >}}).
- [Create a secret]({{< relref path="/guides/core/secrets.md" lang="ko" >}}).