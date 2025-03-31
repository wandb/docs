---
menu:
  default:
    identifier: ja-guides-core-automations-automation-events
    parent: automations
title: Automation events and scopes
weight: 2
---

{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-cloud-only.md" >}}
{{% /pageinfo %}}

An automation can start when a specific event occurs within a project's or registrie's  scope. The *scope* of a project refers to [INSERT tech def of scope]. This page describes the events that can trigger an automation within each scope.

Learn more about automations in the [Automations overview]({{< relref path="/guides/core/automations/" lang="ja" >}}) or [Create an automation]({{< relref path="create-automations/" lang="ja" >}}).

## Registry
This section describes the scopes and events for an automation in a [Registry]({{< relref path="/guides/core/registry/" lang="ja" >}}).

1. Navigate to the **Registry** App at https://wandb.ai/registry/.
1. Click the name of a registry, then view and create automations in the **Automations** tab.

Learn more about [creating automations]({{< relref path="create-automations/" lang="ja" >}}).

### Scopes
You can create a Registry automation at these scopes:
- [Registry]({{< relref path="/guides/core/registry/" lang="ja" >}}) level: The automation watches for the event taking place on any collection within a specific registry, including collections added in the future.
- Collection level: A single collection in a specific registry.

### Events
A Registry automation can watch for these events:
- **Linking a new artifact to a collection**: Test and validate new models or datasets when they are added to a registry.
- **Adding a new alias to a version of an artifact**: Trigger a specific step of your workflow when a new artifact version has a specific alias applied. For example, deploy a model when it has the `production` alias applied.

## Project
This section describes the scopes and events for an automation in a [project]({{< relref path="/guides/models/track/project-page.md" lang="ja" >}}).

1. Navigate to your W&B project on the W&B App at `https://wandb.ai/<team>/<project-name>`.
1. View and create automations in the **Automations** tab.

Learn more about [creating automations]({{< relref path="create-automations/" lang="ja" >}}).

### Scopes
You can create a project automation at these scopes:
- Project level: The automation watches for the event taking place on any collection in the project.
- Collection level: All collections in the project that match the filter you specify.

### Events
A project automation can watch for these events:
- **A new version of an artifact is created in a collection**: Apply recurring actions to each version of an artifact. Specifying a collection is optional. For example, start a training job when a new dataset artifact version is created.
- **An artifact alias is added**: Trigger a specific step of your workflow when a new artifact version in a project or collection has a specific alias applied. For example, run a series of downstream processing steps when an artifact has the `test-set-quality-check` alias applied.

## Next steps
- [Create a Slack automation]({{< relref path="create-automations/slack.md" lang="ja" >}})
- [Create a webhook automation]({{< relref path="create-automations/webhook.md" lang="ja" >}})