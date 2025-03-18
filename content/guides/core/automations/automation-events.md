---
menu:
  default:
    identifier: automation-scopes
    parent: automations
title: Automation events and scopes
weight: 2
---
An automation can run when a specific event occurs at a given scope, either a registry or a project. This page lists the events that can trigger an automation at each scope.

## Registry
For an artifact in [Registry]({{< relref "/guides/core/registry/">}}), you can configure an automation to run when:

- **Linking a new artifact to a collection**: Test and validate new models or datasets when they are registered.
- **Adding a new alias to a version of an artifact**: Trigger a special step of your workflow when a new artifact version has a specific alias applied. For example, deploy a model when it has the `deploy` alias applied.

## Project
For an artifact in a project or a collection, you can configure an automation to run on these events:

- **Linking a new artifact**: Test and validate new models, datasets or dataset automatically.
- **Creating a new version of an artifact**: Apply recurring actions to each version of an artifact. For example, start a training job when a new dataset artifact version is created.
- **Adding a new alias to a version of an artifact**: Trigger a special step of your workflow when a new artifact version in a project or collection has a specific label or alias applied. For example, run a series of downstream processing steps when an artifact has the `test-set-quality-check` alias applied.

## Next steps
- [Create a Slack automation]({{< relref "create-automations/slack.md" >}})
- [Create a webhook automation]({{< relref "create-automations/webhook.md" >}})