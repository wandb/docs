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
This section describes the scopes and events for an automation in a Registry.

### Scopes
You create a Registry automation at these scopes:
- A [Registry]({{< relref "/guides/core/registry/">}}). The automation watches for the event taking place on any collection in the registry.
- A single collection in a registry.

### Events
A Registry automation can watch for these events:
- **Linking a new artifact to a collection**: Test and validate new models or datasets when they are registered.
- **Adding a new alias to a version of an artifact**: Trigger a special step of your workflow when a new artifact version has a specific alias applied. For example, deploy a model when it has the `production` alias applied.

## Project
This section describes the scopes and events for an automation in a project.

### Scopes
You can create a project automation at these scopes:
- The project. The automation watches for the event taking place on any collection in the project.
- All collections in the project that match the filter you supply.

### Artifact events
An automation can watch for these [artifact]({{< relref "/guides/core/artifacts/" >}}) events:
- **A new version of an artifact is created in a collection**: Apply recurring actions to each version of an artifact. Specifying a collection is optional. For example, start a training job when a new dataset artifact version is created.
- **An artifact alias is added**: Trigger a special step of your workflow when a new artifact version in a project or collection has a specific alias applied. For example, run a series of downstream processing steps when an artifact has the `test-set-quality-check` alias applied.

### Run metrics events
An automation can watch for these run metrics events:

- **Run metrics threshold met**: Trigger a workflow when, for a given metric, a run or the average of a number of runs meets the threshold you specify for a given metric.
- **Run metrics change threshold met**: Trigger a workflow when, for a given metric, a run or the average of a number of runs increases or decreases by the threshold you specify for a given metric.

To set up the metric for a run metrics event, you specify the metric, the threshold, and the comparison to make. Refer to [Run metrics comparison choices]({{< relref "#run-metrics-comparison-choices" >}}).

#### Run metrics comparison choices
When configuring an automation for a run metric event, you can configure how to compare the run metric value with the threshold you specify. Your choices depend on the event type.

For **Run metrics threshold met** events, you can configure:
1. The number of logged values to average across (defaults to 5).
1. How to compare the values with the threshold:
    - Above
    - Above or equal to
    - Below
    - Below or equal to
    - Not equal to
    - Equal to

For **Run metrics change threshold met**, you can configure:
1. The current window of logged values to average across (defaults to 10).
1. The prior window of logged values to average across (defaults to 50).
1. Whether to evaluate the values as relative or absolute (defaults to **Relative**).
1. How to compare the values with the threshold:
      - Increases by at least
      - Decreases by at least
      - Increases or decreases by at least

## Next steps
- [Create a Slack automation]({{< relref "create-automations/slack.md" >}})
- [Create a webhook automation]({{< relref "create-automations/webhook.md" >}})