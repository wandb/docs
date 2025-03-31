---
menu:
  default:
    identifier: automation-scopes
    parent: automations
title: Automation events and scopes
weight: 2
---
{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-cloud-only.md" >}}
{{% /pageinfo %}}

An automation can start when a specific event occurs within a project's or registrie's  scope. The *scope* of a project refers to [INSERT tech def of scope]. This page describes the events that can trigger an automation within each scope.

Learn more about automations in the [Automations overview]({{< relref "/guides/core/automations/" >}}) or [Create an automation]({{< relref "create-automations/" >}}).

## Registry
This section describes the scopes and events for an automation in a [Registry]({{< relref "/guides/core/registry/">}}).

1. Navigate to the **Registry** App at https://wandb.ai/registry/.
1. Click the name of a registry, then view and create automations in the **Automations** tab.

Learn more about [creating automations]({{< relref "create-automations/" >}}).

### Scopes
You can create a Registry automation at these scopes:
- [Registry]({{< relref "/guides/core/registry/">}}) level: The automation watches for the event taking place on any collection within a specific registry, including collections added in the future.
- Collection level: A single collection in a specific registry.

### Events
A Registry automation can watch for these events:
- **Linking a new artifact to a collection**: Test and validate new models or datasets when they are added to a registry.
- **Adding a new alias to a version of an artifact**: Trigger a specific step of your workflow when a new artifact version has a specific alias applied. For example, deploy a model when it has the `production` alias applied.

## Project
This section describes the scopes and events for an automation in a [project]({{< relref "/guides/models/track/project-page.md" >}}).

1. Navigate to your W&B project on the W&B App at `https://wandb.ai/<team>/<project-name>`.
1. View and create automations in the **Automations** tab.

Learn more about [creating automations]({{< relref "create-automations/" >}}).

### Scopes
You can create a project automation at these scopes:
- Project level: The automation watches for the event taking place on any collection in the project.
- Collection level: All collections in the project that match the filter you specify.

### Artifact events
This section describes the events related to an artifact that can trigger an automation.

- **Linking a new artifact**: Test and validate new models, datasets or dataset automatically.
- **Creating a new version of an artifact**: Apply recurring actions to each version of an artifact. For example, start a training job when a new dataset artifact version is created.
- **Adding a new alias to a version of an artifact**: Trigger a special step of your workflow when a new artifact version in a project or collection has a specific label or alias applied. For example, run a series of downstream processing steps when an artifact has the `test-set-quality-check` alias applied.

### Run metrics events
An automation can watch for changes in a run metric:

- **Run metrics threshold met**: Trigger a workflow when for a given metric, a single logged value or the average logged values meets the threshold you specify.
- **Run metrics change threshold met**: Trigger a workflow when for a given metric, a single logged value or the average logged value changes by the threshold you specify.

Each run is considered separately, even if the automation has a run filter that matches with multiple runs. Each run's values are put into a separate window and compared to the threshold separately. If an automation starts due to the result of one run, it can still start for another matching run. A particular run metric automation runs only once per run per 24-hour period.

{{% alert %}}
You can create automations triggered by system metrics such as the `cpu` metric, which tracks the percentage of CPU utilization. W&B logs system metrics automatically every 15 seconds. Find each metric's name and details in [System metrics]({{< relref "/guides/models/app/settings-page/system-metrics.md" >}}).
{{% /alert %}}

To set up a run metric automation, you configure how to compare the run metric value with the threshold you specify. Your choices depend on the event type.

#### Threshold
For **Run metrics threshold met** events, you configure:
1. The number of logged values to average across (defaults to 5).
1. How to compare the values with the threshold.

For example, trigger an automation when `accuracy` exceeds `.6`.

#### Change threshold

For **Run metrics change threshold met** events, you can configure:
1. The current window of logged values to average across (defaults to the 10 most recent logged values).
1. The prior window of logged values to average across (defaults to the 50 most recent logged values prior to the current window).
1. Whether to evaluate the values as relative or absolute (defaults to **Relative**).
1. How to compare the values with the threshold:
      - Increases by at least
      - Decreases by at least
      - Increases or decreases by at least

The current and prior window do not overlap. By default, the current window includes the 10 most recent logged values, and the prior window includes the 50 logged values prior to the current window.

## Next steps
- [Create a Slack automation]({{< relref "create-automations/slack.md" >}})
- [Create a webhook automation]({{< relref "create-automations/webhook.md" >}})