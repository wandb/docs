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

An automation can start when a specific event occurs within a project or registry. This diagram shows the relationship between automation events and actions.

{{< img src="/images/automations/automation_events_actions.png" alt="Diagram showing the relationship between automation events and actions" >}}

This page describes the events that can trigger an automation within each scope. Learn more about automations in the [Automations overview]({{< relref "/guides/core/automations/" >}}) or [Create an automation]({{< relref "create-automations/" >}}).

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

- **Linking a new artifact**: Test and validate new models or datasets automatically.
- **Creating a new version of an artifact**: Apply recurring actions to each version of an artifact. For example, start a training job when a new dataset artifact version is created.
- **Adding a new alias to a version of an artifact**: Trigger a specific step of your workflow when a new artifact version in a project or collection has a specific label or alias applied. For example, run a series of downstream processing steps when an artifact has the `test-set-quality-check` alias applied.

### Run events
From the project's **Automations** tab or directly from a line plot panel, you can create a run metric automation triggered by:
- A metric in a run's history.
- A [system metric]({{< relref "/guides/models/app/settings-page/system-metrics.md" >}}) such as `cpu`, which tracks the percentage of CPU utilization. W&B logs system metrics automatically every 15 seconds.

The notification can watch the metric for these events:
- **Run metrics threshold met**: Trigger a workflow when the average, minimum, or maximum value for a given metric meets the threshold you specify.
- **Run metrics change threshold met**: Trigger a workflow when the average logged value for a given metric changes by the threshold you specify.

To set up a run metric automation, you configure how to compare the metric's value with the threshold you specify. Your choices depend on the event type and on any filters you specify.

{{% alert %}}
Run metric automations are currently available only in [W&B Multi-tenant Cloud]({{< relref "/guides/hosting/#wb-multi-tenant-cloud" >}}).
{{% /alert %}}

#### Threshold
For **Run metrics threshold met** events, you configure:
1. The window of most recently logged values to consider (defaults to 5).
1. Whether to evaluate the **Average**, **Min**, or **Max** value within the window.
1. The comparison to make:
      - Above
      - Above or equal to
      - Below
      - Below or equal to
      - Not equal to
      - Equal to

For example, trigger an automation when average `accuracy` exceeds `.6`.

#### Change threshold
For **Run metrics change threshold met** events, the automation uses two "windows" of values to check whether to start:

- The _current window_ averages the 10 most recently logged values by default.
- The _prior window_ averages the 50 most recently logged values prior to the current window.

The windows are consecutive and do not overlap.

To create the automation, you configure:
1. The current window of logged values (defaults to 10).
1. The prior window of logged values (defaults to 50).
1. Whether to evaluate the values as relative or absolute (defaults to **Relative**).
1. The comparison to make:
      - Increases by at least
      - Decreases by at least
      - Increases or decreases by at least

#### Run filters
This section describes how the automation selects runs to evaluate.

- By default, any run in the project triggers the animation when the event occurs. To consider only specific runs, specify a run filter.
- Each run is considered individually and can potentially trigger the automation.
- Each run's values are put into a separate window and compared to the threshold separately.
- In a 24 hour period, a particular automation can fire at most once per run.

## Next steps
- [Create a Slack automation]({{< relref "create-automations/slack.md" >}})
- [Create a webhook automation]({{< relref "create-automations/webhook.md" >}})