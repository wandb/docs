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

{{< img src="/images/automations/automation_events_actions.png" alt="Automation events and actions" >}}

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
An automation can be triggered by a change in a [run's status]({{< relref "/guides/models/track/runs/#run-states" >}}) or a change in a [metric value]({{< relref "/guides/models/track/log/#what-data-is-logged-with-specific-wb-api-calls" >}}).

#### Run status change
{{% alert %}}
- Currently available only in [W&B Multi-tenant Cloud]({{< relref "/guides/hosting/#wb-multi-tenant-cloud" >}}).
- A run with **Killed** status cannot trigger an automation. This status indicates that the run was stopped forcibly by an admin user.
{{% /alert %}}

Trigger a workflow when a run changes its [status]({{< relref "/guides/models/track/runs/_index.md#run-states" >}}) to **Running**, **Finished**, or **Failed**. Optionally, you can further limit the runs that can trigger an automation by filtering by the user that started a run or the run's name.

Because run status is a property of the entire run, you can create a run status automation only from the the **Automations** page, not from a workspace.

#### Run metrics change
{{% alert %}}
Currently available only in [W&B Multi-tenant Cloud]({{< relref "/guides/hosting/#wb-multi-tenant-cloud" >}}).
{{% /alert %}}

Trigger a workflow based on a logged value for a metric, either a metric in a run's history or a [system metric]({{< relref "/guides/models/app/settings-page/system-metrics.md" >}}) such as `cpu`, which tracks the percentage of CPU utilization. W&B logs system metrics automatically every 15 seconds.

You can create a run metrics automation from the project's **Automations** tab or directly from a line plot panel in a workspace.

To set up a run metric automation, you configure how to compare the metric's value with the threshold you specify. Your choices depend on the event type and on any filters you specify.

Optionally, you can further limit the runs that can trigger an automation by filtering by the user that started a run or the run's name.

##### Threshold
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

For example, trigger an automation when average `accuracy` is above `.6`.

##### Change threshold
For **Run metrics change threshold met** events, the automation uses two "windows" of values to check whether to start:

- The _current window_ of recently logged values to consider (defaults to 10).
- The _prior window_ of recently logged values to consider (defaults to 50).

The current and prior windows are consecutive and do not overlap.

To create the automation, you configure:
1. The current window of logged values (defaults to 10).
1. The prior window of logged values (defaults to 50).
1. Whether to evaluate the values as relative or absolute (defaults to **Relative**).
1. The comparison to make:
      - Increases by at least
      - Decreases by at least
      - Increases or decreases by at least

      For example, trigger an automation when `loss` decreases by at least `.25`.

#### Run filters
This section describes how the automation selects runs to evaluate.

- By default, any run in the project triggers the automation when the event occurs. To consider only specific runs, specify a run filter.
- Each run is considered individually and can potentially trigger the automation.
- Each run's values are put into a separate window and compared to the threshold separately.
- In a 24 hour period, a particular automation can fire at most once per run.

## Next steps
- [Create a Slack automation]({{< relref "create-automations/slack.md" >}})
- [Create a webhook automation]({{< relref "create-automations/webhook.md" >}})