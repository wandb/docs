---
title: Automations
module: wandb.automations
weight: 4
no_list: true
---

The W&B Automations API enables programmatic creation and management of automated workflows that respond to events in your ML pipeline. Configure actions to trigger when specific conditions are met, such as model performance thresholds or artifact creation.


### Core Classes

| Class | Description |
|-------|-------------|
| [`Automation`](./automation/) | Represents a saved automation instance with its configuration. |
| [`NewAutomation`](./newautomation/) | Builder class for creating new automations. |

### Events (Triggers)

| Event | Description |
|-------|-------------|
| [`OnRunMetric`](./onrunmetric/) | Trigger when a run metric satisfies a defined condition (threshold, change, etc.). |
| [`OnCreateArtifact`](./oncreateartifact/) | Trigger when a new artifact is created in a collection. |
| [`OnLinkArtifact`](./onlinkartifact/) | Trigger when an artifact is linked to a registry. |
| [`OnAddArtifactAlias`](./onaddartifactalias/) | Trigger when an alias is added to an artifact. |

### Actions

| Action | Description |
|--------|-------------|
| [`SendNotification`](./sendnotification/) | Send notifications via Slack or other integrated channels. |
| [`SendWebhook`](./sendwebhook/) | Send HTTP webhook requests to external services. |
| [`DoNothing`](./donothing/) | Placeholder action for testing automation configurations. |

### Filters

| Filter | Description |
|--------|-------------|
| [`MetricThresholdFilter`](./metricthresholdfilter/) | Filter runs based on metric value comparisons against thresholds. |
| [`MetricChangeFilter`](./metricchangefilter/) | Filter runs based on metric value changes over time. |

## Common Use Cases

### Model Performance Monitoring
- Alert when model accuracy drops below a threshold
- Notify team when training loss plateaus
- Trigger retraining pipelines based on performance metrics

### Artifact Management
- Send notifications when new model versions are created
- Trigger deployment workflows when artifacts are tagged
- Automate downstream processing when datasets are updated

### Experiment Tracking
- Alert on failed or crashed runs
- Notify when long-running experiments complete
- Send daily summaries of experiment metrics

### Integration Workflows
- Update external tracking systems via webhooks
- Sync model registry with deployment platforms
- Trigger CI/CD pipelines based on W&B events

## Example Usage

The following example creates an automation that sends a Slack notification whenever a metric called `custom-metric` exceeds 10. `custom-metric` is expected to be logged during training using `wandb.Run.log({"custom-metric": value })`.

```python
import wandb
from wandb.automations import OnRunMetric, RunEvent, SendNotification

api = wandb.Api()

project = api.project("<my-project>", entity="<my-team>")

# Use the first Slack integration for the team
slack_hook = next(api.slack_integrations(entity="<my-team>"))

# Create a trigger event
event = OnRunMetric(
     scope=project,
     filter=RunEvent.metric("custom-metric") > 10,
)

# Create an action that responds to the event
action = SendNotification.from_integration(slack_hook)

# Create the automation
automation = api.create_automation(
     event >> action,
     name="my-automation",
     description="Send a Slack message whenever 'custom-metric' exceeds 10.",
)
```

For more information about using the Automations API, see the [Automations Guide]({{< relref "/guides/core/automations" >}}).