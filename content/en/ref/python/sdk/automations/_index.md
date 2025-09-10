---
title: Automations
module: wandb.automations
weight: 4
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

## Configuration

### Setting Up Integrations

```python
# Configure Slack integration
from wandb.automations import SlackIntegration

slack = SlackIntegration(
    webhook_url="https://hooks.slack.com/services/..."
)

# Use in notification action
notification = SendNotification.from_integration(
    integration=slack,
    title="ML Alert",
    text="Training completed",
    level="INFO"
)
```

## Example Usage

```python
import wandb
from wandb.automations import OnRunMetric, SendNotification, MetricThresholdFilter

# Initialize W&B
wandb.login()

# Create an automation that alerts when accuracy exceeds 0.95
automation = OnRunMetric(
    filter=MetricThresholdFilter(
        name="accuracy",
        cmp=">",
        threshold=0.95
    ),
    scope="entity/project"
).then(
    SendNotification(
        title="High Accuracy Achieved",
        message="Model accuracy exceeded 95%",
        severity="INFO"
    )
)

# Save the automation
automation.save(name="accuracy-alert", enabled=True)

# Create an automation for artifact creation
artifact_automation = OnCreateArtifact(
    scope="entity/project/artifact-collection"
).then(
    SendWebhook.from_integration(
        integration=webhook_integration,
        payload={"event": "new_artifact", "collection": "models"}
    )
)

# Save with description
artifact_automation.save(
    name="model-webhook",
    description="Notify external service on new model creation",
    enabled=True
)

# Query existing automations
from wandb.apis.public import Api
api = Api()
automations = api.project("entity/project").automations()

for auto in automations:
    print(f"Automation: {auto.name}")
    print(f"Enabled: {auto.enabled}")
    print(f"Event: {auto.event}")
```

