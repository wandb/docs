---
menu:
  default:
    identifier: create-slack-automations
    parent: create-automations
title: Create a Slack automation
weight: 1
---
{{% pageinfo color="info" %}}
{{< readfile file="/_includes/enterprise-cloud-only.md" >}}
{{% /pageinfo %}}

This page shows how to create a Slack [automation]({{< relref "/guides/core/automations/" >}}> ). To create a webhook automation, refer to [Create a webhook automation]({{< relref "/guides/core/automations/create-automations/webhook.md" >}}) instead.

At a high level, to create a Slack automation, you take these steps:
1. [Add a Slack integration]({{< relref "#add-a-slack-integration" >}}), which authorizes W&B to post to the Slack instance and channel.
1. [Create the automation]({{< relref "#create-an-automation" >}}), which defines the [event]({{< relref "/guides/core/automations/automation-events.md" >}}) to watch for and the channel to notify.

## Add a Slack integration
A team admin can add a Slack integration to the team.

1. Log in to W&B and go to **Team Settings**.
1. In the **Slack channel integrations** section, click **Connect Slack** to add a new Slack instance. To add a channel for an existing Slack instance, click **New integration**.

    ![Screenshot showing two Slack integrations in a Team](/images/automations/slack_integrations.png)
1. If necessary, sign in to Slack in your browser. When prompted, grant W&B permission to post to the Slack channel you select. Read the page, then click **Search for a channel** and begin typing the channel name. Select the channel from the list, then click **Allow**.
1. In Slack, go to the channel you selected. If you see a post like `[Your Slack handle] added an integration to this channel: Weights & Biases`, the integration is configured correctly.

Now you can [create an automation]({{< relref "#create-an-automation" >}}) that notifies the Slack channel you configured.

{{% alert %}}
**For programmatic automation creation**: Slack integrations must be configured through the UI due to OAuth requirements. However, once configured, you can use the API to create multiple automations that use the same Slack integration. This separation allows secure channel authorization while enabling flexible automation management via code.
{{% /alert %}}

## View and manage Slack integrations
A team admin can view and manage the team's Slack instances and channels.

1. Log in to W&B and go to **Team Settings**.
1. View each Slack destination in the **Slack channel integrations** section.
1. Delete a destination by clicking its trash icon.

## Create an automation
After you [add a Slack integration]({{< relref "#add-a-slack-integreation" >}}), select **Registry** or **Project**, then follow these steps to create an automation that notifies the Slack channel.

{{< tabpane text=true >}}
{{% tab "Registry" %}}
A Registry admin can create automations in that registry.

1. Log in to W&B.
1. Click the name of a registry to view its details, 
1. To create an automation scoped to the registry, click the **Automations** tab, then click **Create automation**. An automation that is scoped to a registry is automatically applied to all of its collections (including those created in the future).

    To create an automation scoped only to a specific collection in the registry, click the collection's action `...` menu, then click **Create automation**. Alternatively, while viewing a collection, create an automation for it using the **Create automation** button in the **Automations** section of the collection's details page.
1. Choose the [event]({{< relref "/guides/core/automations/automation-events.md" >}}) to watch for.

    Fill in any additional fields that appear, which depend upon the event. For example, if you select **An artifact alias is added**, you must specify the **Alias regex**.

    Click **Next step**.
1. Select the team that owns the [Slack integration]({{< relref "#add-a-slack-integration" >}}).
1. Set **Action type** to **Slack notification**. Select the Slack channel, then click **Next step**.
1. Provide a name for the automation. Optionally, provide a description.
1. Click **Create automation**.

{{% /tab %}}
{{% tab "Project" %}}
A W&B admin can create automations in a project.

1. Log in to W&B.
1. Go the project page and click the **Automations** tab, then click **Create automation**.

    Or, from a line plot in the workspace, you can quickly create a [run metric automation]({{< relref "/guides/core/automations/automation-events.md#run-events" >}}) for the metric it shows. Hover over the panel, then click the bell icon at the top of the panel.
    {{< img src="/images/automations/run_metric_automation_from_panel.png" alt="Automation bell icon location" >}}
1. Choose the [event]({{< relref "/guides/core/automations/automation-events.md" >}}) to watch for.

    Fill in any additional fields that appear, which depend upon the event. For example, if you select **An artifact alias is added**, you must specify the **Alias regex**.

    Click **Next step**.
1. Select the team that owns the [Slack integration]({{< relref "#add-a-slack-integration" >}}).
1. Set **Action type** to **Slack notification**. Select the Slack channel, then click **Next step**.
1. Provide a name for the automation. Optionally, provide a description.
1. Click **Create automation**.

{{% /tab %}}
{{< /tabpane >}}

## Create automations programmatically

You can also create Slack automations using the W&B API. This enables you to automate the creation and management of automations as part of your MLOps workflows.

### Prerequisites

Before creating automations programmatically:
1. Ensure you have a [Slack integration configured]({{< relref "#add-a-slack-integration" >}}) in your team settings
2. Install the W&B SDK: `pip install wandb`
3. Authenticate with your W&B API key

**Note**: Slack integrations must be created through the W&B UI due to the OAuth authentication flow required by Slack. Once configured, you can use the API to list available integrations and create automations that use them.

### Checking for Slack Integrations

Before creating automations, verify that you have Slack integrations configured:

```python
import wandb

# Initialize the API
api = wandb.Api()

# List all Slack integrations for your team
slack_integrations = list(api.slack_integrations(entity="your-team"))

if not slack_integrations:
    print("❌ No Slack integrations found!")
    print("Please configure a Slack integration in your team settings:")
    print(f"https://wandb.ai/{your-team}/settings/integrations")
else:
    print(f"✅ Found {len(slack_integrations)} Slack integration(s):")
    for integration in slack_integrations:
        print(f"   - Channel: {integration.channel_name}")
        print(f"     ID: {integration.id}")
        print(f"     Team: {integration.slack_team_name}")
```

### Helper Function for Integration Selection

```python
def get_slack_integration(entity, channel_pattern=None):
    """Get a Slack integration, optionally filtered by channel name pattern"""
    integrations = list(api.slack_integrations(entity=entity))
    
    if not integrations:
        raise ValueError(f"No Slack integrations found for {entity}. "
                        "Please configure one in team settings.")
    
    if channel_pattern:
        # Filter by channel name pattern
        matching = [i for i in integrations if channel_pattern in i.channel_name]
        if matching:
            return matching[0]
    
    # Return first integration if no pattern or no matches
    return integrations[0]

# Example usage
try:
    # Get integration for alerts channel
    alerts_integration = get_slack_integration("my-team", "alerts")
    print(f"Using Slack channel: {alerts_integration.channel_name}")
except ValueError as e:
    print(e)
```

{{< tabpane text=true >}}
{{% tab "Registry" %}}

#### Example: Registry automation for artifact events

Create an automation that sends a Slack notification when a new model version is added to a registry collection:

```python
import wandb
from wandb.automations import OnAddArtifactAlias, SendNotification

# Initialize the W&B API
api = wandb.Api()

# Get the registry and collection
registry = api.registry("my-registry", entity="my-team")
collection = registry.collection("my-model-collection")

# Get the Slack integration
slack_integration = next(api.slack_integrations(entity="my-team"))

# Define the event: Trigger when "production" alias is added
event = OnAddArtifactAlias(
    scope=collection,
    filter={"alias": "production"}
)

# Define the action: Send notification to Slack
action = SendNotification.from_integration(
    slack_integration,
    title="Production Model Updated",
    text="A new model version has been promoted to production.",
    level="INFO"
)

# Create the automation
automation = api.create_automation(
    event >> action,
    name="production-model-alert",
    description="Notify team when a model is promoted to production"
)

print(f"Created automation: {automation.name}")
```

#### Example: Monitor multiple collections

Create an automation that monitors all collections in a registry:

```python
import wandb
from wandb.automations import OnCreateArtifact, SendNotification

# Initialize the W&B API
api = wandb.Api()

# Get the registry
registry = api.registry("my-registry", entity="my-team")

# Get the Slack integration  
slack_integration = next(api.slack_integrations(entity="my-team"))

# Define event at registry scope (applies to all collections)
event = OnCreateArtifact(
    scope=registry,
    filter={"artifact_type": "model"}
)

# Define the action
action = SendNotification.from_integration(
    slack_integration,
    title="New Model Registered",
    text="A new model artifact has been added to the registry."
)

# Create the automation
automation = api.create_automation(
    event >> action,
    name="registry-wide-model-monitor",
    description="Monitor all model artifacts across the registry"
)
```

{{% /tab %}}
{{% tab "Project" %}}

#### Example: Project automation for run metrics

Create an automation that alerts when model accuracy drops below a threshold:

```python
import wandb
from wandb.automations import OnRunMetric, RunEvent, SendNotification

# Initialize the W&B API
api = wandb.Api()

# Get the project
project = api.project("my-ml-project", entity="my-team")

# Get the Slack integration
slack_integration = next(api.slack_integrations(entity="my-team"))

# Define the event: Trigger when accuracy drops below 0.85
event = OnRunMetric(
    scope=project,
    filter=RunEvent.metric("accuracy") < 0.85
)

# Define the action: Send alert to Slack
action = SendNotification.from_integration(
    slack_integration,
    title="⚠️ Low Accuracy Alert",
    text="Model accuracy has dropped below 85%",
    level="WARN"
)

# Create the automation
automation = api.create_automation(
    event >> action,
    name="low-accuracy-alert",
    description="Alert when model accuracy drops below threshold"
)

print(f"Created automation: {automation.name}")
```

#### Example: Monitor training completion

Create an automation that notifies when long-running experiments complete:

```python
import wandb
from wandb.automations import OnRunStateChange, RunState, SendNotification

# Initialize the W&B API
api = wandb.Api()

# Get the project
project = api.project("long-running-experiments", entity="my-team")

# Get the Slack integration
slack_integration = next(api.slack_integrations(entity="my-team"))

# Define the event: Trigger when a run finishes
event = OnRunStateChange(
    scope=project,
    to_state=RunState.finished
)

# Define the action
action = SendNotification.from_integration(
    slack_integration,
    title="✅ Training Complete",
    text="Your experiment has finished running.",
    level="INFO"
)

# Create the automation
automation = api.create_automation(
    event >> action,
    name="training-completion-notifier",
    description="Notify when training runs complete"
)
```

{{% /tab %}}
{{< /tabpane >}}

### Managing automations via API

You can also list, update, and delete automations programmatically:

```python
# List all automations for a project
automations = project.automations()
for automation in automations:
    print(f"- {automation.name}: {automation.description}")

# Update an automation (enable/disable)
automation.enabled = False
automation.save()

# Delete an automation
automation.delete()
```

### Advanced patterns

#### Using multiple filters

Combine multiple conditions in your automation triggers:

```python
from wandb.automations import OnRunMetric, RunEvent, SendNotification

# Trigger when accuracy is high AND loss is low
event = OnRunMetric(
    scope=project,
    filter=(RunEvent.metric("accuracy") > 0.95) & 
           (RunEvent.metric("loss") < 0.1)
)
```

#### Dynamic Slack channel selection

Route notifications to different channels based on your team structure:

```python
# Find specific Slack integration by channel name
slack_integrations = api.slack_integrations(entity="my-team")
dev_channel = next(
    (ig for ig in slack_integrations if ig.channel_name == "ml-dev-alerts"),
    None
)

if dev_channel:
    action = SendNotification.from_integration(dev_channel)
```

## View and manage automations

{{< tabpane text=true >}}
{{% tab "Registry" %}}

- Manage the registry's automations from the registry's **Automations** tab.
- Mamage a collection's automations from the **Automations** section of the collection's details page.

From either of these pages, a Registry admin can manage existing automations:
- To view an automation's details, click its name.
- To edit an automation, click its action `...` menu, then click **Edit automation**.
- To delete an automation, click its action `...` menu, then click **Delete automation**. Confirmation is required.


{{% /tab %}}
{{% tab "Project" %}}
A W&B admin can view and manage a project's automations from the project's **Automations** tab.

- To view an automation's details, click its name.
- To edit an automation, click its action `...` menu, then click **Edit automation**.
- To delete an automation, click its action `...` menu, then click **Delete automation**. Confirmation is required.
{{% /tab %}}
{{< /tabpane >}}
