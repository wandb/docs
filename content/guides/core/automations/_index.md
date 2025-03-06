---
menu:
  default:
    identifier: automations
    parent: core
title: Automations
weight: 4
---

This page describes _automations_ in W&B and shows how to create and manage them. Create an automation to trigger workflow steps, such as automated model testing and deployment, based on an event in W&B, such as when an [artifact]({{< relref "/guides/core/artifacts" >}}) or a [registered model is changed]({{< relref "/guides/models/registry/" >}}).

An automation defines the [event scopes and types]({{< relref "#event-scopes-and-types" >}}) to watch for and the [action]({{< relref "#actions" >}}) to take when the event occurs, such as running a webhook or posting to a Slack channel.

## Events and event scopes

An automation can run when a specific event occurs at a given scope, either a registry or a project.

### Registry
For an artifact in [Registry]({{< relref "/guides/models/registry/">}}), you can configure an automation to run on these events:

- **Linking a new artifact to a collection**: Test and validate new models, datasets, etc.
- **Adding a new alias to a version of an artifact**: Trigger a special step of your workflow when a new artifact version has a specific alias applied. For example, deploy a model when it has the `deploy` alias applied.

### Project
For an artifact in a project or a collection, you can configure an automation to run on these events:

- **Linking a new artifact**: Test and validate new models, datasets, etc.
- **Creating a new version of an artifact**: Apply recurring actions to each version of an artifact. For example, start a training job when a new dataset artifact version is created.
- **Adding a new alias to a version of an artifact**: Trigger a special step of your workflow when a new artifact version in a project or collection has a specific label or alias applied. For example, , run a series of downstream processing steps when an artifact has the `test-set-quality-check` alias applied.

## Actions

When the configured event occurs, an automation runs the specified action, either by posting to a Slack channel or by running a webhook.

## Create an automation
These sections show how to configure a Slack automation or a webhook automation.

### Configure a Slack automation
Configuring a Slack integration takes multiple steps:
1. [Add one or more Slack channels]({{< relref "#add-a-slack-channel" >}}) as alert destinations.
1. [Create the Slack automation]({{< relref "#create-slack-automation" >}}).

#### Add a Slack destination

1. Log in to W&B and go to Team Settings page.
1. To integrate with a new Slack workspace and channel, click **Connect Slack**.
    {{% alert %}}
    To integrate with a new Slack channel in a workspace that is already set up, click **New integration**.
    {{% /alert %}}

  If necessary, sign in to your Slack instance. A browser window appears, asking that you grant Weights and Biases permission to post to the Slack channel you select. Read the page, then click **Search for a channel** and begin typing the channel name. Select the channel from the list, then click **Allow**.

1. In Slack, go to the channel you selected. If you see a post like `[Your Slack handle] added an integration to this channel: Weights & Biases`, the integration is configured correctly.

Now you can [create an automation that uses the Slack integration]({{< relref "#create-slack-automation" >}}).

#### Create a Slack automation {#create-slack-automation}

After you [configure a Slack integration]({{< relref "#add-a-slack-integration" >}}), follow these steps to create a Slack automation that uses it.

1. Log in to W&B and go to the project page.
1. In the sidebar, click **Automations**.
1. Click **Create automation**.
1. Choose the [**Event**]({{< relref "#events-and-event-scopes" >}}) which triggers the automation. Fill in any additional fields that appear, which depend upon the event. Click **Next step**.
1. Select the team where you added the Slack integration.
1. Set **Action type** to **Slack notification**.
1. Select the Slack channel, then click **Next step**.
1. Provide a name for the automation. Optionally, provide a description.
1. Click **Create automation**.

### Configure a webhook automation

Configuring a webhook integration tales multiple steps:

1. If your webhook requires any sensitive strings, such as a Bearer token for authorization, [add them as secrets]({{< relref "/guides/core/secrets.md#add-a-secret" >}}) before creating the webhook.
1. [Create the webook]({{< relref "#add-a-webhook" >}}). The webhook includes the endpoint URL and the authentication details for the service. Grant it access to any secrets it requires, including those you just created.
1. [Create an automation]({{< relref "#create-webhook-automation" >}}) that uses the webhook. The automation defines the payload the webhook sends to the service when the automation is triggered.

#### Add the webhook {#add-a-webhook}
A W&B Admin can configure a webhook for a team. The webhook defines the endpoint URL and the authentication and authorization details, but does not define the payload sent to the endpoint.

{{% alert %}}
If the webhook requires a Bearer token, [create a secret that contains it]({{< relref "/guides/core/secrets.md#add-a-secret" >}}) before creating the webhoook.
{{% /alert %}}

This section shows how to configure a webhook's URL and Bearer token, as well as any access tokens and secrets required by the webhook and the automation that uses it.

1. Log in to W&B.
1. If the webhook requires any access tokens or other sensitive strings, create a secret in W&B for each sensitive string, and make a note of the secret's name.
1. Go to the **Settings** page.
1. In the **Webhooks** section, click **New webhook**.
1. Provide a name for the webhook. 
1. Provide the endpoint URL for the webhook.
1. If the webhook requires an access token or any other [secrets]({{< relref "/guides/core/secrets.md" >}}), grant the webhook access to each secret by setting **Secret** to the secret's name. When you configure the automation that uses the webhook, you can access the secret in the payload by prefixing its name with `$`.
1. If the webhook authenticates using an access token, set **Access token** to the name of the secret that contains it. When you configure an automation that uses the webhook, you can access the token in the `$ACCESS_TOKEN` environment variable, and the HTTP header sets `Authorization: Bearer` to the access token.
1. Click **Test** to test authenticating to the webhook's endpoint.

    Optionally, provide a payload to test. To refer to a secret the webhook has access to in the payload, prefix its name with `$`. This payload is only used for testing and is not saved.  You configure an automation's payload when you [create the automation]({{< relref "#create-webhook-automation" >}}).

{{% alert %}}
See [Troubleshoot your webhook]({{< relref "#troubleshoot-your-webhook" >}}) to view where the secret and access token are specified in
the POST request.
{{% /alert %}}

Now you can [create a webhook automation]({{< relref "#create-webhook-automation" >}}).

#### Create the automation {#create-webhook-automation}
1. Log in to W&B and go to the project page.
1. In the sidebar, click **Automations**.
1. Click **Create automation**.
1. Choose the **Event** which triggers the automation. Fill in any additional fields that appear, which depend upon the event. Click **Next step**.
1. Select the team where you added the webhook.
1. Set **Action type** to **Webhook**, then select the webhook.
1. Provide the payload for the webhook in **Payload**. Refer to [Example webhook payloads]({{< relref "#example-webhook-payloads" >}}) for variables you can use. If the [webhook]({{< relref "#add-a-webhook" >}}) has access to any secrets, you can access the secret in the payload by prefixing its name with `$`. If the webhook has an access token, you can access the token in the `$ACCESS_TOKEN` environment variable, and the HTTP header sets `Authorization: Bearer` to the access token.
1. Click **Next step**.
1. Provide a name for the automation. Optionally, provide a description.
1. Click **Create automation**.

### Example webhook payloads

The following tabs demonstrate example payloads based on common use cases. Within the examples they reference the following keys to refer to condition objects in the payload parameters:
* `${event_type}` Refers to the type of event that triggered the action.
* `${event_author}` Refers to the user that triggered the action.
* `${artifact_version}` Refers to the specific artifact version that triggered the action. Passed as a [`Wandb.Artifact`]({{< relref "/ref/python/artifact/" >}}).
* `${artifact_version_string}` Refers to the specific artifact version that triggered the action. Passed as a string.
* `${artifact_collection_name}` Refers to the name of the artifact collection that the artifact version is linked to.
* `${project_name}` Refers to the name of the project owning the mutation that triggered the action.
* `${entity_name}` Refers to the name of the entity owning the mutation that triggered the action.

{{< tabpane text=true >}}
{{% tab header="GitHub repository dispatch" value="github" %}}

{{% alert %}}
Verify that your access tokens have required set of permissions to trigger your GHA workflow. For more information, [see these GitHub Docs](https://docs.github.com/en/rest/repos/repos?#create-a-repository-dispatch-event). 
{{% /alert %}}
  
Send a repository dispatch from W&B to trigger a GitHub action. For example, suppose you have workflow that accepts a repository dispatch as a trigger for the `on` key:

```yaml
on:
repository_dispatch:
  types: BUILD_AND_DEPLOY
```

The payload for the repository might look something like:

```json
{
  "event_type": "BUILD_AND_DEPLOY",
  "client_payload": 
  {
    "event_author": "${event_author}",
    "artifact_version": "${artifact_version}",
    "artifact_version_string": "${artifact_version_string}",
    "artifact_collection_name": "${artifact_collection_name}",
    "project_name": "${project_name}",
    "entity_name": "${entity_name}"
    }
}
```

{{% alert %}}
The `event_type` key in the webhook payload must match the `types` field in the GitHub workflow YAML file.
{{% /alert %}}

The contents and positioning of rendered template strings depends on the event or model version the automation is configured for. `${event_type}` will render as either `LINK_ARTIFACT` or `ADD_ARTIFACT_ALIAS`. See below for an example mapping:

```text
${event_type} --> "LINK_ARTIFACT" or "ADD_ARTIFACT_ALIAS"
${event_author} --> "<wandb-user>"
${artifact_version} --> "wandb-artifact://_id/QXJ0aWZhY3Q6NTE3ODg5ODg3""
${artifact_version_string} --> "<entity>/model-registry/<registered_model_name>:<alias>"
${artifact_collection_name} --> "<registered_model_name>"
${project_name} --> "model-registry"
${entity_name} --> "<entity>"
```

Use template strings to dynamically pass context from W&B to GitHub Actions and other tools. If those tools can call Python scripts, they can consume the registered model artifacts through the [W&B API]({{< relref "/guides/core/artifacts/download-and-use-an-artifact.md" >}}).

For more information about repository dispatch, see the [official documentation on the GitHub Marketplace](https://github.com/marketplace/actions/repository-dispatch).  

Watch the videos [Webhook Automations for Model Evaluation](https://www.youtube.com/watch?v=7j-Mtbo-E74&ab_channel=Weights%26Biases) and [Webhook Automations for Model Deployment](https://www.youtube.com/watch?v=g5UiAFjM2nA&ab_channel=Weights%26Biases), which guide you to create automations for model evaluation and deployment. 

Review a W&B [report](https://wandb.ai/wandb/wandb-model-cicd/reports/Model-CI-CD-with-W-B--Vmlldzo0OTcwNDQw), which illustrates how to use a Github Actions webhook automation for Model CI. Check out this [GitHub repository](https://github.com/hamelsmu/wandb-modal-webhook) to learn how to create model CI with a Modal Labs webhook. 

{{% /tab %}}

{{% tab header="Microsoft Teams notification" value="microsoft"%}}

Configure an ‘Incoming Webhook' to get the webhook URL for your Teams Channel by configuring. The following is an example payload:

```json 
{
"@type": "MessageCard",
"@context": "http://schema.org/extensions",
"summary": "New Notification",
"sections": [
  {
    "activityTitle": "Notification from WANDB",
    "text": "This is an example message sent via Teams webhook.",
    "facts": [
      {
        "name": "Author",
        "value": "${event_author}"
      },
      {
        "name": "Event Type",
        "value": "${event_type}"
      }
    ],
    "markdown": true
  }
]
}
```

You can use template strings to inject W&B data into your payload at the time of execution (as shown in the Teams example above).

{{% /tab %}}

{{% tab header="Slack notifications" value="slack"%}}

{{% alert %}}
This section is provided for historical purposes. If you currently use a webhook to integrate with Slack, W&B recommends that you update your configuration to use the [new Slack integration]({{ relref "#create-a-slack-automation"}}) instead.
{{% /alert %}}

Set up your Slack app and add an incoming webhook integration with the instructions highlighted in the [Slack API documentation](https://api.slack.com/messaging/webhooks). Ensure that you have the secret specified under `Bot User OAuth Token` as your W&B webhook’s access token. 

The following is an example payload:

```json
{
    "text": "New alert from WANDB!",
"blocks": [
    {
            "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": "Registry event: ${event_type}"
        }
    },
        {
            "type":"section",
            "text": {
            "type": "mrkdwn",
            "text": "New version: ${artifact_version_string}"
        }
        },
        {
        "type": "divider"
    },
        {
            "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": "Author: ${event_author}"
        }
        }
    ]
}
```

{{% /tab %}}
{{< /tabpane >}}

### Troubleshoot your webhook

Interactively troubleshoot your webhook with the W&B App UI or programmatically with a Bash script. You can troubleshoot a webhook when you create a new webhook or edit an existing webhook.


{{< tabpane text=true >}}
{{% tab header="W&B App UI" value="app" %}}

Interactively test a webhook with the W&B App UI. 

1. Navigate to your W&B Team Settings page.
2. Scroll to the **Webhooks** section.
3. Click on the horizontal three docs (meatball icon) next to the name of your webhook.
4. Select **Test**.
5. From the UI panel that appears, paste your POST request to the field that appears. 
    {{< img src="/images/models/webhook_ui.png" alt="" >}}
6. Click on **Test webhook**.

Within the W&B App UI, W&B posts the response made by your endpoint.

{{< img src="/images/models/webhook_ui_testing.gif" alt="" >}}

See [Testing Webhooks in Weights & Biases](https://www.youtube.com/watch?v=bl44fDpMGJw&ab_channel=Weights%26Biases) YouTube video to view a real-world example.
{{% /tab %}}

{{% tab header="Bash script" value="bash"%}}

The following bash script generates a POST request similar to the POST request W&B sends to your webhook automation when it is triggered.

Copy and paste the code below into a shell script to troubleshoot your webhook. Specify your own values for the following:

* `ACCESS_TOKEN`
* `SECRET`
* `PAYLOAD`
* `API_ENDPOINT`

{{< prism file="/webhook_test.sh" title="webhook_test.sh">}}{{< /prism >}}

{{% /tab %}}
{{< /tabpane >}}

## View and manage automations

1. Log in to W&B and go to the project page.
1. In the sidebar, click **Automations**.
1. To view an automation's details, click the action `...` menu and click **View details**.
1. To delete an automation, click the action `...` menu and click **Delete automation**.

    After you delete an automation, go to the **Settings** page to delete any Slack integrations, webhooks, or secrets that are no longer required.