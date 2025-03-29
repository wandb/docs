---
menu:
  default:
    identifier: create-webhook-automations
    parent: automations
title: Create a webhook automation
weight: 3
---

This page shows how to create a webhook [automation]({{< relref "/guides/core/automations/" >}}> ). To create a Slack automation, refer to [Create a Slack automation]({{< relref "/guides/core/automations/create-automations/slack.md" >}}) instead.

At a high level, to create a webhook automation, you take these steps:
1. If necessary, [create a W&B secret]({{< relref "/guides/core/secrets.md" >}}) for each sensitive string required by the automation, such as an access token, password, or SSH key. Secrets are defined in your team settings.
1. [Create a webhook]({{< relref "#create-a-webhook" >}}) to define the endpoint and authorization details and grant the integration access to any secrets it needs.
1. [Create the automation]({{< relref "#create-an-automation" >}}) to define the [event]({{< relref "/guides/core/automations/automation-events.md" >}}) to watch for and the payload W&B will send. Grant the automation access to any secrets it needs for the payload.

## Create a webhook
A team admin can add a webhook for the team.

{{% alert %}}
If the webhook requires a Bearer token or its payload requires a sensitive string, [create a secret that contains it]({{< relref "/guides/core/secrets.md#add-a-secret" >}}) before creating the webhook. You can configure at most one access token and one other secret for a webhook. Your webhook's authentication and authorization requirements are determined by the webhook's service.
{{% /alert %}}

1. Log in to W&B and go to Team Settings page.
1. In the **Webhooks** section, click **New webhook**.
1. Provide a name for the webhook. 
1. Provide the endpoint URL for the webhook.
1. If the webhook requires a Bearer token, set **Access token** to the [secret]({{< relref "/guides/core/secrets.md" >}})that contains it. When running the webhook automation, W&B sets the `Authorization: Bearer` HTTP header to the access token, and you can access the token in the `${ACCESS_TOKEN}` [payload variable]({{< relref "#payload-variables" >}}).
1. If the webhook requires a password or other sensitive string in its payload, set **Secret** to the secret that contains it. When you configure the automation that uses the webhook, you can access the secret as a [payload variable]({{< relref "#payload-variables" >}}) by prefixing its name with `$`.

    If the webhook's access token is stored in a secret, you must _also_ complete the next step to specify the secret as the access token.
1. To verify that the W&B can connect and authenticate to the endpoint:
    1. Optionally, provide a payload to test. To refer to a secret the webhook has access to in the payload, prefix its name with `$`. This payload is only used for testing and is not saved. You configure an automation's payload when you [create the automation]({{< relref "#create-a-webhook-automation" >}}). See [Troubleshoot your webhook]({{< relref "#troubleshoot-your-webhook" >}}) to view where the secret and access token are specified in the `POST` request.
    1. Click **Test**. W&B attempts to connect to the webhook's endpoint using the credentials you configured. If you provided a payload, W&B sends it.

    If the test does not succeed, verify the webhook's configuration and try again. If necessary, refer to [Troubleshoot your webhook]({{< relref "#troubleshoot-your-webhook" >}}).

Now you can [create an automation]({{< relref "#create-a-webhook-automation" >}}) that uses the webhook.

## Create an automation
After you [configure a webhook]({{< relref "#reate-a-webhook" >}}), select **Registry** or **Project**, then follow these steps to create an automation that triggers the webhook.

{{< tabpane text=true >}}
{{% tab "Registry" %}}
A Registry admin can create automations in that registry. Registry automations are applied to all collections in the registry, including those added in the future.

1. Log in to W&B.
1. Click the name of a registry to view its details, 
1. To create an automation scoped to the registry, click the **Automations** tab, then click **Create automation**. An automation that is scoped to a registry is automatically applied to all of its collections (including those created in the future).

    To create an automation scoped only to a specific collection in the registry, click the collection's action `...` menu, then click **Create automation**. Alternatively, while viewing a collection, create an automation for it using the **Create automation** button in the **Automations** section of the collection's details page.
1. Choose the [**Event**]({{< relref "/guides/core/automations/automation-events.md" >}}) to watch for.

    1. Fill in any additional fields that appear, which depend upon the event. For example, if you select **An artifact alias is added**, you must specify the **Alias regex**.
    
    Click **Next step**.
1. Select the team that owns the [webhook]({{< relref "#create-a-webhook >}}).
1. Set **Action type** to **Webhooks**. then select the [webhook]({{< relref "#create-a-webhook" >}}) to use.
1. If you configured an access token for the webhook, you can access the token in the `${ACCESS_TOKEN}` [payload variable]({{< relref "#payload-variables" >}}). If you configured a secret for the webhook, you can access it in the payload by prefixing its name with `$`. Your webhook's requirements are determined by the webhook's service.
1. Click **Next step**.
1. Provide a name for the automation. Optionally, provide a description. Click **Create automation**.

{{% /tab %}}
{{% tab "Project" %}}
A W&B admin can create automations in a project.

1. Log in to W&B and go to the project page.
1. In the sidebar, click **Automations**.
1. Click **Create automation**.
1. Choose the [**Event**]({{< relref "/guides/core/automations/automation-events.md" >}}) to watch for.

    1. Fill in any additional fields that appear, which depend upon the event. For example, if you select **An artifact alias is added**, you must specify the **Alias regex**.

    1. Optionally specify a collection filter. Otherwise, the automation is applied to all collections in the project, including those added in the future.
    
    Click **Next step**.
1. Select the team that owns the [webhook]({{< relref "#create-a-webhook" >}}).
1. Set **Action type** to **Webhooks**. then select the [webhook]({{< relref "#create-a-webhook" >}}) to use. 
1. If your webhook requires a payload, construct it and paste it into the **Payload** field. If you configured an access token for the webhook, you can access the token in the `${ACCESS_TOKEN}` [payload variable]({{< relref "#payload-variables" >}}). If you configured a secret for the webhook, you can access it in the payload by prefixing its name with `$`. Your webhook's requirements are determined by the webhook's service.
1. Click **Next step**.
1. Provide a name for the automation. Optionally, provide a description. Click **Create automation**.

{{% /tab %}}
{{< /tabpane >}}


## View and manage automations
{{< tabpane text=true >}}
{{% tab "Registry" %}}

- Manage a registry's automations from the registry's **Automations** tab.
- Manage a collection's automations from the **Automations** section of the collection's details page.

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

## Payload reference
Use these sections to construct your webhoook's payload. For details about testing your webhook and its payload, refer to [Troubleshoot your webhook]({{< relref "#troubleshoot-your-webhook" >}}).

### Payload variables
This section describes the variables you can use to construct your webhook's payload.

| Variable | Details |
|----------|---------|
| `${project_name}`             | The name of the project that owns the mutation that triggered the action. |
| `${entity_name}`              | The name of the entity or team that owns the mutation that triggered the action.
| `${event_type}`               | The type of event that triggered the action. |
| `${event_author}`             | The user that triggered the action. |
| `${artifact_collection_name}` | The name of the artifact collection that the artifact version is linked to. |
| `${artifact_metadata.<KEY>}`  | The value of an arbitrary top-level metadata key from the artifact version that triggered the action. Replace `<KEY>` with the name of a top-level metadata key. Only top-level metadata keys are available in the webhook's payload. |
| `${artifact_version}`         | The [`Wandb.Artifact`]({{< relref "/ref/python/artifact/" >}}) representation of the artifact version that triggered the action. |
| `${artifact_version_string}` | The `string` representation of the artifact version that triggered the action. |
| `${ACCESS_TOKEN}` | The value of the access token configured in the [webhook]({{< relref "#create-a-webhook" >}}), if an access token is configured. The access token is automatically passed in the `Authorization: Bearer` HTTP header. |
| `${SECRET_NAME}` | If configured, the value of a secret configured in the [webhook]({{< relref "#create-a-webhook" >}}). Replace `SECRET_NAME` with the name of the secret. |

### Payload examples
This section includes examples of webhook payloads for some common use cases. The examples demonstrate how to use [payload variables]({{< relref "#payload-variables" >}}).

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

- For more information about repository dispatch, see the [official documentation on the GitHub Marketplace](https://github.com/marketplace/actions/repository-dispatch).

- Watch the videos [Webhook Automations for Model Evaluation](https://www.youtube.com/watch?v=7j-Mtbo-E74&ab_channel=Weights%26Biases) and [Webhook Automations for Model Deployment](https://www.youtube.com/watch?v=g5UiAFjM2nA&ab_channel=Weights%26Biases), which guide you to create automations for model evaluation and deployment. 

- Review a W&B [report](https://wandb.ai/wandb/wandb-model-cicd/reports/Model-CI-CD-with-W-B--Vmlldzo0OTcwNDQw), which illustrates how to use a Github Actions webhook automation for Model CI. Check out this [GitHub repository](https://github.com/hamelsmu/wandb-modal-webhook) to learn how to create model CI with a Modal Labs webhook. 

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

## Troubleshoot your webhook
Interactively troubleshoot your webhook with the W&B App UI or programmatically with a Bash script. You can troubleshoot a webhook when you create a new webhook or edit an existing webhook.

{{< tabpane text=true >}}
{{% tab header="W&B App UI" value="app" %}}

A team admin can test a webhook interactively with the W&B App UI. 

1. Navigate to your W&B Team Settings page.
2. Scroll to the **Webhooks** section.
3. Click on the horizontal three docs (meatball icon) next to the name of your webhook.
4. Select **Test**.
5. From the UI panel that appears, paste your POST request to the field that appears. 
    {{< img src="/images/models/webhook_ui.png" alt="Demo of testing a webhook payload" >}}
6. Click on **Test webhook**. Within the W&B App UI, W&B posts the response from your endpoint.
    {{< img src="/images/models/webhook_ui_testing.gif" alt="Demo of testing a webhook" >}}

Watch the video [Testing Webhooks in Weights & Biases](https://www.youtube.com/watch?v=bl44fDpMGJw&ab_channel=Weights%26Biases) for a demonstration.
{{% /tab %}}

{{% tab header="Bash script" value="bash"%}}

This shell script shows one method to generate a `POST` request similar to the request W&B sends to your webhook automation when it is triggered.

Copy and paste the code below into a shell script to troubleshoot your webhook. Specify your own values for:

* `ACCESS_TOKEN`
* `SECRET`
* `PAYLOAD`
* `API_ENDPOINT`

{{< prism file="/webhook_test.sh" title="webhook_test.sh">}}{{< /prism >}}

{{% /tab %}}
{{< /tabpane >}}