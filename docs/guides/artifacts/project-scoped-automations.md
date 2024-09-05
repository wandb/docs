---
description: Use an project scoped artifact automation in your project to trigger actions when aliases or versions in an artifact collection are created or changed. 
title: Artifact automations
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


Use artifact automations when you want CI/CD actions to trigger when an artifact is changed.

## Trigger CI/CD Events with Artifact Changes

To create an automation, define the desired [action](#action-types) based on an [event type](#event-types).

Common use cases for artifact-triggered automations include:

- When a new version of an evaluation or holdout dataset is uploaded, [trigger a launch job](#create-a-launch-automation) to perform inference using the best training model in the model registry and generate a performance report.
- When a new version of the training dataset is marked "production," [trigger a retraining job](#create-a-launch-automation) using the configurations from the current best-performing model.

:::info
Artifact automations are scoped to a project. Only events within a project can trigger an artifact automation.

Automations in the W&B Model Registry, however, are scoped to the Model Registry and are triggered by events on model versions linked to it. For more details, see [Automations for Model CI/CD](../model_registry/model-registry-automations.md) in the [Model Registry chapter](../model_registry/intro.md).
:::

## Event Types

An *event* is a change that occurs in the W&B ecosystem. You can define two event types for artifact collections in your project:

- **A new version of an artifact is created in a collection**: Use this to apply recurring actions to each new version of an artifact, like starting a training job when a new dataset artifact version is created.
  
- **An artifact alias is added**: Use this to trigger an action when a specific alias is applied to an artifact version, such as adding the alias "test-set-quality-check" to trigger downstream processing on that dataset.

## Action Types

An *action* is a response triggered by an event. There are two types of actions you can create for events on artifact collections:

- **Webhooks**: Send HTTP requests to an external web server from W&B.
- **W&B Launch Jobs**: [Jobs](../launch/create-launch-job.md) are reusable, configurable templates that allow you to launch new [runs](../runs/intro.md) locally or on external compute resources like Kubernetes, Amazon SageMaker, etc.

The following sections explain how to create an automation using webhooks or W&B Launch Jobs.

## Create a Webhook Automation

To automate a webhook, first create a webhook in the W&B App UI, then configure the webhook automation.

:::info
Ensure the webhook endpoint has an Address record (A record). W&B does not support endpoints exposed directly by IP addresses (e.g., `[0-255].[0-255].[0-255].[0.255]`) or `localhost`. This restriction protects against server-side request forgery (SSRF) attacks and similar threats.
:::

### Add a Secret for Authentication or Authorization

Secrets are team-level variables used to securely store sensitive data like credentials, API keys, passwords, and tokens. W&B recommends using secrets for any string that needs to be protected.

To use a secret in your webhook, add it to your team's secret manager.

:::info
- Only W&B Admins can create, edit, or delete secrets.
- Skip this section if your external server doesn't use secrets.
- Secrets are available for deployments on Azure, GCP, AWS, and more. Contact your W&B account team to learn how to use secrets with other deployment types.
:::

W&B recommends creating two types of secrets for webhook automation:

- **Access Tokens**: Authorize senders to secure webhook requests.
- **Secrets**: Ensure the authenticity and integrity of transmitted data.

To create a webhook:

1. Go to the W&B App UI.
2. Click **Team Settings**.
3. Scroll to the **Team secrets** section.
4. Click **New secret**.
5. In the modal, enter a name in the **Secret name** field.
6. Enter your secret in the **Secret** field.
7. (Optional) Repeat steps 5 and 6 to add additional secrets (such as access tokens).

Specify the secrets for your webhook automation when configuring the webhook. See the [Configure a Webhook](#configure-a-webhook) section for details.

:::tip
You can access secrets in your W&B workflows using `$`.
:::

### Configure a Webhook

To use a webhook, first configure it in the W&B App UI.

:::info
* Only W&B Admins can configure webhooks for a W&B Team.
* If your webhook requires secret keys or tokens, [create one or more secrets](#add-a-secret-for-authentication-or-authorization) beforehand.
:::

1. Go to the W&B App UI.
2. Click **Team Settings**.
3. Scroll to the **Webhooks** section.
4. Click **New webhook**.
5. Enter a name in the **Name** field.
6. Enter the endpoint URL in the **URL** field.
7. (Optional) Select a secret from the **Secret** dropdown for webhook authentication.
8. (Optional) Select an access token from the **Access token** dropdown for authorization.
9. (Optional) Select additional secret keys or tokens from the **Access token** dropdown if needed.

:::note
Refer to the [Troubleshoot your webhook](#troubleshoot-your-webhook) section to see where secrets and access tokens are specified in the POST request.
:::

### Add a Webhook

Once your webhook is configured, navigate to your project workspace and click the **Automations** tab in the left sidebar.

1. Choose an [event type](#event-types) from the **Event type** dropdown.
   
   ![](/images/artifacts/artifact_webhook_select_event.png)
   
2. If you selected **A new version of an artifact is created in a collection**, choose the artifact collection from the **Artifact collection** dropdown.

   ![](/images/artifacts/webhook_new_version_artifact.png)

3. Choose **Webhooks** from the **Action type** dropdown.
4. Click **Next step**.
5. Select a webhook from the **Webhook** dropdown.

   ![](/images/artifacts/artifacts_webhooks_select_from_dropdown.png)

6. (Optional) Enter a payload in the JSON expression editor. See [Example payloads](#example-payloads) for common examples.
7. Click **Next step**.
8. Enter a name for the automation in the **Automation name** field.

   ![](/images/artifacts/artifacts_webhook_name_automation.png)

9. (Optional) Add a description for your webhook.
10. Click **Create automation**.

### Example Payloads

The following examples show payloads for common use cases. The examples reference the following keys in payload parameters:

* `${event_type}`: The type of event triggering the action.
* `${event_author}`: The user triggering the action.
* `${artifact_version}`: The specific artifact version triggering the action, passed as an artifact instance.
* `${artifact_version_string}`: The specific artifact version triggering the action, passed as a string.
* `${artifact_collection_name}`: The name of the artifact collection linked to the artifact version.
* `${project_name}`: The project name associated with the triggering action.
* `${entity_name}`: The entity name associated with the triggering action.


<Tabs
  defaultValue="github"
  values={[
    {label: 'GitHub repository dispatch', value: 'github'},
    {label: 'Microsoft Teams notification', value: 'microsoft'},
    {label: 'Slack notifications', value: 'slack'},
  ]}>
  <TabItem value="github">

:::info
Verify that your access tokens have required set of permissions to trigger your GHA workflow. For more information, [see these GitHub Docs](https://docs.github.com/en/rest/repos/repos?#create-a-repository-dispatch-event). 
:::

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

:::note
The `event_type` key in the webhook payload must match the `types` field in the GitHub workflow YAML file.
:::

  The contents and positioning of rendered template strings depends on the event or model version the automation is configured for. `${event_type}` will render as either "LINK_ARTIFACT" or "ADD_ARTIFACT_ALIAS". See below for an example mapping:

  ```json
  ${event_type} --> "LINK_ARTIFACT" or "ADD_ARTIFACT_ALIAS"
  ${event_author} --> "<wandb-user>"
  ${artifact_version} --> "wandb-artifact://_id/QXJ0aWZhY3Q6NTE3ODg5ODg3""
  ${artifact_version_string} --> "<entity>/<project_name>/<artifact_name>:<alias>"
  ${artifact_collection_name} --> "<artifact_collection_name>"
  ${project_name} --> "<project_name>"
  ${entity_name} --> "<entity>"
  ```

  Use template strings to dynamically pass context from W&B to GitHub Actions and other tools. If those tools can call Python scripts, they can consume W&B artifacts through the [W&B API](../artifacts/download-and-use-an-artifact.md).

  For more information about repository dispatch, see the [official documentation on the GitHub Marketplace](https://github.com/marketplace/actions/repository-dispatch).  

  </TabItem>
  <TabItem value="microsoft">

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


  </TabItem>
  <TabItem value="slack">

  Setup your Slack app and add an incoming webhook integration with the instructions highlighted in the [Slack API documentation](https://api.slack.com/messaging/webhooks). Ensure that you have the secret specified under `Bot User OAuth Toke`n as your W&B webhook’s access token. 
  
  The following is an example payload:

  ```json
    {
        "text": "New alert from WANDB!",
    "blocks": [
        {
                "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "Artifact event: ${event_type}"
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

  </TabItem>
</Tabs>

### Troubleshoot Your Webhook

You can troubleshoot your webhook using the W&B App UI or a Bash script. This can be done when creating a new webhook or editing an existing one.

<Tabs
  defaultValue="app"
  values={[
    {label: 'W&B App UI', value: 'app'},
    {label: 'Bash script', value: 'bash'},
  ]}>
  <TabItem value="app">

## Test a Webhook Interactively

To test a webhook using the W&B App UI:

1. Go to your W&B Team Settings page.
2. Scroll to the **Webhooks** section.
3. Click the kebab menu (three vertical dots) next to your webhook.
4. Select **Test**.
5. In the panel that appears, paste your POST request into the provided field.
   ![Webhook test screen](/images/models/webhook_ui.png)
6. Click **Test webhook**.

W&B will display the response from your endpoint.

![Webhooks testing screen](/images/models/webhook_ui_testing.gif)

For a real-world example, watch the [Testing Webhooks in Weights & Biases](https://www.youtube.com/watch?v=bl44fDpMGJw&ab_channel=Weights%26Biases) video.

  </TabItem>
  <TabItem value="bash">

The following bash script generates a POST request similar to the POST request W&B sends to your webhook automation when it is triggered.

Copy and paste the code below into a shell script to troubleshoot your webhook. Specify your own values for the following:

* `ACCESS_TOKEN`
* `SECRET`
* `PAYLOAD`
* `API_ENDPOINT`


```sh title="webhook_test.sh"
#!/bin/bash

# Your access token and secret
ACCESS_TOKEN="your_api_key" 
SECRET="your_api_secret"

# The data you want to send (for example, in JSON format)
PAYLOAD='{"key1": "value1", "key2": "value2"}'

# Generate the HMAC signature
# For security, Wandb includes the X-Wandb-Signature in the header computed 
# from the payload and the shared secret key associated with the webhook 
# using the HMAC with SHA-256 algorithm.
SIGNATURE=$(echo -n "$PAYLOAD" | openssl dgst -sha256 -hmac "$SECRET" -binary | base64)

# Make the cURL request
curl -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "X-Wandb-Signature: $SIGNATURE" \
  -d "$PAYLOAD" API_ENDPOINT
```

  </TabItem>
</Tabs>


## Create a Launch Automation

To automatically start a W&B job:

:::info
Ensure you have created a job, a queue, and have an active agent polling. For details, see the [W&B Launch docs](../launch/intro.md).
:::

1. Select an event type from the **Event type** dropdown. Refer to the [Event Types](#event-types) section for supported events.
2. (Optional) If you chose **A new version of an artifact is created in a collection**, select the artifact collection from the **Artifact collection** dropdown.
3. Choose **Jobs** from the **Action type** dropdown.
4. Click **Next step**.
5. Select a W&B Launch job from the **Job** dropdown.
6. Choose a version from the **Job version** dropdown.
7. (Optional) Provide hyperparameter overrides.
8. Select a project from the **Destination project** dropdown.
9. Choose a queue for your job.
10. Click **Next step**.
11. Enter a name for the automation in the **Automation name** field.
12. (Optional) Add a description.
13. Click **Create automation**.

## Viewing an Automation

To view automations associated with an artifact:

1. Open your project workspace in the W&B App.
2. Click the **Automations** tab on the left sidebar.

![Automations Sidebar](/images/artifacts/automations_sidebar.gif)

In the Automations section, you can see these properties for each automation:

- **Trigger Type**: The configured trigger type.
- **Action Type**: The action that triggers the automation, such as Webhooks or Launch.
- **Action Name**: The name given to the action when the automation was created.
- **Queue**: The name of the queue for the job. This is empty if a webhook action type was selected.


## Delete an Automation

To delete an automation associated with an artifact:

1. Open your project workspace in the W&B App.
2. Click the **Automations** tab on the left sidebar.
3. Select the automation you wish to delete.
4. Hover over the automation name and click the kebab menu (three vertical dots).
5. Choose **Delete**.


