---
description: Use an project scoped artifact automation in your project to trigger
  actions when aliases or versions in an artifact collection are created or changed.
menu:
  default:
    identifier: project-scoped-automations
    parent: automations
title: Trigger CI/CD events when artifact changes
url: guides/artifacts/project-scoped-automations
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

Create an automation that triggers when an artifact is changed. Use artifact automations when you want to automate downstream actions for versioning artifacts. To create an automation, define the action you want to occur based on an [event type](#event-types).  

{{% alert %}}
Artifact automations are scoped to a project. This means that only events within a project will trigger an artifact automation.

This is in contrast to automations created in the W&B Model Registry. Automations created in the model registry are in scope of the Model Registry; they are triggered when events are performed on model versions linked to the [Model Registry](../model_registry/intro.md). For information on how to create an automations for model versions, see the [Automations for Model CI/CD](../model_registry/model-registry-automations.md) page in the [Model Registry chapter](../model_registry/intro.md).
{{% /alert %}}


## Event types
An *event* is a change that takes place in the W&B ecosystem. You can define two different event types for artifact collections in your project: **A new version of an artifact is created in a collection** and **An artifact alias is added**.

{{% alert %}}
Use the **A new version of an artifact is created in a collection** event type for applying recurring actions to each version of an artifact. For example, you can create an automation that automatically starts a training job when a new dataset artifact version is created.

Use the **An artifact alias is added** event type to create an automation that activates when a specific alias is applied to an artifact version. For example, you could create an automation that triggers an action when someone adds "test-set-quality-check" alias to an artifact that then triggers downstream processing on that dataset. 
{{% /alert %}}

## Create a webhook automation 
Automate a webhook based on an action with the W&B App UI. To do this, you will first establish a webhook, then you will configure the webhook automation. 

{{% alert %}}
Specify an endpoint for your webhook that has an Address record (A record). W&B does not support connecting to endpoints that are exposed directly with IP addresses such as `[0-255].[0-255].[0-255].[0.255]` or endpoints exposed as `localhost`. This restriction helps protect against server-side request forgery (SSRF) attacks and other related threat vectors.
{{% /alert %}}

### Add a secret for authentication or authorization
Secrets are team-level variables that let you obfuscate private strings such as credentials, API keys, passwords, tokens, and more. W&B recommends you use secrets to store any string that you want to protect the plain text content of.

To use a secret in your webhook, you must first add that secret to your team's secret manager.

{{% alert %}}
* Only W&B Admins can create, edit, or delete a secret.
* Skip this section if the external server you send HTTP POST requests to does not use secrets.
* Secrets are also available if you use [W&B Server](../hosting/intro.md) in an Azure, GCP, or AWS deployment. Connect with your W&B account team to discuss how you can use secrets in W&B if you use a different deployment type.
{{% /alert %}}



There are two types of secrets W&B suggests that you create when you use a webhook automation:

* **Access tokens**: Authorize senders to help secure webhook requests 
* **Secret**: Ensure the authenticity and integrity of data transmitted from payloads

Follow the instructions below to create a webhook:

1. Navigate to the W&B App UI.
2. Click on **Team Settings**.
3. Scroll down the page until you find the **Team secrets** section.
4. Click on the **New secret** button.
5. A modal will appear. Provide a name for your secret in the **Secret name** field.
6. Add your secret into the **Secret** field. 
7. (Optional) Repeat steps 5 and 6 to create another secret (such as an access token) if your webhook requires additional secret keys or tokens to authenticate your webhook.

Specify the secrets you want to use for your webhook automation when you configure the webhook. See the [Configure a webhook](#configure-a-webhook) section for more information. 

{{% alert %}}
Once you create a secret, you can access that secret in your W&B workflows with `$`.
{{% /alert %}}

### Configure a webhook
Before you can use a webhook, you will first need to configure that webhook in the W&B App UI.

{{% alert %}}
* Only W&B Admins can configure a webhook for a W&B Team.
* Ensure you already [created one or more secrets](#add-a-secret-for-authentication-or-authorization) if your webhook requires additional secret keys or tokens to authenticate your webhook.
{{% /alert %}}

1. Navigate to the W&B App UI.
2. Click on **Team Settings**.
4. Scroll down the page until you find the **Webhooks** section.
5. Click on the **New webhook** button.  
6. Provide a name for your webhook in the **Name** field.
7. Provide the endpoint URL for the webhook in the **URL** field.
8. (Optional) From the **Secret** dropdown menu, select the secret you want to use to authenticate the webhook payload.
9. (Optional) From the **Access token** dropdown menu, select the access token you want to use to authorize the sender.
9. (Optional) From the **Access token** dropdown menu select additional secret keys or tokens required to authenticate a webhook  (such as an access token).

{{% alert %}}
See the [Troubleshoot your webhook](#troubleshoot-your-webhook) section to view where the secret and access token are specified in
the POST request.
{{% /alert %}}
### Add a webhook 
Once you have a webhook configured and (optionally) a secret, navigate to your project workspace. Click on the **Automations** tab on the left sidebar.

1. From the **Event type** dropdown, select an [event type](#event-types).
{{< img src="/images/artifacts/artifact_webhook_select_event.png" alt="" >}}
2. If you selected **A new version of an artifact is created in a collection** event, provide the name of the artifact collection that the automation should respond to from the **Artifact collection** dropdown. 
{{< img src="/images/artifacts/webhook_new_version_artifact.png" alt="" >}}
3. Select **Webhooks** from the **Action type** dropdown. 
4. Click on the **Next step** button.
5. Select a webhook from the **Webhook** dropdown.
{{< img src="/images/artifacts/artifacts_webhooks_select_from_dropdown.png" alt="" >}}
6. (Optional) Provide a payload in the JSON expression editor. See the [Example payload](#example-payloads) section for common use case examples.
7. Click on **Next step**.
8. Provide a name for your webhook automation in the **Automation name** field. 
{{< img src="/images/artifacts/artifacts_webhook_name_automation.png" alt="" >}}
9. (Optional) Provide a description for your webhook. 
10. Click on the **Create automation** button.

### Example payloads

The following tabs demonstrate example payloads based on common use cases. Within the examples they reference the following keys to refer to condition objects in the payload parameters:
* `${event_type}` Refers to the type of event that triggered the action.
* `${event_author}` Refers to the user that triggered the action.
* `${artifact_version}` Refers to the specific artifact version that triggered the action. Passed as an artifact instance.
* `${artifact_version_string}` Refers to the specific artifact version that triggered the action. Passed as a string.
* `${artifact_collection_name}` Refers to the name of the artifact collection that the artifact version is linked to.
* `${project_name}` Refers to the name of the project owning the mutation that triggered the action.
* `${entity_name}` Refers to the name of the entity owning the mutation that triggered the action.


<Tabs
  defaultValue="github"
  values={[
    {label: 'GitHub repository dispatch', value: 'github'},
    {label: 'Microsoft Teams notification', value: 'microsoft'},
    {label: 'Slack notifications', value: 'slack'},
  ]}>
  <TabItem value="github">

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

### Troubleshoot your webhook

Interactively troubleshoot your webhook with the W&B App UI or programmatically with a Bash script. You can troubleshoot a webhook when you create a new webhook or edit an existing webhook.

<Tabs
  defaultValue="app"
  values={[
    {label: 'W&B App UI', value: 'app'},
    {label: 'Bash script', value: 'bash'},
  ]}>
  <TabItem value="app">

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

  </TabItem>
  <TabItem value="bash">

The following bash script generates a POST request similar to the POST request W&B sends to your webhook automation when it is triggered.

Copy and paste the code below into a shell script to troubleshoot your webhook. Specify your own values for the following:

* `ACCESS_TOKEN`
* `SECRET`
* `PAYLOAD`
* `API_ENDPOINT`

{{< prism file="/webhook_test.sh" title="webhook_test.sh">}}{{< /prism >}}

  </TabItem>
</Tabs>

## View an automation

View automations associated to an artifact from the W&B App UI. 

1. Navigate to your project workspace on the W&B App. 
2. Click on the **Automations** tab on the left sidebar.

{{< img src="/images/artifacts/automations_sidebar.gif" alt="" >}}

Within the Automations section you can find the following properties for each automations that was created in your project"

- **Trigger type**: The type of trigger that was configured.
- **Action type**: The action type that triggers the automation.
- **Action name**: The action name you provided when you created the automation.
- **Queue**: The name of the queue the job was enqueued to. This field is left empty if you selected a webhook action type.

## Delete an automation
Delete an automation associated with a artifact. Actions in progress are not affected if you delete that automation before the action completes. 

1. Navigate to your project workspace on the W&B App. 
2. Click on the **Automations** tab on the left sidebar.
3. From the list, select the name of the automation you want to view.
4. Hover your mouse next to the name of the automation and click on the kebob (three vertical dots) menu. 
5. Select **Delete**.