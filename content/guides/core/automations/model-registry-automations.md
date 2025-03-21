---
description: Use an Automation for model CI (automated model evaluation pipelines)
  and model deployment.
menu:
  default:
    identifier: model-registry-automations
    parent: automations
title: Model registry automations
url: guides/core/model_registry/model-registry-automations
---

Create an automation to trigger workflow steps, such as automated model testing and deployment. To create an automation, define the action you want to occur based on an [event type]({{< relref "#event-types" >}}).

For example, you can create a trigger that automatically deploys a model to GitHub when you add a new version of a registered model.

{{% alert %}}
Looking for companion tutorials for automations? 
1. [This](https://wandb.ai/wandb/wandb-model-cicd/reports/Model-CI-CD-with-W-B--Vmlldzo0OTcwNDQw) tutorial shows you how to set up an automation that triggers a Github Action for model evaluation and deployment
2. [This](https://youtube.com/playlist?list=PLD80i8An1OEGECFPgY-HPCNjXgGu-qGO6&feature=shared) video series shows webhook basics and how to set them up in W&B.
3. [This](https://www.youtube.com/watch?v=s5CMj_w3DaQ) demo details how to setup an automation to deploy a model to a Sagemaker Endpoint
{{% /alert %}}

## Event types
An *event* is a change that takes place in the W&B ecosystem. The Model Registry supports two event types:

- Use **Linking a new artifact to a registered model** to test new model candidates.
- Use **Adding a new alias to a version of the registered model** to specify an alias that represents a special step of your workflow, like `deploy`, and any time a new model version has that alias applied.

See [Link a model version]({{< relref "/guides/core/registry/link_version.md" >}}) and [Create a custom alias]({{< relref "/guides/core/artifacts/create-a-custom-alias.md" >}}).


## Create a webhook automation 
Automate a webhook based on an action with the W&B App UI. To do this, first establish a webhook, then configure the webhook automation. 

{{% alert %}}
Your webhook's endpoint must have a fully qualified domain name. W&B does not support connecting to an endpoint by IP address or by a hostname such as `localhost`. This restriction helps protect against server-side request forgery (SSRF) attacks and other related threat vectors.
{{% /alert %}}

### Add a secret for authentication or authorization
Secrets are team-level variables that let you obfuscate private strings such as credentials, API keys, passwords, tokens, and more. W&B recommends you use secrets to store any string that you want to protect the plain text content of.

To use a secret in your webhook, you must first add that secret to your team's secret manager.

{{% alert %}}
* Only W&B Admins can create, edit, or delete a secret.
* Skip this section if the external server you send HTTP POST requests to does not use secrets.  
* Secrets are also available if you use [W&B Server]({{< relref "/guides/hosting/" >}}) in an Azure, GCP, or AWS deployment. Connect with your W&B account team to discuss how you can use secrets in W&B if you use a different deployment type.
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

Specify the secrets you want to use for your webhook automation when you configure the webhook. See the [Configure a webhook]({{< relref "#configure-a-webhook" >}}) section for more information. 

{{% alert %}}
Once you create a secret, you can access that secret in your W&B workflows with `$`.
{{% /alert %}}

{{% alert color="secondary" %}}
If you use secrets in W&B Server, you are responsible for configuring security measures that satisfy your security needs. 

W&B strongly recommends that you store secrets in a W&B instance of a cloud secrets manager provided by AWS, GCP, or Azure. Secret managers provided by AWS, GCP, and Azure are configured with advanced security capabilities.

W&B does not recommend that you use a Kubernetes cluster as the backend of your secrets store. Consider a Kubernetes cluster only if you are not able to use a W&B instance of a cloud secrets manager (AWS, GCP, or Azure), and you understand how to prevent security vulnerabilities that can occur if you use a cluster.
{{% /alert %}}

### Configure a webhook
Before you can use a webhook, first configure that webhook in the W&B App UI. 

{{% alert %}}
* Only W&B Admins can configure a webhook for a W&B Team.
* Ensure you already [created one or more secrets]({{< relref "#add-a-secret-for-authentication-or-authorization" >}}) if your webhook requires additional secret keys or tokens to authenticate your webhook.
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
See the [Troubleshoot your webhook]({{< relref "#troubleshoot-your-webhook" >}}) section to view where the secret and access token are specified in
the POST request.
{{% /alert %}}


### Add a webhook 
Once you have a webhook configured and (optionally) a secret, navigate to the Model Registry App at [https://wandb.ai/registry/model](https://wandb.ai/registry/model).

1. From the **Event type** dropdown, select an [event type]({{< relref "#event-types" >}}).
{{< img src="/images/models/webhook_select_event.png" alt="" >}}
2. (Optional) If you selected **A new version is added to a registered model** event, provide the name of a registered model from the **Registered model** dropdown. 
{{< img src="/images/models/webhook_new_version_reg_model.png" alt="" >}}
3. Select **Webhooks** from the **Action type** dropdown. 
4. Click on the **Next step** button.
5. Select a webhook from the **Webhook** dropdown.
{{< img src="/images/models/webhooks_select_from_dropdown.png" alt="" >}}
6. (Optional) Provide a payload in the JSON expression editor. See the [Example payload]({{< relref "#example-payloads" >}}) section for common use case examples.
7. Click on **Next step**.
8. Provide a name for your webhook automation in the **Automation name** field. 
{{< img src="/images/models/webhook_name_automation.png" alt="" >}}
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

  ```json
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
     {{< img src="/images/models/webhook_ui.png" >}}
  6. Click on **Test webhook**.

  Within the W&B App UI, W&B posts the response made by your endpoint.

  {{< img src="/images/models/webhook_ui_testing.gif" alt="" >}}

  Watch the video [Testing webhooks in Weights & Biases](https://www.youtube.com/watch?v=bl44fDpMGJw&ab_channel=Weights%26Biases) for a real-world example.

{{% /tab %}}

{{% tab header="Bash script" value="bash" %}}

  The following bash script generates a POST request similar to the POST request W&B sends to your webhook automation when it is triggered.

  Copy and paste the code below into a shell script to troubleshoot your webhook. Specify your own values for the following:

  * `ACCESS_TOKEN`
  * `SECRET`
  * `PAYLOAD`
  * `API_ENDPOINT`

  ```sh { title = "webhook_test.sh" }
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

{{% /tab %}}
{{< /tabpane >}}


## View automation

View automations associated to a registered model from the W&B App UI. 

1. Navigate to the Model Registry App at [https://wandb.ai/registry/model](https://wandb.ai/registry/model).
2. Select on a registered model. 
3. Scroll to the bottom of the page to the **Automations** section.

Within the Automations section you can find the following properties of automations created for the model you selected:

- **Trigger type**: The type of trigger that was configured.
- **Action type**: The action type that triggers the automation. 
- **Action name**: The action name you provided when you created the automation.
- **Queue**: The name of the queue the job was enqueued to. This field is left empty if you selected a webhook action type.

## Delete an automation
Delete an automation associated with a model. Actions in progress are not affected if you delete that automation before the action completes. 

1. Navigate to the Model Registry App at [https://wandb.ai/registry/model](https://wandb.ai/registry/model).
2. Click on a registered model. 
3. Scroll to the bottom of the page to the **Automations** section.
4. Hover your mouse next to the name of the automation and click on the kebob (three vertical dots) menu. 
5. Select **Delete**.