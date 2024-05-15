---
description: Use an Automation to retrain models as new data comes in
title: Artifact automations
displayed_sidebar: ja
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';



# Artifact automations

Create an automation to trigger workflow steps. To create an automation, define the [action](#action-types) you want to occur based on an [event type](#event-types).

For example, you can create a trigger that automatically retrains a model when a dataset is updated. 

## Event types
An *event* is a change that takes place in the W&B ecosystem. You can define two different event types: **A new version of an artifact is added in a collection** and **An artifact alias is created**.


:::tip
Use the [INSERT].

Use the **A new version of an artifact is added in a collection** event type to specify an alias that represents a special step of your workflow, such as `staging`.
:::


## Action types
An action is a responsive mutation (internal or external) that occurs as a result of some trigger. There are two types of actions you can create for artifacts: webhooks and [W&B Launch Jobs](../launch/intro.md).

* Webhooks: Communicate with an external web server from W&B with HTTP requests.
* W&B Launch job: [Jobs](../launch/create-job.md) are reusable, configurable run templates that allow you to quickly launch new [runs](../runs/intro.md) locally on your desktop or external compute resources such as Kubernetes on EKS, Amazon SageMaker, and more. 

<!-- :::tip
Question: When should I use a webhook as opposed to a W&B Launch job? Answer: [INSERT]
::: -->

The following sections describe how to create an automation with webhooks and W&B Launch.

## Create a webhook automation 
Automate a webhook based on an action with the W&B App UI. To do this, you will first establish a webhook, then you will configure the webhook automation. 

### Add a secret for authentication
Define a team secret to ensure the authenticity and integrity of data transmitted from payloads. 

:::note
* Secrets are available if you use:
  * W&B SaaS public cloud; or
  * W&B Server in a Kubernetes cluster
* Skip this section if the external server you send HTTP POST requests to does not use secrets.  
:::


1. Navigate to the W&B App UI.
2. Click on **Team Settings**.
3. Scroll down the page until you find the **Team secrets** section.
4. Click on the **New secret** button.
5. A modal will appear. Provide a name for your secret in the **Secret name** field.
6. Add your secret into the **Secret** field. 

:::info
Only W&B Admins can create, edit, or delete a secret.
:::

Once you create a secret, you can access that secret in your W&B workflows with `$`.

### Configure a webhook
Before you can use a webhook, you will first need to configure that webhook in the W&B App UI.

:::info
Only W&B Admins can configure a webhook for a W&B Team.
:::

1. Navigate to the W&B App UI.
2. Click on **Team Settings**.
4. Scroll down the page until you find the **Webhooks** section.
5. Click on the **New webhook** button.  
6. Provide a name for your webhook in the **Name** field.
7. Provide the endpoint URL for the webhook in the **URL** field.


### Add a webhook 
Once you have a webhook configured and (optionally) a secret, navigate to your project workspace. Click on the **Automations** tab on the left sidebar.

1. From the **Event type** dropdown, select an [event type](#event-types).
2. (Optional) If you selected **A new version of an artifacts is created in a collection** event, provide the name of a registered model from the **Artifact collection** dropdown. 
3. Select **Webhooks** from the **Action type** dropdown. 
4. Click on the **Next step** button.
5. Select a webhook from the **Webhook** dropdown.
6. (Optional) Provide a payload in the JSON expression editor. See the [Example payload](#example-payloads) section for common use case examples.
7. Click on **Next step**.
8. Provide a name for your webhook automation in the **Automation name** field. 
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

  
  Send a repository dispatch from W&B to trigger a GitHub action. For example, suppose you have workflow that accepts a repository dispatch as a trigger for the `on` key:

  ```yaml
  on:
    repository_dispatch:
      types: LINK_ARTIFACT
  ```

  The payload for the repository might look something like:

  ```json
  {
    "event_type": "${event_type}",
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

  Where template strings render depending on the event or artifact version the automation is configured for. `${event_type}` will render as either "LINK_ARTIFACT" or "ADD_ARTIFACT_ALIAS". See below for an example mapping:

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

## Create a launch automation
Automatically start a W&B Job. 

:::info
This section assumes you already have created a job, a queue, and have an active agent polling. For more information, see the [W&B Launch docs](../launch/intro.md). 
:::


1. From the **Event type** dropdown, select an event type. See the [Event type](#event-types) section for information on supported events.
2. (Optional) If you selected **A new version of an artifact is added in a collection** event, provide the name of the artifact collection from the **Artifact collection** dropdown. 
3. Select **Jobs** from the **Action type** dropdown. 
4. Click **Next step**.
4. Select a W&B Launch job from the **Job** dropdown.  
5. Select a version from the **Job version** dropdown.
6. (Optional) Provide hyperparameter overrides for the new job.
7. Select a project from the **Destination project** dropdown.
8. Select a queue to enqueue your job to.  
9. Click on **Next step**.
10. Provide a name for your webhook automation in the **Automation name** field. 
11. (Optional) Provide a description for your webhook. 
12. Click on the **Create automation** button.


## View automation

View automations associated to an artifact from the W&B App UI. 

1. Navigate to your project workspace on the W&B App. 
2. Click on the **Automations** tab on the left sidebar.
3. From the list, select the name of the automation you want to view.

Within the Automations section you can find the following properties of automations created for the artifact you selected:

- **Trigger type**: The type of trigger that was configured.
- **Action type**: The action type that triggers the automation. Available options are Webhooks and Launch.
- **Action name**: The action name you provided when you created the automation.
- **Queue**: The name of the queue the job was enqueued to. This field is left empty if you selected a webhook action type.

## Delete an automation
Delete an automation associated with a artifact. Actions in progress are not affected if you delete that automation before the action completes. 

1. Navigate to your project workspace on the W&B App. 
2. Click on the **Automations** tab on the left sidebar.
3. From the list, select the name of the automation you want to view.
4. Hover your mouse next to the name of the automation and click on the kebob (three vertical dots) menu. 
5. Select **Delete**.

