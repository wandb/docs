---
description: Use an project scoped artifact automation in your project to trigger actions when aliases or versions in an artifact collection are created or changed. 
title: Artifact automations
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';



# Trigger CI/CD events with artifact changes


Create an automation that triggers when an artifact is changed. Use artifact automations when you want to automate downstream actions for versioning artifacts that are not of type "model".

For example, you can create an automation that triggers a [launch job](../launch/intro.md) when a new artifact version is added. 

To create an automation, define the [action](#action-types) you want to occur based on an [event type](#event-types).  

:::info
Artifact automations are scoped to a project. This means that only events within a project will trigger an artifact automation.

This is in contrast to automations created in the W&B Model Registry. Automations created in the model registry are in scope of the Model Registry; they are triggered when events are performed on model versions linked to the [Model Registry](../model_registry/intro.md). For information on how to create an automations for model versions, see the [Automations for Model CI/CD](../model_registry/automation.md) page in the [Model Registry chapter](../model_registry/intro.md).
:::


## Event types
An *event* is a change that takes place in the W&B ecosystem. You can define two different event types for artifact collections in your project: **A new version of an artifact is added in a collection** and **An artifact alias is created**.

:::tip
Use the **A new version of an artifact is added in a collection** event type for applying recurring actions to each version of an artifact. For example, you can create an automation that automatically starts a training job when a new dataset artifact version is created.

Use the **An artifact alias is added** event type to create an automation that activates when a specific alias is applied to an artifact version. For example, you could create an automation that triggers an action when someone adds "test-set-quality-check" alias to an artifact that then triggers downstream processing on that dataset. 
:::



## Action types
An action is a responsive mutation (internal or external) that occurs as a result of some trigger. There are two types of actions you can create in response to events on artifact collections in your project: webhooks and [W&B Launch Jobs](../launch/intro.md).

* Webhooks: Communicate with an external web server from W&B with HTTP requests.
* W&B Launch job: [Jobs](../launch/create-launch-job.md) are reusable, configurable run templates that allow you to quickly launch new [runs](../runs/intro.md) locally on your desktop or external compute resources such as Kubernetes on EKS, Amazon SageMaker, and more. 


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
![](/images/artifacts/artifact_webhook_select_event.png)
2. If you selected **A new version of an artifact is created in a collection** event, provide the name of the artifact collection that the automation should respond to from the **Artifact collection** dropdown. 
![](/images/artifacts/webhook_new_version_artifact.png)
3. Select **Webhooks** from the **Action type** dropdown. 
4. Click on the **Next step** button.
5. Select a webhook from the **Webhook** dropdown.
![](/images/artifacts/artifacts_webhooks_select_from_dropdown.png)
6. (Optional) Provide a payload in the JSON expression editor. See the [Example payload](#example-payloads) section for common use case examples.
7. Click on **Next step**.
8. Provide a name for your webhook automation in the **Automation name** field. 
![](/images/artifacts/artifacts_webhook_name_automation.png)
9. (Optional) Provide a description for your webhook. 
10. Click on the **Create automation** button.


<!-- INSERT -->

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
      types: ADD_ARTIFACT_ALIAS
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

  Where template strings render depending on the event or artifact version the automation is configured for. `${event_type}` will render as an "LINK_ARTIFACT" or "ADD_ARTIFACT_ALIAS". See below for an example mapping:

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

## View an automation

View automations associated to an artifact from the W&B App UI. 

1. Navigate to your project workspace on the W&B App. 
2. Click on the **Automations** tab on the left sidebar.

![](/images/artifacts/automations_sidebar.gif)

Within the Automations section you can find the following properties for each automations that was created in your project"

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


