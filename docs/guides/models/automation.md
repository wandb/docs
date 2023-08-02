---
description: Use an Automation for model CI (automated model evaluation pipelines) and model deployment.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';



# Model CI/CD

Create an automation to trigger workflow steps, such as automated model testing and deployment. To create an automation, define the [action](#action-types) you want to occur based on an [event type](#event-types).

For example, you can create a trigger that automatically deploys a model to GitHub when you add a new version of a registered model.

## Event types
An *event* is a change that takes place in the W&B ecosystem. The Model Registry supports two event types: **Linking a new artifact to a registered model** and **Adding a new alias to a version of the registered model**.

:::tip
Use the **Linking a new artifact to a registered model** event type to test new model candidates. Use the **Adding a new alias to a version of the registered model** event type to specify an alias that represents a special step of your workflow, like `deploy`, and any time a new model version has that alias applied.
:::


## Action types
An action is a responsive mutation (internal or external) that occurs as a result of some trigger. There are two types of actions you can create in the Model Registry: webhooks and [W&B Launch Jobs](../launch/intro.md).

* Webhooks: Communicate with an external web server from W&B with HTTP requests.
* W&B Launch job: [Jobs](../launch/create-job.md) are reusable, configurable run templates that allow you to quickly launch new [runs](../runs/intro.md) locally on your desktop or external compute resources such as Kubernetes on EKS, Amazon SageMaker, and more. 

:::tip
Question: When should I use a webhook as opposed to a W&B Launch job? Answer: [INSERT]
:::

The following sections describe how to create webhook or launch automation.

## Create a webhook automation 
Automate a webhook based on an action with the W&B App UI. To do this, you will first establish a webhook, then you will configure the webhook automation.

### Establish a webhook
Configure a webhook in the W&B App UI.
1. Navigate to the W&B App UI.
2. Click on **Team Settings**.
4. Scroll down the page until you find the **Webhooks** section.
5. Click on the **New webhook** button.  
6. Provide a name for your webhook in the **Name** field.
7. Provide the endpoint URL for the webhook in the **URL** field.

### Add a secret for authentication
Define a secret to ensure the authenticity and integrity of data transmitted from payloads. Skip this section if the external server you will send the HTTP POST request does not have secrets.

1. Navigate to the W&B App UI.
2. Click on **Team Settings**.
3. Scroll down the page until you find the **Team secrets** section.
4. Click on the **New secret** button.
5. A modal will appear. Provide a name for your secret in the **Secret name** field.
6. Add your secret into the **Secret** field. 

Once you create a secret, you can access that secret in your W&B workflows with `$`.

### Configure a webhook 
Once you have a webhook, navigate to the Model Registry App at [https://wandb.ai/registry/model](https://wandb.ai/registry/model).

1. From the **Event type** dropdown, select an [event type](#event-types).
![](/images/models/webhook_select_event.png)
2. (Optional) If you selected **A new version is added to a registered model** event, provide the name of a registered model from the **Registered model** dropdown. 
![](/images/models/webhook_new_version_reg_model.png)
3. Select **Webhooks** from the **Action type** dropdown. 
4. Click on the **Next step** button.
5. Select a webhook from the **Webhook** dropdown.
![](/images/models/webhooks_select_from_dropdown.png)
6. (Optional) Provide a payload in the JSON expression editor. See the [Example payload](#example-payloads) section for some common examples. 
7. Click on **Next step**.
8. Provide a name for your webhook automation in the **Automation name** field. 
![](/images/models/webhook_name_automation.png)
9. (Optional) Provide a description for your webhook. 
10. Click on the **Create automation** button.

### Example payloads
The following tabs demonstrate example payloads based on common use case. 


<Tabs
  defaultValue="github"
  values={[
    {label: 'GitHub repository dispatch', value: 'github'},
    {label: 'Microsoft Teams Notification', value: 'microsoft'},
    {label: 'Slack notifications', value: 'slack'},
  ]}>
  <TabItem value="github">

  
  Send a repository dispatch from W&B to trigger a GitHub action. For example, suppose you have workflow that accepts a repository dispatch as a trigger for the `on` key:

  ```yaml
  on:
    repository_dispatch:
  ```

  The payload for the repository might look something like:

  ```json
{
	"event_type": "my_event_type",
	"client_payload": {
		"champion": "Nature Classification:production",
	  "challenger": "Nature Classification:staging"
		}
}
  ```

  Where the keys are defined as:
  * `event-type`: A custom webhook event name.
  * `client-payload`: JSON payload with extra information about the webhook event that your action or workflow may use.
  * `Champion`:
  * `Challenger`: 
  

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

  </TabItem>
</Tabs>




## Create a launch automation
Automatically start a W&B Job. 

:::info
This section assumes you already have created a job, a queue, and have an active agent polling. For more information, see the [W&B Launch docs](../launch/intro.md). 
:::


1. From the **Event type** dropdown, select an event type. See the [Event type](link to future doc) section for information on supported events.
2. (Optional) If you selected **A new version is added to a registered model** event, provide the name of a registered model from the **Registered model** dropdown. 
3. Select **Jobs** from the **Action type** dropdown. 
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

View automations associated to a registered model from the W&B App UI. 

1. Navigate to the Model Registry App at [https://wandb.ai/registry/model](https://wandb.ai/registry/model).
2. Select on a registered model. 
3. Scroll to the bottom of the page to the **Automations** section.

Within the Automations section you can find the following properties of automations created for the model you selected:

- **Automation**:
- **Trigger type**:
- **Action type**: The action type that triggers the automation. Available options are Webhooks and Launch.
- **Action name**: The action name you provided when you created the automation.
- **Queue**: The name of the queue the job was enqueued to. This field is left empty if you selected a webhook action type.

## Delete an automation
Delete an automation associated with a model. Actions in progress are not affected if you delete that automation before the action completes. 

1. Navigate to the Model Registry App at [https://wandb.ai/registry/model](https://wandb.ai/registry/model).
2. Click on a registered model. 
3. Scroll to the bottom of the page to the **Automations** section.
4. Hover your mouse next to the name of the automation and click on the kebob (three vertical dots) menu. 
5. Select **Delete**.

<!-- # Automate workflows
Create an automation to trigger workflow steps based on an event you configure. For example, you can create an event that automatically tests new models versions added to a registered model. Automations are executed on your own infrastructure with [W&B Launch](../launch/intro.md).  

:::tip
Before you get started, ensure you create a W&B Launch [job](../launch/create-job.md), [queue](../launch/create-queue.md), and have an [agent polling](../launch/run-agent.md). For more information, see the [Launch documentation](../launch/intro.md).

:::

## Automation event types
There are two types of events that can trigger an action:

1. **A new version is added to a registered model**: Each time a new model version is linked, this automation will run. This is useful for Model CI — run testing on each new model candidate. 

2. **An artifact alias is added**: Specify an alias that represents a special step of your workflow, like `deploy`, and any time a new model version has that alias applied, it will automatically run this automation. This would let you automatically trigger a deployment job.

![](/images/models/automations_sidebar_step_1.png)





## Create an automation
1. Navigate to the W&B Model Registry app at [https://wandb.ai/registry/model](https://wandb.ai/registry/model).
2. Select **View details** next to the name of the registered model you want to create an automation for.
3. Scroll down the page to the **Automations** section.
![](/images/models/automations_section_in_registry.png)
4. Click on the **New automation** button. 
5. A UI modal will appear. Complete the steps described in the modal.

<!-- Will complete this with the new webhook docs  -->

