---
description: Use an Automation for model CI (automated model evaluation pipelines) and model deployment.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';



# Model CI/CD

Create an automation to trigger workflow steps, such as automated model testing and deployment.  The [action](#action-types) that is automatically started occurs when an [event type](#event-types) that you define is completed.  


## Event types
An *event* is a [INSERT]. There are two event types that can trigger an action:

1. **A new version is added to a registered model**: Each time a new model version is linked, this automation will run. This is useful for Model CI — run testing on each new model candidate.
2. **An artifact alias is added**: Specify an alias that represents a special step of your workflow, like `deploy`, and any time a new model version has that alias applied, it will automatically run this automation. This would let you automatically trigger a deployment job.



## Action types
An *action* is a [INSERT]. There are two types of actions you can create in the Model Registry: webhooks and [W&B Launch Jobs](../launch/intro.md).

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
2. Click on your user icon located on the top right of the page.
3. From the dropdown, select **User settings**.
4. Scroll down the page until you find the **Webhooks** block.
5. Click on the **New webhook** button.  
6. Provide a name for your webhook in the **Name** field.
7. Provide the endpoint URL for the webhook in the **URL** field.

### Add a secret for authentication
Define a secret to ensure the authenticity and integrity of data transmitted from payloads. Skip this section if the external server you use does not have secrets.

Configure a webhook in the W&B App UI.
1. Navigate to the W&B App UI.
2. Click on your user icon located on the top right of the page.
3. From the dropdown, select **User settings**.
4. Scroll down the page until you find the **Webhooks** block.


## Create a launch automation



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

