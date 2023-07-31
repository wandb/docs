---
description: Use an Automation for model CI (automated model evaluation pipelines) and model deployment.
displayed_sidebar: default
---

# Automate workflows
Create an automation to trigger workflow steps based on an event you configure. For example, you can create an event that automatically tests new models versions added to a registered model. Automations are executed on your own infrastructure with [W&B Launch](../launch/intro.md).  


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
5. A model will appear. From the **Event type** dropdown, select an event type.

Select a tab based on the event type you defined:





<!-- ## Walkthrough
For a step by step guide on how to use Automations for Model CI, check out [this](https://wandb.ai/examples/wandb_automations/reports/Model-CI-with-W-B-Automations--Vmlldzo0NDY5OTIx) report. -->


<!-- Update this later w/ webhook stuff -->
<!-- ## Infrastructure

Use [Launch](../launch/intro.md) to set up a connection to your own compute resources, whether that’s a GPU machine at your desk or a scalable cloud Kubernetes cluster. -->