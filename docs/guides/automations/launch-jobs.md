---
displayed_sidebar: default
---
## What are launch jobs?
Launch jobs are blueprints for reproducing W&B runs. Jobs are W&B Artifacts that capture the source code, dependencies, and inputs required to execute a workload.

## Use cases
- Increased observability: Launch jobs allow you to view and analyze training runs through the W&B platform.
- Alternate credential storage: Store your credentials on an internal machine instead of a secret store.
- Infrastructure analysis: Gain increased visibility into your infrastructure usage.

## Create a launch automation
Automatically trigger a W&B Job. 

:::info
This section assumes you already have created a job, a queue, and have an active agent polling. For more information, see the [W&B Launch docs](../launch/intro.md). 
:::


1. From the **Event type** dropdown, select an event type. See the [Event type](#event-types) section for information on supported events.
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

See this example [report](https://wandb.ai/examples/wandb_automations/reports/Model-CI-with-W-B-Automations--Vmlldzo0NDY5OTIx) for an end to end example on how to create an automation for model CI with W&B Launch.