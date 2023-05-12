---
description: Use automation for model CI (automated model evaluation pipelines) and model deployment.
---

# Automation

Use Automation to easily trigger workflow steps, such as automated model testing and deployment, on your own infrastructure with a lightweight Launch connector.

![](/images/models/automations_section_in_registry.png)

## Types of automation

You can automatically trigger actions based on two types of events:

1. **A new version is added to a registered model**: Each time a new model version is linked, this automation will run. This is useful for Model CI — run testing on each new model candidate. 

2.**An artifact alias is added**: Specify an alias that represents a special step of your workflow, like `deploy`, and any time a new model version has that alias applied, it will automatically run this automation. This would let you automatically trigger a deployment job.

![](/images/models/automations_sidebar_step_1.png)

## Infrastructure

Use [Launch](https://docs.wandb.ai/guides/launch) to set up a connection to your own compute resources, whether that’s a GPU machine at your desk or a scalable cloud Kubernetes cluster.