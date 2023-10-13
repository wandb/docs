---
description: Use an Automation for model CI (automated model evaluation pipelines) and model deployment.
displayed_sidebar: default
---

# Automations

An Automation is an event, action pair where a specific event in W&B will kick off a downstream action, such as a [Launch Job](../launch/create-launch-job.md). Use Automations to easily trigger workflow steps, such as automated model testing and deployment, on your own infrastructure with a lightweight Launch connector. 

![](/images/models/automations_section_in_registry.png)

## Types of automation

You can automatically trigger actions based on two types of events:

1. **A new version is added to a registered model**: Each time a new model version is linked, this automation will run. This is useful for Model CI — run testing on each new model candidate. 

2. **An artifact alias is added**: Specify an alias that represents a special step of your workflow, like `deploy`, and any time a new model version has that alias applied, it will automatically run this automation. This would let you automatically trigger a deployment job.

![](/images/models/automations_sidebar_step_1.png)

## Infrastructure

Use [Launch](../launch/intro.md) to set up a connection to your own compute resources, whether that’s a GPU machine at your desk or a scalable cloud Kubernetes cluster.

## Walkthrough
For a step by step guide on how to use Automations for Model CI, check out [this](https://wandb.ai/examples/wandb_automations/reports/Model-CI-with-W-B-Automations--Vmlldzo0NDY5OTIx) report.
