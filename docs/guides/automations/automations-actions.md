---
description: Find out how different action types can impact your automations.
title: Actions
displayed_sidebar: default
---

# Actions
An action is an internal or external responsive mutation that occurs as a result of some trigger. For automations, this trigger is an event configured based on your project's needs. 


# Action types
There are two types of actions you can create in response to events on artifact collections in your project: webhooks and [W&B Launch Jobs](../launch/intro.md). Configure both action types in the W&B App UI.

## Webhooks
Webhooks communicate with an external web server from W&B with HTTP requests. This is useful for linking your models or artifacts to external tools like GitHub Actions.

## Launch jobs
[Launch jobs](../launch/create-launch-job.md) are reusable, configurable run templates that allow you to quickly launch new [runs](../runs/intro.md) locally on your desktop or external compute resources such as Kubernetes on EKS and Amazon SageMaker. 