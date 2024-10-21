---
title: "Can you use W&B Sweeps with cloud infrastructures such as AWS Batch, ECS, etc.?"
tags:
   - sweeps
---

In general, you would need a way to publish `sweep_id` to a location that any potential W&B Sweep agent can read and a way for these Sweep agents to consume this `sweep_id` and start running.

In other words, you need something that can invoke `wandb agent`. For instance, bring up an EC2 instance and then call `wandb agent` on it. In this case, you might use an SQS queue to broadcast `sweep_id` to a few EC2 instances and then have them consume the `sweep_id` from the queue and start running.