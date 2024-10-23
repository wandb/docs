---
title: "Can you use W&B Sweeps with cloud infrastructures such as AWS Batch, ECS, etc.?"
tags:
   - sweeps
   - aws
---
To publish the `sweep_id` so that any W&B Sweep agent can access it, implement a method for these agents to read and execute the `sweep_id`.

For example, launch an Amazon EC2 instance and execute `wandb agent` on it. Use an SQS queue to broadcast the `sweep_id` to multiple EC2 instances. Each instance can then retrieve the `sweep_id` from the queue and initiate the process.