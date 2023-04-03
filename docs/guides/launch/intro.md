---
slug: /guides/launch
description: Easily queue and manage jobs using W&B Launch.
---


## What is Launch?

W&B Launch introduces a connective layer between machine learning practitioners and the high-scale, specialized hardware that powers modern machine learning workflows. Easily scale training runs from your desktop to your GPUs, quickly spin up, intensive model evaluation suites, and prepare models for production inference, all without the friction of complex infrastructure.


![](/images/launch/ready_to_launch.png)

## How it works

Launch workflows are powered by three fundamental components: **jobs, queues, and agents**.

![](/images/launch/Launch_Diagram.png)


* **Jobs** are blueprints for configuring and running the tasks in your ML workflow. A job is actually an [Artifact](../../guides/artifacts/intro.md) that is created automatically when you track a run with W&B. Each job contains contextual information about the run it is being created from, including the source code, entrypoint, software dependencies, hyperparameters, dataset version, etc.

* **Launch queues** are first-in, first-out (FIFO) queues where users can configure and submit their jobs to a particular compute resource. Each item in a launch queue consists of a job and settings for the parameters of that job.

* **Launch agents** are long-running processes that poll on one or more launch queues for jobs to run. The agent is capable of building container images to replicate the original environment of the job. The agent can then take the image it has built (or a pre-made image) and execute it on the system targeted by the queue this job was taken from.

## How to get started


To enable W&B Launch:

1. Navigate to https://wandb.ai/settings.
2. Scroll down to the **Beta Features** section and enable **W&B Launch**.

![](/images/launch/toggle_beta_flag.png)

Depending on your use case, explore the following resources to get started with Weights & Biases Launch:

* If this is your first time using W&B Launch, we recommend you go through the [Getting started](./getting-started.md) guide.
* Explore topics about W&B Launch in this Developer Guide, such as:
    * [Prerequisites](../launch/prerequisites.md)  
    * [Create a job](../launch/create-job.md)
    * [Add jobs to your queue](../launch/add-jobs-to-queue.md)
    * [Launch jobs](../launch/launch-jobs.md)
* Discover the [`wandb launch`](../../ref/cli/wandb-launch.md) and [`wandb launch-agent`](../../ref/cli/wandb-launch-agent.md) commands in the CLI Reference.

:::info
Talk to the W&B Sales Team to get W&B Launch set up for your business: [https://wandb.ai/site/pricing](https://wandb.ai/site/pricing).

:::