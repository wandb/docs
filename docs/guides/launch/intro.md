---
slug: /guides/launch
description: Easily queue and manage jobs using W&B Launch.
---

# Launch jobs


W&B Launch is in Public Preview and in active development. 

Before you get started, enable the W&B Launch UI:

1. Go to https://wandb.ai/settings.
2. Scroll down to the **Beta Features** section and enable **W&B Launch**.

![](/images/launch/toggle_beta_flag.png)

## What is Launch?

Launch is a new toolkit from Weights & Biases that makes it easy to build 
ergonomic, portable, and reproducible machine learning workflows.

Launch workflows can be driven from the W&B CLI, UI, or SDK.

![](/images/launch/ready_to_launch.png)


## How it works

Launch workflows are powered by three fundamental compoenents: **jobs, queues, and agents**.

### Jobs

**Jobs** are blueprints for configuring and running the tasks in your ML workflow. A job is actually an [Artifact](../../guides/artifacts/intro.md) that gets automatically created when you track a run with W&B. Each job contains contextual information about the run it is being created from, like the source code, software dependencies, hyperparameters, dataset version, etc. Once a job is created, it can be used as a template to launch new runs.

Jobs make your workloads reproducible, but they also allow you to rapidly iterate on experiments while avoiding bugs that might come from changes to your codebase or software environment. When a job is created, any configuration stored in your [wandb.config](../track/config.md) becomes a parameter of that job.

### Queues

**Launch queues** are first-in, first-out (FIFO) queues where users can configure and submit their jobs. Each item in a launch queue consists of a job and settings for the parameters of that job. Users can submit workloads to launch queues via the W&B UI, CLI, or SDK.

The queue itself corresponds to a particular compute resource, e.g. a kubernetes cluster, an on-prem machine, or an on demand cloud service like AWS sagemaker.

### Agents

A **launch agent** is a long-running process that polls on one or more launch queues for jobs to run.

The agent is capable of building container images to replicate the original environment of the job. The agent can then take the image it has built (or a pre-made image) and execute it on the system targeted by the queue this job was taken from.

## How to get started
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