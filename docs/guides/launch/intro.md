---
slug: /guides/launch
description: Easily queue and manage jobs using W&B Launch.
---

# Launch jobs

Use W&B Launch to start jobs in the compute  environment or infrastructure of choice from one central interface. W&B Launch removes the complexity of using different environments to run training, inference, and more. Weights and Biases will automatically capture the information needed to reproduce the run, containerize the job and launch the job locally or remotely. 

![](/images/launch/ready_to_launch.png)

:::caution
W&B Launch is Public preview in active development. 

Talk to the W&B Sales Team to get W&B Launch set up for your business: [https://wandb.ai/site/pricing](https://wandb.ai/site/pricing).
:::

## How it works
There are three unique concepts to W&B Launch: *jobs*, *queues*, and *agents*: 
* **Job**:  a definition of a computational process. You can think of W&B Jobs as a ‘Run template’. A Job can have one or more versions. Each job version combines source code and environment to produce a template for a reproducible run.
* **Queue**: FIFO queue of W&B Jobs. A launch queue has a compute resource associated to that queue. Launch queues are entity-scoped. All jobs pushed to a queue automatically posses the same compute resource type.
* **Agents**: A launch agent listens the queue and executes jobs added to the queue. The launch agent builds a docker image for you if you do not have one. The agent will build an image for the environment and will send the job to the desired compute runner.


:::tip
Generally, an MLOps Team will configure a queue with a cloud compute resource and activate a launch agent. Once this step is completed, anyone who has access to a W&B entity can add jobs to the queue with the W&B App UI or the CLI.
:::


## How to get started
Depending on your use case, explore the following resources to get started with Weights & Biases Launch:

* If this is your first time using W&B Launch, we recommend you go through the [Getting started](./getting-started.md) guide.
* Explore topics about W&B Launch in this Developer Guide, such as:
    * [Prerequisites](../launch/prerequisites.md)  
    * [Create a job](../launch/create-job.md)
    * [Add jobs to your queue](../launch/add-jobs-to-queue.md)
    * [Launch jobs](../launch/launch-jobs.md)
* Discover the [`wandb launch`](../../ref/cli/wandb-launch.md) and [`wandb launch-agent`](../../ref/cli/wandb-launch-agent.md) in the CLI Reference.