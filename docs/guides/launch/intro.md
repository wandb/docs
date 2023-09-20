---
slug: /guides/launch
description: Easily scale and manage ML jobs using W&B Launch.
displayed_sidebar: default
---
import Translate, {translate} from '@docusaurus/Translate';

# Launch

Easily scale training [runs](../runs/intro.md) from your desktop to a compute resource like Amazon SageMaker, Kubernetes and more with W&B Launch. Once W&B Launch is configured, you can quickly run training scripts, model evaluation suites, prepare models for production inference, and more with a few clicks and commands. 

<!-- ![](/images/launch/ready_to_launch.png) -->

## How it works

Launch is composed of three fundamental components: **launch jobs**, **queues**, and **agents**.

A *launch job*[LINK] is a blueprint for configuring and running tasks in your ML workflow.  Once you have a launch job, you can add it to a *launch queue*[LINK]. A launch queue is a first-in, first-out (FIFO) queue where you can configure and submit your jobs to a particular compute target resource, such as Amazon SageMaker or a Kubernetes cluster. 

<!-- ![](/images/launch/mlOps_flow.png) -->

As jobs are added to the queue, one or more [*launch agents*](./run-agent.md) will poll that queue and execute the job on the system targeted by the queue.

![](/images/launch/ml_user_flow.png)
<!-- ![](/images/launch/Launch_Diagram.png) -->

Based on your use case, you (or someone on your team) will configure the launch queue according to your chosen compute resource target [LINK] (for example Amazon SageMaker) and deploy a launch agent on your own infrastructure. [LINK]



[INSERT NEW IMAGE]

See the [Terms and concepts](./launch_terminology.md) page for more information on launch jobs, how queues work, launch agents, and additional information on how W&B Launch works.

## How to get started

Depending on your use case, explore the following resources to get started with W&B Launch:

* If this is your first time using W&B Launch, we recommend you go through the [Walkthrough](./walkthrough.md) guide.
* Learn how to set up W&B Launch[LINK].
* Create a launch job. [LINK]
* Check out the W&B Launch [public jobs GitHub repo](https://github.com/wandb/launch-jobs) for templates of common tasks like [deploying to Triton](https://github.com/wandb/launch-jobs/tree/main/jobs/deploy_to_nvidia_triton), [evaluating an LLM](https://github.com/wandb/launch-jobs/tree/main/jobs/openai_evals), or more. View launch jobs created from this repo (`wandb/launch-jobs`) in this public [`wandb/jobs` project](https://wandb.ai/wandb/jobs/jobs) W&B project.
<!-- * Explore topics about W&B Launch in this Developer Guide, such as:
    * [Create a job](./create-job.md)
    * [Create a queue](./create-queue.md)
    * [Start an agent](./run-agent.md)
    * [Launch a run](./launch-jobs.md)
    * [Run an agent](./run-agent.md)   -->
<!-- * Discover the [`wandb launch`](../../ref/cli/wandb-launch.md) and [`wandb launch-agent`](../../ref/cli/wandb-launch-agent.md) commands in the CLI Reference. -->
