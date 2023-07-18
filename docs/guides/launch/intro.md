---
slug: /guides/launch
description: Easily scale and manage ML jobs using W&B Launch.
displayed_sidebar: default
---
import Translate, {translate} from '@docusaurus/Translate';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

# Launch

<CTAButtons productLink="https://github.com/wandb/docodile" colabLink="https://www.google.com/"/>

Easily scale training [runs](../runs/intro.md) from your desktop to your GPUs, quickly spin up intensive model evaluation suites, and prepare models for production inference, all without the friction of complex infrastructure.

![](/images/launch/ready_to_launch.png)

## How it works

Launch is composed of three fundamental components: **jobs, queues, and agents**.

A [job](./create-job.md) is a blueprint for configuring and running tasks in your ML workflow.  Once you have a job, you can add it to a [*launch queue*](./create-queue.md). A launch queue is a first-in, first-out (FIFO) queues where you can configure and submit your jobs to a particular compute resource. 

As jobs are added to the queue, a [*launch agent*](./run-agent.md) will poll that queue and execute the job on the system targeted by the queue this job was taken from.

![](/images/launch/Launch_Diagram.png)


## How to get started

Depending on your use case, explore the following resources to get started with W&B Launch:

* If this is your first time using W&B Launch, we recommend you go through the [Quickstart](./quickstart.md) guide.
* Check out the W&B Launch [public jobs repo](https://github.com/wandb/launch-jobs) for templates of common tasks like [deploying to Triton](https://github.com/wandb/launch-jobs/tree/main/jobs/deploy_to_nvidia_triton), [evaluating an LLM](https://github.com/wandb/launch-jobs/tree/main/jobs/openai_evals), or more. 
    * These jobs are also visible in the public [`wandb/jobs` project](https://wandb.ai/wandb/jobs/jobs). (Be sure to uncheck the toggle to see the full list.)
* Explore topics about W&B Launch in this Developer Guide, such as:
    * [Create a job](./create-job.md)
    * [Create a queue](./create-queue.md)
    * [Start an agent](./run-agent.md)
    * [Launch a run](./launch-jobs.md)
    * [Run an agent](./run-agent.md)  
* Discover the [`wandb launch`](../../ref/cli/wandb-launch.md) and [`wandb launch-agent`](../../ref/cli/wandb-launch-agent.md) commands in the CLI Reference.
