---
menu:
  default:
    identifier: serverless-rl
title: Serverless RL
description: Learn about how to more efficiently post-train your models using reinforcement learning.
weight: 5
---

Now in public preview, Serverless RL helps developers post-train LLMs to learn new behaviors and improve reliability, speed, and costs when performing multi-turn agentic tasks. We provision the training infrastructure for you while allowing full flexibility in your environment's setup. Serverless RL gives you instant access to a managed training cluster that elastically auto-scales to dozens of GPUs. By splitting RL workflows into inference and training phases and multiplexing them across jobs, Serverless RL increases GPU utilization and reduces your training time and costs.

Serverless RL is ideal for tasks like:
* Voice agents
* Deep research assistants
* On-prem models
* Content marketing analysis agents

Serverless RL trains low-rank adapters (LoRAs) to specialize a model for your agent's specific task. This extends the original model’s capabilities with on-the-job experience. The LoRAs you train are automatically stored as artifacts in your W&B account, and can be saved locally or to a third party for backup. Models that you train through Serverless RL are also automatically hosted on W&B Inference.

## Why Serverless RL?

Reinforcement learning (RL) is a set of powerful training techniques that you can use in many kinds of training setups, including on GPUs that you own or rent directly. Serverless RL can provide the following advantages in your RL post-training:

* **Lower training costs**: By multiplexing shared infrastructure across many users, skipping the setup process for each job, and scaling your GPU costs down to 0 when you're not actively training, Serverless RL reduces training costs significantly.
* **Faster training time**: By splitting inference requests across many GPUs and immediately provisioning training infrstructure when you need it, Serverless RL speeds up your training jobs and lets you iterate faster.
* **Automatic deployment**: Serverless RL automatically deploys every checkpoint you train, eliminating the need to manually set up hosting infrastructure. Trained models can be accessed and tested immediately in local, staging, or production environments.

## How Serverless RL uses W&B services

Serverless RL uses a combination of the following W&B components to operate:

* [Inference]({{< relref "guides/inference" >}}): To run your models
* [Models]({{< relref "guides/models" >}}): To track performance metrics during the LoRA adapter's training
* [Artifacts]({{< relref "guides/core/artifacts" >}}): To store and version the LoRA adapters
* [Weave (optional)]({{< relref "guides/models" >}}): To gain observability into how the model responds at each step of the training loop

Serverless RL is in public preview. During the preview, you are charged only for the use of inference and the storage of artifacts, during the preview period, not for adapter training. We do not charge for adapter training during the preview period.