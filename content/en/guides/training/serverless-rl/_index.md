---
menu:
  default:
    identifier: serverless-rl
title: Serverless RL
description: Learn about how to more efficiently post-train your models using reinforcement learning.
weight: 5
---

Serverless RL lets you post-train LLMs to learn new behaviors and improve reliability, speed, and costs when performing multi-turn, agentic tasks while we handle the GPUs and infrastructure for you. Serverless RL gives you instant access to GPUs on a fully managed, distributed training cluster that elastically auto-scales to dozens of GPUs or down to zero instantly, matching your reinforcement learning (RL) job at every moment. By automatically splitting RL workflows into inference and training phases and multiplexing them across jobs, Serverless RL maximizes GPU utilization and reduces your training time and costs.

Serverless RL is ideal for:
* Voice agents
* Deep research assistants
* On-prem models
* Content marketing analysis agents

Serverless RL trains low-rank adapters (LoRAs) and then provides you the ability to integrate them into calls to your existing LLM models, such as GPT-4.1. This extends the existing model’s training with data and hyper-parameters that you provide and gives you the ability to access the adapted model from a singular endpoint. You can also store versions of your adapters as artifacts stored in your W&B account.

Serverless RL uses a combination of the following W&B services to operate:

* [Inference]({{< relref "guides/inference" >}}): To run your models
* [Models]({{< relref "guides/models" >}}): To track performance metrics during the LoRA adapter's training
* [Artifacts]({{< relref "guides/core/artifacts" >}}): To store and version the LoRA adapters
* [Weave]({{< relref "guides/models" >}}): To gain observability into how the model responds at each step of the training loop

Serverless RL is currently in public preview. We only charge for the use of inference and the storage of artifacts. We do not charge for adapter training during the preview period. We will make training pricing details available when the service is released for general availability. 
