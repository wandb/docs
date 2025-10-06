---
menu:
  default:
    identifier: serverless-rl
title: Serverless RL
description: Learn about how to more efficiently post-train your models using reinforcement learning.
weight: 5
---

Serverless RL helps developers post-train LLMs to learn new behaviors and improve reliability, speed, and costs when performing multi-turn agentic tasks. We provision the training infrastructure for you while allowing full flexibility in your environment's setup. Serverless RL gives you instant access to a managed training cluster that elastically auto-scales to dozens of GPUs. By splitting RL workflows into inference and training phases and multiplexing them across jobs, Serverless RL increases GPU utilization and reduces your training time and costs.

Serverless RL is ideal for tasks like:
* Voice agents
* Deep research assistants
* On-prem models
* Content marketing analysis agents

Serverless RL trains low-rank adapters (LoRAs) to specialize a model for your agent's specific task. This extends the original model’s capabilities with on-the-job experience. The LoRAs you train will automatically be stored as artifacts in your W&B account, and can be saved locally or to a third party for backup. Models that you train through Serverless RL will also be automatically hosted on W&B Inference.

Serverless RL uses a combination of the following W&B services to operate:

* [Inference]({{< relref "guides/inference" >}}): To run your models
* [Models]({{< relref "guides/models" >}}): To track performance metrics during the LoRA adapter's training
* [Artifacts]({{< relref "guides/core/artifacts" >}}): To store and version the LoRA adapters
* [Weave (optional)]({{< relref "guides/models" >}}): To gain observability into how the model responds at each step of the training loop

Serverless RL is currently in public preview. We only charge for the use of inference and the storage of artifacts. We do not charge for adapter training during the preview period. We will make training pricing details available in the next few weeks.
