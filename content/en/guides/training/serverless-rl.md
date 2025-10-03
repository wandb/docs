---
description: Learn about Serverless RL and how to start using it.
title: Use Serverless RL
weight: 100
---

W&B Serverless RL (reinforcement learning) allows you to post-train LLMs to learn new behavior and improve their reliability performing multi-turn, agentic tasks while also increasing speed and reducing costs. Serverless RL runs your reinforcement learning (RL) jobs on a large, managed cluster of GPUs that give you the freedom to scale your training run up to dozens of GPUs or down to zero instantly.

Serverless RL is ideal for training:

* Voice agents
* Deep research assistants
* On-prem models
* Content marketing analysis agents

Serverless RL trains [low-rank adapters (LoRAs)](https://en.wikipedia.org/wiki/Fine-tuning_(deep_learning)#Low-rank_adaptation) and then provides you the ability to integrate them into calls to your existing LLM models, such as GPT-4.1. This extends the existing model's training with data and hyper-parameters that you provide and gives you the ability to access the adapted model from a singular endpoint. You can also store versions of your adapters as artifacts stored in your W&B account.

Serverless RL uses a combination of the following W&B services to operate:

* [Inference]({{< relref "guides/inference" >}}): To run your models
* [Models]({{< relref "guides/models" >}}): To track performance metrics during the LoRA adapter's training
* [Artifacts]({{< relref "guides/core/artifacts" >}}): To store and version the LoRA adapters
* [Weave]({{< relref "guides/models" >}}): To gain observability into how the model responds at each step of the training loop

We charge for the use of inference and the storage of artifacts. We do not currently charge to train the adapters. See the [Serverless RL pricing page](#update-link) for more details.

## Use Serverless RL

Serverless RL is currently only supported through [OpenPipe's ART framework](https://art.openpipe.ai/getting-started/about) and a [W&B API](#UPDATE-LINK). See the [OpenPipe's Serverless RL quickstart](#UPDATE-LINK) for code examples and workflows.

See the [Serverless RL API]({{< relref "ref/training" >}}) reference for more information on the service's endpoints.