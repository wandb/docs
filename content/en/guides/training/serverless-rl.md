---
description: Learn about Serverless RL and how to start using it.
title: Use Serverless RL
weight: 100
---

W&B Training's Serverless RL (reinforncement learning) allows you to post-train LLMs to extend their knowledge and improve their reliability performing multi-turn, agentic tasks while also increasing speed and reducing costs. For example, instead of building your own model or retraining an entire model to use HIPAA compliant medical terminology or legal citation formats, you can use Serverless RL to train a small module about these subjects and then integrate it into your existing LLM workflows to extend the existing model's capabilities.

Serverless RL trains [low-rank adaptors (LoRAs)](https://en.wikipedia.org/wiki/Fine-tuning_(deep_learning)#Low-rank_adaptation) and then provides you the ability to integrate them into calls to your existing LLM models, such as GPT-4.1. This extends the existing model's training with data and hyper-parameters that you provide and gives you the ability to access the adapted model from a singular endpoint. You can also store versions of your adaptors as artifacts stored in your W&B account.

Serverless RL uses a combination of the following W&B services to operate:

* [Inference]({{< relref "guides/inference" >}}): To access your models
* [Models]({{< relref "guides/models" >}}): To train the LoRA adaptors
* [Artifacts]({{< relref "guides/core/artifacts" >}}): To store and version the LoRA adaptors

We charge for the use of inference and the storage of artifacts. We do not currently charge to train the adaptors. See the [Serverless RL pricing page](#update-link) for more details.

## Use Serverless RL

Serverless RL is currently only supported through [OpenPipe's ART framework](https://art.openpipe.ai/getting-started/about). See the [OpenPipe's Serverless RL quickstart](#update-link) for code examples and workflows.

See the [Serverless RL API]({{< relref "ref/training" >}}) reference for more information on the service's endpoints.