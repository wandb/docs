---
menu:
  default:
    identifier: training
title: W&B Training
description: Post-train your models using reinforcement learning.
weight: 60
---

Now in public preview, W&B Training offers serverless reinforcement learning (RL) for post-training large language models (LLMs) to improve their reliability performing multi-turn, agentic tasks while also increasing speed and reducing costs. RL is a training technique where models learn to improve their behavior through feedback on their outputs. 

W&B Training includes integration with:

* [ART](https://art.openpipe.ai/getting-started/about), a flexible RL fine-tuning framework.
* [RULER](https://openpipe.ai/blog/ruler), a universal verifier. 
* A fully-managed backend on CoreWeave cloud.

To get started, satisfy the [prerequisites]({{< relref "/guides/training/prerequisites" >}}) to start using the service and then see [OpenPipe's Serverless RL quickstart](https://art.openpipe.ai/getting-started/quick-start) to learn how to post-train your models.