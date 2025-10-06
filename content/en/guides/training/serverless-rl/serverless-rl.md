---
description: Learn how to start using it.
title: Use Serverless RL
weight: 10
---

Serverless RL is supported through [OpenPipe's ART framework](https://art.openpipe.ai/getting-started/about) and the W&B Training API. 

### Why Serverless RL?

RL is a set of powerful training techniques that can be utilized in many kinds of training setups, including on GPUs that you own or rent directly. In fact, much of the code that makes Serverless RL so effective is already open source in the [ART SDK](https://github.com/openpipe/art). So why use Serverless RL?

* Lower training costs
  * By multiplexing shared infra across many users, skipping the setup process for each job, and scaling your GPU costs down to 0 when you're not actively training, Serverless RL reduces training costs significantly.
* Faster training time
  * By splitting inference requests across many GPUs and immediately provisioning training infra when you need it, Serverless RL speeds up your training jobs and lets you iterate faster. Shorter feedback cycles means more chances to find and fix bugs, quicker results, and happier developers.
* Automatic deployment
  * Once you train your model, you need to somehow host it on a public-facing service that serves traffic at a high concurrency. Serverless RL solves that problem for you by **automatically deploying every checkpoint you train**. That means you can go right from training your model in a sandboxed environment to deploying testing it locally, in staging, or even production.

### Get Started

  To start using Serverless RL, check out the ART [quickstart](https://art.openpipe.ai/getting-started/quick-start) for code examples and workflows. To learn about Serverless RL's API endpoints, see the [W&B Training API]({{< relref "ref/training" >}}).