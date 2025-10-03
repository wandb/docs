---
title: "Usage Information and Limits"
linkTitle: "Usage & Limits"
weight: 30
description: >
  Understand pricing, usage limits, and account restrictions for W&B Serverless RL.
---

## Pricing

Pricing has three components: inference, training, and storage. Learn more: https://wandb.ai/site/pricing/reinforcement-learning 

### Inference

Trajectories are batched at each training step and, together with rewards from your reward function (often an LLM-as-judge), are used to update a training copy of the LoRA. The training workload runs on a separate GPU cluster within Serverless RL. During the public preview, training is free. Pricing for training will be announced at general availability (GA).

* Check the [model-specific costs](https://wandb.ai/site/pricing/inference) for more details. Learn more about purchasing credits, account tiers, and usage caps in the [W&B Inference docs]({{< relref "/guides/inference/usage-limits/#purchase-more-credits" >}}).

### Training

At each training step, Serverless RL batches trajectories and combines them with rewards from your reward function (typically an LLM-as-judge). These batched trajectories and rewards are then used to update the LoRA adapter weights on a training copy of the base model. All training workloads run on a dedicated GPU cluster managed by Serverless RL.

During the public preview, training is free. Pricing for training will be announced at general availability (GA).

### Model Storage

Serverless RL stores checkpoints of your trained LoRA heads so you can evaluate, serve, or continue training them at any time. Storage is billed monthly based on total checkpoint size and your [pricing plan](https://wandb.ai/site/pricing). Every plan includes free storage.


## Concurrency limits

If you exceed the rate limit, the API returns a `429 Concurrency limit reached for requests` response. To fix this error, reduce the number of concurrent requests.
W&B applies rate limits per project. For example, if you have three projects in a team, each project has its own rate limit quota. The default rate limit is 2000 concurrent requests.

## Personal entities unsupported

Serverless RL and W&B Inference don't support personal entities (personal accounts). To access Serverless RL, switch to a non-personal account by [creating a Team]({{< relref "/guides/hosting/iam/access-management/manage-organization/#add-and-manage-teams" >}}). Personal entities (personal accounts) were deprecated in May 2024, so this only applies to legacy accounts. 

## Geographic restrictions

Serverless RL is only available from supported geographic locations. For more information, see the [Terms of Service](https://docs.coreweave.com/docs/policies/terms-of-service/terms-of-use#geographic-restrictions).

