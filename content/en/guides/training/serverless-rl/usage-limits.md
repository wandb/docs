---
title: "Usage information and limits"
linkTitle: "Usage & limits"
weight: 30
description: >
  Understand pricing, usage limits, and account restrictions for W&B Serverless RL.
---

## Pricing

Pricing has three components: inference, training, and storage. For specific billing rates, visit our [pricing page](https://wandb.ai/site/pricing/reinforcement-learning).

### Inference

Pricing for Serverless RL inference requests matches W&B Inference pricing. See [model-specific costs](https://wandb.ai/site/pricing/inference) for more details. Learn more about purchasing credits, account tiers, and usage caps in the [W&B Inference docs]({{< relref "/guides/inference/usage-limits/#purchase-more-credits" >}}).

### Training

At each training step, Serverless RL collects batches of trajectories that include your agent's outputs and associated rewards (calculated by your reward function). The batched trajectories are then used to update the weights of a LoRA adapter that specializes a base model for your task. The training jobs to update these LoRAs run on dedicated GPU clusters managed by Serverless RL.

Training is free during the public preview period.

### Model storage

Serverless RL stores checkpoints of your trained LoRAs so you can evaluate, serve, or continue training them at any time. Storage is billed monthly based on total checkpoint size and your [pricing plan](https://wandb.ai/site/pricing). Every plan includes at least 5GB of free storage, which is enough for roughly 30 LoRAs. We recommend deleting low-performing LoRAs to save space. See the [ART SDK](https://art.openpipe.ai/features/checkpoint-deletion) for instructions on how to do this.

## Limits

* **Inference concurrency limits**: By default, Serverless RL currently supports up to 2000 concurrent requests per user and 6000 per project. If you exceed your rate limit, the Inference API returns a `429 Concurrency limit reached for requests` response. To avoid this error, reduce the number of concurrent requests your training job or production workload makes at once. If you need a higher rate limit, you can request one at support@wandb.com.

* **Personal entities unsupported**: Serverless RL and W&B Inference don't support personal entities (personal accounts). To access Serverless RL, switch to a non-personal account by [creating a Team]({{< relref "/guides/hosting/iam/access-management/manage-organization/#add-and-manage-teams" >}}). Personal entities (personal accounts) were deprecated in May 2024, so this advisory only applies to legacy accounts.

* **Geographic restrictions**: Serverless RL is only available in supported geographic locations. For more information, see the [Terms of Service](https://site.wandb.ai/terms/).