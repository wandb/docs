---
description: 'Understand pricing, usage limits, and account restrictions for W&B Inference

  '
linkTitle: Usage & Limits
menu:
  default:
    identifier: ja-guides-inference-usage-limits
title: Usage Information and Limits
weight: 20
---

Learn about pricing, limits, and other important usage information before using W&B Inference.

## Pricing

For detailed model pricing information, visit [W&B Inference pricing](https://wandb.ai/site/pricing/inference).

## Purchase more credits

W&B Inference credits come with Free, Pro, and Academic plans for a limited time. Enterprise availability may vary. When credits run out:

- **Free accounts** must upgrade to a paid plan to continue using W&B Inference. [Upgrade to Pro or Enterprise](https://wandb.ai/subscriptions)
- **Pro plan users** are billed for overages monthly, based on [model-specific pricing](https://wandb.ai/site/pricing/inference)
- **Enterprise accounts** should contact their account executive

## Account tiers and default usage caps

Each account tier has a default spending cap to help manage costs and prevent unexpected charges. W&B requires prepayment for paid Inference access.

Some users may need to change their cap. Contact your account executive or support to adjust your limit.

| Account Tier | Default Cap | How to Change Limit |
|--------------|-------------|---------------------|
| Pro | $6,000/month | Contact your account executive or support for manual review |
| Enterprise | $700,000/year | Contact your account executive or support for manual review |

## Concurrency limits

If you exceed the rate limit, the API returns a `429 Concurrency limit reached for requests` response. To fix this error, reduce the number of concurrent requests. For detailed troubleshooting, see [W&B Inference support articles](/support/inference/).

W&B applies rate limits per W&B project. For example, if you have 3 projects in a team, each project has its own rate limit quota.

## Personal entities unsupported

{{< alert title="Note" >}}
Personal entities were deprecated in May 2024, so this only applies to legacy accounts.
{{< /alert >}}

Personal accounts (personal entities) don't support W&B Inference. To access W&B Inference, switch to a non-personal account by creating a Team.

## Geographic restrictions

The Inference service is only available from supported geographic locations. For more information, see the [Terms of Service](https://docs.coreweave.com/docs/policies/terms-of-service/terms-of-use#geographic-restrictions).

## Next steps

- Review the [prerequisites]({{< relref path="prerequisites" lang="ja" >}}) before getting started
- See [available models]({{< relref path="models" lang="ja" >}}) and their specific costs