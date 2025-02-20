---
menu:
  default:
    identifier: ja-guides-hosting-_index
no_list: true
title: W&B Platform
weight: 6
---

W&B Platform is the foundational infrastructure, tooling and governance scaffolding which supports the W&B products like [Core]({{< relref path="/guides/core/" lang="ja" >}}), [Models]({{< relref path="/guides/models/" lang="ja" >}}) and [Weave]({{< relref path="/guides/weave/" lang="ja" >}}). 

W&B Platform is available in three different deployment options:

* [W&B Multi-tenant Cloud]({{< relref path="#wb-multi-tenant-cloud" lang="ja" >}})
* [W&B Dedicated Cloud]({{< relref path="#wb-dedicated-cloud" lang="ja" >}})
* [W&B Customer-managed]({{< relref path="#wb-customer-managed" lang="ja" >}})


The following responsibility matrix outlines some of the key differences between the different options:
{{< img src="/images/hosting/shared_responsibility_matrix.png" alt="" >}}

## Deployment options
The following sections provide an overview of each deployment type. 

### W&B Multi-tenant Cloud
W&B Multi-tenant Cloud is a fully managed service deployed in W&B's cloud infrastructure, where you can seamlessly access the W&B products at the desired scale, with cost-efficient options for pricing, and with continuous updates for the latest features and functionalities. W&B recommends to use the Multi-tenant Cloud for your product trial, or to manage your production AI workflows if you do not need the security of a private deployment, self-service onboarding is important, and cost efficiency is critical.

See [W&B Multi-tenant Cloud]({{< relref path="./hosting-options/saas_cloud.md" lang="ja" >}}) for more information. 

### W&B Dedicated Cloud
W&B Dedicated Cloud is a single-tenant, fully managed service deployed in W&B's cloud infrastructure. It is the best place to onboard W&B if your organization requires conformance to strict governance controls including data residency, have need of advanced security capabilities, and are looking to optimize their AI operating costs by not having to build & manage the required infrastructure with security, scale & performance characteristics.

See [W&B Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ja" >}}) for more information.

### W&B Customer-Managed
With this option, you can deploy and manage W&B Server on your own managed infrastructure. W&B Server is a self-contained packaged mechanism to run the W&B Platform & its supported W&B products. W&B recommends this option if all your existing infrastructure is on-prem, or your organization has strict regulatory needs that are not satisfied by W&B Dedicated Cloud. With this option, you are fully responsible to manage the provisioning, and continuous maintenance & upgrades of the infrastructure required to support W&B Server.

See [W&B Self Managed]({{< relref path="/guides/hosting/hosting-options/self-managed/" lang="ja" >}}) for more information.

## Next steps

If you're looking to try any of the W&B products, W&B recommends using the [Multi-tenant Cloud](https://wandb.ai/home). If you're looking for an enterprise-friendly setup, choose the appropriate deployment type for your trial [here](https://wandb.ai/site/enterprise-trial).