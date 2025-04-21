---
menu:
  default:
    identifier: deployment-options
    parent: w-b-platform
title: Deployment options
weight: 1
---
This section describes the different ways can you can deploy W&B.

## W&B Multi-tenant Cloud
[W&B Multi-tenant Cloud]({{< relref "saas_cloud.md" >}}) is fully managed by W&B, including upgrades, maintenance, platform security, and capacity planning. Multi-tenant Cloud is deployed in W&B's Google Cloud Platform (GCP) account in [GPC's North America regions](https://cloud.google.com/compute/docs/regions-zones).

[Learn more about W&B Multi-tenant Cloud]({{< relref "saas_cloud.md" >}}) or [get started for free](https://app.wandb.ai/login?signup=true).

## W&B Dedicated Cloud
[W&B Dedicated Cloud]({{< relref "dedicated_cloud/" >}}) is a single-tenant, fully managed platform designed with enterprise organizations in mind. W&B Dedicated Cloud is deployed in W&B's AWS, GCP or Azure account. Upgrades, maintenance, platform security, and capacity planning are managed by W&B. Each Dedicated Cloud instance has its own isolated network, compute and storage from other W&B Dedicated Cloud instances. Your W&B specific metadata and data is stored in an isolated cloud storage and is processed using isolated cloud compute services. Dedicated Cloud provides more flexibility than Multi-tenant Cloud, but less complexity than W&B Self-Hosted.

W&B Dedicated Cloud includes an [enterprise license]({{< relref "self-managed/server-upgrade-process.md" >}}), which includes support for important security and other enterprise-friendly capabilities.

For organizations with advanced security or compliance requirements, features such as HIPAA compliance, Single Sign On, or Customer Managed Encryption Keys (CMEK) are available with **Enterprise** support. [Request more information](https://wandb.ai/site/contact).

[Learn more about W&B Dedicated Cloud]({{< relref "dedicated_cloud/" >}}) or [get started for free](https://app.wandb.ai/login?signup=true).

## W&B Self-Managed
[W&B Self-Managed]({{< relref "self-managed/" >}}) is entirely managed by you, either on your premises or in cloud infrastructure that you manage. Your IT/DevOps/MLOps team is responsible for:
- Provisioning your deployment.
- Securing your infrastructure in accordance with your organization's policies and [Security Technical Implementation Guidelines (STIG)](https://en.wikipedia.org/wiki/Security_Technical_Implementation_Guide), if applicable.
- Managing upgrades and applying patches.
- Continuously maintaining your self managed W&B Server instance.

You can optionally obtain an enterprise license for W&B Self-Managed. An enterprise license includes support for important security and other enterprise-friendly capabilities.

[Learn more about W&B Self-Managed]({{< relref "self-managed/" >}}) or review the [reference architecture]({{< relref "self-managed/ref-arch.md" >}}) guidelines.