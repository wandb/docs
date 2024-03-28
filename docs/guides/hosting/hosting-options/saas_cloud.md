---
title: SaaS Cloud
displayed_sidebar: default
---

# SaaS Cloud (Multi-tenant SaaS)

W&B SaaS Cloud is a multi-tenant, fully-managed platform deployed in W&B's Google Cloud Platform (GCP) account in [GPC's North America regions](https://cloud.google.com/compute/docs/regions-zones). 

W&B SaaS Cloud utilizes GCP's autoscaling feature to ensure that W&B scales appropriately based on increases or decreases in traffic. 

:::tip
Since W&B SaaS Cloud is managed by W&B, you do not incur the overhead and costs of provisioning and maintaining the W&B platform.
:::

Metadata from W&B and data of other W&B SaaS Cloud customers is stored in a shared cloud storage for non enterprise customers. Metadata from W&B and W&B SaaS Cloud customer data is also processed with shared cloud compute services. Depending on your pricing plan, you may be subject to storage limits if you use the default W&B managed storage bucket to store your files.

With the [enterprise pricing plan](https://wandb.ai/site/pricing), you can bring your own bucket(BYOB) using the [secure storage connector](../secure-storage-connector.md) to store a W&B Team's data such as models, datasets and more. You can configure secure storage connector for one or more W&B teams in your account, or you can use separate buckets for different teams. For more information, see [BYOB(Secure storage connector)](../secure-storage-connector.md).

For example, the following image [INSERT]:
![](/images/hosting/saas_cloud_arch.png)

You can also use the security capabilities like Single-Sign On, Role-based Access Control, Restricted Projects with the enterprise pricing plan. 

Submit [this form](https://wandb.ai/site/for-enterprise/multi-tenant-saas-trial) for more information about the W&B Enterprise plans.

:::info
Security controls for W&B SaaS Cloud are periodically audited internally and externally. Reach out to the W&B Security Team at https://security.wandb.ai/ to request the SOC2 report and other security and compliance documents.
:::