---
title: SaaS Cloud
displayed_sidebar: default
---

# SaaS Cloud (Multi-tenant SaaS)

W&B SaaS Cloud is the multi-tenant, fully-managed platform deployed in W&B's Google Cloud Platform (GCP) account. It has autoscaling infrastructure in GCP's North America regions and benefits from the economies of scale, providing a simpler and cost-efficient way to use the W&B products. Since it's fully-managed, you do not incur the overhead and costs of provisioning and maintaining the W&B platform.

Being a multi-tenant service, W&B specific metadata and data of all customers on SaaS Cloud is stored in the shared cloud storage, and is processed using the shared cloud compute services. Security controls for the SaaS Cloud are perodically audited internally and externally, and you can [request](https://security.wandb.ai/) the SOC2 report and other security & compliance documents if needed.

With the [enterprise pricing plan](https://wandb.ai/site/pricing), you can bring your own bucket using the [secure storage connector](../secure-storage-connector.md) at the team level to store a team's files including datasets, models etc. You can configure secure storage connector for one or more W&B teams in your account, optionally using separate buckets for different teams. If you use the default W&B managed storage bucket to store files, you may be subject to limits depending on your pricing plan.

![](/images/hosting/saas_cloud_arch.png)

You can also use the security capabilities like Single-Sign On, Role-based Access Control, Restricted Projects with the enterprise pricing plan. Submit [this form](https://wandb.ai/site/for-enterprise/multi-tenant-saas-trial) if you are interested in the enterprise plan.