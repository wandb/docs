---
displayed_sidebar: default
---

# Dedicated cloud (Single-tenant SaaS)

W&B Dedicated Cloud is the single-tenant, fully-managed platform deployed in W&B's AWS, GCP or Azure cloud accounts. You can choose from the available clouds and regions depending on your cloud strategy and data residency requirements. W&B supports multiple global regions in each cloud, with each of those regions satisfying the minimum requirements of the platform. Just like with SaaS Cloud, you do not incur the overhead and costs of provisioning and maintaining the W&B platform with Dedicated Cloud.

Each Dedicated Cloud instance has its own isolated network, compute and storage from the other instances. Your W&B specific metadata and data is stored in the isolated cloud storage, and is processed using the isolated cloud compute services. Security controls for the SaaS Cloud are perodically audited internally and externally, and you can [request](https://security.wandb.ai/) the SOC2 report and other security & compliance documents if needed.

You can bring your own bucket using the [secure storage connector](../secure-storage-connector.md) at both instance and team levels to store your files including datasets, models etc. Just like with SaaS Cloud, you can configure secure storage connector for one or more W&B teams in your Dedicated Cloud instance, optionally using separate buckets for different teams. If you do not configure secure storage connector for a team, it's data is stored in the instance level bucket.

![](/images/hosting/dedicated_cloud_arch.png)

Beyond the basic security capabilities like Single-Sign On, Role-based Access Control and Restricted Projects, you can also opt in to use the features like [Audit logs](../audit-logging.md), Secure private connectivity etc. Submit [this form](https://wandb.ai/site/for-enterprise/dedicated-saas-trial) if you are interested in using Dedicated Cloud.

To understand how W&B manages updates on Dedicated Cloud, refer to the [server release process](../server-release-process.md).