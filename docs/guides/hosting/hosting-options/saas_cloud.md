---
title: SaaS Cloud
displayed_sidebar: default
---

# SaaS Cloud (Multi-tenant SaaS)

W&B SaaS Cloud is a multi-tenant, fully-managed platform deployed in W&B's Google Cloud Platform (GCP) account in [GPC's North America regions](https://cloud.google.com/compute/docs/regions-zones). W&B SaaS Cloud utilizes autoscaling in GCP to ensure that W&B scales appropriately based on increases or decreases in traffic. 

![](/images/hosting/saas_cloud_arch.png)
## Data security

For non enterprise users, data from both W&B and other W&B SaaS Cloud customers are stored in a shared cloud storage. That data stored in the shared cloud storage from both W&B and W&B SaaS Cloud customer are processed with shared cloud compute services. Depending on your pricing plan, you may be subject to storage limits if you use the default W&B managed storage bucket to store your files.


For enterprise users, you can [bring your own bucket(BYOB) using the secure storage connector](../secure-storage-connector.md) to store files at the [Team level](../secure-storage-connector.md#configuration-options). You can configure a single bucket for multiple teams or you can use separate buckets for different W&B Teams. For more information, see the [BYOB(Secure storage connector)](../secure-storage-connector.md).

:::note
W&B SaaS Cloud only supports BYOB at the [Team level](../secure-storage-connector.md#configuration-options). 
:::


## Identity and access management (IAM)
For enterprise users, you can use identity and access managements tools to authenticate and authorize users in your W&B Organization. The following tools are available for IAM in W&B Dedicated Cloud deployments:

* Authenticate with [SSO using OpenID Connect (OIDC)](../iam/sso.md) or with an [LDAP server](../iam/ldap.md).
* Define the scope of a W&B project to limit who can view, edit, and submit W&B runs to it with [restricted projects](../restricted-projects.md).


## Monitor
View organization usage of W&B with [W&B Organization Dashboards](../org_dashboard.md).

## Maintenance  
W&B SaaS Cloud is a multi-tenant, fully-managed platform. Since W&B SaaS Cloud is managed by W&B, you do not incur the overhead and costs of provisioning and maintaining the W&B platform.

## Compliance 
Security controls for W&B SaaS Cloud are periodically audited internally and externally. Reach out to the W&B Security Team at https://security.wandb.ai/ to request the SOC2 report and other security and compliance documents.

## Migration options
Migration to and from W&B SaaS Cloud to other deployment types is currently unavailable. 

## Next steps
Submit [this form](https://wandb.ai/site/for-enterprise/multi-tenant-saas-trial) for more information about the W&B Enterprise plans.