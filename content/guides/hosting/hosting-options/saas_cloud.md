---
menu:
  default:
    identifier: saas_cloud
    parent: deployment-options
title: Use W&B Multi-tenant SaaS
weight: 1
---

W&B Multi-tenant Cloud is a fully managed platform deployed in W&B's Google Cloud Platform (GCP) account in [GPC's North America regions](https://cloud.google.com/compute/docs/regions-zones). W&B Multi-tenant Cloud utilizes autoscaling in GCP to ensure that the platform scales appropriately based on increases or decreases in traffic. 

{{< img src="/images/hosting/saas_cloud_arch.png" alt="SaaS Cloud architecture diagram" >}}

W&B Multi-tenant Cloud scales to meet your organization's needs, and supports logging up to 250,000 metrics per project with up to 1 million data points per metric. For larger deployments, contact [support](mailto:support@wandb.com).

## Data security

For non enterprise plan users, all data is only stored in the shared cloud storage and is processed with shared cloud compute services. Depending on your pricing plan, you may be subject to storage limits.

Enterprise plan users can [bring their own bucket (BYOB) using the secure storage connector]({{< relref "/guides/hosting/data-security/secure-storage-connector.md" >}}) at the [team level]({{< relref "/guides/hosting/data-security/secure-storage-connector.md#configuration-options" >}}) to store their files such as models, datasets, and more. You can configure a single bucket for multiple teams or you can use separate buckets for different W&B Teams. If you do not configure secure storage connector for a team, that data is stored in the shared cloud storage.

You are responsible for ensuring that your deployment complies with your organization's policies and [Security Technical Implementation Guidelines (STIG)](https://en.wikipedia.org/wiki/Security_Technical_Implementation_Guide), if applicable.

## Identity and access management (IAM)
If you are on enterprise plan, you can use the identity and access managements capabilities for secure authentication and effective authorization in your W&B Organization. The following features are available for IAM in Multi-tenant Cloud:

* SSO authentication with OIDC or SAML. Reach out to your W&B team or support if you would like to configure SSO for your organization.
* [Configure appropriate user roles]({{< relref "/guides/hosting/iam/access-management/manage-organization.md#assign-or-update-a-users-role" >}}) at the scope of the organization and within a team.
* Define the scope of a W&B project to limit who can view, edit, and submit W&B runs to it with [restricted projects]({{< relref "/guides/hosting/iam/access-management/restricted-projects.md" >}}).

## Monitor
Organization admins can manage usage and billing for their account from the `Billing` tab in their account view. If using the shared cloud storage on Multi-tenant Cloud, an admin can optimize storage usage across different teams in their organization.

## Maintenance
W&B Multi-tenant Cloud is a multi-tenant, fully managed platform. Since W&B Multi-tenant Cloud is managed by W&B, you do not incur the overhead and costs of provisioning and maintaining the W&B platform.

## Compliance 
Security controls for Multi-tenant Cloud are periodically audited internally and externally. Refer to the [W&B Security Portal](https://security.wandb.ai/) to request the SOC2 report and other security and compliance documents.

## Next steps
Access [Multi-tenant Cloud directly](https://wandb.ai) if you are looking for non-enterprise capabilities. To start with the enterprise plan, submit [this form](https://wandb.ai/site/for-enterprise/multi-tenant-saas-trial).