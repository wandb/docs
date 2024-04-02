---
displayed_sidebar: default
---

# Dedicated cloud (Single-tenant SaaS)

W&B Dedicated Cloud is a single-tenant, fully-managed platform deployed in W&B's AWS, GCP or Azure cloud accounts. Each Dedicated Cloud instance has its own isolated network, compute and storage from other W&B Dedicated Cloud instances. Your W&B specific metadata and data is stored in an isolated cloud storage and is processed using isolated cloud compute services. 

W&B Dedicated Cloud is available in [multiple global regions for each cloud provider](./dedicated_regions.md).




## Data security 
Configure a bucket to store artifacts and other sensitive data, restrict who can access your deployment with IP allowlisting, and how users connect to your deployment with a secure private connection. 

You can bring your own bucket(BYOB) using the [secure storage connector](../secure-storage-connector.md) at the [instance and team levels](../secure-storage-connector.md#configuration-options) to store your files such as models, datasets, and more.

Similar to W&B SaaS Cloud, you can configure a single bucket for multiple teams or you can use separate buckets for different W&B Teams. For more information, see the [BYOB(Secure storage connector)](../secure-storage-connector.md). If you do not configure secure storage connector for a team, that data is stored in the instance level bucket.

![](/images/hosting/dedicated_cloud_arch.png)

In addition to BYOB with secure storage connector, you can utilize [IP allowlisting](../ip-allowlisting.md) to restrict access to your Dedicated Cloud instance. 

Connect to your W&B Dedicated Cloud deployment with a [cloud provider's secure private network](../private-connectivity.md). This feature is currently available for AWS instances of Dedicated Cloud that use [AWS PrivateLink](https://aws.amazon.com/privatelink/).


## Identity and access management (IAM)
Use identity and access managements tools to authenticate and authorize users in your W&B Organization. The following tools are available for IAM in W&B Dedicated Cloud deployments:

* Authenticate with [SSO using OpenID Connect (OIDC)](../iam/sso.md) or with an [LDAP server](../iam/ldap.md).
* Define the scope of a W&B project to limit who can view, edit, and submit W&B runs to it with [restricted projects](../restricted-projects.md).

## Monitor
Use [Audit logs](../audit-logging.md) to track user activity within your teams, and to conform to your enterprise governance requirements. View organization usage of W&B with [W&B Organization Dashboards](../org_dashboard.md).


Monitor your W&B Dedicated Cloud deployment with [Prometheus](../prometheus-logging.md).



## Maintenance
Similar to W&B SaaS Cloud, you do not incur the overhead and costs of provisioning and maintaining the W&B platform with Dedicated Cloud.

To understand how W&B manages updates on Dedicated Cloud, refer to the [server release process](../server-release-process.md).

## Compliance 
Security controls for W&B Dedicated Cloud are periodically audited internally and externally. Reach out to the W&B Security Team at https://security.wandb.ai/ to request the SOC2 report and other security and compliance documents.

## Migration options
Migration to and from W&B Dedicated Cloud to Private Cloud and On-prem is supported.

## Next steps
Submit [this form](https://wandb.ai/site/for-enterprise/dedicated-saas-trial) if you are interested in using Dedicated Cloud.

