---
menu:
  default:
    identifier: dedicated-cloud
    parent: deployment-options
title: Dedicated Cloud
url: guides/hosting/hosting-options/dedicated_cloud
---

## Use dedicated cloud (Single-tenant SaaS)

W&B Dedicated Cloud is a single-tenant, fully managed platform deployed in W&B's AWS, GCP or Azure cloud accounts. Each Dedicated Cloud instance has its own isolated network, compute and storage from other W&B Dedicated Cloud instances. Your W&B specific metadata and data is stored in an isolated cloud storage and is processed using isolated cloud compute services. 

W&B Dedicated Cloud is available in [multiple global regions for each cloud provider]({{< relref "./dedicated_regions.md" >}})

## Data security 

You can bring your own bucket (BYOB) using the [secure storage connector]({{< relref "/guides/hosting/data-security/secure-storage-connector.md" >}}) at the [instance and team levels]({{< relref "/guides/hosting/data-security/secure-storage-connector.md#configuration-options" >}}) to store your files such as models, datasets, and more.

Similar to W&B Multi-tenant Cloud, you can configure a single bucket for multiple teams or you can use separate buckets for different teams. If you do not configure secure storage connector for a team, that data is stored in the instance level bucket.

{{< img src="/images/hosting/dedicated_cloud_arch.png" alt="" >}}

In addition to BYOB with secure storage connector, you can utilize [IP allowlisting]({{< relref "/guides/hosting/data-security/ip-allowlisting.md" >}}) to restrict access to your Dedicated Cloud instance from only trusted network locations. 

You can also privately connect to your Dedicated Cloud instance using [cloud provider's secure connectivity solution]({{< relref "/guides/hosting/data-security/private-connectivity.md" >}}).

## Identity and access management (IAM)

Use the identity and access management capabilities for secure authentication and effective authorization in your W&B Organization. The following features are available for IAM in Dedicated Cloud instances:

* Authenticate with [SSO using OpenID Connect (OIDC)]({{< relref "/guides/hosting/iam/authentication/sso.md" >}}) or with [LDAP]({{< relref "/guides/hosting/iam/authentication/ldap.md" >}}).
* [Configure appropriate user roles]({{< relref "/guides/hosting/iam/access-management/manage-organization.md#assign-or-update-a-users-role" >}}) at the scope of the organization and within a team.
* Define the scope of a W&B project to limit who can view, edit, and submit W&B runs to it with [restricted projects]({{< relref "/guides/hosting/iam/access-management/restricted-projects.md" >}}).
* Leverage JSON Web Tokens with [identity federation]({{< relref "/guides/hosting/iam/authentication/identity_federation.md" >}}) to access W&B APIs.

## Monitor

Use [Audit logs]({{< relref "/guides/hosting/monitoring-usage/audit-logging.md" >}}) to track user activity within your teams and to conform to your enterprise governance requirements. Also, you can view organization usage in our Dedicated Cloud instance with [W&B Organization Dashboard]({{< relref "/guides/hosting/monitoring-usage/org_dashboard.md" >}}).

## Maintenance

Similar to W&B Multi-tenant Cloud, you do not incur the overhead and costs of provisioning and maintaining the W&B platform with Dedicated Cloud.

To understand how W&B manages updates on Dedicated Cloud, refer to the [server release process]({{< relref "/guides/hosting/hosting-options/self-managed/server-upgrade-process.md" >}}).

## Compliance

Security controls for W&B Dedicated Cloud are periodically audited internally and externally. Refer to the [W&B Security Portal](https://security.wandb.ai/) to request the security and compliance documents for your product assessment exercise.

## Migration options

Migration to Dedicated Cloud from a [Self-managed instance]({{< relref "/guides/hosting/hosting-options/self-managed/" >}}) or [Multi-tenant Cloud]({{< relref "/guides/hosting/hosting-options/saas_cloud.md" >}}) is supported.

## Next steps

Submit [this form](https://wandb.ai/site/for-enterprise/dedicated-saas-trial) if you are interested in using Dedicated Cloud.