---
menu:
  default:
    identifier: w-b-platform
title: W&B Platform
weight: 6
no_list: true
---
W&B Platform is the foundational infrastructure, tooling and governance scaffolding which supports the W&B products like [Core]({{< relref "/guides/core" >}}), [Models]({{< relref "/guides/models/" >}}) and [Weave]({{< relref "/guides/weave/" >}}). 

W&B Platform is available in three different deployment options:

* [W&B Multi-tenant Cloud]({{< relref "#wb-multi-tenant-cloud" >}})
* [W&B Dedicated Cloud]({{< relref "#wb-dedicated-cloud" >}})
* [W&B Customer-managed]({{< relref "#wb-customer-managed" >}})

The following responsibility matrix outlines some of the key differences:

|                                      | Multi-tenant Cloud                | Dedicated Cloud                                                     | Customer-managed |
|--------------------------------------|-----------------------------------|---------------------------------------------------------------------|------------------|
| MySQL / DB management                | Fully hosted and managed by W&B     | Fully hosted & managed by W&B on cloud or region of customer choice | Fully hosted and managed by customer |
| Object Storage (S3/GCS/Blob storage) | **Option 1**: Fully hosted by W&B<br />**Option 2**: Customer can configure their own bucket per team, using the [Secure Storage Connector]({{< relref "/guides/hosting/data-security/secure-storage-connector.md" >}})  | **Option 1**: Fully hosted by W&B<br />**Option 2**: Customer can configure their own bucket per instance or team, using the [Secure Storage Connector]({{< relref "/guides/hosting/data-security/secure-storage-connector.md" >}}) | Fully hosted and managed by customer |
| SSO Support                          | W&B managed via Auth0             | **Option 1**: Customer managed<br />**Option 2**: Managed by W&B via Auth0 | Fully managed by customer   |
| W&B Service (App)                    | Fully managed by W&B              | Fully managed by W&B                                                | Fully managed by customer          |
| App security                         | Fully managed by W&B              | Shared responsibility of W&B and customer                           | Fully manbaged by customer         |
| Maintenance (upgrades, backups, etc.)| Managed by W&B | Managed by W&B | Managed by customer |
| Support                              | Support SLA                       | Support SLA                                                         | Support SLA |
| Supported cloud infrastructure       | GCP                               | AWS, GCP, Azure                                                     | AWS, GCP, Azure, On-Prem bare-metal |

## Deployment options
The following sections provide an overview of each deployment type. 

### W&B Multi-tenant Cloud
W&B Multi-tenant Cloud is a fully managed service deployed in W&B's cloud infrastructure, where you can seamlessly access the W&B products at the desired scale, with cost-efficient options for pricing, and with continuous updates for the latest features and functionalities. W&B recommends to use the Multi-tenant Cloud for your product trial, or to manage your production AI workflows if you do not need the security of a private deployment, self-service onboarding is important, and cost efficiency is critical.

See [W&B Multi-tenant Cloud]({{< relref "./hosting-options/saas_cloud.md" >}}) for more information. 

### W&B Dedicated Cloud
W&B Dedicated Cloud is a single-tenant, fully managed service deployed in W&B's cloud infrastructure. It is the best place to onboard W&B if your organization requires conformance to strict governance controls including data residency, have need of advanced security capabilities, and are looking to optimize their AI operating costs by not having to build & manage the required infrastructure with security, scale & performance characteristics.

See [W&B Dedicated Cloud]({{< relref "/guides/hosting/hosting-options/dedicated_cloud/" >}}) for more information.

### W&B Customer-Managed
With this option, you can deploy and manage W&B Server on your own managed infrastructure. W&B Server is a self-contained packaged mechanism to run the W&B Platform & its supported W&B products. W&B recommends this option if all your existing infrastructure is on-prem, or your organization has strict regulatory needs that are not satisfied by W&B Dedicated Cloud. With this option, you are fully responsible to manage the provisioning, and continuous maintenance & upgrades of the infrastructure required to support W&B Server.

See [W&B Self Managed]({{< relref "/guides/hosting/hosting-options/self-managed/" >}}) for more information.

## Next steps

If you're looking to try any of the W&B products, W&B recommends using the [Multi-tenant Cloud](https://wandb.ai/home). If you're looking for an enterprise-friendly setup, choose the appropriate deployment type for your trial [here](https://wandb.ai/site/enterprise-trial).