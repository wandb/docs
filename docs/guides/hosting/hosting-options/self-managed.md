---
description: Deploying W&B in production
displayed_sidebar: default
---

# Self managed
Deploy W&B Server on your bare servers (on-prem) or within AWS, GCP, or Azure cloud that you manage. 

:::info
W&B recommends fully managed platforms such as [SaaS Cloud](../hosting-options/saas_cloud.md) or [Dedicated Cloud](../hosting-options//dedicated_cloud.md) deployment types. W&B fully managed services are simple and secure to use, with minimum to no configuration required.
:::

Check [Obtain your W&B Server license](#obtain-your-wb-server-license) to complete the setup.

## Self managed cloud accounts

Deploy W&B Server in your own managed cloud accounts in AWS, GCP, or Azure cloud.  W&B recommends that you use official W&B Terraform scripts to deploy W&B Server into AWS, GCP, or Azure.

- [Amazon Web Services (AWS)](https://github.com/wandb/terraform-aws-wandb)
- [Google Cloud Platform (GCP)](https://github.com/wandb/terraform-google-wandb)
- [Microsoft Azure](https://github.com/wandb/terraform-azurerm-wandb)

<!-- You can deploy W&B Server in the region of your choice, provided the required W&B services are available in the chosen region.  -->

See specific cloud provider documentation for more information on how to set up W&B Server in [AWS](../how-to-guides/aws-tf.md), [GCP](../how-to-guides/gcp-tf.md) or [Azure](../how-to-guides/azure-tf.md).


:::note
Your IT/DevOps/MLOps team is responsible for provisioning your deployment, managing upgrades, and continuously maintenance of your self managed W&B Server instance.
:::
## On-prem or non-supported cloud

Deploy W&B Server runs on your on-prem, bare-metal infrastructure, or on a cloud that's not supported in Dedicated Cloud or with Terraform scripts. You need to configure several infrastructure components in order to set up W&B Server on the required infrastructure. Some of those components include include, but are not limited to: 

- [Strongly recommended] Kubernetes cluster
- MySQL 8 database cluster
- Amazon S3-compatible object storage
- Redis cache cluster

W&B can provide recommendations for the different components and provide guidance through the installation process.

Refer to the [On Prem / Baremetal](../how-to-guides/bare-metal.md) documentation.


## Obtain your W&B Server license

You need a W&B trial license to complete your configuration of the W&B server. Open the [Deploy Manager](https://deploy.wandb.ai/deploy) to generate a free trial license. 

:::note
If you do not already have a W&B SaaS Cloud account then you will need to create one to generate your free license.
:::

The URL will redirect you to a **Get a License for W&B Local** form. Provide the following information:

1. Choose a deployment type from the **Choose Platform** step.
2. Select the owner of the license or add a new organization in the **Basic Information** step.
3. Provide a name for the instance in the **Name of Instance** field and optionally provide a description in the **Description** field in the **Get a License** step.
4. Select the **Generate License Key** button.

A page with an overview of your deployment along with the license associated to the instance will be displayed.

:::info
If you need an enterprise license for W&B Server which includes support for important security & other enterprise-fiendly capabilities, [submit this form](https://wandb.ai/site/for-enterprise/self-hosted-trial) or reach out to your W&B team.
:::