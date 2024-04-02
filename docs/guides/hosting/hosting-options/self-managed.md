---
title: Self managed
description: Deploying W&B in production
displayed_sidebar: default
---

:::info
W&B recommends fully managed deployment options such as [W&B SaaS Cloud](../hosting-options/saas_cloud.md) or [W&B Dedicated Cloud](../hosting-options//dedicated_cloud.md) deployment types. W&B fully managed services are simple and secure to use, with minimum to no configuration required.
:::

Deploy W&B Server on your [bare servers (on-prem)](#on-prem-or-non-supported-cloud) or within [AWS, GCP, or Azure cloud that you manage](#self-managed-cloud-accounts). 

Your IT/DevOps/MLOps team is responsible for provisioning your deployment, managing upgrades, and continuously maintaining your self managed W&B Server instance.


<!-- Check [Obtain your W&B Server license](#obtain-your-wb-server-license) to complete the setup. -->

## Deploy W&B Server within self managed cloud accounts

Deploy and manage W&B Server with AWS, GCP, or Azure cloud. W&B recommends that you use official W&B Terraform scripts to deploy W&B Server into AWS, GCP, or Azure.

See specific cloud provider documentation for more information on how to set up W&B Server in [AWS](../self-managed/aws-tf.md), [GCP](../self-managed/gcp-tf.md) or [Azure](../self-managed/azure-tf.md).

## Deploy W&B Server on premises

Deploy W&B Server on your on-prem, bare-metal infrastructure with Terraform scripts. 

You need to configure several infrastructure components in order to set up W&B Server on the required infrastructure. Some of those components include include, but are not limited to: 

- (Strongly recommended) Kubernetes cluster
- MySQL 8 database cluster
- Amazon S3-compatible object storage
- Redis cache cluster

See [Install on on-prem infrastructure](../self-managed/bare-metal.md) for more information on how to to install W&B Server on your on-prem infrastructure. W&B can provide recommendations for the different components and provide guidance through the installation process.


## Deploy W&B Server on a custom cloud platform
Deploy W&B Server with Terraform scripts managed by W&B to a cloud platform that is not AWS, GCP, or Azure. Similar to on-premises requirements, you must configure several infrastructure components to set up W&B Server on your  infrastructure. Some of those components include include, but are not limited to: 

- (Strongly recommended) Kubernetes cluster
- MySQL 8 database cluster
- Amazon S3-compatible object storage
- Redis cache cluster

See [Install on on-prem infrastructure](../self-managed/bare-metal.md) for more information on how to to install W&B Server on your on-prem infrastructure. W&B can provide recommendations for the different components and provide guidance through the installation process.




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
If you need an enterprise license for W&B Server which includes support for important security & other enterprise-friendly capabilities, [submit this form](https://wandb.ai/site/for-enterprise/self-hosted-trial) or reach out to your W&B team.
:::