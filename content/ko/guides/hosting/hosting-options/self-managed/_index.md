---
cascade:
- url: guides/hosting/self-managed/:filename
description: Deploying W&B in production
menu:
  default:
    identifier: ko-guides-hosting-hosting-options-self-managed-_index
    parent: deployment-options
title: Self-managed
url: guides/hosting/hosting-options/self-managed
---

## Use self-managed cloud or on-prem infrastructure

{{% alert %}}
W&B recommends fully managed deployment options such as [W&B Multi-tenant Cloud]({{< relref path="../saas_cloud.md" lang="ko" >}}) or [W&B Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ko" >}}) deployment types. W&B fully managed services are simple and secure to use, with minimum to no configuration required.
{{% /alert %}}

Deploy W&B Server on your [AWS, GCP, or Azure cloud account]({{< relref path="#deploy-wb-server-within-self-managed-cloud-accounts" lang="ko" >}}) or within your [on-premises infrastructure]({{< relref path="#deploy-wb-server-in-on-prem-infrastructure" lang="ko" >}}). 

Your IT/DevOps/MLOps team is responsible for provisioning your deployment, managing upgrades, and continuously maintaining your self managed W&B Server instance.

<!-- Check [Obtain your W&B Server license]({{< relref path="#obtain-your-wb-server-license" lang="ko" >}}) to complete the setup. -->

## Deploy W&B Server within self managed cloud accounts

W&B recommends that you use official W&B Terraform scripts to deploy W&B Server into your AWS, GCP, or Azure cloud account.

See specific cloud provider documentation for more information on how to set up W&B Server in [AWS]({{< relref path="/guides/hosting/hosting-options/self-managed/install-on-public-cloud/aws-tf.md" lang="ko" >}}), [GCP]({{< relref path="/guides/hosting/hosting-options/self-managed/install-on-public-cloud/gcp-tf.md" lang="ko" >}}) or [Azure]({{< relref path="/guides/hosting/hosting-options/self-managed/install-on-public-cloud/azure-tf.md" lang="ko" >}}).

## Deploy W&B Server in on-prem infrastructure

You need to configure several infrastructure components in order to set up W&B Server in your on-prem infrastructure. Some of those components include include, but are not limited to: 

- (Strongly recommended) Kubernetes cluster
- MySQL 8 database cluster
- Amazon S3-compatible object storage
- Redis cache cluster

See [Install on on-prem infrastructure]({{< relref path="/guides/hosting/hosting-options/self-managed/bare-metal.md" lang="ko" >}}) for more information on how to install W&B Server on your on-prem infrastructure. W&B can provide recommendations for the different components and provide guidance through the installation process.

## Deploy W&B Server on a custom cloud platform

You can deploy W&B Server to a cloud platform that is not AWS, GCP, or Azure. Requirements for that are similar to that for deploying in [on-prem infrastructure]({{< relref path="#deploy-wb-server-in-on-prem-infrastructure" lang="ko" >}}).

## Obtain your W&B Server license

You need a W&B trial license to complete your configuration of the W&B server. Open the [Deploy Manager](https://deploy.wandb.ai/deploy) to generate a free trial license. 

{{% alert %}}
If you do not already have a W&B account, create one to generate your free license.

If you need an enterprise license for W&B Server which includes support for important security & other enterprise-friendly capabilities, [submit this form](https://wandb.ai/site/for-enterprise/self-hosted-trial) or reach out to your W&B team.
{{% /alert %}}

The URL redirects you to a **Get a License for W&B Local** form. Provide the following information:

1. Choose a deployment type from the **Choose Platform** step.
2. Select the owner of the license or add a new organization in the **Basic Information** step.
3. Provide a name for the instance in the **Name of Instance** field and optionally provide a description in the **Description** field in the **Get a License** step.
4. Select the **Generate License Key** button.

A page displays with an overview of your deployment along with the license associated with the instance.