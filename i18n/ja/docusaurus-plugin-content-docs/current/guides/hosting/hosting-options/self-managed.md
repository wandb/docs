---
description: Deploying W&B in production
displayed_sidebar: default
---

# Self managed

:::info
We recommend that you consider leveraging [W&B managed hosting options](./wb-managed.md) before privately hosting a W&B server on your infrastructure. The W&B cloud is simple and secure, with minimum to no configuration required.
:::

## On-prem Private Cloud

Deploy W&B Server in your private cloud infrastructure. W&B recommends that you use the official W&B Terraform scripts to deploy into AWS, GCP, or Azure. You can choose to deploy W&B Server in the region of your choice. Ensure that all of your W&B services are accessibly and available in the same region. 

Your company or W&B can provision your environment with Terraform and Kubernetes.
The environment can be provisioned by us or by your company, using a toolset comprised of Terraform and Kubernetes. Upgrades and maintenance of the instance would need to be handled by customer's IT/DevOps/MLOps teams.

The simplest way to configure infrastructure is by using W&B's official terraform scripts:

- [Amazon Web Services (AWS)](https://github.com/wandb/terraform-aws-wandb)
- [Google Cloud Platform (GCP)](https://github.com/wandb/terraform-google-wandb)
- [Microsoft Azure](https://github.com/wandb/terraform-azurerm-wandb)

## On-prem Bare Metal
W&B Server runs on your on-prem, bare-metal infrastructure. There are several infrastructure pieces that you need to configure in order to set up W&B Server on your on-prem bare metal infrastructure. Some requirements you must satisfy include, but is not limited to: 

- a fully scalable MySQL 8 database
- an Amazon S3-compatible object storage
- a message queue and
- (optionally) a redis cache 

W&B can provide recommendations for compatible database engines, object stores and have an experienced team to help with the installation process. The complexity of administrating a database, creating and maintaining a distributed object storage system adds additional overhead to the customer's IT/DevOps/MLOps teams. 

:::info
When possible, W&B recommends using W&B managed cloud solutions for better user experience.
:::

### Contact

See the [On Prem / Baremetal](../how-to-guides/bare-metal.md) documentation if you have questions about planning an on premises installation of W&B and reach out to W&B Supported at support@wandb.com.
