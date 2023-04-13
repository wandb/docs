---
description: Deploying W&B in production
---

# Self managed hosting

:::info
We recommend that you consider leveraging [W&B managed hosting options](./wb-managed.md) before privately hosting a W&B server on your infrastructure. The W&B cloud is simple and secure, with minimum to no configuration required.
:::

## On-prem Private Cloud

On-prem Private Cloud is a fully self-hosted solution where W&B server is running in a scalable deployment on customer's private cloud infrastructure. W&B recommends customers to use the official W&B terraform scripts to deploy into AWS/GCP/Azure. Customers can choose to deploy this in a region of their choice where all W&B services are available. The environment can be provisioned by us or by your company, using a toolset comprised of Terraform and Kubernetes. Upgrades and maintenance of the instance would need to be handled by customer's IT/DevOps/MLOps teams.

The simplest way to configure infrastructure is by using W&B's official terraform scripts:

- [Amazon Web Services (AWS)](https://github.com/wandb/terraform-aws-wandb)
- [Google Cloud Platform (GCP)](https://github.com/wandb/terraform-google-wandb)
- [Microsoft Azure](https://github.com/wandb/terraform-azurerm-wandb)

## On-prem Bare Metal

This is a fully self-hosted solution where W&B server is running in a scalable deployment on customer's on-prem bare-metal infrastructure. There are several infrastructure pieces needed to setup and configure W&B server in an on-prem bare metal installation including but not limited to

- a fully scalable MySQL 8 database
- an S3-compatible object storage
- a message queue and
- a redis cache (optional)

W&B can provide recommendations for compatible database engines, object stores and have an experienced team to help with the installation process. The complexity of administrating a database, creating and maintaining a distributed object storage system adds additional overhead to the customer's IT/DevOps/MLOps teams. When possible, W&B recommends using W&B managed cloud solutions for better user experience.
