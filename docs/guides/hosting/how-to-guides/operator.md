---
description: Hosting W&B Server via Kubernetes Operator
displayed_sidebar: default
---

# Operator

# Understanding the W&B Operator

The W&B Kubernetes Operator is designed to leverage the Kubernetes [operator pattern](https://kubernetes.io/docs/concepts/extend-kubernetes/operator/) that significantly simplifies the deployment, administration, troubleshooting, and scalability of the W&B Server deployments on Kubernetes. The operator facilitates managing the lifecycle of different services making the product that is W&B Server more seamless. It is based on standardized predefined configurations, thus acting as a smart assistant for your W&B instance.

The operator allows for consistency in provisioning & operating the W&B Server deployments across public cloud and private infrastructure. W&B uses the operator to deploy and manage the Dedicated Cloud instances on AWS, GCP and Azure public clouds, and encourages to use it for customer-managed deployments too.

W&B Server architecture & design continuously evolves to expand the AI developer tooling capabilities for the users, and to have good primitives for high performance, better scalability, and easier administration. That evolution applies to the compute services, relevant storage and the connectivity between them. Operator is the standardized interface to easily roll out those improvements to customers across deployment types. At a future date, W&B shall deprecate the deployment mechanisms that do not use operator. If you self-manage your W&B Server instances, transitioning to the operator is strongly recommended. W&B could then roll out newer services & products to your instance more seamlessly, and provide better troubleshooting & support.

:::note
Operator for self-managed W&B Server deployments is in private preview. Reach out to [Customer Support](mailto:support@wandb.com) or your W&B team if you have any questions.
:::

## Pre-baked in Cloud Terraform Modules

The W&B Kubernetes Operator is pre-baked with the official W&B cloud-specific Terraform Modules with the following versions:

| Terraform Module                                 | Version |
| ------------------------------------------------ | ------- |
| https://github.com/wandb/terraform-aws-wandb     | v4.0.0+ |
| https://github.com/wandb/terraform-azurerm-wandb | v2.0.0+ |
| https://github.com/wandb/terraform-google-wandb  | v2.0.0+ |

This integration ensures that W&B Kubernetes Operator is ready to use for your instance with minimal setup, providing a streamlined path to deploying and managing W&B Server in your cloud environment.

## Deploying with Helm Terraform Module

Install the W&B Kubernetes Operator with the official W&B Terraform Module [terraform-helm-wandb](https://github.com/wandb/terraform-helm-wandb).

This method allows for customized deployments tailored to specific requirements, leveraging Terraform's infrastructure-as-code approach for consistency and repeatability.

:::note
For detailed instructions on how to use the operator, reach out to [Customer Support](mailto:support@wandb.com) or your W&B team.
:::
