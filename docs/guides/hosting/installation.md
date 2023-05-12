---
slug: /guides/hosting/installation
description: W&B Server Install
---

# Install

## W&B Production and Develpoment

In this page you will find all available installation methods of W&B server.

The production installation types available are the following:

- [AWS](/guides/hosting/installation/aws-tf)
- [Azure](/guides/hosting/installation/azure-tf)
- [GCP](/guides/hosting/installation/gcp-tf)
- [Bare Metal](/guides/hosting/installation/bare-metal)

For all of our Cloud Deployments, we rely on [Terraform](https://developer.hashicorp.com/terraform/intro) as a tool to provision all infrastructure components necessary to execute the W&B Server reliably.

:::info
We recommend you choose one of the [remote backends](https://developer.hashicorp.com/terraform/language/settings/backends/configuration) available for Terraform to store the [State File](https://developer.hashicorp.com/terraform/language/state).

The State File is the necessary resource to roll out upgrades or make changes in your deployment without recreating all components.
:::

To allow the users try Weights and Biases server without having to provisioning the whole infrastructure, it's possible to run W&B Server locally.

- [Development Setup](/guides/hosting/installation/dev-setup)

This mode is **NOT RECOMMENDED FOR PRODUCTION**

# Upgrade

- [Upgrade W&B Server](/guides/hosting/installation/upgrade) 
