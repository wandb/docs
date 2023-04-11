---
slug: /guides/hosting/intro
description: Deploying W&B in production
---

# W&B Server
Deploy Weights & Biases in a resource isolate environment managed by W&B or by yourself. W&B Server is shipped as a packaged Docker image that can be deployed easily into any underlying infrastructure. There are several ways to install and host the W&B Server in different environments. 

:::info
W&B Server is an option available only for Enterprise-tier customers.
:::


With W&B Server you can configure:
- Secure Storage Connector
- Role-based Access Control via LDAP integrations
- Single Sign-On
- LDAP integration
- Monitoring & Alerting and more.

The following guide describes different options on how to install W&B Server, the shared responsibility model, step-by-step installation and configuration guides.

## Recommendations
W&B recommends the following when configuring W&B Server:

1. Run the W&B Server Docker container with an external storage and an external MySQL database in order to preserve the state outside of a container. This protects the data from being accidentally deleted if the container dies or crashes.
2. Leverage Kubernetes to run the W&B Server Docker image and expose the `wandb` service. 
3. Set up and manage a scale-able file system if you plan on using W&B Server for production-related work.



## System Requirements

W&B Server requires a machine with at least 4 cores and 8GB of memory to run. Your W&B data will be saved on a persistent volume or external database, ensuring that it is preserved across different versions of the container.

:::tip
For enterprise customers, W&B offers extensive technical support and frequent installation updates for privately hosted instances. 
:::

## Releases
Subscribe to receive notifications from the [W&B Server GitHub repository](https://github.com/wandb/server/releases) when a new W&B Server release comes out. 

To subscribe, select the **Watch** button at the top of the GitHub page and select **All Activity**. 
