---
slug: /guides/hosting/intro
description: Deploying W&B in production
---

# W&B Server

W&B server is an option available for Enterprise customers who prefer to deploy Weights & Biases in a resource isolated environment either managed by W&B or by the customer themselves. W&B server comes with several Enterprise level options such as:

- Secure Storage Connector
- Role-based Access Control
- Single Sign-On
- LDAP integration
- Monitoring & Alerting
- …and much more

W&B server is shipped as a packaged docker image that can be deployed easily into any underlying infrastructure. In order to preserve state outside of the container, it's highly recommended to run the docker container with an external storage and an external Mysql database. This protects the data from getting accidentally deleted if the container dies or crashes. W&B also highly recommends leveraging Kubernetes to run the docker image and expose the `wandb` service.There are several ways to install and host the W&B server in various different environments. The following sections talk about options available to install W&B server, the shared responsibility model, step by step installation and configuration guides.

**System Requirements:**

W&B Server requires a machine with at least 4 cores and 8GB of memory to run. Your W&B data will be saved on a persistent volume or external database, ensuring that it is preserved across different versions of the container.

For enterprise customers, we offer extensive technical support and frequent installation updates for privately hosted instances. If you are planning on using the W&B Server for production related work, we recommend setting up and managing a scalable file system.

**Releases**

You can find information about our latest releases on our [official github repo here](https://github.com/wandb/server/releases). You can subscribe to automatic notifications on the releases by simply clicking `Watch` > `All Activity` on the page above.
