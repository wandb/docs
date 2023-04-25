---
slug: /guides/hosting
description: Deploying W&B in production
---

# W&B Server

There are three ways to deploy W&B Server:

- **W&B managed Dedicated Cloud**: A fully managed solution hosted and maintained by W&B on W&B Cloud
- **Customer managed OnPrem Private Cloud**: A self managed solution hosted and maintained by customer on their Private Cloud
- **Customer managed OnPrem Bare Metal**: A self managed solution hosted and maintained by customer on their Bare Metal infrastructure

## Shared Responsibility Matrix

The following shared responsibility matrix outlines the respective responsibilities of W&B and the customer for each of the hosting options specified above.

![](/images/hosting/shared_responsibility_matrix.png)

## Obtain your license

Talk to your W&B Sales Team to obtain a license. The sales team will provide a URL that you can use to create a local license and deployment.

The URL will redirect you to a **Get a License for W&B Local** form. Provide the following information:

1. Choose a deployment type from the **Choose Platform** step.
2. Select the owner of the license or add a new organization in the **Basic Information** step.
3. Provide a name for the instance in the **Name of Instance** field and optionally provide a description in the **Description** field in the **Get a License** step.
4. Select the **Generate License Key** button.

A page with an overview of your deployment along with licenses associated to the instance will be displayed.

For information on how to set up your deployment type, see [our How-to guides](/guides/hosting/installation) section.

## W&B managed hosting

### SaaS Cloud

Our most popular deployment option. A Multi-Tenant SaaS offering that allows you access to a fast, secure version of W&B with all of the latest features.

:::info
We recommend that you consider using the wandb.ai cloud before privately hosting a W&B server on your infrastructure. The cloud is simple and secure, with no configuration required. [Click here](../../quickstart.md) to learn more.
:::

### Dedicated Cloud

Dedicated Cloud is a fully managed solution offered by W&B for organizations with sensitive use cases and stringent enterprise security controls. In Dedicated Cloud, W&B server is hosted in a dedicated virtual private network on W&B's single-tenant AWS, GCP or Azure infrastructure in your choice of cloud region. Customers have the option to use our Secure Storage Connector to connect your data to a scalable data store hosted on your company's private cloud.

Talk to our sales team by reaching out to contact@wandb.com.

## Self managed hosting

:::info
We recommend that you consider leveraging [W&B managed hosting options](#wb-managed-hosting) before privately hosting a W&B server on your infrastructure. The W&B cloud is simple and secure, with minimum to no configuration required.
:::

### On-prem Private Cloud

On-prem Private Cloud is a fully self-hosted solution where W&B server is running in a scalable deployment on customer's private cloud infrastructure. W&B recommends customers to use the official W&B terraform scripts to deploy into AWS/GCP/Azure. Customers can choose to deploy this in a region of their choice where all W&B services are available. The environment can be provisioned by us or by your company, using a toolset comprised of Terraform and Kubernetes. Upgrades and maintenance of the instance would need to be handled by customer's IT/DevOps/MLOps teams.

The simplest way to configure infrastructure is by using W&B's official terraform scripts:

- [Amazon Web Services (AWS)](https://registry.terraform.io/modules/wandb/wandb/aws/latest)
- [Google Cloud Platform (GCP)](https://registry.terraform.io/modules/wandb/wandb/google/latest)
- [Microsoft Azure](https://registry.terraform.io/modules/wandb/wandb/azurerm/latest)

### On-prem Bare Metal

Deploy Weights & Biases in a resource isolated environment managed by W&B or by yourself. W&B Server is shipped as a packaged Docker image that can be deployed easily into any underlying infrastructure. There are several ways to install and host the W&B Server in different environments.

:::info
Production-grade features for W&B Server are available for enterprise-tier only.

See the [Development Setup guide](/guides/hosting/installation/dev-setup) to set up a developer or trial environment.
:::

This is a fully self-hosted solution where W&B server is running in a scalable deployment on customer's on-prem bare-metal infrastructure. There are several infrastructure pieces needed to setup and configure W&B server in an on-prem bare metal installation including but not limited to

- a fully scalable MySQL 8 database
- an S3-compatible object storage
- a message queue and
- a redis cache (optional)

W&B can provide recommendations for compatible database engines, object stores and have an experienced team to help with the installation process. The complexity of administrating a database, creating and maintaining a distributed object storage system adds additional overhead to the customer's IT/DevOps/MLOps teams. When possible, W&B recommends using W&B managed cloud solutions for better user experience.

With W&B Server you can configure and leverage features such as:

- [Audit Logs](/guides/hosting/configuration/audit-logging)
- [Role-based Access Control via LDAP](/guides/hosting/configuration/ldap)
- [User Management](/guides/hosting/configuration/manage-users)
- [Prometheus Monitoring](/guides/hosting/configuration/prometheus-logging)
- [Secure Storage Connector](/guides/hosting/configuration/secure-storage-connector)
- [Slack Alerts](/guides/hosting/configuration/slack-alerts)
- [Single Sign-On](/guides/hosting/configuration/sso)
- Check the [Configurations](/guides/hosting/configuration) page for all available configuration

The following sections of the documentation describes different options on how to install W&B Server, the shared responsibility model, step-by-step installation and configuration guides.

## Recommendations

W&B recommends the following when configuring W&B Server:

1. Run the W&B Server Docker container with an external storage and an external MySQL database in order to preserve the state outside of a container. This protects the data from being accidentally deleted if the container dies or crashes.
2. Leverage Kubernetes to run the W&B Server Docker image and expose the `wandb` service.
3. Set up and manage a scale-able file system if you plan on using W&B Server for production-related work.

## System Requirements

W&B Server

- 4 cores of CPU
- 8GB of memory (RAM)
- 50GB of disk space

MySQL Database

- 4 cores of CPU
- 16GB of memory (RAM)
- 100GB of disk space

:::info
* These are minimum recommended resources and will vary according to the number of users logging experiments in parallel.
* When running [W&B for development](installation/dev-setup.md), your data will be saved on a local persistent volume.
* For [production-grade installation](/guides/hosting/installation), S3-compatible object storage and an external MySQL database are highly recommended.
:::

:::tip
For enterprise customers, W&B offers extensive technical support and frequent installation updates for privately hosted instances.
:::

## Releases

Subscribe to receive notifications from the [W&B Server GitHub repository](https://github.com/wandb/server/releases) when a new W&B Server release comes out.

To subscribe, select the **Watch** button at the top of the GitHub page and select **All Activity**.
