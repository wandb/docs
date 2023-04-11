---
slug: /guides/hosting-options/intro
description: Deploying W&B in production
---

# W&B Server Hosting Options

There are three ways to deploy W&B Server:

- W&B managed Dedicated Cloud
- Customer managed OnPrem Private Cloud
- Customer managed OnPrem Bare Metal

## W&B Managed Dedicated Cloud

_Shared responsibility:_

_W&B_

- W&B hosts the docker container and runs it on Kubernetes
- W&B hosts the MySQL database that stores the high level metadata information about the experiments
- W&B performs the upgrades and maintenance of the systems
- (Optional - Recommended) W&B hosts the object storage that stores model artifact data like saved models, checkpoint files, model weights etc.

_Customer Cloud_

- (Optional) Customers have the option to bring their own S3 compatible object storage bucket and connect to the rest of the systems running on W&B cloud.
- Customers can bring their own KMS key. W&B uses this key to encrypt all communication between W&B services. We recommend customers to setup their own encryption key so that they can continue to adhere to any internal security policies of key rotation, compromised key replacements etc.

## Customer Managed On-Prem Private Cloud

_W&B:_

- W&B provides official terraform scripts to help customers deploy into their own on premises Private Cloud (currently supports AWS, GCP & Azure)

_Customer:_

- Customer is responsible for installing W&B server in a fully scalable environment with support from W&B and using the official terraform scripts.
- While official terraform scripts can be modified to fit specific needs, W&B strongly recommends using the terraform as-is to avoid any discrepancies when debugging issues in the future.
- Customer would be responsible for regular updates and maintenance of the systems including the docker container, database and the storage bucket.
- W&B releases latest docker containers with new features and bug fixes every 2 weeks. Customers are expected to upgrade at this cadence to avoid any issues.
- Customer would be responsible for tuning the MySQL database as per W&B recommendations.
- Customer would be responsible for setting up Monitoring and Alerting on important events like:
  - high CPU resources utilization
  - high memory utilization

## Customer Managed On-Prem Bare Metal

_W&B:_

- W&B advises on all the architectural components that need to be installed for W&B to be fully operational and scalable.
- W&B, in some cases, also provides recommendations of compatible solutions that customers can use to work with the W&B application.

_Customer:_

- Customer is responsible for installing W&B server docker container either in an on-prem Kubernetes environment (recommended) or in a docker environment.
- Customer is also responsible for setting up and external MySQL 8 database - this is used to store the metadata from the W&B application.
- Customer would also be responsible for setting up a fully external and scalable S3-compatible object storage solution - this is used to store most file like data logged to the W&B application.
- Customer would be responsible for regular updates and maintenance of the systems including the docker container, database and the storage bucket.
- Customer would be responsible for tuning the MySQL database as per W&B recommendations.
- Customer would be responsible for setting up Monitoring and Alerting on important events like:
  - high CPU resources utilization
  - high memory utilization

## Obtain your license

Talk to your W&B Sales Team to obtain a license. The sales team will provide a URL that you can use to create a local license and deployment.

The URL will redirect you to a **Get a License for W&B Local** form. Provide the following information:

1. Choose a deployment type from the **Choose Platform** step.
2. Select the owner of the license or add a new organization in the **Basic Information** step.
3. Provide a name for the instance in the **Name of Instance** field and optionally provide a description in the **Description** field in the **Get a License** step.
4. Select the **Generate License Key** button.

A page with an overview of your deployment along with licenses associated to the instance will be displayed.

For information on how to set up your deployment type, see [INSERT].
