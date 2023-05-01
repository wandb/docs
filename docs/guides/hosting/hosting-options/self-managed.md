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

### Infrastructure Guidelines

It is highly recommended to deploy W&B into a kubernetes cluster to fully take advantage of all the features of the product and to utilize the standardized [helm interface](https://github.com/wandb/helm-charts) that we provide.

If kubernetes is unavailable, W&B can be installed onto a bare-metal server and configured manually, however, future features may be broken out into k8s native or custom resource definitions and may not be backported into the standalone Docker Container. 

### Application Server

We recommend deploying W&B Application into its own namespace and a two availability zone node group with the following specifications to provide the best performance, reliability, unavailability:

| Specification              | Value                             |
|----------------------------|-----------------------------------|
| Bandwidth                  | Dual 10 Gigabit+ Ethernet Network |
| Root Disk Bandwidth (Mbps) | 4,750+                            |
| Root Disk Provision (GB)   | 100+                              |
| Core Count                 | 4                                 |
| Memory (GiB)               | 8                                 |

This ensures that W&B has sufficient disk space to process W&B server application data and store temporary logs before they are externalized. It also ensures fast and reliable data transfer, the necessary processing power and memory for smooth operation, and that W&B will not be affected by any noisy neighbors. 

It is important to keep in mind that these specifications are minimum requirements, and actual resource needs may vary depending on the specific usage and workload of the W&B application. Monitoring the resource usage and performance of the application is critical to ensure that it operates optimally and to make adjustments as necessary.


### Database Server

W&B recommends a [MySQL 8](../how-to-guides/bare-metal.md#mysql-80) database as a metadata store of all project, report, artifact, and run data. The shape of the ML practitioners parameters and metadata will greatly affect the performance of the database. The database is typically incrementally written to as SMLE’s track their training runs and is more read heavy when queries are executed in reports and dashboard.

To ensure optimal performance we recommend deploying the W&B database on to a server with the following starting specs:

| Specification              | Value                             |
|--------------------------- |-----------------------------------|
| Bandwidth                  | Dual 10 Gigabit+ Ethernet Network |
| Root Disk Bandwidth (Mbps) | 4,750+                            |
| Root Disk Provision (GB)   | 1000+                              |
| Core Count                 | 4                                 |
| Memory (GiB)               | 32                                |


Again, we recommend monitoring the resource usage and performance of the database to ensure that it operates optimally and to make adjustments as necessary.

Additionally, we recommend the following [parameter overrides](../how-to-guides/bare-metal.md#mysql-80) to tune the DB for MySQL 8.

### Object Storage

W&B has no hardware requirements for the on premises storage backend as long as it supports an S3 API interface, Signed URL’s, and a CORS policy. We recommend specing the storage array to the current needs of your ML practitioners and to capacity plan on a regular cadence.

The [following script](https://gist.github.com/vanpelt/2e018f7313dabf7cca15ad66c2dd9c5b) can be used to validate that the Object Store supports Signed URL’s.

The following CORS policy needs to be applied to the S3 Bucket.

``` xml
<?xml version="1.0" encoding="UTF-8"?>
<CORSConfiguration xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
<CORSRule>
    <AllowedOrigin>http://YOUR-W&B-SERVER-IP</AllowedOrigin>
    <AllowedMethod>GET</AllowedMethod>
    <AllowedMethod>PUT</AllowedMethod>
    <AllowedMethod>HEAD</AllowedMethod>
    <AllowedHeader>*</AllowedHeader>
</CORSRule>
</CORSConfiguration>
```

Some tested and working providers:
- [MinIO](https://min.io/)
- [Ceph](https://ceph.io/)
- [NetApp](https://www.netapp.com/)
- [Pure Storage](https://www.purestorage.com/)

##### Secure Storage Connector

The [Secure Storage Connector](../secure-storage-connector.md) is not available for teams at this time.

### Contact

If you have questions about planning an onpremises installation of W&B please take a look at our [On Prem / Baremetal](../how-to-guides/bare-metal.md) docs and reach out to us at support@wandb.com.
