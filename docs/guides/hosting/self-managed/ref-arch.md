---
title: Reference Architecture
description: W&B Reference Architecture.
displayed_sidebar: default
---

# Reference Architecture

Weights & Biases provides a reference architecture outlining the recommended infrastructure and resources needed to support a deployment of the platform.

Depending on your chosen deployment environment for Weights & Biases (W&B), various services are available to enhance the resiliency of your setup. For instance, major cloud providers offer robust managed database services, removing the complexity of configuring and maintaining database clustering and failover mechanisms. With these capabilities in mind, we’ve developed a reference architecture tailored to the most common deployment scenarios, ensuring seamless integration with cloud vendor services for optimal performance and reliability.

## Before You Start

Consider carefully whether a self-managed approach with W&B is suitable for your team and specific requirements.

Running any application in production comes with its own set of challenges, and W&B is no exception. While we aim to streamline the process, certain complexities may arise depending on your unique architecture and design decisions. Typically, managing a production deployment involves overseeing various components, including hardware, operating systems, networking, storage, security, the W&B platform itself, and other dependencies. This responsibility extends to both the initial setup of the environment and its ongoing maintenance.

It is essential to have a strong understanding of running and maintaining production-grade applications if you choose to self-manage W&B. If your team needs assistance, our Professional Services team and partners offer support for implementation and optimization. For those who prefer a more managed and worry-free solution, we offer alternatives such as [W&B Multi-tenant Cloud](../hosting-options/saas_cloud.md) and [W&B Dedicated Cloud](../hosting-options/dedicated_cloud.md) deployment types.

## Infrastructure diagram

![](/images/hosting/reference_architecture.png)

### Application layer

The application layer consists of a Kubernetes cluster, which should include multiple nodes to ensure resilience against node failures. W&B runs as pods on the Kubernetes cluster.

### Storage layer

The storage layer consists of a MySQL database and object storage. The MySQL database stores metadata and the object storage stores artifacts (models, datasets, and so on).

## Infrastructure Requirements

### Kubernetes
W&B requires a Kubernetes cluster with a deployed, configured and fully functioning Ingress controller (for example Contour, Nginx) as the W&B server application comes in the form of a [Kubernetes Operator](../operator.md). 

### MySQL
W&B requires a MySQL database as a metadata store. The shape of the model parameters and related metadata impact the performance of the database. The database size grows as the ML practitioners track more training runs, and incurs read heavy load when queries are executed in run tables, users workspaces, and reports.

Consider the following when you run your own MySQL database:

1. **Backups**. You should  periodically back up the database to a separate facility. W&B recommends daily backups with at least 1 week of retention.
2. **Performance.** The disk the server is running on should be fast. W&B recommends running the database on an SSD or accelerated NAS.
3. **Monitoring.** The database should be monitored for load. If CPU usage is sustained at > 40% of the system for more than 5 minutes it is likely a good indication the server is resource starved.
4. **Availability.** Depending on your availability and durability requirements you might want to configure a hot standby on a separate machine that streams all updates in realtime from the primary server and can be used to failover to in the event that the primary server crashes or become corrupted.


### Object storage
W&B requires object storage (Amazon S3, Azure Cloud Storage, Google Cloud Storage, or any S3-compatible storage service) with Pre-signed URL and CORS support.

### Versions
* Kubernetes: at least version 1.29.
* MySQL: at least 8.0.

## Other considerations

### Networking

In non-airgapped deployments, egress to the following endpoints during installation and during runtime is required:
    * deploy.wandb.ai
    * charts.wandb.ai
    * docker.io
    * quay.io
    * gcr.io

The training infrastracture as well as each desktop that tracks experiments needs access to W&B and to the object storage.

### DNS
The fully qualified domain name of W&B should resolve to the IP address of the ingress/load balancer using an A record.

### SSL/TLS
A valid, signed SSL/TLS certificate is required for secure communication between clients and W&B. SSL/TLS termination must be done on the ingress/load balancer. W&B server application does not terminate SSL/TLS.

Please note: We do not recommend the use self-signed certificates and custom CAs.

### Supported CPU architectures
Only x86 architecture is supported.

## Infrastructure provisioning
The recommended way to deploy W&B for production is through the use of Terraform configuration that defines the required resources, their references to other resources and dependencies. We provide Terraform modules for the major clouds. Please refer to [Deploy W&B Server within self managed cloud accounts](../hosting-options/self-managed#deploy-wb-server-within-self-managed-cloud-accounts).

## Sizing

The tables below offer general guidelines to use as a starting point. We recommend monitoring all components closely and adjusting the sizing based on actual usage patterns.

### W&B platform

#### Kubernetes worker nodes

| Environment      | CPU	            | Memory	         | Disk               | 
| ---------------- | ------------------ | ------------------ | ------------------ | 
| Test/Dev         | 2 cores            | 16 GB              | 100 GB             |
| Production       | 4 cores            | 32 GB              | 100 GB             |

Recommendations are per Kubernetes worker node.

#### MySQL

| Environment      | CPU	            | Memory	         | Disk               | 
| ---------------- | ------------------ | ------------------ | ------------------ | 
| Test/Dev         | 4 cores            | 16 GB              | 100 GB             |
| Production       | 8 cores            | 32 GB              | 500 GB             |

Recommendations are per MySQL node.

### W&B platform with Weave

#### Kubernetes worker nodes

| Environment      | CPU	            | Memory	         | Disk               | 
| ---------------- | ------------------ | ------------------ | ------------------ | 
| Test/Dev         | 4 cores            | 32 GB              | 100 GB             |
| Production       | 8 cores            | 64 GB              | 100 GB             |

Recommendations are per Kubernetes worker node.

#### MySQL

| Environment      | CPU	            | Memory	         | Disk               | 
| ---------------- | ------------------ | ------------------ | ------------------ | 
| Test/Dev         | 4 cores            | 16 GB              | 100 GB             |
| Production       | 8 cores            | 32 GB              | 500 GB             |

Recommendations are per MySQL node.


## Recommended Cloud Provider Services and Sizes

### Services

| Cloud       | Kubernetes	 | MySQL	                | Object Storage             |   
| ----------- | ------------ | ------------------------ | -------------------------- | 
| AWS         | EKS          | RDS Aurora               | S3                         |
| GCP         | GKE          | Google Cloud SQL - Mysql | Google Cloud Storage (GCS) |
| Azure       | AKS          | Azure Database for Mysql | Azure Blob Storage         |


### Flavors

| Cloud  | K8s worker nodes	  | MySQL	            |  
| ------ | ------------------ | ------------------- | 
| AWS    | m5.xlarge          | db.r5.large         | 
| GCP    | n1-standard-4      | db-n1-standard-2    | 
| Azure  | Standard_D4s_v3    | GP_Standard_D4ds_v4 | 

