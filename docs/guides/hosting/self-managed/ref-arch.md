---
title: Reference Architecture
description: W&B Reference Architecture.
displayed_sidebar: default
---


This page describes a reference architecture for a Weights & Biases deployment and outlines the recommended infrastructure and resources to support a production deployment of the platform.

Depending on your chosen deployment environment for Weights & Biases (W&B), various services can help to enhance the resiliency of your deployment.

For instance, major cloud providers offer robust managed database services, which remove the complexity of configuring and maintaining database clustering and failover mechanisms.

This reference architecture addresses some common deployment scenarios, ensuring seamless integration with cloud vendor services for optimal performance and reliability.

## Before you start

Running any application in production comes with its own set of challenges, and W&B is no exception. While we aim to streamline the process, certain complexities may arise depending on your unique architecture and design decisions. Typically, managing a production deployment involves overseeing various components, including hardware, operating systems, networking, storage, security, the W&B platform itself, and other dependencies. This responsibility extends to both the initial setup of the environment and its ongoing maintenance.

Consider carefully whether a self-managed approach with W&B is suitable for your team and specific requirements.

A strong understanding of how to run and maintain  production-grade application is an important prerequisite before you deploy self-managed W&B. If your team needs assistance, our Professional Services team and partners offer support for implementation and optimization.

For those who prefer a managed solution over a self-hosted deployment, we offer alternatives such as [W&B Multi-tenant Cloud](../hosting-options/saas_cloud.md) and [W&B Dedicated Cloud](../hosting-options/dedicated_cloud.md).

## Infrastructure

![W&B infrastructure diagram](/images/hosting/reference_architecture.png)

### Application layer

The application layer consists of a multi-node Kubernetes cluster, with resilience against node failures. The Kubernetes cluster runs and maintains W&B's pods.

### Storage layer

The storage layer consists of a MySQL database and object storage. The MySQL database stores metadata and the object storage stores artifacts such as models and datasets.

## Infrastructure requirements

### Kubernetes
The W&B server application is deployed as a [Kubernetes Operator](../operator.md) that deploys multiple Pods. For this reason, W&B requires a Kubernetes cluster with:
- A fully configured and functioning Ingress controller
- The capability to provision Persistent Volumes.

### MySQL
W&B uses a MySQL database as a metadata store. The shape of the model parameters and related metadata impact the performance of the database. The database size grows as the ML practitioners track more training runs, and incurs read heavy load when queries are executed in run tables, users workspaces, and reports.

Consider the following when you deploy a self-managed MySQL database:

- **Backups**. You should periodically back up the database to a separate facility. W&B recommends daily backups with at least 1 week of retention.
- **Performance.** The disk the server is running on should be fast. W&B recommends running the database on an SSD or accelerated NAS.
- **Monitoring.** The database should be monitored for load. If CPU usage is sustained at > 40% of the system for more than 5 minutes it is likely a good indication the server is resource starved.
- **Availability.** Depending on your availability and durability requirements you might want to configure a hot standby on a separate machine that streams all updates in realtime from the primary server and can be used to failover to in the event that the primary server crashes or become corrupted.


### Object storage
W&B requires object storage with Pre-signed URL and CORS support, deployed in Amazon S3, Azure Cloud Storage, Google Cloud Storage, or a storage service compatible with Amazon S3.service) 

### Versions
* Kubernetes: at least version 1.29.
* MySQL: at least 8.0.


### Networking

In a deployment connected a public or private network, egress to the following endpoints is required during installation _and_ during runtime:
    * `https://deploy.wandb.ai`
    * `https://charts.wandb.ai`
    * `https://docker.io`
    * `https://quay.io`
    * `https://gcr.io`

Access to W&B and to the object storage is required for the training infrastructure and for each system that tracks the needs of experiments.

### DNS
The fully qualified domain name (FQDN) of the W&B deployment must resolve to the IP address of the ingress/load balancer using an A record.

### SSL/TLS
W&B requires a valid signed SSL/TLS certificate for secure communication between clients and the server. SSL/TLS termination must occur on the ingress/load balancer. The W&B server application does not terminate SSL or TLS connections.

Please note: W&B does not recommend the use self-signed certificates and custom CAs.

### Supported CPU architectures
W&B runs on the Intel (x86) CPU architecture. ARM is not supported.

## Infrastructure provisioning
Terraform is the recommended way to deploy W&B for production. Using Terraform, you define the required resources, their references to other resources, and their dependencies. W&B provides Terraform modules for the major clouds. For details, refer to [Deploy W&B Server within self managed cloud accounts](../hosting-options/self-managed#deploy-wb-server-within-self-managed-cloud-accounts).

## Sizing

Use the following general guidelines as a starting point when planning a deployment. W&B recommends that you monitor all components of a new deployment closely and that you make adjustments based on observed usage patterns.

### Models only

#### Kubernetes

| Environment      | CPU	            | Memory	         | Disk               | 
| ---------------- | ------------------ | ------------------ | ------------------ | 
| Test/Dev         | 2 cores            | 16 GB              | 100 GB             |
| Production       | 8 cores            | 64 GB              | 100 GB             |

Numbers are per Kubernetes worker node.

#### MySQL

| Environment      | CPU	            | Memory	         | Disk               | 
| ---------------- | ------------------ | ------------------ | ------------------ | 
| Test/Dev         | 2 cores            | 16 GB              | 100 GB             |
| Production       | 8 cores            | 64 GB              | 500 GB             |

Numbers are per MySQL node.


### Weave only

#### Kubernetes

| Environment      | CPU                | Memory             | Disk               | 
| ---------------- | ------------------ | ------------------ | ------------------ | 
| Test/Dev         | 4 cores            | 32 GB              | 100 GB             |
| Production       | 12 cores           | 96 GB              | 100 GB             |

Numbers are per Kubernetes worker node.

#### MySQL

| Environment      | CPU                | Memory             | Disk               | 
| ---------------- | ------------------ | ------------------ | ------------------ | 
| Test/Dev         | 2 cores            | 16 GB              | 100 GB             |
| Production       | 8 cores            | 64 GB              | 500 GB             |

Numbers are per MySQL node.

### Models and Weave

#### Kubernetes

| Environment      | CPU                | Memory             | Disk               | 
| ---------------- | ------------------ | ------------------ | ------------------ | 
| Test/Dev         | 4 cores            | 32 GB              | 100 GB             |
| Production       | 16 cores           | 128 GB             | 100 GB             |

Numbers are per Kubernetes worker node.

#### MySQL

| Environment      | CPU                | Memory             | Disk               | 
| ---------------- | ------------------ | ------------------ | ------------------ | 
| Test/Dev         | 2 cores            | 16 GB              | 100 GB             |
| Production       | 8 cores            | 64 GB              | 500 GB             |

Numbers are per MySQL node.

## Recommended cloud provider services and machine types

### Services

| Cloud       | Kubernetes	 | MySQL	                | Object Storage             |   
| ----------- | ------------ | ------------------------ | -------------------------- | 
| AWS         | EKS          | RDS Aurora               | S3                         |
| GCP         | GKE          | Google Cloud SQL - Mysql | Google Cloud Storage (GCS) |
| Azure       | AKS          | Azure Database for Mysql | Azure Blob Storage         |


### Machine types

#### AWS

| Environment | K8s (Models only)  | K8s (Weave only)   | K8s (Models&Weave)  | MySQL	           |  
| ----------- | ------------------ | ------------------ | ------------------- | ------------------ |  
| Test/Dev    | r6i.large          | r6i.xlarge         | r6i.xlarge          | db.r6g.large       | 
| Production  | r6i.2xlarge        | r6i.4xlarge        | r6i.4xlarge         | db.r6g.2xlarge     | 

Machine type is per node.


#### GCP

| Environment | K8s (Models only)  | K8s (Weave only)   | K8s (Models&Weave)  | MySQL              |  
| ----------- | ------------------ | ------------------ | ------------------- | ------------------ |  
| Test/Dev    | n2-highmem-2       | n2-highmem-4       | n2-highmem-4        | db-n1-highmem-2    | 
| Production  | n2-highmem-8       | n2-highmem-16      | n2-highmem-16       | db-n1-highmem-8    | 

Machine type is per node.


#### Azure

| Environment | K8s (Models only)  | K8s (Weave only)   | K8s (Models&Weave)  | MySQL               |  
| ----------- | ------------------ | ------------------ | ------------------- | ------------------- |  
| Test/Dev    | Standard_E2_v5     | Standard_E4_v5     | Standard_E4_v5      | MO_Standard_E2ds_v4 | 
| Production  | Standard_E8_v5     | Standard_E16_v5    | Standard_E16_v5     | MO_Standard_E8ds_v4 | 

Machine type is per node.
