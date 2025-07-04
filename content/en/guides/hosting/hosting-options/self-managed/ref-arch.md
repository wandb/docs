---
title: Reference Architecture
description: W&B Reference Architecture
menu:
  default:
    identifier: ref-arch
    parent: self-managed
weight: 1
---


This page describes a reference architecture for a Weights & Biases deployment and outlines the recommended infrastructure and resources to support a production deployment of the platform.

Depending on your chosen deployment environment for Weights & Biases (W&B), various services can help to enhance the resiliency of your deployment.

For instance, major cloud providers offer robust managed database services which help to reduce the complexity of database configuration, maintenance, high availability, and resilience.

This reference architecture addresses some common deployment scenarios and shows how you can integrate your W&B deployment with cloud vendor services for optimal performance and reliability.

## Before you start

Running any application in production comes with its own set of challenges, and W&B is no exception. While we aim to streamline the process, certain complexities may arise depending on your unique architecture and design decisions. Typically, managing a production deployment involves overseeing various components, including hardware, operating systems, networking, storage, security, the W&B platform itself, and other dependencies. This responsibility extends to both the initial setup of the environment and its ongoing maintenance.

Consider carefully whether a self-managed approach with W&B is suitable for your team and specific requirements.

A strong understanding of how to run and maintain production-grade application is an important prerequisite before you deploy self-managed W&B. If your team needs assistance, our Professional Services team and partners offer support for implementation and optimization.

To learn more about managed solutions for running W&B instead of managing it yourself, refer to [W&B Multi-tenant Cloud]({{< relref "/guides/hosting/hosting-options/saas_cloud.md" >}}) and [W&B Dedicated Cloud]({{< relref "/guides/hosting/hosting-options/dedicated_cloud.md" >}}).

## Infrastructure

{{< img src="/images/hosting/reference_architecture.png" alt="W&B infrastructure diagram" >}}

### Application layer

The application layer consists of a multi-node Kubernetes cluster, with resilience against node failures. The Kubernetes cluster runs and maintains W&B's pods.

### Storage layer

The storage layer consists of a MySQL database and object storage. The MySQL database stores metadata and the object storage stores artifacts such as models and datasets.

## Infrastructure requirements

### Kubernetes
The W&B Server application is deployed as a [Kubernetes Operator]({{< relref "kubernetes-operator/" >}}) that deploys multiple pods. For this reason, W&B requires a Kubernetes cluster with:
- A fully configured and functioning Ingress controller.
- The capability to provision Persistent Volumes.

### MySQL
W&B stores metadata in a MySQL database. The database's performance and storage requirements depend on the shapes of the model parameters and related metadata. For example, the database grows in size as you track more training runs, and load on the database increases based on queries in run tables, user workspaces, and reports.

Consider the following when you deploy a self-managed MySQL database:

- **Backups**. You should periodically back up the database to a separate facility. W&B recommends daily backups with at least 1 week of retention.
- **Performance.** The disk the server is running on should be fast. W&B recommends running the database on an SSD or accelerated NAS.
- **Monitoring.** The database should be monitored for load. If CPU usage is sustained at > 40% of the system for more than 5 minutes it is likely a good indication the server is resource starved.
- **Availability.** Depending on your availability and durability requirements you might want to configure a hot standby on a separate machine that streams all updates in realtime from the primary server and can be used to failover to in the event that the primary server crashes or become corrupted.

### Object storage
W&B requires object storage with pre-signed URL and CORS support, deployed in one of:

- [CoreWeave AI Object Storage](https://docs.coreweave.com/docs/products/storage/object-storage) is a high-performance, S3-compatible object storage service optimized for AI workloads.
- [Amazon S3](https://aws.amazon.com/s3/) is an object storage service offering industry-leading scalability, data availability, security, and performance.
- [Google Cloud Storage](https://cloud.google.com/storage) is a managed service for storing unstructured data at scale.
- [Azure Blob Storage](https://azure.microsoft.com/products/storage/blobs) is a cloud-based object storage solution for storing massive amounts of unstructured data like text, binary data, images, videos, and logs.
- S3-compatible storage like [MinIO](https://github.com/minio/minio) hosted in your cloud or infrastructure on your premises.

### Versions
| Software     | Minimum version                              |
| ------------ | -------------------------------------------- |
| Kubernetes   | v1.29                                        |
| MySQL        | v8.0.0, "General Availability" releases only |

### Networking

For a networked deployment, egress to these endpoints is required during _both_ installation and runtime:
* https://deploy.wandb.ai
* https://charts.wandb.ai
* https://docker.io
* https://quay.io
* `https://gcr.io`

To learn about air-gapped deployments, refer to [Kubernetes operator for air-gapped instances]({{< relref "kubernetes-operator/operator-airgapped.md" >}}).
Access to W&B and to the object storage is required for the training infrastructure and for each system that tracks the needs of experiments.

### DNS
The fully qualified domain name (FQDN) of the W&B deployment must resolve to the IP address of the ingress/load balancer using an A record.

### SSL/TLS
W&B requires a valid signed SSL/TLS certificate for secure communication between clients and the server. SSL/TLS termination must occur on the ingress/load balancer. The W&B Server application does not terminate SSL or TLS connections.

Please note: W&B does not recommend the use self-signed certificates and custom CAs.

### Supported CPU architectures
W&B runs on the Intel (x86) CPU architecture. ARM is not supported.

## Infrastructure provisioning
Terraform is the recommended way to deploy W&B for production. Using Terraform, you define the required resources, their references to other resources, and their dependencies. W&B provides Terraform modules for the major cloud providers. For details, refer to [Deploy W&B Server within self managed cloud accounts]({{< relref "/guides/hosting/hosting-options/self-managed.md#deploy-wb-server-within-self-managed-cloud-accounts" >}}).

## Sizing
Use the following general guidelines as a starting point when planning a deployment. W&B recommends that you monitor all components of a new deployment closely and that you make adjustments based on observed usage patterns. Continue to monitor production deployments over time and make adjustments as needed to maintain optimal performance.

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

## Cloud provider instance recommendations

### Services

| Cloud       | Kubernetes	 | MySQL	                | Object Storage             |   
| ----------- | ------------ | ------------------------ | -------------------------- | 
| AWS         | EKS          | RDS Aurora               | S3                         |
| GCP         | GKE          | Google Cloud SQL - Mysql | Google Cloud Storage (GCS) |
| Azure       | AKS          | Azure Database for Mysql | Azure Blob Storage         |


### Machine types

These recommendations apply to each node of a self-managed deployment of W&B in cloud infrastructure.

#### AWS

| Environment | K8s (Models only)  | K8s (Weave only)   | K8s (Models&Weave)  | MySQL	           |  
| ----------- | ------------------ | ------------------ | ------------------- | ------------------ |  
| Test/Dev    | r6i.large          | r6i.xlarge         | r6i.xlarge          | db.r6g.large       | 
| Production  | r6i.2xlarge        | r6i.4xlarge        | r6i.4xlarge         | db.r6g.2xlarge     | 

#### GCP

| Environment | K8s (Models only)  | K8s (Weave only)   | K8s (Models&Weave)  | MySQL              |  
| ----------- | ------------------ | ------------------ | ------------------- | ------------------ |  
| Test/Dev    | n2-highmem-2       | n2-highmem-4       | n2-highmem-4        | db-n1-highmem-2    | 
| Production  | n2-highmem-8       | n2-highmem-16      | n2-highmem-16       | db-n1-highmem-8    | 

#### Azure

| Environment | K8s (Models only)  | K8s (Weave only)   | K8s (Models&Weave)  | MySQL               |  
| ----------- | ------------------ | ------------------ | ------------------- | ------------------- |  
| Test/Dev    | Standard_E2_v5     | Standard_E4_v5     | Standard_E4_v5      | MO_Standard_E2ds_v4 | 
| Production  | Standard_E8_v5     | Standard_E16_v5    | Standard_E16_v5     | MO_Standard_E8ds_v4 | 
