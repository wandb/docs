---
title: Reference Architecture
description: W&B Reference Architecture.
displayed_sidebar: default
---

# Reference Architecture

Weights & Biases provides a reference architecture outlining the recommended infrastructure and resources needed to support a deployment of the platform.

Depending on where you choose to deploy Weights & Biases (W&B), there are different services available to enhance the resiliency of your setup. For example, most major cloud providers offer resilient managed database services, eliminating the need for users to manage complex database clustering and failover configurations. We’ve taken this into account and designed a reference architecture for the most common deployment scenarios, leveraging cloud vendor services effectively for optimal performance and reliability.

## Before You Start

Consider carefully whether a self-managed approach with W&B is suitable for your team and specific requirements.

Running any application in production presents inherent challenges, and W&B is no exception. We strive to simplify the process; however, complexities will arise due to your specific architecture and design choices. Typically, you will need to manage various aspects, such as hardware, operating systems, networking, storage, security, the W&B platform, and other dependencies. This includes both the initial environment setup and the ongoing maintenance and upgrades.

It is essential to have a strong understanding of running and maintaining production-grade applications if you choose to self-manage W&B. If your team needs assistance, our Professional Services team offers support for implementation and optimization. For those who prefer a more managed and worry-free solution, we offer alternatives such as [W&B Multi-tenant Cloud](../hosting-options/saas_cloud.md) and or [W&B Dedicated Cloud](../hosting-options/dedicated_cloud.md) deployment types.

## Infrastructure diagram

### Application layer

The application layer consists of a Kubernetes cluster, which should include multiple nodes to ensure resilience against node failures.

### Storage layer

The storage layer consists of a MySQL database and object storage. The MySQL database stores metadata and the object storage storage artifacts (models, datasets, and so on).


## Infrastructure Requirements

### Kubernetes

W&B requires a Kubernetes cluster with a deployed, configured and fully functioning Ingress controller (for example Contour, Nginx) as W&B comes in the form of a Kubernetes Operator. More details can be found here: [W&B Kubernetes Operator](../operator.md). 

### MySQL

W&B requires a MySQL database as a metadata store. The shape of the model parameters and related metadata impact the performance of the database. The database size grows as the ML practitioners track more training runs, and incurs read heavy load when queries are executed in run tables, users workspaces, and reports.

Consider the following when you run your own MySQL database:

1. **Backups**. You should  periodically back up the database to a separate facility. W&B recommends daily backups with at least 1 week of retention.
2. **Performance.** The disk the server is running on should be fast. W&B recommends running the database on an SSD or accelerated NAS.
3. **Monitoring.** The database should be monitored for load. If CPU usage is sustained at > 40% of the system for more than 5 minutes it is likely a good indication the server is resource starved.
4. **Availability.** Depending on your availability and durability requirements you might want to configure a hot standby on a separate machine that streams all updates in realtime from the primary server and can be used to failover to in the event that the primary server crashes or become corrupted.


### Object storage

W&B requires an object storage (Amazon S3, Azure Cloud Storage, Google Cloud Storage, or any S3-compatible storage service) with Pre-signed URL and CORS support.

## Other considerations

### Networking

In non-airgapped deployments, egress to the following endpoints during installation and during runtime is required:
    * deploy.wandb.ai
    * charts.wandb.ai
    * docker.io
    * quay.io
    * gcr.io

### DNS
The fully qualified domain name of W&B should resolve to the IP address of the ingress/load balancer using an A record. Creating the required DNS entry is outside the scope of this guide.

### SSL/TLS
A valid, signed SSL/TLS certificate is required for secure communication between clients and W&B. Requesting a certificate is outside the scope of this guide. SSL/TLS termination musst be done on the ingress/load balancer.

## Versions

* Kubernetes: at least version 1.29.
* MySQL: at least 8.0.


## Sizing

The table below offers general guidelines to use as a starting point. We recommend monitoring all components closely and adjusting the sizing based on actual usage patterns.

Please note that we only support x86 architecture.

### Models only

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

### Models and Weave

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


### Flavors



