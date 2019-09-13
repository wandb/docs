---
title: W&B Enterprise
sidebar_label: About
---

W&B Enterprise is the on-premises version of [Weights & Biases](app.wandb.ai). It makes collaborative experiment tracking possible for enterprise machine learning teams, offering confidence that all training data and metadata are protected within your organization's network.

<p style="text-align: center;">
  <a href="https://www.wandb.com/demo" style="background: #444; color: white !important; display: inline-block; padding: 1rem; margin: 1rem auto; font-size: 24px;">
    Request a demo to try out W&B Enterprise
  </a>
</p>

Most enterprise customers will use a [W&B Enterprise Server](#enterprise-server). This is a single virtual machine containing all of W&B's systems and storage. You can provision a W&B server on any cloud environment or local hardware or virtual server.

We also offer [W&B Enterprise Cloud](#enterprise-cloud), which runs a completely scalable infrastructure within your company's AWS or GCP account. This system can scale to any level of usage.

## Features

* Unlimited runs, experiments, and reports
* Keep your data safe on your own company's network
* Integrate with your company's authentication system
* Premier support by the W&B engineering team

## Enterprise Server

The Enterprise Server consists of a single virtual machine, saved as a bootable image in the format of your cloud platform. Your W&B data is saved on a separate drive from the server softare so data can be preserved across VM versions.

We support the following environments:

| **Platform**          | **Image Format**  |
|-----------------------|---------------|
| Amazon Web Services   | AMI           |
| Microsoft Azure       | Managed Image |
| Google Cloud Platform | GCE Image     |
| VMware                | OVF           |
| Vagrant               | Vagrant Box   |

### Server Requirements

The W&B Enterprise server requires a virtual machine with at least 4 cores and 16GB memory.

## Enterprise Cloud

Our W&B Enterprise Cloud offering consists of a fully scalable W&B cluster provisioned on your AWS or GCP environment.

### Cloud Requirements

W&B Enterprise Cloud requires the following cloud resources in your account. The size can be configured for your level of usage.

* A Kubernetes cluster (EKS or GKE)
* A SQL database (RDS or Google Cloud SQL)
* A columnar store (HBase or BigTable)
