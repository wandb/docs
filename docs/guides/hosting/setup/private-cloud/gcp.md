---
description: Hosting W&B Server on GCP.
---

# GCP

## Introduction

We recommend the usage of the [Terraform Module](https://registry.terraform.io/modules/wandb/wandb/google/latest) developed by Weights and Biases to deploy the product on Google Cloud.

The module documentation is extensive and contains all available options that can be used. We will cover some deployment options in this document.

Before you start, we recommend you choose one of the [remote backends](https://developer.hashicorp.com/terraform/language/settings/backends/configuration) available for Terraform to store the [State File](https://developer.hashicorp.com/terraform/language/state).

The State File is the necessary resource to roll out upgrades or make changes in your deployment without recreating all components.

The Terraform Module will deploy the following `mandatory` components:

- VPC
- Cloud SQL for MySQL
- Cloud Storage Bucket
- Google Kubernetes Engine
- KMS Crypto Key
- Load Balancer

Other deployment options can also include the following optional components:

- Memory store for Redis
- Pub/Sub messages system

## **Pre-requisite permissions**

The account that will run the terraform need to have the role `roles/owner` in the GCP project used.

## General steps

The steps on this topic are common for any deployment option covered by this documentation.

1. Prepare the development environment.
    - Install [Terraform](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli)
    - We recommend creating a Git repository with the code that will be used, but you can keep your files locally.
    - Create a project in [Google Cloud Console](https://console.cloud.google.com/)
    - Authenticate with GCP (make sure to [install gcloud](https://cloud.google.com/sdk/docs/install) before)
        
        `gcloud auth application-default login`
        
2. Create the `terraform.tfvars` file.
    
    The `tvfars` file content can be customized according to the installation type, but the minimum recommended will look like the example below.
    
    ```bash
    project_id  = "wandb-project"
    region      = "europe-west2"
    zone        = "europe-west2-a"
    namespace   = "wandb"
    license     = "xxxxxxxxxxyyyyyyyyyyyzzzzzzz"
    subdomain   = "wandb-gcp"
    domain_name = "wandb.ml"
    ```
    
    The variables defined here need to be decided before the deployment because. The `namespace` variable will be a string that will prefix all resources created by Terraform.
    
    The combination of `subdomain` and `domain` will form the FQDN that W&B will be configured. In the example above, the W&B FQDN will be `wandb-gcp.wandb.ml`
    
3. Create the file `versions.tf`
    
    This file will contain the Terraform and Terraform provider versions required to deploy W&B in GCP
    
    ```
    terraform {
      required_version = "~> 1.0"
      required_providers {
        google = {
          source  = "hashicorp/google"
          version = "~> 4.15"
        }
        kubernetes = {
          source  = "hashicorp/kubernetes"
          version = "~> 2.9"
        }
      }
    }
    ```
    
    Optionally, **but highly recommended**, you can add the [remote backend configuration](https://developer.hashicorp.com/terraform/language/settings/backends/configuration) mentioned at the beginning of this documentation.
    
4. Create the file `variables.tf`
    
    For every option configured in the `terraform.tfvars` Terraform requires a correspondent variable declaration.
    
    ```
    variable "project_id" {
      type        = string
      description = "Project ID"
    }
    
    variable "region" {
      type        = string
      description = "Google region"
    }
    
    variable "zone" {
      type        = string
      description = "Google zone"
    }
    
    variable "namespace" {
      type        = string
      description = "Namespace prefix used for resources"
    }
    
    variable "domain_name" {
      type        = string
      description = "Domain name for accessing the Weights & Biases UI."
    }
    
    variable "subdomain" {
      type        = string
      description = "Subdomain for access the Weights & Biases UI."
    }
    
    variable "license" {
      type        = string
      description = "W&B License"
    }
    ```
    

## Deployment - Recommended (~20 mins)

This is the most straightforward deployment option configuration that will create all `Mandatory` components and install in the `Kubernetes Cluster` the latest version of `W&B`.

1. Create the `main.tf`
    
    In the same directory where you created the files in the `General Steps`, create a file `main.tf` with the following content:
    
    ```
    data "google_client_config" "current" {}
    
    provider "kubernetes" {
      host                   = "https://${module.wandb.cluster_endpoint}"
      cluster_ca_certificate = base64decode(module.wandb.cluster_ca_certificate)
      token                  = data.google_client_config.current.access_token
    }
    
    # Spin up all required services
    module "wandb" {
      source  = "wandb/wandb/google"
      version = "1.12.2"
    
      namespace   = var.namespace
      license     = var.license
      domain_name = var.domain_name
      subdomain   = var.subdomain
    
    }
    
    # You'll want to update your DNS with the provisioned IP address
    output "url" {
      value = module.wandb.url
    }
    
    output "address" {
      value = module.wandb.address
    }
    
    output "bucket_name" {
      value = module.wandb.bucket_name
    }
    ```
    
2. Deploy W&B
    
    To deploy W&B, execute the following commands:
    
    ```
    terraform init
    terraform apply -var-file=terraform.tfvars
    ```
    

## Enabling REDIS

Another deployment option uses `Redis` to cache the SQL queries and speedup the application response when loading the metrics for the experiments.

You need to add the option `create_redis = true` to the same `main.tf` file we worked on in `Deployment option 1` to enable the cache.

```
[...]

module "wandb" {
  source  = "wandb/wandb/google"
  version = "1.12.2"

  namespace    = var.namespace
  license      = var.license
  domain_name  = var.domain_name
  subdomain    = var.subdomain
  #Enable Redis
  create_redis = true

}
[...]
```

## Enabling message broker (queue)

Deployment option 3 consists of enabling the external `message broker`. This is optional because the W&B brings embedded a broker. This option doesn't bring a performance improvement.

The GCP resource that provides the message broker is the `Pub/Sub`, and to enable it, you will need to add the option `use_internal_queue = false` to the same `main.tf` that we worked on the `Deployment option 1`

```
[...]

module "wandb" {
  source  = "wandb/wandb/google"
  version = "1.12.2"

  namespace          = var.namespace
  license            = var.license
  domain_name        = var.domain_name
  subdomain          = var.subdomain
  #Create and use Pub/Sub
  use_internal_queue = false

}

[...]

```

## Other deployment options

You can combine all three deployment options adding all configurations to the same file.
The [Terraform Module](https://github.com/wandb/terraform-google-wandb) provides several options that can be combined along with the standard options and the minimal configuration found in `Deployment - Recommended`

## Links and references?

- List some links.