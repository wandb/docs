---
description: Hosting W&B Server on AWS.
---

# AWS

We recommend the usage of the [Terraform Module](https://registry.terraform.io/modules/wandb/wandb/aws/latest) developed by Weights and Biases to deploy the W&B server on AWS.

The module documentation is extensive and contains all available options that can be used. We will cover some deployment options in this document.

Before you start, we recommend you choose one of the [remote backends](https://developer.hashicorp.com/terraform/language/settings/backends/configuration) available for Terraform to store the [State File](https://developer.hashicorp.com/terraform/language/state).

The State File is the necessary resource to roll out upgrades or make changes in your deployment without recreating all components.

The Terraform Module will deploy the following `mandatory` components:

- Load Balancer
- AWS Identity & Access Management (IAM)
- AWS Key Management System (KMS)
- Amazon Aurora MySQL
- Amazon VPC
- Amazon S3
- Amazon Route53
- Amazon Certificate Manager (ACM)
- Amazon Elastic Loadbalancing (ALB)
- Amazon Secrets Manager

Other deployment options can also include the following optional components:

- Elastic Cache for Redis
- SQS

## **Pre-requisite permissions**

The account that will run the Terraform needs to be able to create all components described in the Introduction and permission to create **IAM Policies** and **IAM Roles** and assign roles to resources.

## General steps

The steps on this topic are common for any deployment option covered by this documentation.

1. Prepare the development environment.
   - Install [Terraform](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli)
   - We recommend creating a Git repository with the code that will be used, but you can keep your files locally.
2. Create the `terraform.tfvars` file.

   The `tvfars` file content can be customized according to the installation type, but the minimum recommended will look like the example below.

   ```bash
   namespace     = "wandb"
   wandb_license = "xxxxxxxxxxyyyyyyyyyyyzzzzzzz"
   subdomain     = "wandb-aws"
   domain_name   = "wandb.ml"
   zone_id       = "xxxxxxxxxxxxxxxx"
   ```

   The variables defined here need to be decided before the deployment because. The `namespace` variable will be a string that will prefix all resources created by Terraform.

   The combination of `subdomain` and `domain` will form the FQDN that W&B will be configured. In the example above, the W&B FQDN will be `wandb-aws.wandb.ml` and the DNS `zone_id` where the FQDN record will be created.

3. Create the file `versions.tf`

   This file will contain the Terraform and Terraform provider versions required to deploy W&B in AWS

   ```bash
   terraform {
     required_providers {
       aws = {
         source  = "hashicorp/aws"
         version = "~> 3.60"
       }
     }
   }

   provider "aws" {
     region = "eu-central-1"

     default_tags {
       tags = {
         GithubRepo = "terraform-aws-wandb"
         GithubOrg  = "wandb"
         Enviroment = "Example"
         Example    = "PublicDnsExternal"
       }
     }
   }
   ```

   Please, refer to the [Terraform Official Documentation](https://registry.terraform.io/providers/hashicorp/aws/latest/docs#provider-configuration) to configure the AWS provider.

   Optionally, **but highly recommended**, you can add the [remote backend configuration](https://developer.hashicorp.com/terraform/language/settings/backends/configuration) mentioned at the beginning of this documentation.

4. Create the file `variables.tf`

   For every option configured in the `terraform.tfvars` Terraform requires a correspondent variable declaration.

   ```
   variable "namespace" {
     type        = string
     description = "Name prefix used for resources"
   }

   variable "domain_name" {
     type        = string
     description = "Domain name used to access instance."
   }

   variable "subdomain" {
     type        = string
     default     = null
     description = "Subdomain for accessing the Weights & Biases UI."
   }

   variable "license" {
     type = string
   }

   variable "zone_id" {
     type        = string
     description = "Domain for creating the Weights & Biases subdomain on."
   }
   ```

## Deployment - Recommended (~20 mins)

This is the most straightforward deployment option configuration that will create all `Mandatory` components and install in the `Kubernetes Cluster` the latest version of `W&B`.

1. Create the `main.tf`

   In the same directory where you created the files in the `General Steps`, create a file `main.tf` with the following content:

   ```
   module "wandb_infra" {
     source  = "wandb/wandb/aws"
     version = "1.6.0"

     namespace   = var.namespace
     domain_name = var.domain_name
     subdomain   = var.subdomain
     zone_id     = var.zone_id
   }

   data "aws_eks_cluster" "app_cluster" {
     name = module.wandb_infra.cluster_id
   }

   data "aws_eks_cluster_auth" "app_cluster" {
     name = module.wandb_infra.cluster_id
   }

   provider "kubernetes" {
     host                   = data.aws_eks_cluster.app_cluster.endpoint
     cluster_ca_certificate = base64decode(data.aws_eks_cluster.app_cluster.certificate_authority.0.data)
     token                  = data.aws_eks_cluster_auth.app_cluster.token
   }

   module "wandb_app" {
     source  = "wandb/wandb/kubernetes"
     version = "1.5.0"

     license                    = var.license
     host                       = module.wandb_infra.url
     bucket                     = "s3://${module.wandb_infra.bucket_name}"
     bucket_aws_region          = module.wandb_infra.bucket_region
     bucket_queue               = "internal://"
     database_connection_string = "mysql://${module.wandb_infra.database_connection_string}"

     # If we dont wait, tf will start trying to deploy while the work group is
     # still spinning up
     depends_on = [module.wandb_infra]
   }

   output "bucket_name" {
     value = module.wandb_infra.bucket_name
   }

   output "url" {
     value = module.wandb_infra.url
   }
   ```

2. Deploy W&B

   To deploy W&B, execute the following commands:

   ```
   terraform init
   terraform apply -var-file=terraform.tfvars
   ```

## Enabling REDIS

Another deployment option uses `Redis` to cache the SQL queries and speed up the application response when loading the metrics for the experiments.

You need to add the option `create_elasticache_subnet = true` to the same `main.tf` file we worked on in `Recommended Deployment` to enable the cache.

```
module "wandb_infra" {
  source  = "wandb/wandb/aws"
  version = "1.6.0"

  namespace   = var.namespace
  domain_name = var.domain_name
  subdomain   = var.subdomain
  zone_id     = var.zone_id
	**create_elasticache_subnet = true**
}
[...]
```

## Enabling message broker (queue)

Deployment option 3 consists of enabling the external `message broker`. This is optional because the W&B brings embedded a broker. This option doesn't bring a performance improvement.

The AWS resource that provides the message broker is the `SQS`, and to enable it, you will need to add the option `use_internal_queue = false` to the same `main.tf` that we worked on the `Recommended Deployment`

```
module "wandb_infra" {
  source  = "wandb/wandb/aws"
  version = "1.6.0"

  namespace   = var.namespace
  domain_name = var.domain_name
  subdomain   = var.subdomain
  zone_id     = var.zone_id
  **use_internal_queue = false**

[...]

```

## Other deployment options

You can combine all three deployment options adding all configurations to the same file.
The [Terraform Module](https://github.com/wandb/terraform-aws-wandb) provides several options that can be combined along with the standard options and the minimal configuration found in `Deployment - Recommended`

<!-- ## Upgrades (coming soon) -->
