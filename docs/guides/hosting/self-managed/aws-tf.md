---
description: W&B サーバーを AWS でホスティングする
title: AWS
displayed_sidebar: default
---

:::info
W&B recommends fully managed deployment options such as [W&B Multi-tenant Cloud](../hosting-options/saas_cloud.md) or [W&B Dedicated Cloud](../hosting-options//dedicated_cloud.md) deployment types. W&B fully managed services are simple and secure to use, with minimum to no configuration required.
:::

W&B recommends using the [W&B Server AWS Terraform Module](https://registry.terraform.io/modules/wandb/wandb/aws/latest) to deploy the platform on AWS.

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
   namespace                  = "wandb"
   license                    = "xxxxxxxxxxyyyyyyyyyyyzzzzzzz"
   subdomain                  = "wandb-aws"
   domain_name                = "wandb.ml"
   zone_id                    = "xxxxxxxxxxxxxxxx"
   allowed_inbound_cidr       = ["0.0.0.0/0"]
   allowed_inbound_ipv6_cidr  = ["::/0"]
   ```

   Ensure to define variables in your `tvfars` file before you deploy because the `namespace` variable is a string that prefixes all resources created by Terraform.



   The combination of `subdomain` and `domain` will form the FQDN that W&B will be configured. In the example above, the W&B FQDN will be `wandb-aws.wandb.ml` and the DNS `zone_id` where the FQDN record will be created.

   Both `allowed_inbound_cidr` and `allowed_inbound_ipv6_cidr` also require setting. In the module, this is a mandatory input. The proceeding example permits access from any source to the W&B installation.

3. Create the file `versions.tf`

   This file will contain the Terraform and Terraform provider versions required to deploy W&B in AWS

   ```bash
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

   variable "allowed_inbound_cidr" {
    description = "CIDRs allowed to access wandb-server."
    nullable    = false
    type        = list(string)
   }

   variable "allowed_inbound_ipv6_cidr" {
    description = "CIDRs allowed to access wandb-server."
    nullable    = false
    type        = list(string)
   }
   ```

## Deployment - Recommended (~20 mins)

This is the most straightforward deployment option configuration that will create all `Mandatory` components and install in the `Kubernetes Cluster` the latest version of `W&B`.

1. Create the `main.tf`

   In the same directory where you created the files in the `General Steps`, create a file `main.tf` with the following content:

   ```
   module "wandb_infra" {
     source  = "wandb/wandb/aws"
     version = "~>2.0"

     namespace   = var.namespace
     domain_name = var.domain_name
     subdomain   = var.subdomain
     zone_id     = var.zone_id

     allowed_inbound_cidr           = var.allowed_inbound_cidr
     allowed_inbound_ipv6_cidr      = var.allowed_inbound_ipv6_cidr

     public_access                  = true
     external_dns                   = true
     kubernetes_public_access       = true
     kubernetes_public_access_cidrs = ["0.0.0.0/0"]
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
     version = "~>1.0"

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

## Enable REDIS

Another deployment option uses `Redis` to cache the SQL queries and speed up the application response when loading the metrics for the experiments.

You need to add the option `create_elasticache_subnet = true` to the same `main.tf` file we worked on in `Recommended Deployment` to enable the cache.

```
module "wandb_infra" {
  source  = "wandb/wandb/aws"
  version = "~>2.0"

  namespace   = var.namespace
  domain_name = var.domain_name
  subdomain   = var.subdomain
  zone_id     = var.zone_id
	**create_elasticache_subnet = true**
}
[...]
```

## Enable message broker (queue)

Deployment option 3 consists of enabling the external `message broker`. This is optional because the W&B brings embedded a broker. This option doesn't bring a performance improvement.

The AWS resource that provides the message broker is the `SQS`, and to enable it, you will need to add the option `use_internal_queue = false` to the same `main.tf` that we worked on the `Recommended Deployment`

```
module "wandb_infra" {
  source  = "wandb/wandb/aws"
  version = "~>2.0"

  namespace   = var.namespace
  domain_name = var.domain_name
  subdomain   = var.subdomain
  zone_id     = var.zone_id
  **use_internal_queue = false**

[...]
}
```

## Other deployment options

You can combine all three deployment options adding all configurations to the same file.
The [Terraform Module](https://github.com/wandb/terraform-aws-wandb) provides several options that can be combined along with the standard options and the minimal configuration found in `Deployment - Recommended`

## Manual configuration

To use an Amazon S3 bucket as a file storage backend for W&B, you will need to:

* [Create an Amazon S3 Bucket and Bucket Notifications](#create-an-s3-bucket-and-bucket-notifications)
* [Create SQS Queue](#create-an-sqs-queue)
* [Grant Permissions to Node Running W&B](#grant-permissions-to-node-running-wb)


 you'll need to create a bucket, along with an SQS queue configured to receive object creation notifications from that bucket. Your instance will need permissions to read from this queue.

### Create an S3 Bucket and Bucket Notifications

Follow the procedure bellow to create an Amazon S3 bucket and enable bucket notifications.

1. Navigate to Amazon S3 in the AWS Console.
2. Select **Create bucket**.
3. Within the **Advanced settings**, select **Add notification** within the **Events** section.
4. Configure all object creation events to be sent to the SQS Queue you configured earlier.

![Enterprise file storage settings](/images/hosting/s3-notification.png)

Enable CORS access. Your CORS configuration should look like the following:

```markup
<?xml version="1.0" encoding="UTF-8"?>
<CORSConfiguration xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
<CORSRule>
    <AllowedOrigin>http://YOUR-W&B-SERVER-IP</AllowedOrigin>
    <AllowedMethod>GET</AllowedMethod>
    <AllowedMethod>PUT</AllowedMethod>
    <AllowedHeader>*</AllowedHeader>
</CORSRule>
</CORSConfiguration>
```

### Create an SQS Queue

Follow the procedure below to create an SQS Queue:

1. Navigate to Amazon SQS in the AWS Console.
2. Select **Create queue**.
3. From the **Details** section, select a **Standard** queue type.
4. Within the Access policy section, add permission to the following principals:
* `SendMessage`
* `ReceiveMessage`
* `ChangeMessageVisibility`
* `DeleteMessage`
* `GetQueueUrl`

Optionally add an advanced access policy in the **Access Policy** section. For example, the policy for accessing Amazon SQS with a statement is as follows:

```json
{
    "Version" : "2012-10-17",
    "Statement" : [
      {
        "Effect" : "Allow",
        "Principal" : "*",
        "Action" : ["sqs:SendMessage"],
        "Resource" : "<sqs-queue-arn>",
        "Condition" : {
          "ArnEquals" : { "aws:SourceArn" : "<s3-bucket-arn>" }
        }
      }
    ]
}
```

### Grant Permissions to Node Running W&B

The node where W&B server is running must be configured to permit access to Amazon S3 and Amazon SQS. Depending on the type of server deployment you have opted for, you may need to add the following policy statements to your node role:

```json
{
   "Statement":[
      {
         "Sid":"",
         "Effect":"Allow",
         "Action":"s3:*",
         "Resource":"arn:aws:s3:::<WANDB_BUCKET>"
      },
      {
         "Sid":"",
         "Effect":"Allow",
         "Action":[
            "sqs:*"
         ],
         "Resource":"arn:aws:sqs:<REGION>:<ACCOUNT>:<WANDB_QUEUE>"
      }
   ]
}
```

### Configure W&B server
Finally, configure your W&B Server.

1. Navigate to the W&B settings page at `http(s)://YOUR-W&B-SERVER-HOST/system-admin`. 
2. Enable the ***Use an external file storage backend* option/
3. Provide information about your Amazon S3 bucket, region, and Amazon SQS queue in the following format:
* **File Storage Bucket**: `s3://<bucket-name>`
* **File Storage Region (AWS only)**: `<region>`
* **Notification Subscription**: `sqs://<queue-name>`

![](/images/hosting/configure_file_store.png)

4. Select **Update settings** to apply the new settings.

## Upgrade your W&B version

Follow the steps outlined here to update W&B:

1. Add `wandb_version` to your configuration in your `wandb_app` module. Provide the version of W&B you want to upgrade to. For example, the following line specifies W&B version `0.48.1`:

  ```
  module "wandb_app" {
      source  = "wandb/wandb/kubernetes"
      version = "~>1.0"

      license       = var.license
      wandb_version = "0.48.1"
  ```

  :::info
  Alternatively, you can add the `wandb_version` to the `terraform.tfvars` and create a variable with the same name and instead of using the literal value, use the `var.wandb_version`
  :::

2. After you update your configuration, complete the steps described in the [Deployment section](#deployment---recommended-20-mins).
Here is a chunk of documentation in docusaurus Markdown format to translate.

```markdown
### Utilizing Weights & Biases to Its Full Potential

When working on machine learning projects, efficiency and organization are crucial. Leveraging the right tools can significantly improve workflow and performance. Weights & Biases is a comprehensive platform designed to help you manage your machine learning life cycle effectively.

#### Key Features of Weights & Biases

1. **Experiment Tracking**: Monitor and visualize your machine learning Runs easily.
2. **Dataset Versioning**: Keep track of Datasets and their versions.
3. **Model Management**: Organize and version different Models.
4. **Collaborative Reports**: Share insights and results through detailed Reports.
5. **Hyperparameter Sweeps**: Automate the process of finding the best hyperparameters.

#### Getting Started

1. Signup for a W&B account
2. Initialize W&B in your project
3. Track & visualize your Experiments

Refer to the [Quickstart Guide](link/quickstart) for detailed instructions.

#### Managing Your Projects

Effectively managing Projects involves organizing related Runs, Datasets, and Models. Detailed documentation is available [here](link/documentation).

For any support, please refer to our [Support Page](link/support).

### Additional Benefits

Using W&B, individuals and Teams can ensure reproducibility and transparency in their machine learning workflows.

#### Community and Contributions

We encourage Users to contribute by providing feedback or participating in the community discussions. Explore our [Community Forum](link/forum) to get involved.
```

### Weights & Biasesを最大限に活用する方法

機械学習プロジェクトを進める際、効率と組織力が重要です。適切なツールを活用することで、ワークフローとパフォーマンスが大幅に向上します。Weights & Biasesは、機械学習のライフサイクルを効果的に管理するための包括的なプラットフォームです。

#### Weights & Biasesの主な機能

1. **Experiment Tracking**: 機械学習のRunsを簡単にモニタリングし、視覚化します。
2. **Dataset Versioning**: Datasetsとそのバージョンを管理します。
3. **Model Management**: さまざまなModelsを整理し、バージョン管理します。
4. **Collaborative Reports**: 詳細なReportsを通じて洞察と結果を共有します。
5. **Hyperparameter Sweeps**: 最適なハイパーパラメータを見つけるプロセスを自動化します。

#### 始め方

1. W&Bアカウントにサインアップする
2. プロジェクトにW&Bを初期化する
3. Experimentsを追跡し、視覚化する

詳細な手順については、[Quickstart Guide](link/quickstart)を参照してください。

#### Projectsの管理

Projectsを効果的に管理するためには、関連するRuns、Datasets、Modelsを整理することが重要です。詳細なドキュメントは[こちら](link/documentation)からご覧いただけます。

サポートが必要な場合は、[Support Page](link/support)を参照してください。

### 追加の利点

W&Bを使用することで、個人やTeamsは機械学習ワークフローにおける再現性と透明性を確保できます。

#### コミュニティと貢献

Usersがフィードバックを提供したり、コミュニティディスカッションに参加したりすることを奨励します。[Community Forum](link/forum)で詳細を確認し、参加してください。


