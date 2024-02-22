---
description: Hosting W&B Server on AWS.
displayed_sidebar: default
---

# AWS

Weights & Biases에서 개발한 [Terraform Module](https://registry.terraform.io/modules/wandb/wandb/aws/latest)을 사용하여 AWS에 W&B 서버를 배포하는 것이 좋습니다.

해당 모듈 문서는 매우 상세하며 사용 가능한 모든 옵션을 포함하고 있습니다. 본 문서에서는 일부 배포 옵션에 대해 다룰 것입니다.

시작하기 전에, Terraform이 [State File](https://developer.hashicorp.com/terraform/language/state)을 저장하기 위해 사용 가능한 [remote backends](https://developer.hashicorp.com/terraform/language/settings/backends/configuration) 중 하나를 선택하는 것이 좋습니다.

State File은 모든 구성 요소를 다시 생성하지 않고 배포에서 업그레이드를 진행하거나 변경 사항을 적용할 때 필요한 리소스입니다.

Terraform 모듈은 다음 `필수` 구성 요소를 배포합니다:

- 로드 밸런서
- AWS Identity & Access Management (IAM)
- AWS Key Management System (KMS)
- Amazon Aurora MySQL
- Amazon VPC
- Amazon S3
- Amazon Route53
- Amazon Certificate Manager (ACM)
- Amazon Elastic Loadbalancing (ALB)
- Amazon Secrets Manager

기타 배포 옵션은 다음 선택 구성 요소를 포함할 수도 있습니다:

- Redis를 위한 Elastic Cache
- SQS

## **필요한 권한**

Terraform을 실행할 계정은 소개에서 설명한 모든 구성 요소를 생성할 수 있어야 하며 **IAM 정책** 및 **IAM 역할**을 생성하고 리소스에 역할을 할당할 수 있는 권한이 있어야 합니다.

## 일반 단계

이 주제의 단계는 본 문서에서 다루는 모든 배포 옵션에 공통적입니다.

1. 개발 환경 준비.
   - [Terraform](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli) 설치
   - 코드가 포함된 Git 저장소를 생성하는 것이 좋지만, 파일을 로컬에 보관할 수도 있습니다.
2. `terraform.tfvars` 파일 생성.

   `tvfars` 파일 내용은 설치 유형에 따라 커스터마이징될 수 있지만, 최소 권장 사항은 아래 예제와 같습니다.

   ```bash
   namespace                  = "wandb"
   license                    = "xxxxxxxxxxyyyyyyyyyyyzzzzzzz"
   subdomain                  = "wandb-aws"
   domain_name                = "wandb.ml"
   zone_id                    = "xxxxxxxxxxxxxxxx"
   allowed_inbound_cidr       = ["0.0.0.0/0"]
   allowed_inbound_ipv6_cidr  = ["::/0"]
   ```

   배포하기 전에 `tvfars` 파일에서 변수를 정의해야 합니다. 왜냐하면 `namespace` 변수는 Terraform이 생성하는 모든 리소스에 접두사로 사용되는 문자열이기 때문입니다.

   `subdomain`과 `domain`의 조합은 W&B가 구성될 FQDN을 형성할 것입니다. 위 예제에서 W&B FQDN은 `wandb-aws.wandb.ml`이 될 것이며 FQDN 레코드가 생성될 DNS `zone_id`입니다.

   `allowed_inbound_cidr` 및 `allowed_inbound_ipv6_cidr`도 설정이 필요합니다. 모듈에서는 이것이 필수 입력입니다. 앞의 예제는 W&B 설치에 대한 모든 출처에서의 접근을 허용합니다.

3. `versions.tf` 파일 생성

   이 파일은 AWS에서 W&B를 배포하기 위해 필요한 Terraform 및 Terraform 제공자 버전을 포함할 것입니다.

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

   AWS 제공자를 구성하는 방법에 대해서는 [Terraform 공식 문서](https://registry.terraform.io/providers/hashicorp/aws/latest/docs#provider-configuration)를 참조하십시오.

   선택적으로, **하지만 강력히 권장됩니다**, 본 문서 초반에 언급된 [remote backend configuration](https://developer.hashicorp.com/terraform/language/settings/backends/configuration)을 추가할 수 있습니다.

4. `variables.tf` 파일 생성

   `terraform.tfvars`에 구성된 모든 옵션에 대해 Terraform은 해당 변수 선언이 필요합니다.

   ```
   variable "namespace" {
     type        = string
     description = "리소스에 사용되는 이름 접두사"
   }

   variable "domain_name" {
     type        = string
     description = "인스턴스에 액세스하기 위해 사용되는 도메인 이름."
   }

   variable "subdomain" {
     type        = string
     default     = null
     description = "Weights & Biases UI에 액세스하기 위한 서브도메인."
   }

   variable "license" {
     type = string
   }

   variable "zone_id" {
     type        = string
     description = "Weights & Biases 서브도메인을 생성하기 위한 도메인."
   }

   variable "allowed_inbound_cidr" {
    description = "wandb-서버에 액세스할 수 있는 CIDR."
    nullable    = false
    type        = list(string)
   }

   variable "allowed_inbound_ipv6_cidr" {
    description = "wandb-서버에 액세스할 수 있는 CIDR."
    nullable    = false
    type        = list(string)
   }
   ```

## 배포 - 추천 (~20분)

이것은 모든 `필수` 구성 요소를 생성하고 `Kubernetes 클러스터`에 `W&B`의 최신 버전을 설치할 가장 간단한 배포 옵션 구성입니다.

1. `main.tf` 생성

   `일반 단계`에서 생성한 파일과 동일한 디렉터리에 다음 내용을 가진 `main.tf` 파일을 생성합니다:

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

     # 작업 그룹이 아직 시작중인 동안 기다리지 않으면, tf가 배포를 시작하려고 시도합니다
     depends_on = [module.wandb_infra]
   }

   output "bucket_name" {
     value = module.wandb_infra.bucket_name
   }

   output "url" {
     value = module.wandb_infra.url
   }
   ```

2. W&B 배포

   W&B를 배포하려면 다음 명령을 실행합니다:

   ```
   terraform init
   terraform apply -var-file=terraform.tfvars
   ```

## REDIS 활성화

다른 배포 옵션은 `Redis`를 사용하여 SQL 쿼리를 캐시하고 실험에 대한 메트릭을 로드할 때 애플리케이션 응답을 가속화합니다.

캐시를 활성화하려면 `create_elasticache_subnet = true` 옵션을 `추천 배포`에서 작업한 동일한 `main.tf` 파일에 추가해야 합니다.

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

## 메시지 브로커 (큐) 활성화

배포 옵션 3은 외부 `메시지 브로커`를 활성화하는 것으로 구성됩니다. W&B는 내장된 브로커를 제공하기 때문에 이것은 선택 사항입니다. 이 옵션은 성능 향상을 제공하지 않습니다.

메시지 브로커를 제공하는 AWS 리소스는 `SQS`이며, 이를 활성화하려면 `use_internal_queue = false` 옵션을 `추천 배포`에서 작업한 동일한 `main.tf`에 추가해야 합니다.

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

## 기타 배포 옵션

세 가지 배포 옵션을 모두 결합하여 동일한 파일에 모든 구성을 추가할 수 있습니다.
[Terraform 모듈](https://github.com/wandb/terraform-aws-wandb)은 표준 옵션과 `추천 배포`에서 찾을 수 있는 최소 구성과 함께 결합될 수 있는 여러 옵션을 제공합니다.

## 수동 구성

W&B를 위한 파일 저장 백엔드로 Amazon S3 버킷을 사용하려면 다음을 수행해야 합니다:

* [Amazon S3 버킷 및 버킷 알림 생성](#create-an-s3-bucket-and-bucket-notifications)
* [SQS 큐 생성](#create-an-sqs-queue)
* [W&B를 실행하는 노드에 권한 부여](#grant-permissions-to-node-running-wb)

버킷을 생성하고, 해당 버킷에서 객체 생성 알림을 수신하도록 구성된 SQS 큐를 생성해야 합니다. 인스턴스는 이 큐에서 읽을 수 있는 권한이 필요합니다.

### S3 버킷 및 버킷 알림 생성

다음 절차를 따라 Amazon S3 버킷을 생성하고 버킷 알림을 활성화하세요.

1. AWS 콘솔에서 Amazon S3로 이동합니다.
2. **버킷 생성**을 선택합니다.
3. **고급 설정** 내에서 **이벤트** 섹션 내 **알림 추가**를 선택합니다.
4. 모든 객체 생성 이벤트가 앞서 구성한 SQS 큐로 전송되도록 구성합니다.

![엔터프라이즈 파일 저장소 설정](/images/hosting/s3-notification.png)

CORS 액세스를 활성화하세요. CORS 구성은 다음과 같아야 합니다:

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

### SQS 큐 생성

다음 절차를 따라 SQS 큐를 생성하세요:

1. AWS 콘솔에서 Amazon SQS로 이동합니다.
2. **큐 생성**을 선택합니다.
3. **세부 정보** 섹션에서 **표준** 큐 유형을 선택합니다.
4. 액세스 정책 섹션 내에서 다음 주체에 대한 권한을 추가합니다:
* `SendMessage`
* `ReceiveMessage`
* `ChangeMessageVisibility`
* `DeleteMessage`
* `GetQueueUrl`

선택적으로 **액세스 정책** 섹션에서 고급 액세스 정책을 추가할 수 있습니다. 예를 들어, Amazon SQS에 대한 액세스 정책은 다음과 같습니다:

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

### W&B를 실행하는 노드에 권한 부여

W&B 서버가 실행되는 노드는 Amazon S3 및 Amazon SQS에 액세스할 수 있도록 구성되어야 합니다. 선택한 서버 배포 유형에 따라 노드 역할에 다음 정책 문을 추가해야 할 수 있습니다:

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

### W&B 서버 구성
마지막으로, W&B 서버를 구성하세요.

1. `http(s)://YOUR-W&B-SERVER-HOST/system-admin`에서 W&B 설정 페이지로 이동합니다.
2. ***외부 파일 저장소 백엔드 사용* 옵션을 활성화합니다.
3. 다음 형식으로 Amazon S3 버킷, 지역 및 Amazon SQS 큐에 대한 정보를 제공합니다:
* **파일 저장소 버킷**: `s3://<bucket-name>`
* **파일 저장소 지역 (AWS 전용)**: `<region>`
* **알림 구독**: `sqs://<queue-name>`

![](/images/hosting/configure_file_store.png)

4. 새 설정을 적용하려면 **설정 업데이트**를 선택합니다.

## W&B 버전 업그레이드

W&B를 업데이트하려면 여기에 설명된 단계를 따르세요:

1. `wandb_app` 모듈 구성에 `wandb_version`을 추가합니다. 업그레이드하려는 W&B의 버전을 제공하세요. 예를 들어, 다음 줄은 W&B 버전 `0.48.1`을 지정합니다:

  ```
  module "wandb_app" {
      source  = "wandb/wandb/kubernetes"
      version = "~>1.0"

      license       = var.license
      wandb_version = "0.48.1"
  ```

  :::info
  대안으로, `terraform.tfvars`에