---
title: Deploy W&B Platform on AWS
description: AWS에서 W&B 서버 호스팅.
displayed_sidebar: default
---

:::info
W&B는 [W&B Multi-tenant Cloud](../hosting-options/saas_cloud.md) 또는 [W&B Dedicated Cloud](../hosting-options//dedicated_cloud.md) 배포 유형과 같은 완전 관리형 배포 옵션을 권장합니다. W&B 완전 관리형 서비스는 간단하고 안전하게 사용할 수 있으며, 별도의 설정이 거의 필요하지 않습니다.
:::

W&B는 AWS 상의 플랫폼 배포를 위해 [W&B Server AWS Terraform 모듈](https://registry.terraform.io/modules/wandb/wandb/aws/latest)을 사용할 것을 권장합니다.

시작하기 전에, W&B는 Terraform의 [원격 백엔드](https://developer.hashicorp.com/terraform/language/backend)를 선택하여 [State File](https://developer.hashicorp.com/terraform/language/state)을 저장할 것을 권장합니다.

State File은 모든 컴포넌트를 재생성하지 않고 배포에서 업그레이드나 변경 작업을 수행하는 데 필요한 리소스입니다.

Terraform 모듈은 다음의 `필수` 컴포넌트를 배포합니다:

- Load Balancer
- AWS Identity & Access Management (IAM)
- AWS Key Management System (KMS)
- Amazon Aurora MySQL
- Amazon VPC
- Amazon S3
- Amazon Route53
- Amazon Certificate Manager (ACM)
- Amazon Elastic Load Balancing (ALB)
- Amazon Secrets Manager

다른 배포 옵션에서는 다음의 선택적 컴포넌트를 포함할 수 있습니다:

- Redis용 Elastic Cache
- SQS

## 사전 요구 권한

Terraform을 실행하는 계정은 `도입`에서 설명한 모든 컴포넌트를 생성할 수 있어야 하며, **IAM 정책**과 **IAM 역할**을 생성하고 리소스에 할당할 수 있어야 합니다.

## 일반 단계

이 문서에서 다루는 모든 배포 옵션에 공통된 단계입니다.

1. 개발 환경 준비.
   - [Terraform](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli) 설치
   - W&B는 버전 관리를 위해 Git 저장소를 생성할 것을 권장합니다.
2. `terraform.tfvars` 파일 생성.

   `tfvars` 파일의 내용은 설치 유형에 따라 사용자 정의될 수 있지만, 최소 권장 사항은 아래 예시와 같습니다.

   ```bash
   namespace                  = "wandb"
   license                    = "xxxxxxxxxxyyyyyyyyyyyzzzzzzz"
   subdomain                  = "wandb-aws"
   domain_name                = "wandb.ml"
   zone_id                    = "xxxxxxxxxxxxxxxx"
   allowed_inbound_cidr       = ["0.0.0.0/0"]
   allowed_inbound_ipv6_cidr  = ["::/0"]
   ```

   배포 전에 `tfvars` 파일 내 변수를 정의해야 합니다. `namespace` 변수는 Terraform이 생성한 모든 리소스에 접두사를 붙이는 문자열입니다.

   `subdomain`과 `domain`의 조합은 W&B에 구성될 FQDN을 형성합니다. 위 예시에서 W&B FQDN은 `wandb-aws.wandb.ml`이 되며, FQDN 레코드가 생성될 DNS `zone_id`입니다.

   `allowed_inbound_cidr`와 `allowed_inbound_ipv6_cidr`도 설정이 필요합니다. 모듈에서는 필수 입력입니다. 진행 예시는 W&B 설치에 대한 모든 소스의 엑세스를 허용합니다.

3. `versions.tf` 파일 생성

   이 파일은 AWS에서 W&B를 배포하기 위한 Terraform 및 Terraform 제공자 버전을 포함합니다.

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

   AWS 제공자를 구성하려면 [Terraform 공식 문서](https://registry.terraform.io/providers/hashicorp/aws/latest/docs#provider-configuration)를 참조하세요.

   선택 사항이지만 강력히 권장되는 [원격 백엔드 설정](https://developer.hashicorp.com/terraform/language/settings/backends/configuration)을 이 문서 초반에 언급된 대로 추가하세요.

4. `variables.tf` 파일 생성

   `terraform.tfvars`에 설정된 각 옵션에 대해 Terraform은 해당 변수 선언이 필요합니다.

   ```
   variable "namespace" {
     type        = string
     description = "리소스에 사용된 이름 접두사"
   }

   variable "domain_name" {
     type        = string
     description = "인스턴스에 엑세스하는 데 사용된 도메인 이름."
   }

   variable "subdomain" {
     type        = string
     default     = null
     description = "Weights & Biases UI에 엑세스하기 위한 서브도메인."
   }

   variable "license" {
     type = string
   }

   variable "zone_id" {
     type        = string
     description = "Weights & Biases 서브도메인이 생성될 도메인."
   }

   variable "allowed_inbound_cidr" {
    description = "wandb-server에 엑세스할 수 있는 CIDR."
    nullable    = false
    type        = list(string)
   }

   variable "allowed_inbound_ipv6_cidr" {
    description = "wandb-server에 엑세스할 수 있는 CIDR."
    nullable    = false
    type        = list(string)
   }
   ```

## 권장 배포 옵션

이것은 모든 `필수` 컴포넌트를 생성하고 최신 버전의 `W&B`를 `Kubernetes 클러스터`에 설치하는 가장 간단한 배포 옵션 설정입니다.

1. `main.tf` 파일 생성

   `일반 단계`에서 파일을 생성한 디렉토리에서, 다음 내용을 포함하는 `main.tf` 파일을 생성하세요:

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

     # TF attempts to deploy while the work group is
     # still spinning up if you do not wait
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

   W&B를 배포하려면 다음 명령을 실행하세요:

   ```
   terraform init
   terraform apply -var-file=terraform.tfvars
   ```

## REDIS 활성화

다른 배포 옵션에서는 `Redis`를 사용하여 SQL 쿼리를 캐시하고 애플리케이션 응답을 가속화할 수 있습니다. 캐시를 활성화하려면 [권장 배포](#recommended-deployment-option) 섹션에서 설명한 대로 `main.tf` 파일에 `create_elasticache_subnet = true` 옵션을 추가해야 합니다.

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

## 메시지 브로커 (queue) 활성화

배포 옵션 3은 외부 `메시지 브로커`를 활성화하는 것입니다. 이는 선택 사항으로, W&B에는 임베디드 브로커가 포함되어 있습니다. 이 옵션은 성능 개선을 가져오지 않습니다.

`SQS`는 메시지 브로커를 제공하는 AWS 리소스이며, 이를 활성화하려면 [권장 배포](#recommended-deployment-option) 섹션에서 설명한 대로 동일 `main.tf`에 `use_internal_queue = false` 옵션을 추가해야 합니다.

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

## 다른 배포 옵션

세 가지 배포 옵션을 모두 결합하여 동일한 파일에 모든 설정을 추가할 수 있습니다. [Terraform 모듈](https://github.com/wandb/terraform-aws-wandb)은 표준 옵션과 `배포 - 권장`에서 찾을 수 있는 최소한의 구성과 함께 결합할 수 있는 여러 옵션을 제공합니다.

## 수동 설정

W&B에 대한 파일 저장소 백엔드로 Amazon S3 버킷을 사용하려면 다음을 수행해야 합니다:

* [Amazon S3 버킷 및 버킷 알림 생성](#create-an-s3-bucket-and-bucket-notifications)
* [SQS 큐 생성](#create-an-sqs-queue)
* [W&B를 실행하는 노드에 권한 부여](#grant-permissions-to-node-that-runs-wb)

버킷과 해당 버킷에서 오브젝트 생성 알림을 받을 수 있도록 구성된 SQS 큐를 만들어야 합니다. 인스턴스는 이 큐에서 읽을 수 있는 권한이 필요합니다.

### S3 버킷 및 버킷 알림 생성

아래 절차를 따라 Amazon S3 버킷을 생성하고 버킷 알림을 활성화하세요.

1. AWS 콘솔에서 Amazon S3로 이동하세요.
2. **버킷 생성**을 선택합니다.
3. **고급 설정** 내에서 **알림 추가**를 **이벤트** 섹션 내에서 선택합니다.
4. 모든 오브젝트 생성 이벤트를 이전에 구성한 SQS 큐로 전송하도록 구성하세요.

![Enterprise file storage settings](/images/hosting/s3-notification.png)

CORS 엑세스를 활성화하세요. CORS 설정은 다음과 같습니다:

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

아래 절차를 따라 SQS 큐를 생성하세요:

1. AWS 콘솔에서 Amazon SQS로 이동하세요.
2. **큐 생성**을 선택합니다.
3. **세부 정보** 섹션에서, **표준** 큐 유형을 선택합니다.
4. 엑세스 정책 섹션 내에서 다음 기본 사용자에게 권한을 추가합니다:
* `SendMessage`
* `ReceiveMessage`
* `ChangeMessageVisibility`
* `DeleteMessage`
* `GetQueueUrl`

선택적으로 **액세스 정책** 섹션에서 고급 엑세스 정책을 추가하세요. 예를 들어, Amazon SQS에 엑세스하기 위한 정책은 다음과 같습니다:

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

W&B 서버가 실행 중인 노드는 Amazon S3 및 Amazon SQS에 엑세스할 수 있도록 설정되어야 합니다. 선택한 서버 배포 유형에 따라 노드 역할에 다음 정책 선언을 추가해야 할 수 있습니다:

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

1. W&B 설정 페이지 `http(s)://YOUR-W&B-SERVER-HOST/system-admin`로 이동합니다.
2. ***외부 파일 저장소 백엔드 사용* 옵션을 활성화합니다.
3. 다음 형식으로 Amazon S3 버킷, 지역 및 Amazon SQS 큐에 대한 정보를 제공하세요:
* **파일 저장소 버킷**: `s3://<bucket-name>`
* **파일 저장소 지역 (AWS 전용)**: `<region>`
* **알림 구독**: `sqs://<queue-name>`

![](/images/hosting/configure_file_store.png)

4. **업데이트 설정**을 선택하여 새 설정을 적용합니다.

## W&B 버전 업그레이드

W&B를 업데이트하려면 다음 단계를 따르세요:

1. `wandb_app` 모듈의 설정에 `wandb_version`을 추가합니다. 업그레이드하려는 W&B 버전을 제공합니다. 예를 들어, 다음 줄은 W&B 버전 `0.48.1`을 지정합니다:

  ```
  module "wandb_app" {
      source  = "wandb/wandb/kubernetes"
      version = "~>1.0"

      license       = var.license
      wandb_version = "0.48.1"
  ```

  :::info
  또는 `wandb_version`을 `terraform.tfvars`에 추가하고 동일한 이름의 변수를 생성한 후, 리터럴 값을 사용하는 대신 `var.wandb_version`을 사용하세요.
  :::

2. 설정을 업데이트한 후, [권장 배포 섹션](#recommended-deployment-option)에서 설명한 단계를 완료하세요.

## operator 기반 AWS Terraform 모듈로 마이그레이션

이 섹션에서는 [terraform-aws-wandb](https://registry.terraform.io/modules/wandb/wandb/aws/latest) 모듈을 사용하여 _pre-operator_ 환경에서 _post-operator_ 환경으로 업그레이드하는 데 필요한 단계를 자세히 설명합니다.

:::info
Kubernetes [operator](https://kubernetes.io/docs/concepts/extend-kubernetes/operator/) 패턴으로의 전환은 W&B 아키텍처에 필수적입니다. 아키텍처 변경의 자세한 설명은 [이 섹션](../operator.md#reasons-for-the-architecture-shift)을 참조하세요.
:::


### 전과 후의 아키텍처

이전에는 W&B 아키텍처가 다음을 사용했습니다:

```hcl
module "wandb_infra" {
  source  = "wandb/wandb/aws"
  version = "1.16.10"
  ...
}
```

인프라를 제어하기 위해:

![pre-operator-infra](/images/hosting/pre-operator-infra.svg)

W&B 서버를 배포하기 위해 다음 모듈을 사용했습니다:

```hcl
module "wandb_app" {
  source  = "wandb/wandb/kubernetes"
  version = "1.12.0"
}
```

![pre-operator-k8s](/images/hosting/pre-operator-k8s.svg)

전환 후, 아키텍처는 다음을 사용합니다:

```hcl
module "wandb_infra" {
  source  = "wandb/wandb/aws"
  version = "4.7.2"
  ...
}
```

인프라 및 W&B 서버 설치를 Kubernetes 클러스터에 관리함으로써 `post-operator.tf`에서 `module "wandb_app"`의 필요성을 제거합니다.

![post-operator-k8s](/images/hosting/post-operator-k8s.svg)

이 아키텍처 변경은 추가적인 기능(예: OpenTelemetry, Prometheus, HPAs, Kafka 및 이미지 업데이트)을 수동 Terraform 작업 없이 SRE/인프라 팀이 사용하도록 활성화합니다.

W&B Pre-Operator의 기본 설치를 시작하려면, `post-operator.tf`에 `.disabled` 파일 확장자를 추가하고 `pre-operator.tf`가 활성 상태인지 확인하세요 (활성 상태가 아닌 파일은 `.disabled` 확장자가 없습니다). 해당 파일은 [여기](https://github.com/wandb/terraform-aws-wandb/tree/main/docs/operator-migration)에서 찾을 수 있습니다.

### 요구 사항

마이그레이션 프로세스를 시작하기 전에 다음 요구 사항이 충족되었는지 확인하세요:

- **Egress**: 배포는 공중두절이 불가능합니다. 최신 **_Release Channel_** 사양을 얻기 위해 [deploy.wandb.ai](https://deploy.wandb.ai)에 엑세스해야 합니다.
- **AWS Credentials**: AWS 리소스와 상호 작용할 수 있도록 적절한 AWS 자격 증명이 구성되어 있어야 합니다.
- **Terraform Installed**: 최신 버전의 Terraform이 시스템에 설치되어 있어야 합니다.
- **Route53 Hosted Zone**: 애플리케이션이 제공될 도메인에 해당하는 Route53 호스팅 영역이 존재해야 합니다.
- **Pre-Operator Terraform Files**: `pre-operator.tf`와 `pre-operator.tfvars`와 같은 관련 변수 파일이 올바르게 설정되어 있는지 확인하세요.

### Pre-Operator 설정

Pre-Operator 설정을 위해 구성 초기화 및 적용을 위한 다음 Terraform 명령을 실행하세요:

```bash
terraform init -upgrade
terraform apply -var-file=./pre-operator.tfvars
```

`pre-operator.tf`는 다음과 같이 나타납니다:

```ini
namespace     = "operator-upgrade"
domain_name   = "sandbox-aws.wandb.ml"
zone_id       = "Z032246913CW32RVRY0WU"
subdomain     = "operator-upgrade"
wandb_license = "ey..."
wandb_version = "0.51.2"
```

`pre-operator.tf` 구성은 두 개의 모듈을 호출합니다:

```hcl
module "wandb_infra" {
  source  = "wandb/wandb/aws"
  version = "1.16.10"
  ...
}
```

이 모듈은 인프라를 활성화합니다.

```hcl
module "wandb_app" {
  source  = "wandb/wandb/kubernetes"
  version = "1.12.0"
}
```

이 모듈은 애플리케이션을 배포합니다.

### Post-Operator 설정

`pre-operator.tf`가 `.disabled` 확장자를 갖도록 하여 비활성화하고, `post-operator.tf`가 활성화된 상태인지 확인하세요.

`post-operator.tfvars`에는 추가 변수가 포함됩니다:

```ini
...
# wandb_version = "0.51.2"는 이제 Release Channel 또는 User Spec에서 관리됩니다.

# 필요한 Operator 변수들:
size                 = "small"
enable_dummy_dns     = true
enable_operator_alb  = true
custom_domain_filter = "sandbox-aws.wandb.ml"
```

Post-Operator 구성을 초기화하고 적용하기 위해 다음 명령을 실행하세요:

```bash
terraform init -upgrade
terraform apply -var-file=./post-operator.tfvars
```

계획 및 적용 단계는 다음 리소스를 업데이트합니다:

```yaml
actions:
  create:
    - aws_efs_backup_policy.storage_class
    - aws_efs_file_system.storage_class
    - aws_efs_mount_target.storage_class["0"]
    - aws_efs_mount_target.storage_class["1"]
    - aws_eks_addon.efs
    - aws_iam_openid_connect_provider.eks
    - aws_iam_policy.secrets_manager
    - aws_iam_role_policy_attachment.ebs_csi
    - aws_iam_role_policy_attachment.eks_efs
    - aws_iam_role_policy_attachment.node_secrets_manager
    - aws_security_group.storage_class_nfs
    - aws_security_group_rule.nfs_ingress
    - random_pet.efs
    - aws_s3_bucket_acl.file_storage
    - aws_s3_bucket_cors_configuration.file_storage
    - aws_s3_bucket_ownership_controls.file_storage
    - aws_s3_bucket_server_side_encryption_configuration.file_storage
    - helm_release.operator
    - helm_release.wandb
    - aws_cloudwatch_log_group.this[0]
    - aws_iam_policy.default
    - aws_iam_role.default
    - aws_iam_role_policy_attachment.default
    - helm_release.external_dns
    - aws_default_network_acl.this[0]
    - aws_default_route_table.default[0]
    - aws_iam_policy.default
    - aws_iam_role.default
    - aws_iam_role_policy_attachment.default
    - helm_release.aws_load_balancer_controller

  update_in_place:
    - aws_iam_policy.node_IMDSv2
    - aws_iam_policy.node_cloudwatch
    - aws_iam_policy.node_kms
    - aws_iam_policy.node_s3
    - aws_iam_policy.node_sqs
    - aws_eks_cluster.this[0]
    - aws_elasticache_replication_group.default
    - aws_rds_cluster.this[0]
    - aws_rds_cluster_instance.this["1"]
    - aws_default_security_group.this[0]
    - aws_subnet.private[0]
    - aws_subnet.private[1]
    - aws_subnet.public[0]
    - aws_subnet.public[1]
    - aws_launch_template.workers["primary"]

  destroy:
    - kubernetes_config_map.config_map
    - kubernetes_deployment.wandb
    - kubernetes_priority_class.priority
    - kubernetes_secret.secret
    - kubernetes_service.prometheus
    - kubernetes_service.service
    - random_id.snapshot_identifier[0]

  replace:
    - aws_autoscaling_attachment.autoscaling_attachment["primary"]
    - aws_route53_record.alb
    - aws_eks_node_group.workers["primary"]
```

이렇게 보입니다:

![post-operator-apply](/images/hosting/post-operator-apply.png)

`post-operator.tf`에는 단일 다음과 같은 모듈이 존재합니다:

```hcl
module "wandb_infra" {
  source  = "wandb/wandb/aws"
  version = "4.7.2"
  ...
}
```

#### post-operator 설정에서의 변경 사항:

1. **업데이트 필수 제공자**: `required_providers.aws.version`을 `3.6`에서 `4.0`으로 변경하여 제공자 호환성을 유지합니다.
2. **DNS 및 로드 밸런서 구성**: `enable_dummy_dns` 및 `enable_operator_alb`를 통합하여 인그레스를 통한 DNS 레코드와 AWS 로드 밸런서 설정을 관리합니다.
3. **라이선스 및 크기 구성**: 새로운 운영 요구 사항에 맞게 `wandb_infra` 모듈로 `license` 및 `size` 매개변수를 직접 이동합니다.
4. **커스텀 도메인 처리**: 필요할 경우 `custom_domain_filter`를 사용하여 `kube-system` 네임스페이스 내의 외부 DNS 포드 로그를 통해 DNS 문제를 해결합니다.
5. **헬름 제공자 구성**: `helm` 제공자를 활성화하고 구성하여 Kubernetes 리소스를 효과적으로 관리합니다:

```hcl
provider "helm" {
  kubernetes {
    host                   = data.aws_eks_cluster.app_cluster.endpoint
    cluster_ca_certificate = base64decode(data.aws_eks_cluster.app_cluster.certificate_authority[0].data)
    token                  = data.aws_eks_cluster_auth.app_cluster.token
    exec {
      api_version = "client.authentication.k8s.io/v1beta1"
      args        = ["eks", "get-token", "--cluster-name", data.aws_eks_cluster.app_cluster.name]
      command     = "aws"
    }
  }
}
```

이 포괄적인 설정은 새로운 효율성과 오퍼레이터 모델이 가능하게 하는 기능을 활용하여 Pre-Operator에서 Post-Operator 구성으로 원활한 전환을 보장합니다.