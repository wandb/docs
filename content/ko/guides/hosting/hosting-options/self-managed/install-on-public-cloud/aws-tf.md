---
title: AWS에 W&B 플랫폼 배포하기
description: AWS에서 W&B 서버 호스팅하기.
menu:
  default:
    identifier: ko-guides-hosting-hosting-options-self-managed-install-on-public-cloud-aws-tf
    parent: install-on-public-cloud
weight: 10
---

{{% alert %}}
W&B는 [W&B Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}}) 또는 [W&B Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ko" >}})와 같은 완전 관리형 배포 옵션을 권장합니다. W&B의 완전 관리형 서비스는 사용이 간편하고 안전하며, 최소한의 설정만으로 바로 사용할 수 있습니다.
{{% /alert %}}

W&B 플랫폼을 AWS에 배포하려면 [W&B Server AWS Terraform Module](https://registry.terraform.io/modules/wandb/wandb/aws/latest) 을 사용하는 것이 가장 좋습니다.

시작에 앞서, Terraform의 [State File](https://developer.hashicorp.com/terraform/language/state)을 저장할 수 있도록 [remote backend](https://developer.hashicorp.com/terraform/language/backend) 중 하나의 사용을 권장합니다.

State File은 배포 후 모든 컴포넌트를 재생성하지 않고도 업그레이드나 변경을 적용하는 데 꼭 필요한 리소스입니다.

Terraform Module은 다음과 같은 `필수` 컴포넌트를 배포합니다:

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

다른 배포 옵션의 경우 다음과 같은 `선택` 컴포넌트도 포함할 수 있습니다:

- Elastic Cache for Redis
- SQS

## 사전 필수 권한

Terraform을 실행하는 계정은 Intro에서 설명한 모든 컴포넌트의 생성 권한 뿐 아니라 **IAM Policies** 와 **IAM Roles** 생성과 역할을 리소스에 할당할 수 있는 권한이 있어야 합니다.

## 일반적인 단계

이 섹션의 단계들은 본 문서에서 다루는 모든 배포 방법에 공통적으로 적용됩니다.

1. 개발 환경을 준비합니다.
   - [Terraform 설치](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli)
   - 버전 관리를 위해 Git 저장소 생성을 권장합니다.
2. `terraform.tfvars` 파일을 만듭니다.

   `tfvars` 파일은 설치 방식에 따라 내용을 맞춤 설정할 수 있지만, 최소 권장 예시는 다음과 같습니다.

   ```bash
   namespace                  = "wandb"
   license                    = "xxxxxxxxxxyyyyyyyyyyyzzzzzzz"
   subdomain                  = "wandb-aws"
   domain_name                = "wandb.ml"
   zone_id                    = "xxxxxxxxxxxxxxxx"
   allowed_inbound_cidr       = ["0.0.0.0/0"]
   allowed_inbound_ipv6_cidr  = ["::/0"]
   eks_cluster_version        = "1.29"
   ```

   배포에 앞서 `tfvars` 파일에 변수를 정의해야 합니다. 예를 들어, `namespace` 변수는 Terraform이 생성하는 모든 리소스의 접두사로 사용됩니다.

   `subdomain`과 `domain`을 조합하면 W&B에 설정될 FQDN이 만들어집니다. 위의 예시에서는 `wandb-aws.wandb.ml`이 W&B FQDN이 되고, 해당 FQDN 레코드가 생성될 DNS `zone_id`가 지정됩니다.

   `allowed_inbound_cidr`와 `allowed_inbound_ipv6_cidr` 역시 반드시 입력해야 하며, 위 예시는 W&B 서비스에 모든 소스에서 접속 가능하도록 허용한 설정입니다.

3. `versions.tf` 파일을 만듭니다.

   이 파일에는 AWS에 W&B를 배포하는 데 필요한 Terraform 및 Provider 버전을 지정합니다.

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

   AWS Provider 구성 방법은 [Terraform 공식 문서](https://registry.terraform.io/providers/hashicorp/aws/latest/docs#provider-configuration)를 참고하세요.

   선택 사항이지만, 본 문서 초반에 언급한 [remote backend 구성](https://developer.hashicorp.com/terraform/language/backend)도 적극 권장합니다.

4. `variables.tf` 파일을 만듭니다.

   `terraform.tfvars`에 설정한 모든 옵션마다 Terraform은 대응하는 변수 선언이 필요합니다.

   ```
   variable "namespace" {
     type        = string
     description = "리소스에 사용할 이름 접두사"
   }

   variable "domain_name" {
     type        = string
     description = "인스턴스 엑세스에 사용할 도메인 명"
   }

   variable "subdomain" {
     type        = string
     default     = null
     description = "Weights & Biases UI 엑세스에 사용할 서브도메인"
   }

   variable "license" {
     type = string
   }

   variable "zone_id" {
     type        = string
     description = "Weights & Biases 서브도메인용 도메인"
   }

   variable "allowed_inbound_cidr" {
    description = "wandb-server 엑세스 허용 CIDR 대역"
    nullable    = false
    type        = list(string)
   }

   variable "allowed_inbound_ipv6_cidr" {
    description = "wandb-server 엑세스 허용 CIDR 대역"
    nullable    = false
    type        = list(string)
   }

   variable "eks_cluster_version" {
    description = "EKS 클러스터 쿠버네티스 버전"
    nullable    = false
    type        = string
   }
   ```

## 권장 배포 옵션

이 방법은 모든 `필수` 컴포넌트를 생성하고 `Kubernetes Cluster`에 최신 `W&B` 버전을 설치하는 가장 간단한 배포 방식입니다.

1. `main.tf` 파일 생성

   앞서 만든 파일들과 같은 디렉토리에 아래와 같이 `main.tf` 파일을 만듭니다.

   ```
   module "wandb_infra" {
     source  = "wandb/wandb/aws"
     version = "~>7.0"

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
     eks_cluster_version            = var.eks_cluster_version
   }

    data "aws_eks_cluster" "eks_cluster_id" {
      name = module.wandb_infra.cluster_name
    }

    data "aws_eks_cluster_auth" "eks_cluster_auth" {
      name = module.wandb_infra.cluster_name
    }

    provider "kubernetes" {
      host                   = data.aws_eks_cluster.eks_cluster_id.endpoint
      cluster_ca_certificate = base64decode(data.aws_eks_cluster.eks_cluster_id.certificate_authority.0.data)
      token                  = data.aws_eks_cluster_auth.eks_cluster_auth.token
    }


    provider "helm" {
      kubernetes {
        host                   = data.aws_eks_cluster.eks_cluster_id.endpoint
        cluster_ca_certificate = base64decode(data.aws_eks_cluster.eks_cluster_id.certificate_authority.0.data)
        token                  = data.aws_eks_cluster_auth.eks_cluster_auth.token
      }
    }

    output "url" {
      value = module.wandb_infra.url
    }

    output "bucket" {
      value = module.wandb_infra.bucket_name
    }
   ```

2. W&B 배포

   다음 명령어로 W&B를 배포합니다:

   ```
   terraform init
   terraform apply -var-file=terraform.tfvars
   ```

## REDIS 사용 활성화

다른 배포 옵션으로, `Redis`를 사용하여 SQL 쿼리를 캐싱하고, 실험의 메트릭 로딩 속도를 높일 수 있습니다.

`main.tf` 파일에 `create_elasticache_subnet = true` 옵션을 추가하면 캐시가 활성화됩니다. 방법은 [권장 배포 옵션]({{< relref path="#recommended-deployment-option" lang="ko" >}}) 섹션 설명을 참고하세요.

```
module "wandb_infra" {
  source  = "wandb/wandb/aws"
  version = "~>7.0"

  namespace   = var.namespace
  domain_name = var.domain_name
  subdomain   = var.subdomain
  zone_id     = var.zone_id
	**create_elasticache_subnet = true**
}
[...]
```

## 메시지 브로커(Queue) 활성화

3번째 배포 옵션은 외부 `message broker`를 사용하는 것입니다. 이 옵션은 선택 사항이며, W&B에는 내장 브로커가 포함되어 있습니다. 해당 옵션은 성능 향상을 제공하지는 않습니다.

AWS에서 메시지 브로커는 `SQS` 리소스로 제공되며, 활성화를 위해서는 `main.tf` 파일에 `use_internal_queue = false` 옵션을 추가해야 합니다. 방법은 [권장 배포 옵션]({{< relref path="#recommended-deployment-option" lang="ko" >}}) 섹션을 참고하세요.

```
module "wandb_infra" {
  source  = "wandb/wandb/aws"
  version = "~>7.0"

  namespace   = var.namespace
  domain_name = var.domain_name
  subdomain   = var.subdomain
  zone_id     = var.zone_id
  **use_internal_queue = false**

[...]
}
```

## 기타 배포 옵션

세 가지 배포 옵션 모두 한 파일에 설정을 추가하여 같이 사용할 수도 있습니다. [Terraform Module](https://github.com/wandb/terraform-aws-wandb) 에서는 표준 옵션과 `Deployment - Recommended`의 최소 구성과 함께 조합할 수 있는 다양한 옵션을 제공합니다.

## 수동 설정

Amazon S3 버킷을 W&B의 파일 스토리지 백엔드로 사용하려면 다음 과정을 수행해야 합니다:

* [Amazon S3 Bucket 및 Bucket 알림 생성]({{< relref path="#create-an-s3-bucket-and-bucket-notifications" lang="ko" >}})
* [SQS Queue 생성]({{< relref path="#create-an-sqs-queue" lang="ko" >}})
* [W&B 가 실행되는 Node에 권한 부여]({{< relref path="#grant-permissions-to-node-that-runs-wb" lang="ko" >}})

즉, 버킷을 만들고, 해당 버킷에서 오브젝트가 생성될 때마다 SQS queue로 알림을 보내도록 설정해야 합니다. 인스턴스는 이 queue의 읽기 권한이 필요합니다.

### S3 버킷 및 알림 생성

아래 절차에 따라 Amazon S3 버킷을 만들고 알림을 활성화하세요.

1. AWS Console에서 Amazon S3로 이동합니다.
2. **Create bucket**을 선택합니다.
3. **Advanced settings** 내 **Events** 섹션에서 **Add notification**을 선택합니다.
4. 모든 오브젝트 생성 이벤트가 앞서 설정한 SQS Queue로 전송되도록 구성합니다.

{{< img src="/images/hosting/s3-notification.png" alt="Enterprise file storage settings" >}}

CORS 엑세스를 활성화합니다. CORS 설정 예시는 다음과 같습니다:

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

### SQS Queue 생성

SQS Queue를 생성하려면 아래 절차를 따르세요:

1. AWS Console에서 Amazon SQS로 이동합니다.
2. **Create queue**를 선택합니다.
3. **Details** 섹션에서 **Standard** queue type을 선택합니다.
4. Access policy 섹션에서 다음 principal에 권한을 추가합니다:
* `SendMessage`
* `ReceiveMessage`
* `ChangeMessageVisibility`
* `DeleteMessage`
* `GetQueueUrl`

필요에 따라 **Access Policy**에서 고급 접근 제어 정책을 추가할 수 있습니다. 예시 정책:

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

### W&B가 실행되는 Node에 권한 부여

W&B 서버가 실행 중인 노드는 Amazon S3와 Amazon SQS 엑세스가 가능하도록 구성해야 합니다. 서버 배포 방식에 따라, 노드 롤에 다음 정책 구문을 추가해야 할 수 있습니다:

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

### W&B 서버 설정
마지막으로, W&B Server를 설정합니다.

1. `http(s)://YOUR-W&B-SERVER-HOST/system-admin`의 W&B 설정 페이지로 이동합니다. 
2. ***외부 파일 스토리지 백엔드 사용* 옵션을 활성화합니다.
3. 아래 형식에 맞추어 Amazon S3 버킷, 리전, SQS queue 정보를 입력합니다:
* **File Storage Bucket**: `s3://<bucket-name>`
* **File Storage Region (AWS only)**: `<region>`
* **Notification Subscription**: `sqs://<queue-name>`

{{< img src="/images/hosting/configure_file_store.png" alt="AWS file storage configuration" >}}

4. **Update settings**를 선택하여 새로운 설정을 적용합니다.

## W&B 버전 업그레이드

W&B를 업데이트하려면 아래 단계를 따르세요:

1. `wandb_app` 모듈의 설정에 `wandb_version`을 추가합니다. 업그레이드하고자 하는 W&B 버전을 입력하세요. 아래는 `0.48.1` 버전 사용 예시입니다.

  ```
  module "wandb_app" {
      source  = "wandb/wandb/kubernetes"
      version = "~>1.0"

      license       = var.license
      wandb_version = "0.48.1"
  ```

  {{% alert %}}
  또는 `terraform.tfvars`에 `wandb_version` 값을 추가하고 동일한 이름의 변수를 만들어서, 위 예시처럼 리터럴 값 대신 `var.wandb_version`을 사용할 수도 있습니다.
  {{% /alert %}}

2. 설정을 업데이트한 후, [권장 배포 옵션 섹션]({{< relref path="#recommended-deployment-option" lang="ko" >}})의 절차를 따라 마무리하세요.

## operator 기반 AWS Terraform module로 마이그레이션

이 섹션에서는 [terraform-aws-wandb](https://registry.terraform.io/modules/wandb/wandb/aws/latest) 모듈을 활용해 _pre-operator_에서 _post-operator_ 환경으로 업그레이드 하는 방법을 안내합니다.

{{% alert %}}
Kubernetes [operator](https://kubernetes.io/docs/concepts/extend-kubernetes/operator/) 방식으로의 전환은 W&B 아키텍처에 필수적입니다. 자세한 설명은 [아키텍처 변화 설명]({{< relref path="/guides/hosting/hosting-options/self-managed/kubernetes-operator/#reasons-for-the-architecture-shift" lang="ko" >}})을 참고하세요.
{{% /alert %}}

### 아키텍처 변화 전후

이전 W&B 아키텍처에서는:

```hcl
module "wandb_infra" {
  source  = "wandb/wandb/aws"
  version = "1.16.10"
  ...
}
```

로 인프라를 제어했습니다:

{{< img src="/images/hosting/pre-operator-infra.svg" alt="pre-operator-infra" >}}

그리고 별도의 모듈로 W&B Server를 배포했습니다:

```hcl
module "wandb_app" {
  source  = "wandb/wandb/kubernetes"
  version = "1.12.0"
}
```

{{< img src="/images/hosting/pre-operator-k8s.svg" alt="pre-operator-k8s" >}}

이제 operator 방식 전환 후에는,

```hcl
module "wandb_infra" {
  source  = "wandb/wandb/aws"
  version = "4.7.2"
  ...
}
```

한 번에 인프라 구축과 W&B Server의 Kubernetes 클러스터 설치까지 관리하며, `post-operator.tf`에서 더 이상 `module "wandb_app"`이 필요하지 않습니다.

{{< img src="/images/hosting/post-operator-k8s.svg" alt="post-operator-k8s" >}}

이 아키텍처 변화로 OpenTelemetry, Prometheus, HPA, Kafka, 이미지 업데이트 등 다양한 부가 기능을 SRE/인프라팀의 별도 Terraform 작업 없이 사용할 수 있습니다.

W&B Pre-Operator 기본 설치를 시작하려면, `post-operator.tf` 파일이 `.disabled` 확장자를 갖고 있고, `pre-operator.tf`는 활성화되어 있어야 합니다. 관련 파일은 [여기](https://github.com/wandb/terraform-aws-wandb/tree/main/docs/operator-migration)에서 확인할 수 있습니다.

### 사전 준비

마이그레이션 시작 전, 다음 전제 조건을 갖추세요:

- **Egress**: 배포 환경이 airgapped가 아니어야 합니다. **_Release Channel_** 사양을 최신으로 받기 위해 [deploy.wandb.ai](https://deploy.wandb.ai) 접속이 필요합니다.
- **AWS 자격 증명**: AWS 리소스 접근을 위한 자격 증명이 제대로 설정되어야 합니다.
- **Terraform 설치**: 최신 Terraform 버전이 설치되어 있어야 합니다.
- **Route53 Hosted Zone**: 애플리케이션이 서비스될 도메인에 해당하는 Route53 hosted zone이 필요합니다.
- **Pre-Operator Terraform 파일**: `pre-operator.tf` 와 관련 변수 파일(`pre-operator.tfvars`)이 정확히 설정되어 있어야 합니다.

### Pre-Operator 설정

Terraform 커맨드로 Pre-Operator 구성을 초기화 및 적용하세요:

```bash
terraform init -upgrade
terraform apply -var-file=./pre-operator.tfvars
```

`pre-operator.tf`의 예시는 다음과 같습니다:

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

이 모듈이 인프라를 생성합니다.

```hcl
module "wandb_app" {
  source  = "wandb/wandb/kubernetes"
  version = "1.12.0"
}
```

이 모듈이 애플리케이션을 배포합니다.

### Post-Operator 설정

이제 `pre-operator.tf` 파일에 `.disabled` 확장자를 붙이고, `post-operator.tf`를 활성화합니다.

`post-operator.tfvars`에는 추가 변수가 포함됩니다:

```ini
...
# wandb_version = "0.51.2"는 이제 Release Channel 또는 User Spec에서 관리됩니다.

# 업그레이드를 위한 Operator 필수 변수:
size                 = "small"
enable_dummy_dns     = true
enable_operator_alb  = true
custom_domain_filter = "sandbox-aws.wandb.ml"
```

다음 명령어로 Post-Operator 설정을 초기화 및 적용합니다:

```bash
terraform init -upgrade
terraform apply -var-file=./post-operator.tfvars
```

계획 및 적용 과정에서 아래와 같은 리소스가 업데이트됩니다:

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

아래와 같이 화면을 확인할 수 있습니다:

{{< img src="/images/hosting/post-operator-apply.png" alt="post-operator-apply" >}}

이제 `post-operator.tf`에는 단일:

```hcl
module "wandb_infra" {
  source  = "wandb/wandb/aws"
  version = "4.7.2"
  ...
}
```

#### post-operator 구성 변경점:

1. **필수 Provider 업데이트**: `required_providers.aws.version`을 `3.6`에서 `4.0`으로 변경해 Provider 호환성을 맞춥니다.
2. **DNS 및 Load Balancer 구성**: `enable_dummy_dns`와 `enable_operator_alb`를 추가해 Ingress를 통한 DNS 설정과 AWS Load Balancer를 활성화합니다.
3. **라이선스/사이즈 구성**: 새로운 운영 요구사항에 맞춰 `license`와 `size`를 `wandb_infra` 모듈에 직접 전달하세요.
4. **커스텀 도메인 처리**: 필요에 따라 `custom_domain_filter`를 사용해 도메인 문제를 점검(특히, `kube-system` 네임스페이스의 External DNS pod 로그 확인)할 수 있습니다.
5. **Helm Provider 구성**: Helm provider를 활성화하고, Kubernetes 리소스를 효과적으로 관리하도록 설정합니다:

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

이와 같이 구성하면 Pre-Operator에서 Post-Operator로 매끄럽게 전환하고, operator 모델이 제공하는 새로운 효율성과 기능을 모두 누릴 수 있습니다.