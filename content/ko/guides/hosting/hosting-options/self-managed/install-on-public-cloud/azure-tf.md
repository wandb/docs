---
title: Azure 에 W&B 플랫폼 배포하기
description: Azure 에서 W&B 서버 호스팅하기.
menu:
  default:
    identifier: ko-guides-hosting-hosting-options-self-managed-install-on-public-cloud-azure-tf
    parent: install-on-public-cloud
weight: 30
---

{{% alert %}}
W&B는 [W&B Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}}) 또는 [W&B Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ko" >}})와 같은 완전 관리형 배포 옵션을 권장합니다. W&B의 완전 관리형 서비스는 최소 또는 별도의 설정 없이 간단하고 안전하게 사용할 수 있습니다.
{{% /alert %}}

직접 W&B Server를 관리하기로 결정하셨다면, Azure에서 플랫폼을 배포하기 위해 [W&B Server Azure Terraform Module](https://registry.terraform.io/modules/wandb/wandb/azurerm/latest) 사용을 권장합니다.

해당 모듈 문서는 매우 자세하게 작성되어 있으며, 사용 가능한 모든 옵션을 포함합니다. 이 문서에서는 주요 배포 옵션에 대해 다룹니다.

시작하기 전에, Terraform의 [State File](https://developer.hashicorp.com/terraform/language/state)을 저장하기 위한 [remote backends](https://developer.hashicorp.com/terraform/language/backend) 중 하나를 선택하는 것을 권장합니다.

State File은 모든 컴포넌트를 다시 생성하지 않고도 업그레이드 또는 배포 환경의 변경을 진행할 수 있도록 해주는 필수 리소스입니다.

Terraform Module은 다음과 같은 `필수` 컴포넌트들을 배포합니다:

- Azure Resource Group
- Azure Virtual Network (VPC)
- Azure MySQL Flexible Server
- Azure Storage Account & Blob Storage
- Azure Kubernetes Service
- Azure Application Gateway

추가적으로, 다음과 같은 선택적 컴포넌트들도 배포 옵션에 따라 포함될 수 있습니다:

- Azure Cache for Redis
- Azure Event Grid

## **사전 요구 권한**

AzureRM provider 설정의 가장 쉬운 방법은 [Azure CLI](https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs/guides/azure_cli)를 사용하는 것입니다. 자동화가 필요한 경우 [Azure Service Principal](https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs/guides/service_principal_client_secret)도 유용하게 사용할 수 있습니다.
어떤 인증 메소드를 사용하든, Terraform을 실행하는 계정은 도입부에 설명된 모든 컴포넌트를 생성할 수 있는 권한이 있어야 합니다.

## 일반적인 단계
이 문서에서 다루는 모든 배포 옵션에서 공통적으로 필요한 단계입니다.

1. 개발 환경 준비하기.
  * [Terraform](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli) 설치
  * 사용될 코드를 위한 Git 저장소 생성(권장), 또는 파일을 로컬에 보관할 수도 있음

2. **`terraform.tfvars` 파일 생성**  
   `tfvars` 파일의 내용은 설치 유형에 따라 맞춤화될 수 있지만, 최소 권장 예시는 아래와 같습니다.

   ```bash
    namespace     = "wandb"
    wandb_license = "xxxxxxxxxxyyyyyyyyyyyzzzzzzz"
    subdomain     = "wandb-aws"
    domain_name   = "wandb.ml"
    location      = "westeurope"
   ```

   여기에 정의된 변수는 배포 전에 결정해야 하며, `namespace` 변수는 Terraform이 생성하는 모든 리소스에 접두사를 붙이기 위해 사용됩니다.

   `subdomain`과 `domain`의 조합으로 W&B가 설정될 FQDN이 형성됩니다. 위 예시에서는 W&B의 FQDN이 `wandb-aws.wandb.ml`이 되고, 해당 FQDN 레코드가 생성될 DNS `zone_id`가 필요합니다.

3. **`versions.tf` 파일 생성**  
   이 파일에는 AWS에 W&B를 배포하는 데 필요한 Terraform 및 Provider 버전이 명시되어야 합니다.
  ```bash
  terraform {
    required_version = "~> 1.3"

    required_providers {
      azurerm = {
        source  = "hashicorp/azurerm"
        version = "~> 3.17"
      }
    }
  }
  ```

  AWS provider 설정 방법은 [Terraform 공식 문서](https://registry.terraform.io/providers/hashicorp/aws/latest/docs#provider-configuration)를 참고하세요.

  선택적으로, **강력히 권장**하는 사항으로, 서두에 언급한 [remote backend 설정](https://developer.hashicorp.com/terraform/language/backend)도 추가할 수 있습니다.

4. **`variables.tf` 파일 생성**  
   `terraform.tfvars`에 설정한 각 옵션에 대해 Terraform에서는 대응하는 변수 선언이 필요합니다.

  ```bash
    variable "namespace" {
      type        = string
      description = "접두사로 사용할 문자열입니다."
    }

    variable "location" {
      type        = string
      description = "Azure 리소스 그룹의 위치"
    }

    variable "domain_name" {
      type        = string
      description = "Weights & Biases UI에 엑세스할 도메인"
    }

    variable "subdomain" {
      type        = string
      default     = null
      description = "Weights & Biases UI에 엑세스할 서브도메인. 기본값은 Route53에 레코드를 생성합니다."
    }

    variable "license" {
      type        = string
      description = "wandb/local 라이선스"
    }
  ```

## 권장 배포 방식

이 방식은 모든 `필수` 컴포넌트를 생성하고 `Kubernetes Cluster`에 최신 버전의 `W&B`를 설치하는 가장 간단한 배포 옵션입니다.

1. **`main.tf` 파일 생성**  
   `일반적인 단계`에서 파일을 만든 동일한 디렉토리에 아래 내용으로 `main.tf` 파일을 생성하세요.

  ```bash
  provider "azurerm" {
    features {}
  }

  provider "kubernetes" {
    host                   = module.wandb.cluster_host
    cluster_ca_certificate = base64decode(module.wandb.cluster_ca_certificate)
    client_key             = base64decode(module.wandb.cluster_client_key)
    client_certificate     = base64decode(module.wandb.cluster_client_certificate)
  }

  provider "helm" {
    kubernetes {
      host                   = module.wandb.cluster_host
      cluster_ca_certificate = base64decode(module.wandb.cluster_ca_certificate)
      client_key             = base64decode(module.wandb.cluster_client_key)
      client_certificate     = base64decode(module.wandb.cluster_client_certificate)
    }
  }

  # 필수 서비스 준비
  module "wandb" {
    source  = "wandb/wandb/azurerm"
    version = "~> 1.2"

    namespace   = var.namespace
    location    = var.location
    license     = var.license
    domain_name = var.domain_name
    subdomain   = var.subdomain

    deletion_protection = false

    tags = {
      "Example" : "PublicDns"
    }
  }

  output "address" {
    value = module.wandb.address
  }

  output "url" {
    value = module.wandb.url
  }
  ```

2. **W&B 배포**  
   W&B를 배포하려면 다음 명령어를 실행하세요.

   ```
   terraform init
   terraform apply -var-file=terraform.tfvars
   ```

## REDIS Cache를 포함한 배포

다른 배포 옵션으로는, `Redis`를 이용해서 SQL 쿼리의 캐싱과 실험 메트릭 로딩 시 애플리케이션 응답 속도를 개선할 수 있습니다.

캐시를 활성화하려면 [권장 배포]({{< relref path="#recommended-deployment" lang="ko" >}})에서 사용한 동일한 `main.tf` 파일에 `create_redis = true` 옵션을 추가해야 합니다.

```bash
# 필수 서비스 준비
module "wandb" {
  source  = "wandb/wandb/azurerm"
  version = "~> 1.2"


  namespace   = var.namespace
  location    = var.location
  license     = var.license
  domain_name = var.domain_name
  subdomain   = var.subdomain

  create_redis       = true # Redis 생성
  [...]
```

## 외부 큐를 포함한 배포

세 번째 배포 옵션은 외부 `message broker`를 사용하는 방법입니다. W&B 자체적으로 message broker가 포함되어 있으므로, 이 옵션은 필수는 아니며 성능 향상을 보장하지도 않습니다.

Azure에서 message broker 역할을 하는 리소스는 `Azure Event Grid`이며, 이를 활성화하려면 [권장 배포]({{< relref path="#recommended-deployment" lang="ko" >}})와 동일한 `main.tf`에 `use_internal_queue = false` 옵션을 추가하면 됩니다.

```bash
# 필수 서비스 준비
module "wandb" {
  source  = "wandb/wandb/azurerm"
  version = "~> 1.2"


  namespace   = var.namespace
  location    = var.location
  license     = var.license
  domain_name = var.domain_name
  subdomain   = var.subdomain

  use_internal_queue       = false # Azure Event Grid 활성화
  [...]
}
```

## 기타 배포 옵션

세 가지 배포 옵션 모두를 한 파일에 설정을 추가해 조합할 수 있습니다.  
[Terraform Module](https://github.com/wandb/terraform-azure-wandb)은 표준 옵션 및 [권장 배포]({{< relref path="#recommended-deployment" lang="ko" >}})에 포함된 최소 설정과 더불어 여러 가지 추가 옵션을 제공합니다.