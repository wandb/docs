---
description: Hosting W&B Server on Azure.
displayed_sidebar: default
---

# Azure

Weights & Biases에서 개발한 [Terraform 모듈](https://registry.terraform.io/modules/wandb/wandb/azurerm/latest)을 사용하여 Azure에서 W&B 서버를 배포하는 것을 권장합니다.

모듈 문서에는 사용할 수 있는 모든 옵션이 포함되어 있습니다. 이 문서에서는 일부 배포 옵션을 다룰 것입니다.

시작하기 전에, Terraform이 [State File](https://developer.hashicorp.com/terraform/language/state)을 저장하기 위해 사용할 수 있는 [remote backends](https://developer.hashicorp.com/terraform/language/settings/backends/configuration) 중 하나를 선택하는 것이 좋습니다.

State File은 배포를 재구성하지 않고 업그레이드를 진행하거나 변경사항을 적용하는 데 필요한 리소스입니다.

Terraform 모듈은 다음과 같은 `필수` 구성요소를 배포합니다:

- Azure 리소스 그룹
- Azure Virtual Network (VPC)
- Azure MySQL Flexible 서버
- Azure Storage Account & Blob Storage
- Azure Kubernetes 서비스
- Azure Application 게이트웨이

다른 배포 옵션에는 다음과 같은 선택적 구성요소도 포함될 수 있습니다:

- Azure Redis 캐시
- Azure Event Grid

## **사전 요구사항 권한**

AzureRM 제공자를 구성하는 가장 간단한 방법은 [Azure CLI](https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs/guides/azure_cli)를 통한 방법이지만, 자동화의 경우 [Azure 서비스 주체](https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs/guides/service_principal_client_secret)를 사용하는 것도 유용할 수 있습니다.
사용된 인증 방법에 관계없이 Terraform을 실행할 계정은 소개에서 설명한 모든 구성요소를 생성할 수 있어야 합니다.

## 일반 단계
이 주제의 단계는 이 문서에서 다루는 모든 배포 옵션에 공통적입니다.

1. 개발 환경 준비.
  * [Terraform](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli) 설치
  * 사용할 코드를 Git 저장소에 만드는 것을 권장하지만, 파일을 로컬에 보관할 수도 있습니다.

2. **`terraform.tfvars` 파일 생성** `tvfars` 파일 내용은 설치 유형에 따라 맞춤설정할 수 있지만, 아래 예제와 같은 최소 권장 사항이 있습니다.

   ```bash
    namespace     = "wandb"
    wandb_license = "xxxxxxxxxxyyyyyyyyyyyzzzzzzz"
    subdomain     = "wandb-aws"
    domain_name   = "wandb.ml"
    location      = "westeurope"
   ```

   여기서 정의된 변수는 배포 전에 결정해야 합니다. `namespace` 변수는 Terraform이 생성하는 모든 리소스에 접두사로 사용될 문자열입니다.

   `subdomain`과 `domain`의 조합은 W&B가 구성될 FQDN을 형성할 것입니다. 위의 예제에서는 W&B FQDN이 `wandb-aws.wandb.ml`이고 FQDN 레코드가 생성될 DNS `zone_id`가 됩니다.

3. **`versions.tf` 파일 생성** 이 파일에는 AWS에서 W&B를 배포하는 데 필요한 Terraform 및 Terraform 제공자 버전이 포함됩니다.
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

  AWS 제공자를 구성하려면 [Terraform 공식 문서](https://registry.terraform.io/providers/hashicorp/aws/latest/docs#provider-configuration)를 참조하세요.

  선택 사항이지만, **매우 권장되는**, 문서 시작 부분에서 언급한 [remote backend 설정](https://developer.hashicorp.com/terraform/language/settings/backends/configuration)을 추가할 수 있습니다.

4. **`variables.tf` 파일 생성** `terraform.tfvars`에서 구성된 모든 옵션에 대해 Terraform은 해당 변수 선언이 필요합니다.

   ```bash
    variable "namespace" {
      type        = string
      description = "리소스 접두사로 사용되는 문자열."
    }

    variable "location" {
      type        = string
      description = "Azure 리소스 그룹 위치"
    }

    variable "domain_name" {
      type        = string
      description = "Weights & Biases UI에 접속하기 위한 도메인."
    }

    variable "subdomain" {
      type        = string
      default     = null
      description = "Weights & Biases UI에 접속하기 위한 서브 도메인. 기본값은 Route53 Route에서 레코드를 생성합니다."
    }

    variable "license" {
      type        = string
      description = "당신의 wandb/local 라이센스"
    }
  ```

## 배포 - 추천 (~20분)

이것은 모든 `필수` 구성요소를 생성하고 `Kubernetes 클러스터`에 `W&B`의 최신 버전을 설치하는 가장 간단한 배포 옵션 구성입니다.

1. **`main.tf` 생성** `일반 단계`에서 생성한 파일과 같은 디렉토리에 다음 내용을 가진 `main.tf` 파일을 생성하세요:

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

  # 모든 필요한 서비스 실행
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

2. **W&B에 배포** W&B에 배포하려면 다음 명령을 실행하세요:

   ```
   terraform init
   terraform apply -var-file=terraform.tfvars
   ```

## REDIS 캐시와 함께 배포

다른 배포 옵션은 `Redis`를 사용하여 SQL 쿼리를 캐시하고 실험에 대한 메트릭을 로딩할 때 애플리케이션 응답 속도를 높입니다.

[Deployment Recommended](azure-tf.md#deployment---recommended-20-mins)에서 작업한 동일한 `main.tf` 파일에 `create_redis = true` 옵션을 추가하여 캐시를 활성화할 수 있습니다.

```bash
# 모든 필요한 서비스 실행
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

## 외부 큐와 함께 배포

옵션 3은 외부 `메시지 브로커`를 활성화하는 배포 옵션입니다. 이것은 W&B가 내장된 브로커를 제공하기 때문에 선택적입니다. 이 옵션은 성능 개선을 제공하지 않습니다.

메시지 브로커를 제공하는 Azure 리소스는 `Azure Event Grid`이며, 이를 활성화하려면 [Deployment Recommended](azure-tf.md#deployment---recommended-20-mins)에서 작업한 동일한 `main.tf`에 `use_internal_queue = false` 옵션을 추가해야 합니다.
```bash
# 모든 필요한 서비스 실행
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

세 가지 배포 옵션을 모두 결합하여 동일한 파일에 모든 구성을 추가할 수 있습니다.
[Terraform 모듈](https://github.com/wandb/terraform-azure-wandb)은 표준 옵션 및 [Deployment Recommended](azure-tf.md#deployment---recommended-20-mins)에서 찾을 수 있는 최소 구성과 함께 결합할 수 있는 여러 옵션을 제공합니다.