---
title: Deploy W&B Platform on Azure
description: Azure에서 W&B 서버 호스팅하기.
displayed_sidebar: default
---

:::info
W&B는 [W&B Multi-tenant Cloud](../hosting-options/saas_cloud.md) 또는 [W&B 전용 클라우드](../hosting-options//dedicated_cloud.md) 배포 유형과 같은 완전 관리형 배포 옵션을 권장합니다. W&B의 완전 관리형 서비스는 최소한의 설정 또는 설정 없이도 간단하고 안전하게 사용할 수 있습니다.
:::

자체 관리형 W&B 서버를 선택한 경우, W&B는 Azure에 플랫폼을 배포하기 위해 [W&B Server Azure Terraform Module](https://registry.terraform.io/modules/wandb/wandb/azurerm/latest)을 사용하는 것을 권장합니다.

모듈 문서는 방대하며 사용할 수 있는 모든 옵션을 포함하고 있습니다. 이 문서에서는 몇 가지 배포 옵션을 다룰 것입니다.

시작하기 전에, Terraform의 [원격 백엔드](https://developer.hashicorp.com/terraform/language/settings/backends/configuration) 중 하나를 선택하여 [상태 파일](https://developer.hashicorp.com/terraform/language/state)을 저장하는 것을 권장합니다.

상태 파일은 모든 구성 요소를 다시 생성하지 않고도 배포에서 업그레이드를 실행하거나 변경을 수행하는 데 필요한 자원입니다.

Terraform 모듈은 다음과 같은 `필수` 구성 요소를 배포할 것입니다:

- Azure Resource 그룹
- Azure 가상 네트워크 (VPC)
- Azure MySQL 유연한 서버
- Azure 스토리지 계정 및 Blob 스토리지
- Azure Kubernetes 서비스
- Azure 애플리케이션 게이트웨이

다른 배포 옵션은 다음과 같은 선택적 구성 요소를 포함할 수도 있습니다:

- Azure Redis 캐시
- Azure 이벤트 그리드

## **사전 요구 사항 권한**

AzureRM 공급자를 설정하는 가장 간단한 방법은 [Azure CLI](https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs/guides/azure_cli)를 사용하는 것이지만 자동화를 위해 [Azure 서비스 주체](https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs/guides/service_principal_client_secret)를 사용하는 것도 유용할 수 있습니다. 인증 방법에 관계없이 Terraform을 실행할 계정이 도입에 설명된 모든 구성 요소를 생성할 수 있어야 합니다.

## 일반 단계
이 주제의 단계는 이 문서에서 설명하는 모든 배포 옵션에 공통적입니다.

1. 개발 환경을 준비합니다.
  * [Terraform](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli)을 설치하세요.
  * 사용할 코드를 포함하는 Git 리포지토리를 만드는 것을 추천하지만, 로컬에 파일을 유지할 수도 있습니다.

2. **`terraform.tfvars` 파일 생성** `tfvars` 파일의 내용은 설치 유형에 따라 사용자 지정할 수 있지만, 최소 권장 사항은 아래 예시와 같이 보일 것입니다.

   ```bash
    namespace     = "wandb"
    wandb_license = "xxxxxxxxxxyyyyyyyyyyyzzzzzzz"
    subdomain     = "wandb-aws"
    domain_name   = "wandb.ml"
    location      = "westeurope"
   ```

   여기 정의된 변수들은 배포 전에 결정되어야 합니다. `namespace` 변수는 Terraform이 생성하는 모든 리소스에 접두어로 사용할 문자열입니다.

   `subdomain`과 `domain`의 조합은 W&B가 구성할 FQDN을 형성합니다. 위의 예제에서는 W&B FQDN이 `wandb-aws.wandb.ml`이 되고 FQDN 레코드가 생성될 DNS `zone_id`가 됩니다.

3. **`versions.tf` 파일 생성** 이 파일은 AWS에 W&B를 배포하는 데 필요한 Terraform 및 Terraform 공급자 버전을 포함할 것입니다.
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

  AWS 공급자를 구성하기 위해 [Terraform 공식 문서](https://registry.terraform.io/providers/hashicorp/aws/latest/docs#provider-configuration)를 참조하세요.

  선택 사항으로, **강력히 추천하지만**, 문서 시작 부분에 언급된 [원격 백엔드 구성](https://developer.hashicorp.com/terraform/language/settings/backends/configuration)을 추가할 수 있습니다.

4. **`variables.tf` 파일 생성** `terraform.tfvars`에 구성된 각 옵션에 대해, Terraform은 상응하는 변수 선언이 필요합니다.

  ```bash
    variable "namespace" {
      type        = string
      description = "자원을 위한 접두어로 사용되는 문자열입니다."
    }

    variable "location" {
      type        = string
      description = "Azure Resource Group 위치"
    }

    variable "domain_name" {
      type        = string
      description = "Weights & Biases UI에 엑세스하기 위한 도메인."
    }

    variable "subdomain" {
      type        = string
      default     = null
      description = "Weights & Biases UI에 엑세스하기 위한 서브도메인. 기본값은 Route53 경로에 기록을 만듭니다."
    }

    variable "license" {
      type        = string
      description = "wandb/local 라이센스입니다."
    }
  ```

## 권장되는 배포

이것은 모든 `필수` 구성 요소를 생성하고 `Kubernetes 클러스터`에 `W&B`의 최신 버전을 설치하는 가장 간단한 배포 옵션 구성입니다.

1. **`main.tf` 생성** `일반 단계`에서 파일을 생성한 디렉토리와 동일한 디렉토리에 `main.tf` 파일을 다음 내용으로 생성하십시오.

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

  # 필요한 모든 서비스 시작
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

2. **W&B에 배포** W&B를 배포하려면 다음 명령을 실행하십시오:

   ```
   terraform init
   terraform apply -var-file=terraform.tfvars
   ```

## REDIS 캐시를 사용한 배포

다른 배포 옵션에서는 `Redis`를 사용하여 SQL 쿼리를 캐시하고 실험의 메트릭을 로드할 때 애플리케이션 응답을 가속화합니다.

캐시를 활성화하려면 [권장되는 배포](#recommended-deployment)에서 사용한 동일한 `main.tf` 파일에 `create_redis = true` 옵션을 추가해야 합니다.

```bash
# 필요한 모든 서비스 시작
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

## 외부 큐를 사용한 배포

배포 옵션 3은 외부 `메시지 브로커`를 활성화하는 것으로 구성됩니다. 이는 W&B에는 브로커가 내장되어 있기 때문에 선택적입니다. 이 옵션은 성능 향상을 가져오지 않습니다.

메시지 브로커를 제공하는 Azure 리소스는 `Azure Event Grid`이며, 이를 활성화하려면 [권장되는 배포](#recommended-deployment)에서 사용한 동일한 `main.tf` 파일에 `use_internal_queue = false` 옵션을 추가해야 합니다.
```bash
# 필요한 모든 서비스 시작
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

## 다른 배포 옵션

모든 구성 옵션을 동일한 파일에 추가하여 세 가지 배포 옵션을 결합할 수 있습니다. [Terraform 모듈](https://github.com/wandb/terraform-azure-wandb)은 표준 옵션 및 [권장되는 배포](#recommended-deployment)에서 찾을 수 있는 최소 구성과 함께 결합할 수 있는 다양한 옵션을 제공합니다.