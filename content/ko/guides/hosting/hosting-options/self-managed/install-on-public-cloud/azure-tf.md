---
title: Deploy W&B Platform on Azure
description: Azure에서 W&B 서버 호스팅하기.
menu:
  default:
    identifier: ko-guides-hosting-hosting-options-self-managed-install-on-public-cloud-azure-tf
    parent: install-on-public-cloud
weight: 30
---

{{% alert %}}
Weights & Biases는 [W&B Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}}) 또는 [W&B Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ko" >}}) 배포 유형과 같은 완전 관리형 배포 옵션을 권장합니다. Weights & Biases 완전 관리형 서비스는 사용하기 간단하고 안전하며, 필요한 설정이 최소화되어 있거나 전혀 필요하지 않습니다.
{{% /alert %}}

W&B Server를 자체 관리하기로 결정했다면, [W&B Server Azure Terraform Module](https://registry.terraform.io/modules/wandb/wandb/azurerm/latest)을 사용하여 Azure에 플랫폼을 배포하는 것이 좋습니다.

모듈 문서는 광범위하며 사용 가능한 모든 옵션이 포함되어 있습니다. 이 문서에서는 몇 가지 배포 옵션을 다룹니다.

시작하기 전에 Terraform에서 사용 가능한 [원격 백엔드](https://developer.hashicorp.com/terraform/language/backend) 중 하나를 선택하여 [State File](https://developer.hashicorp.com/terraform/language/state)을 저장하는 것이 좋습니다.

State File은 모든 구성 요소를 다시 만들지 않고도 업그레이드를 롤아웃하거나 배포를 변경하는 데 필요한 리소스입니다.

Terraform Module은 다음과 같은 `필수` 구성 요소를 배포합니다.

- Azure 리소스 그룹
- Azure Virtual Network (VPC)
- Azure MySQL Fliexible Server
- Azure Storage Account & Blob Storage
- Azure Kubernetes Service
- Azure Application Gateway

다른 배포 옵션에는 다음과 같은 선택적 구성 요소도 포함될 수 있습니다.

- Azure Cache for Redis
- Azure Event Grid

## **사전 필수 권한**

AzureRM 공급자를 구성하는 가장 간단한 방법은 [Azure CLI](https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs/guides/azure_cli)를 사용하는 것이지만, [Azure Service Principal](https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs/guides/service_principal_client_secret)을 사용한 자동화의 경우에도 유용할 수 있습니다.
사용된 인증 방법에 관계없이 Terraform을 실행할 계정은 도입부에 설명된 모든 구성 요소를 만들 수 있어야 합니다.

## 일반적인 단계
이 항목의 단계는 이 문서에서 다루는 모든 배포 옵션에 공통적입니다.

1. 개발 환경을 준비합니다.
  * [Terraform](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli)을 설치합니다.
  * 사용할 코드로 Git 저장소를 만드는 것이 좋지만, 파일을 로컬에 보관할 수도 있습니다.

2. **`terraform.tfvars` 파일 생성** `tvfars` 파일 내용은 설치 유형에 따라 사용자 정의할 수 있지만, 최소 권장 사항은 아래 예제와 같습니다.

   ```bash
    namespace     = "wandb"
    wandb_license = "xxxxxxxxxxyyyyyyyyyyyzzzzzzz"
    subdomain     = "wandb-aws"
    domain_name   = "wandb.ml"
    location      = "westeurope"
   ```

   여기에 정의된 변수는 배포 전에 결정해야 합니다. `namespace` 변수는 Terraform에서 생성한 모든 리소스에 접두사로 붙는 문자열입니다.

   `subdomain`과 `domain`의 조합은 Weights & Biases가 구성될 FQDN을 형성합니다. 위 예에서 W&B FQDN은 `wandb-aws.wandb.ml`이 되고 FQDN 레코드가 생성될 DNS `zone_id`가 됩니다.

3. **`versions.tf` 파일 생성** 이 파일에는 AWS에 W&B를 배포하는 데 필요한 Terraform 및 Terraform 공급자 버전이 포함됩니다.
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

  AWS 공급자를 구성하려면 [Terraform 공식 문서](https://registry.terraform.io/providers/hashicorp/aws/latest/docs#provider-configuration)를 참조하십시오.

  선택적으로, **하지만 적극 권장되는** 사항으로, 이 문서의 시작 부분에서 언급한 [원격 백엔드 구성](https://developer.hashicorp.com/terraform/language/backend)을 추가할 수 있습니다.

4. **`variables.tf` 파일 생성**. `terraform.tfvars`에 구성된 모든 옵션에 대해 Terraform은 해당 변수 선언이 필요합니다.

  ```bash
    variable "namespace" {
      type        = string
      description = "리소스 접두사에 사용되는 문자열입니다."
    }

    variable "location" {
      type        = string
      description = "Azure 리소스 그룹 위치"
    }

    variable "domain_name" {
      type        = string
      description = "Weights & Biases UI에 엑세스하기 위한 도메인입니다."
    }

    variable "subdomain" {
      type        = string
      default     = null
      description = "Weights & Biases UI에 엑세스하기 위한 서브 도메인입니다. 기본값은 Route53 Route에 레코드를 생성합니다."
    }

    variable "license" {
      type        = string
      description = "귀하의 wandb/local 라이센스"
    }
  ```

## 권장 배포

이는 모든 `필수` 구성 요소를 생성하고 `Kubernetes Cluster`에 최신 버전의 `W&B`를 설치하는 가장 간단한 배포 옵션 구성입니다.

1. **`main.tf` 생성** `General Steps`에서 파일을 만든 동일한 디렉토리에 다음 내용으로 `main.tf` 파일을 만듭니다.

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

  # Spin up all required services
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

2. **W&B에 배포** W&B를 배포하려면 다음 코맨드를 실행합니다.

   ```
   terraform init
   terraform apply -var-file=terraform.tfvars
   ```

## REDIS 캐시를 사용한 배포

또 다른 배포 옵션은 `Redis`를 사용하여 SQL 쿼리를 캐시하고 Experiments에 대한 메트릭을 로드할 때 애플리케이션 응답 속도를 높이는 것입니다.

캐시를 활성화하려면 [권장 배포]({{< relref path="#recommended-deployment" lang="ko" >}})에서 사용한 것과 동일한 `main.tf` 파일에 `create_redis = true` 옵션을 추가해야 합니다.

```bash
# Spin up all required services
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

배포 옵션 3은 외부 `message broker`를 활성화하는 것으로 구성됩니다. W&B가 broker를 내장하고 있기 때문에 이는 선택 사항입니다. 이 옵션은 성능 향상을 가져오지 않습니다.

message broker를 제공하는 Azure 리소스는 `Azure Event Grid`이며, 이를 활성화하려면 [권장 배포]({{< relref path="#recommended-deployment" lang="ko" >}})에서 사용한 것과 동일한 `main.tf`에 `use_internal_queue = false` 옵션을 추가해야 합니다.
```bash
# Spin up all required services
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

동일한 파일에 모든 구성을 추가하여 세 가지 배포 옵션을 모두 결합할 수 있습니다.
[Terraform Module](https://github.com/wandb/terraform-azure-wandb)은 [권장 배포]({{< relref path="#recommended-deployment" lang="ko" >}})에서 찾을 수 있는 표준 옵션 및 최소 구성과 함께 결합할 수 있는 여러 옵션을 제공합니다.
