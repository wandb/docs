---
title: GCP 에 W&B 플랫폼 배포하기
description: GCP에서 W&B 서버 호스팅하기
menu:
  default:
    identifier: ko-guides-hosting-hosting-options-self-managed-install-on-public-cloud-gcp-tf
    parent: install-on-public-cloud
weight: 20
---

{{% alert %}}
W&B는 [W&B Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}}) 또는 [W&B Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ko" >}})와 같은 완전 관리형 배포 옵션을 권장합니다. W&B 완전 관리형 서비스는 최소 또는 별도의 설정 없이 간편하고 안전하게 사용할 수 있습니다.
{{% /alert %}}

직접 W&B Server를 운영하기로 결정하셨다면, GCP에 플랫폼을 배포하기 위해 [W&B Server GCP Terraform Module](https://registry.terraform.io/modules/wandb/wandb/google/latest)을 사용하는 것을 권장드립니다.

이 모듈의 문서는 매우 상세하며, 사용할 수 있는 모든 옵션이 포함되어 있습니다.

시작하기 전에, Terraform의 [원격 백엔드](https://developer.hashicorp.com/terraform/language/backend/remote) 중 하나를 선택하여 [State File](https://developer.hashicorp.com/terraform/language/state)을 저장하는 것을 추천합니다.

State File은 배포 시 모든 컴포넌트를 다시 생성하지 않고, 업그레이드나 변경사항을 적용할 때 반드시 필요한 리소스입니다.

Terraform 모듈은 다음의 `필수` 컴포넌트들을 자동으로 배포합니다:

- VPC
- Cloud SQL for MySQL
- Cloud Storage Bucket
- Google Kubernetes Engine
- KMS Crypto Key
- Load Balancer

다른 옵션을 추가할 경우, 다음과 같은 선택적 컴포넌트도 포함될 수 있습니다:

- Redis용 Memory store
- Pub/Sub 메시지 시스템

## 사전 요구 권한

Terraform을 실행할 계정이 사용할 GCP 프로젝트에서 `roles/owner` 역할을 가지고 있어야 합니다.

## 일반 절차

이 항목의 모든 단계는 본 문서에서 소개하는 어떤 배포 방식에도 공통적으로 적용됩니다.

1. 개발 환경을 준비합니다.
   - [Terraform](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli) 설치
   - 사용할 코드를 관리할 Git 저장소 생성을 권장하지만, 파일을 로컬에만 보관해도 됩니다.
   - [Google Cloud Console](https://console.cloud.google.com/)에서 프로젝트 생성
   - GCP에 인증 (사전에 [gcloud 설치](https://cloud.google.com/sdk/docs/install) 필요)
     `gcloud auth application-default login`
2. `terraform.tfvars` 파일을 생성합니다.

   `tfvars` 파일의 내용은 설치 형태에 따라 맞춤 구성할 수 있습니다. 최소 권장 내용은 아래와 같습니다.

   ```bash
   project_id  = "wandb-project"
   region      = "europe-west2"
   zone        = "europe-west2-a"
   namespace   = "wandb"
   license     = "xxxxxxxxxxyyyyyyyyyyyzzzzzzz"
   subdomain   = "wandb-gcp"
   domain_name = "wandb.ml"
   ```

   여기 정의된 변수들은 배포 전 결정되어야 합니다. `namespace` 변수는 Terraform이 생성하는 모든 리소스 이름 앞에 붙는 문자열입니다.

   `subdomain`과 `domain` 조합으로 W&B가 설정될 FQDN이 만들어집니다. 위의 예시에서는 W&B FQDN이 `wandb-gcp.wandb.ml`이 됩니다.

3. `variables.tf` 파일을 생성합니다.

   `terraform.tfvars`에 정의된 각 옵션은 Terraform에 변수 선언이 필요합니다.

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

## 배포 - 권장 방식 (~20분 소요)

가장 단순한 구성의 배포 옵션으로 모든 `필수` 컴포넌트 생성과 `Kubernetes Cluster`에 최신 버전의 `W&B` 설치를 포함합니다.

1. `main.tf` 파일을 만듭니다.

   [일반 절차]({{< relref path="#general-steps" lang="ko" >}})에서 생성한 파일들과 동일한 디렉토리에, 아래와 같이 `main.tf` 파일을 생성하세요:

   ```
   provider "google" {
    project = var.project_id
    region  = var.region
    zone    = var.zone
   }

   provider "google-beta" {
    project = var.project_id
    region  = var.region
    zone    = var.zone
   }

   data "google_client_config" "current" {}

   provider "kubernetes" {
     host                   = "https://${module.wandb.cluster_endpoint}"
     cluster_ca_certificate = base64decode(module.wandb.cluster_ca_certificate)
     token                  = data.google_client_config.current.access_token
   }

   # 필요한 모든 서비스 생성
   module "wandb" {
     source  = "wandb/wandb/google"
     version = "~> 5.0"

     namespace   = var.namespace
     license     = var.license
     domain_name = var.domain_name
     subdomain   = var.subdomain
   }

   # 프로비저닝된 IP 어드레스로 DNS를 업데이트하세요
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

2. W&B 배포

   W&B를 배포하려면, 다음 명령어를 실행하세요:

   ```
   terraform init
   terraform apply -var-file=terraform.tfvars
   ```

## REDIS Cache를 활용한 배포

SQL 쿼리를 캐시하여 실험 메트릭 로딩 시 애플리케이션 응답 속도를 향상시키기 위한 추가 배포 옵션입니다.

`create_redis = true` 옵션을 [권장 배포 방식]({{< relref path="#deployment---recommended-20-mins" lang="ko" >}})에 설명된 `main.tf` 파일에 추가하면 캐시 기능을 활성화할 수 있습니다.

```
[...]

module "wandb" {
  source  = "wandb/wandb/google"
  version = "~> 1.0"

  namespace    = var.namespace
  license      = var.license
  domain_name  = var.domain_name
  subdomain    = var.subdomain
  allowed_inbound_cidrs = ["*"]
  # Redis 활성화
  create_redis = true

}
[...]
```

## 외부 큐를 사용한 배포

세 번째 배포 옵션은 외부 `message broker` 활성화입니다. 이는 선택 사항으로 W&B가 기본적으로 내장 브로커를 제공하기 때문에 성능 향상과는 무관합니다.

GCP에서 메시지 브로커 리소스는 `Pub/Sub`이며, 이를 사용하려면 권장 [배포 옵션]({{< relref path="#deployment---recommended-20-mins" lang="ko" >}})에서 설명한 동일한 `main.tf` 파일에 `use_internal_queue = false` 옵션을 추가하세요.

```
[...]

module "wandb" {
  source  = "wandb/wandb/google"
  version = "~> 1.0"

  namespace          = var.namespace
  license            = var.license
  domain_name        = var.domain_name
  subdomain          = var.subdomain
  allowed_inbound_cidrs = ["*"]
  # Pub/Sub 생성 및 사용
  use_internal_queue = false

}

[...]

```

## 추가 배포 옵션

세 가지 배포 옵션을 모두 결합하여 하나의 파일에 구성 추가가 가능합니다.
[Terraform Module](https://github.com/wandb/terraform-google-wandb)은 여러 다양한 옵션을 제공하며, 표준 및 최소 구성과 함께 조합할 수 있습니다.




## 수동 설정

GCP Storage bucket을 W&B의 파일 스토리지 백엔드로 사용하려면 다음을 생성해야 합니다:

* [PubSub Topic 및 Subscription 생성]({{< relref path="#create-pubsub-topic-and-subscription" lang="ko" >}})
* [Storage Bucket 생성]({{< relref path="#create-storage-bucket" lang="ko" >}})
* [PubSub Notification 생성]({{< relref path="#create-pubsub-notification" lang="ko" >}})


### PubSub Topic 및 Subscription 생성

아래 절차대로 PubSub 토픽 및 구독을 생성합니다.

1. GCP 콘솔에서 Pub/Sub 서비스로 이동합니다.
2. **Create Topic**을 선택하고 토픽 이름을 입력합니다.
3. 페이지 하단에서 **Create subscription**을 선택합니다. **Delivery Type**은 반드시 **Pull**로 설정해야 합니다.
4. **Create**를 클릭합니다.

인스턴스가 실행되는 서비스 계정 또는 계정에 이 subscription에 대한 `pubsub.admin` 역할이 부여되어야 합니다. 자세한 내용은 https://cloud.google.com/pubsub/docs/access-control#console 을 참고하세요.

### Storage Bucket 생성

1. **Cloud Storage Buckets** 페이지로 이동합니다.
2. **Create bucket**을 선택 후 이름을 입력합니다. [스토리지 클래스](https://cloud.google.com/storage/docs/storage-classes) 중 반드시 **Standard**를 선택하세요.

인스턴스가 실행되는 서비스 계정 혹은 계정에는 다음 권한이 필요합니다:
* 방금 만든 버킷에 대한 엑세스 권한
* 해당 버킷에 대해 `storage.objectAdmin` 역할 (https://cloud.google.com/storage/docs/access-control/using-iam-permissions#bucket-add)

{{% alert %}}
서명된 파일 URL 생성을 위해 인스턴스에 GCP의 `iam.serviceAccounts.signBlob` 권한이 필요합니다. 서비스 계정 또는 인스턴스가 실행 중인 IAM 멤버에 `Service Account Token Creator` 역할을 추가하세요. 
{{% /alert %}}

3. CORS 엑세스를 활성화합니다. 이는 커맨드라인에서만 가능합니다. 먼저 아래와 같이 CORS 설정이 지정된 JSON 파일을 생성하세요.

```
cors:
- maxAgeSeconds: 3600
  method:
   - GET
   - PUT
     origin:
   - '<YOUR_W&B_SERVER_HOST>'
     responseHeader:
   - Content-Type
```

origin에 대한 scheme, host, 포트는 실제 값과 정확히 일치해야 합니다.

4. `gcloud`가 설치되어 있고, 올바른 GCP 프로젝트에 로그인되어 있는지 확인하십시오.
5. 그리고 다음을 실행하세요:

```bash
gcloud storage buckets update gs://<BUCKET_NAME> --cors-file=<CORS_CONFIG_FILE>
```

### PubSub Notification 생성
아래 절차대로 Storage Bucket에서 Pub/Sub 토픽으로 알림 스트림을 커맨드라인에서 생성하세요.

{{% alert %}}
CLI를 사용해서만 알림 스트림을 만들 수 있습니다. `gcloud`가 반드시 설치되어 있어야 합니다.
{{% /alert %}}

1. GCP 프로젝트에 로그인합니다.
2. 터미널에서 다음을 실행하세요:

```bash
gcloud pubsub topics list  # 참고용 토픽 이름 목록 확인
gcloud storage ls          # 참고용 버킷 목록 확인

# 버킷의 알림 생성
gcloud storage buckets notifications create gs://<BUCKET_NAME> --topic=<TOPIC_NAME>
```

[더 자세한 내용은 Cloud Storage 공식 문서를 참고하세요.](https://cloud.google.com/storage/docs/reporting-changes)

### W&B 서버 설정

1. 마지막 단계로, W&B의 `System Connections` 페이지(`http(s)://YOUR-W&B-SERVER-HOST/console/settings/system`)로 이동합니다.
2. provider에서 `Google Cloud Storage (gcs)`를 선택합니다.
3. GCS bucket 이름을 입력하세요.

{{< img src="/images/hosting/configure_file_store_gcp.png" alt="GCP 파일 저장소 설정" >}}

4. **Update settings**를 눌러 변경 사항을 적용하세요.

## W&B Server 업그레이드

아래 절차에 따라 W&B를 업데이트할 수 있습니다.

1. `wandb_app` 모듈의 설정에 `wandb_version`을 추가하고, 업그레이드할 W&B 버전을 지정합니다. 예를 들어, 아래와 같이 W&B 버전 `0.58.1`을 명시할 수 있습니다.

  ```
  module "wandb_app" {
      source  = "wandb/wandb/kubernetes"
      version = "~>5.0"

      license       = var.license
      wandb_version = "0.58.1"
  ```

  {{% alert %}}
  또는, `wandb_version`을 `terraform.tfvars`에 추가하여 동일한 이름으로 변수를 생성한 다음, 리터럴 값 대신 `var.wandb_version`을 사용할 수도 있습니다.
  {{% /alert %}}

2. 구성 수정 후에는 [배포 옵션]({{< relref path="#deployment---recommended-20-mins" lang="ko" >}})에서 안내된 절차를 그대로 따라 실행하세요.