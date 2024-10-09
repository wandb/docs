---
title: Deploy W&B Platform on GCP
description: GCP에서 W&B 서버 호스팅.
displayed_sidebar: default
---

:::info
W&B는 [W&B Multi-tenant Cloud](../hosting-options/saas_cloud.md) 또는 [W&B Dedicated Cloud](../hosting-options//dedicated_cloud.md) 배포 타입과 같은 완전히 관리되는 배포 옵션을 추천합니다. W&B 완전 관리 서비스는 최소한의 설정(또는 설정 없이) 간단하고 안전하게 사용할 수 있습니다.
:::

W&B Server를 자가 관리하기로 결정했다면 GCP에 플랫폼을 배포하기 위해 [W&B Server GCP Terraform Module](https://registry.terraform.io/modules/wandb/wandb/google/latest)을 사용하는 것을 추천합니다.

모듈 문서는 광범위하며 사용할 수 있는 모든 옵션이 포함되어 있습니다.

시작하기 전에, W&B는 Terraform의 [State File](https://developer.hashicorp.com/terraform/language/state)을 저장할 수 있는 [원격 백엔드](https://developer.hashicorp.com/terraform/language/settings/backends/configuration) 중 하나를 선택할 것을 권장합니다.

State File은 배포의 모든 컴포넌트를 재생성하지 않고 업그레이드를 진행하거나 변경할 수 있도록 하는 필수 리소스입니다.

Terraform Module은 다음의 '필수' 컴포넌트를 배포합니다:

- VPC
- Cloud SQL for MySQL
- Cloud Storage Bucket
- Google Kubernetes Engine
- KMS Crypto Key
- Load Balancer

다른 배포 옵션은 다음의 선택적 컴포넌트를 포함할 수 있습니다:

- Redis 용 메모리 저장소
- Pub/Sub 메시지 시스템

## 선행 조건 권한

Terraform을 실행할 계정은 사용되는 GCP 프로젝트에서 `roles/owner` 역할을 가지고 있어야 합니다.

## 일반적인 단계

이 주제의 단계는 이 문서에서 다루는 모든 배포 옵션에 공통적입니다.

1. 개발 환경을 준비합니다.
   - [Terraform](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli) 설치
   - 사용할 코드와 함께 Git 레포지토리를 만드는 것을 권장하지만, 파일을 로컬에 보관할 수 있습니다.
   - [Google Cloud Console](https://console.cloud.google.com/)에서 프로젝트를 생성합니다.
   - GCP에 인증합니다 (완료 전에 [gcloud 설치](https://cloud.google.com/sdk/docs/install)를 확인하세요).
     `gcloud auth application-default login`
2. `terraform.tfvars` 파일을 생성합니다.

   `tfvars` 파일의 내용은 설치 타입에 따라 맞춤화할 수 있지만, 최소 권장 사항은 아래 예시와 같습니다.

   ```bash
   project_id  = "wandb-project"
   region      = "europe-west2"
   zone        = "europe-west2-a"
   namespace   = "wandb"
   license     = "xxxxxxxxxxyyyyyyyyyyyzzzzzzz"
   subdomain   = "wandb-gcp"
   domain_name = "wandb.ml"
   ```

   여기 정의된 변수는 배포 전에 결정되어야 합니다. `namespace` 변수는 Terraform이 생성한 모든 리소스를 접두어로 갖는 문자열입니다.

   `subdomain`과 `domain`의 조합은 W&B가 설정될 FQDN을 형성합니다. 위 예시에서는 W&B FQDN이 `wandb-gcp.wandb.ml`이 됩니다.

3. `variables.tf` 파일을 생성합니다.

   `terraform.tfvars`에 설정된 모든 옵션에 대해 Terraform은 해당 변수 선언이 필요합니다.

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
     description = "Weights & Biases UI에 엑세스하기 위한 도메인 이름."
   }

   variable "subdomain" {
     type        = string
     description = "Weights & Biases UI에 엑세스하기 위한 서브도메인."
   }

   variable "license" {
     type        = string
     description = "W&B 라이센스"
   }
   ```

## 배포 - 추천 (~20분)

이것은 가장 간단한 배포 옵션 설정으로 모든 'Mandatory' 컴포넌트를 생성하고 `Kubernetes Cluster`에 최신 `W&B` 버전을 설치합니다.

1. `main.tf`를 생성합니다.

   [일반적인 단계](#general-steps)에서 파일을 생성한 동일한 디렉토리에 `main.tf` 파일을 아래의 내용으로 생성합니다:

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

   # 필수 서비스를 모두 실행합니다
   module "wandb" {
     source  = "wandb/wandb/google"
     version = "~> 5.0"

     namespace   = var.namespace
     license     = var.license
     domain_name = var.domain_name
     subdomain   = var.subdomain
   }

   # 프로비저닝된 IP 주소로 DNS를 업데이트 해야 합니다
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

   W&B를 배포하려면 다음 코맨드를 실행하십시오:

   ```
   terraform init
   terraform apply -var-file=terraform.tfvars
   ```

## REDIS 캐시를 이용한 배포

다른 배포 옵션은 SQL 쿼리를 캐시하고 실험을 로드할 때 애플리케이션 응답을 가속화하기 위해 `Redis`를 사용합니다.

캐시를 활성화하려면, 추천 [배포 옵션 섹션](#deployment---recommended-20-mins)에 명시된 동일한 `main.tf` 파일에 옵션 `create_redis = true`를 추가해야 합니다.

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

## 외부 큐를 이용한 배포

배포 옵션 3은 외부 `message broker`를 활성화하는 것입니다. W&B는 기본적으로 브로커를 포함하고 있으므로 이는 선택사항입니다. 이 옵션은 성능 개선을 제공하지 않습니다.

GCP 리소스가 제공하는 메시지 브로커는 `Pub/Sub`이며 이를 사용하려면 추천 [배포 옵션 섹션](#deployment---recommended-20-mins)에 명시된 동일한 `main.tf`에 옵션 `use_internal_queue = false`를 추가해야 합니다.

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

## 다른 배포 옵션

모든 배포 옵션을 동일한 파일에 추가하여 결합할 수 있습니다.
[Terraform Module](https://github.com/wandb/terraform-google-wandb)은 여러 옵션을 표준 옵션 및 '배포 - 추천'에서 찾은 최소 설정과 결합할 수 있는 여러 옵션을 제공합니다.

## 수동 구성

W&B의 파일 저장소 백엔드로 GCP Storage bucket을 사용하려면 다음을 생성해야 합니다:

* [PubSub Topic 및 Subscription](#create-pubsub-topic-and-subscription)
* [Storage Bucket](#create-storage-bucket)
* [PubSub 알림](#create-pubsub-notification)

### PubSub Topic 및 Subscription 생성

아래 절차를 따라 PubSub 토픽 및 구독을 생성하십시오:

1. GCP Console 내에서 Pub/Sub 서비스로 이동하십시오.
2. **토픽 생성**을 선택하고 토픽에 이름을 제공하십시오.
3. 페이지 하단에서 **구독 생성**을 선택합니다. **배달 유형**이 **Pull**로 설정되었는지 확인하십시오.
4. **생성**을 클릭합니다.

서비스 계정 또는 인스턴스를 실행하는 계정이 이 구독에서 `pubsub.admin` 역할을 가지고 있는지 확인하십시오. 자세한 내용은 https://cloud.google.com/pubsub/docs/access-control#console를 참조하십시오.

### 스토리지 버킷 생성

1. **Cloud Storage Buckets** 페이지로 이동하십시오.
2. **버킷 생성**을 선택하고 버킷에 이름을 지정하십시오. **표준** [저장소 클래스](https://cloud.google.com/storage/docs/storage-classes)를 선택하십시오.

인스턴스를 실행하는 서비스 계정 또는 계정이 다음을 모두 가지고 있는지 확인하십시오:
* 이전 단계에서 생성한 버킷에 대한 엑세스
* 이 버킷에 대한 `storage.objectAdmin` 역할. 자세한 내용은 https://cloud.google.com/storage/docs/access-control/using-iam-permissions#bucket-add를 참조하십시오.

:::info
인스턴스는 GCP에서 서명된 파일 URL을 생성하기 위해 `iam.serviceAccounts.signBlob` 권한이 필요합니다. 인스턴스를 실행하는 서비스 계정 또는 IAM 멤버에게 `Service Account Token Creator` 역할을 추가하여 권한을 활성화하십시오.
:::

3. CORS 엑세스를 활성화합니다. 이는 커맨드라인을 사용하여서만 가능합니다. 먼저 다음 CORS 설정으로 JSON 파일을 만드십시오.

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

원본에 대한 값의 스킴(host) 및 포트가 정확히 일치해야 합니다.

4. `gcloud`가 설치되어 있으며, 올바른 GCP 프로젝트에 로그인되어 있는지 확인하세요.
5. 다음을 실행합니다:

```bash
gcloud storage buckets update gs://<BUCKET_NAME> --cors-file=<CORS_CONFIG_FILE>
```

### PubSub 알림 생성
스토리지 버킷에서 Pub/Sub 토픽으로의 알림 스트림을 생성하기 위해 커맨드라인으로 아래 절차를 따르세요.

:::info
CLI를 사용하여 알림 스트림을 생성해야 합니다. `gcloud`가 설치되어 있는지 확인하세요.
:::

1. GCP 프로젝트에 로그인합니다.
2. 터미널에서 다음을 실행합니다:

```bash
gcloud pubsub topics list  # 참조용 토픽 이름 목록
gcloud storage ls          # 참조용 버킷 이름 목록

# 버킷 알림 생성
gcloud storage buckets notifications create gs://<BUCKET_NAME> --topic=<TOPIC_NAME>
```

[추가 참조는 Cloud Storage 웹사이트에 제공됩니다.](https://cloud.google.com/storage/docs/reporting-changes)

### W&B 서버 구성

1. 마지막으로, W&B `System Connections` 페이지로 이동하여 `http(s)://YOUR-W&B-SERVER-HOST/console/settings/system`에 접속합니다.
2. 제공 업체로 `Google Cloud Storage (gcs)`를 선택합니다,
3. GCS 버킷의 이름을 제공하세요.

![](/images/hosting/configure_file_store_gcp.png)

4. **설정 업데이트**를 눌러 새로운 설정을 적용합니다.

## W&B Server 업그레이드

W&B를 업데이트하려면 여기 나열된 단계를 따르십시오:

1. 구성에 `wandb_version`을 추가하여 `wandb_app` 모듈에 적용합니다. 업데이트할 W&B의 버전을 제공하십시오. 예를 들어, 다음 라인은 W&B 버전 `0.48.1`을 지정합니다:

  ```
  module "wandb_app" {
      source  = "wandb/wandb/kubernetes"
      version = "~>5.0"

      license       = var.license
      wandb_version = "0.58.1"
  ```

  :::info
  대안적으로, `wandb_version`을 `terraform.tfvars`에 추가하고 동일한 이름으로 변수를 생성할 수 있으며 문자 값을 사용하기 보다는 `var.wandb_version`를 사용하십시오.
  :::

2. 구성을 업데이트한 후, [배포 옵션 섹션](#deployment---recommended-20-mins)에 설명된 단계를 완료합니다.