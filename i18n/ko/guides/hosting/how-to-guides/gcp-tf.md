---
description: Hosting W&B Server on GCP.
displayed_sidebar: default
---

# GCP

Weights & Biases에서 개발한 [Terraform Module](https://registry.terraform.io/modules/wandb/wandb/google/latest)을 사용하여 Google Cloud에 W&B 서버를 배포하는 것을 권장합니다.

모듈 문서에는 사용할 수 있는 모든 옵션이 포함되어 있으며, 이 문서에서는 일부 배포 옵션을 다룰 것입니다.

시작하기 전에, Terraform을 위한 [State File](https://developer.hashicorp.com/terraform/language/state)을 저장할 수 있는 [remote backends](https://developer.hashicorp.com/terraform/language/settings/backends/configuration) 중 하나를 선택하는 것이 좋습니다.

State File은 배포를 업그레이드하거나 변경 사항을 적용할 때 모든 구성 요소를 다시 생성하지 않고도 필요한 자원입니다.

Terraform 모듈은 다음 `필수` 구성 요소를 배포합니다:

- VPC
- Cloud SQL for MySQL
- Cloud Storage Bucket
- Google Kubernetes Engine
- KMS Crypto Key
- Load Balancer

다른 배포 옵션에는 다음과 같은 선택적 구성 요소도 포함될 수 있습니다:

- Redis용 Memory store
- Pub/Sub 메시지 시스템

## **사전 요구 사항 권한**

terraform을 실행할 계정은 사용할 GCP 프로젝트에서 `roles/owner` 역할을 가져야 합니다.

## 일반 단계

이 문서에서 다루는 모든 배포 옵션에 공통적인 단계입니다.

1. 개발 환경을 준비하십시오.
   - [Terraform](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli) 설치
   - 사용할 코드를 Git 저장소에 생성하는 것이 좋지만 파일을 로컬에 유지할 수도 있습니다.
   - [Google Cloud Console](https://console.cloud.google.com/)에서 프로젝트 생성
   - GCP 인증 (`gcloud` [설치](https://cloud.google.com/sdk/docs/install) 후)
     `gcloud auth application-default login`
2. `terraform.tfvars` 파일 생성.

   `tvfars` 파일 내용은 설치 유형에 따라 사용자 정의할 수 있지만, 최소 권장 사항은 아래 예와 같습니다.

   ```bash
   project_id  = "wandb-project"
   region      = "europe-west2"
   zone        = "europe-west2-a"
   namespace   = "wandb"
   license     = "xxxxxxxxxxyyyyyyyyyyyzzzzzzz"
   subdomain   = "wandb-gcp"
   domain_name = "wandb.ml"
   ```

   여기에 정의된 변수들은 배포 전에 결정해야 합니다. `namespace` 변수는 Terraform이 생성하는 모든 리소스에 접두사로 사용될 문자열입니다.

   `subdomain`과 `domain`의 조합은 W&B가 구성될 FQDN을 형성합니다. 예를 들어, 위의 예에서 W&B FQDN은 `wandb-gcp.wandb.ml`이 됩니다.


3. `variables.tf` 파일 생성

   `terraform.tfvars`에 구성된 모든 옵션에 대해 Terraform은 해당 변수 선언이 필요합니다.

   ```
   variable "project_id" {
     type        = string
     description = "프로젝트 ID"
   }

   variable "region" {
     type        = string
     description = "Google 지역"
   }

   variable "zone" {
     type        = string
     description = "Google 존"
   }

   variable "namespace" {
     type        = string
     description = "리소스에 사용되는 네임스페이스 접두사"
   }

   variable "domain_name" {
     type        = string
     description = "Weights & Biases UI에 접근하기 위한 도메인 이름."
   }

   variable "subdomain" {
     type        = string
     description = "Weights & Biases UI에 접근하기 위한 서브도메인."
   }

   variable "license" {
     type        = string
     description = "W&B 라이선스"
   }
   ```

## 배포 - 권장 (~20 분)

이것은 모든 `필수` 구성 요소를 생성하고 `Kubernetes Cluster`에 `W&B`의 최신 버전을 설치하는 가장 간단한 배포 옵션 구성입니다.

1. `main.tf` 생성

   `일반 단계`에서 생성한 파일과 동일한 디렉토리에 다음 내용을 가진 `main.tf` 파일을 생성하십시오.

   ```
   provider "google" {
    project = var.project_id
    region  = var.region
    zone    = var.zone
   }

   provider "google-beta" {
    project = var.project_id
    region  = var.reguion
    zone    = var.zone
   }

   data "google_client_config" "current" {}

   provider "kubernetes" {
     host                   = "https://${module.wandb.cluster_endpoint}"
     cluster_ca_certificate = base64decode(module.wandb.cluster_ca_certificate)
     token                  = data.google_client_config.current.access_token
   }

   # 필요한 모든 서비스 시작
   module "wandb" {
     source  = "wandb/wandb/google"
     version = "~> 1.0"

     namespace   = var.namespace
     license     = var.license
     domain_name = var.domain_name
     subdomain   = var.subdomain
     allowed_inbound_cidrs = ["*"]
   }

   # 할당된 IP 주소로 DNS를 업데이트하고 싶을 것입니다
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

   W&B를 배포하려면 다음 명령을 실행하십시오:

   ```
   terraform init
   terraform apply -var-file=terraform.tfvars
   ```

## REDIS 캐시를 사용한 배포

다른 배포 옵션은 `Redis`를 사용하여 SQL 쿼리를 캐시하고 실험에 대한 메트릭을 로딩할 때 애플리케이션 응답을 가속화합니다.

캐시를 활성화하려면 `Deployment option 1`에서 작업한 동일한 `main.tf` 파일에 `create_redis = true` 옵션을 추가하십시오.

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
  #Redis 활성화
  create_redis = true

}
[...]
```

## 외부 큐 사용 배포

배포 옵션 3은 외부 `메시지 브로커`를 활성화하는 것으로 구성됩니다. W&B는 내장된 브로커를 제공하기 때문에 이 옵션은 선택 사항입니다. 이 옵션은 성능 향상을 제공하지 않습니다.

메시지 브로커를 제공하는 GCP 리소스는 `Pub/Sub`이며, 이를 활성화하려면 `Deployment option 1`에서 작업한 동일한 `main.tf`에 `use_internal_queue = false` 옵션을 추가해야 합니다.

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
  #Pub/Sub 생성 및 사용
  use_internal_queue = false

}

[...]

```

## 기타 배포 옵션

모든 세 가지 배포 옵션을 동일한 파일에 모든 구성을 추가하여 결합할 수 있습니다.
[Terraform Module](https://github.com/wandb/terraform-google-wandb)은 표준 옵션과 `Deployment - Recommended`에서 찾을 수 있는 최소 구성과 함께 결합될 수 있는 여러 옵션을 제공합니다.

## 수동 구성

W&B를 파일 스토리지 백엔드로 GCP Storage 버킷을 사용하려면 다음을 생성해야 합니다:

* [PubSub 주제 및 구독 생성](#create-pubsub-topic-and-subscription)
* [Storage Bucket 생성](#create-storage-bucket)
* [PubSub 알림 생성](#create-pubsub-notification)

### PubSub 주제 및 구독 생성

아래 절차를 따라 PubSub 주제 및 구독을 생성하십시오:

1. GCP Console 내의 Pub/Sub 서비스로 이동합니다.
2. **주제 생성**을 선택하고 주제에 대한 이름을 제공합니다.
3. 페이지 하단에서 **구독 생성**을 선택합니다. **전송 유형**이 **Pull**로 설정되어 있는지 확인하십시오.
4. **생성**을 클릭하십시오.

인스턴스가 실행되는 서비스 계정 또는 계정이 이 구독에 대해 `pubsub.admin` 역할을 가져야 합니다. 자세한 내용은 https://cloud.google.com/pubsub/docs/access-control#console을 참조하십시오.

### Storage Bucket 생성

1. **Cloud Storage Buckets** 페이지로 이동합니다.
2. **버킷 생성**을 선택하고 버킷에 대한 이름을 제공합니다. **Standard** [스토리지 클래스](https://cloud.google.com/storage/docs/storage-classes)를 선택하십시오.

인스턴스가 실행되는 서비스 계정 또는 계정이 다음을 모두 가지고 있는지 확인하십시오:
* 이전 단계에서 생성한 버킷에 대한 엑세스
* 이 버킷에 대한 `storage.objectAdmin` 역할. 자세한 내용은 https://cloud.google.com/storage/docs/access-control/using-iam-permissions#bucket-add를 참조하십시오.

:::info
인스턴스는 또한 GCP에서 서명된 파일 URL을 생성하기 위해 `iam.serviceAccounts.signBlob` 권한이 필요합니다. 인스턴스가 실행 중인 서비스 계정 또는 IAM 멤버에 `Service Account Token Creator` 역할을 추가하여 권한을 활성화하십시오.
:::

3. CORS 엑세스를 활성화하십시오. 이 작업은 커맨드라인을 사용해서만 수행할 수 있습니다. 먼저 다음 CORS 구성이 포함된 JSON 파일을 생성하십시오.

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

origin의 값에 대한 스키마, 호스트 및 포트가 정확히 일치해야 합니다.

4. `gcloud`가 설치되어 있고 올바른 GCP 프로젝트에 로그인되어 있는지 확인하십시오.
5. 다음을 실행하십시오:

```bash
gcloud storage buckets update gs://<BUCKET_NAME> --cors-file=<CORS_CONFIG_FILE>
```

### PubSub 알림 생성
커맨드라인에서 Storage Bucket에서 Pub/Sub 주제로의 알림 스트림을 생성하려면 아래 절차를 따르십시오.

:::info
알림 스트림을 생성하려면 CLI를 사용해야 합니다. `gcloud`가 설치되어 있는지 확인하십시오.
:::

1. GCP 프로젝트에 로그인하십시오.
2. 터미널에서 다음을 실행하십시오:

```bash
gcloud pubsub topics list  # 참조용 주제 이름 나열
gcloud storage ls          # 참조용 버킷 이름 나열

# 버킷 알림 생성
gcloud storage buckets notifications create gs://<BUCKET_NAME> --topic=<TOPIC_NAME>
```

[Cloud Storage 웹사이트에서 추가 참조를 확인할 수 있습니다.](https://cloud.google.com/storage/docs/reporting-changes)

### W&B 서버 구성

1. 마지막으로, `http(s)://YOUR-W&B-SERVER-HOST/system-admin`에서 W&B 설정 페이지로 이동하십시오.
2. "외부 파일 스토리지 백엔드 사용" 옵션을 활성화하고, 버킷 이름, 버킷이 저장된 지역, SQS 큐를 다음 형식으로 제공하십시오:
* **파일 스토리지 버킷**: `gs://<bucket-name>`
* **파일 스토리지 지역**: 비워두기
* **알림 구독**: `pubsub:/<project-name>/<topic-name>/<subscription-name>`

![](/images/hosting/configure_file_store.png)

4. **설정 업데이트**를 눌러 새 설정을 적용하십시오.

## W&B 서버 업그레이드

W&B를 업데이트하려면 여기에 설명된 단계를 따르십시오:

1. `wandb_app` 모듈의 구성에 `wandb_version`을 추가하십시오. 업그레이드하려는 W&B 버전을 제공하십시오. 예를 들어, 다음 줄은 W&B 버전 `0.48.1`을 지정합니다:

  ```
  module "wandb_app" {
      source  = "wandb/wandb/kubernetes"
      version = "~>1.0"

      license       = var.license
      wandb_version = "0.48.1"
  ```

  :::info
  또는 `wandb_version`을 `terraform.tfvars`에 추가하고 동일한 이름의 변수를 생성한 다음 리터럴 값 대신 `var.wandb_version`을 사용할 수 있습니다.
  :::

2. 구성을 업데이트한 후 [배포 섹션](#deployment---recommended-20-mins)에 설명된 단계를 완료하십시오.