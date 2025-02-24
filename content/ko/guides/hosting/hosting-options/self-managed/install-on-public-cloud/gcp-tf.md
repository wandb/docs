---
title: Deploy W&B Platform on GCP
description: GCP에서 W&B 서버 호스팅하기.
menu:
  default:
    identifier: ko-guides-hosting-hosting-options-self-managed-install-on-public-cloud-gcp-tf
    parent: install-on-public-cloud
weight: 20
---

{{% alert %}}
Weights & Biases에서는 [W&B Multi-tenant Cloud]({{< relref path="/guides/hosting/hosting-options/saas_cloud.md" lang="ko" >}}) 또는 [W&B Dedicated Cloud]({{< relref path="/guides/hosting/hosting-options/dedicated_cloud/" lang="ko" >}}) 배포 유형과 같은 완전 관리형 배포 옵션을 권장합니다. Weights & Biases 완전 관리형 서비스는 사용하기 간단하고 안전하며, 필요한 설정이 최소화되어 있습니다.
{{% /alert %}}

W&B Server를 자체 관리하기로 결정한 경우, GCP에 플랫폼을 배포하기 위해 [W&B Server GCP Terraform Module](https://registry.terraform.io/modules/wandb/wandb/google/latest)을 사용하는 것이 좋습니다.

모듈 문서는 광범위하며 사용 가능한 모든 옵션이 포함되어 있습니다.

시작하기 전에 Terraform에서 [State File](https://developer.hashicorp.com/terraform/language/state)을 저장하는 데 사용할 수 있는 [remote backends](https://developer.hashicorp.com/terraform/language/backend/remote) 중 하나를 선택하는 것이 좋습니다.

State File은 모든 구성 요소를 다시 생성하지 않고도 배포에서 업그레이드를 롤아웃하거나 변경하는 데 필요한 리소스입니다.

Terraform Module은 다음과 같은 `필수` 구성 요소를 배포합니다.

- VPC
- MySQL용 Cloud SQL
- Cloud Storage Bucket
- Google Kubernetes Engine
- KMS Crypto Key
- Load Balancer

다른 배포 옵션에는 다음과 같은 선택적 구성 요소가 포함될 수도 있습니다.

- Redis용 메모리 저장소
- Pub/Sub 메시지 시스템

## 사전 필수 권한

terraform을 실행할 계정은 사용된 GCP 프로젝트에서 `roles/owner` 역할을 가지고 있어야 합니다.

## 일반적인 단계

이 주제의 단계는 이 문서에서 다루는 모든 배포 옵션에 공통적입니다.

1. 개발 환경을 준비합니다.
   - [Terraform](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli)을 설치합니다.
   - 사용할 코드로 Git 저장소를 만드는 것이 좋지만, 파일을 로컬에 보관할 수도 있습니다.
   - [Google Cloud Console](https://console.cloud.google.com/)에서 프로젝트를 만듭니다.
   - GCP로 인증합니다 ([gcloud](https://cloud.google.com/sdk/docs/install)를 먼저 설치해야 함).
     `gcloud auth application-default login`
2. `terraform.tfvars` 파일을 만듭니다.

   `tvfars` 파일 내용은 설치 유형에 따라 사용자 정의할 수 있지만, 최소 권장 사항은 아래 예제와 같습니다.

   ```bash
   project_id  = "wandb-project"
   region      = "europe-west2"
   zone        = "europe-west2-a"
   namespace   = "wandb"
   license     = "xxxxxxxxxxyyyyyyyyyyyzzzzzzz"
   subdomain   = "wandb-gcp"
   domain_name = "wandb.ml"
   ```

   여기에 정의된 변수는 배포 전에 결정해야 합니다. `namespace` 변수는 Terraform에서 생성된 모든 리소스의 접두사가 되는 문자열입니다.

   `subdomain`과 `domain`의 조합은 Weights & Biases가 구성될 FQDN을 형성합니다. 위의 예제에서 Weights & Biases FQDN은 `wandb-gcp.wandb.ml`입니다.

3. `variables.tf` 파일을 만듭니다.

   `terraform.tfvars`에서 구성된 모든 옵션에 대해 Terraform은 해당 변수 선언이 필요합니다.

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
     description = "리소스에 사용되는 Namespace 접두사"
   }

   variable "domain_name" {
     type        = string
     description = "Weights & Biases UI에 엑세스하기 위한 도메인 이름입니다."
   }

   variable "subdomain" {
     type        = string
     description = "Weights & Biases UI에 엑세스하기 위한 서브 도메인입니다."
   }

   variable "license" {
     type        = string
     description = "W&B License"
   }
   ```

## 배포 - 권장 (~20분)

이것은 모든 `필수` 구성 요소를 만들고 `Kubernetes Cluster`에 최신 버전의 `W&B`를 설치하는 가장 간단한 배포 옵션 구성입니다.

1. `main.tf`를 만듭니다.

   [일반적인 단계]({{< relref path="#general-steps" lang="ko" >}})에서 파일을 만든 것과 동일한 디렉토리에 다음 내용으로 `main.tf` 파일을 만듭니다.

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

   # Spin up all required services
   module "wandb" {
     source  = "wandb/wandb/google"
     version = "~> 5.0"

     namespace   = var.namespace
     license     = var.license
     domain_name = var.domain_name
     subdomain   = var.subdomain
   }

   # You'll want to update your DNS with the provisioned IP address
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

2. W&B를 배포합니다.

   W&B를 배포하려면 다음 코맨드를 실행합니다.

   ```
   terraform init
   terraform apply -var-file=terraform.tfvars
   ```

## REDIS 캐시를 사용한 배포

또 다른 배포 옵션은 `Redis`를 사용하여 SQL 쿼리를 캐시하고 Experiments에 대한 메트릭을 로드할 때 애플리케이션 응답 속도를 높입니다.

캐시를 활성화하려면 권장 [배포 옵션 섹션]({{< relref path="#deployment---recommended-20-mins" lang="ko" >}})에 지정된 동일한 `main.tf` 파일에 옵션 `create_redis = true`를 추가해야 합니다.

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
  #Enable Redis
  create_redis = true

}
[...]
```

## 외부 큐를 사용한 배포

배포 옵션 3은 외부 `message broker`를 활성화하는 것으로 구성됩니다. W&B에 broker가 내장되어 있기 때문에 이는 선택 사항입니다. 이 옵션은 성능 향상을 제공하지 않습니다.

메시지 broker를 제공하는 GCP 리소스는 `Pub/Sub`이며, 이를 활성화하려면 권장 [배포 옵션 섹션]({{< relref path="#deployment---recommended-20-mins" lang="ko" >}})에 지정된 동일한 `main.tf`에 옵션 `use_internal_queue = false`를 추가해야 합니다.

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
  #Create and use Pub/Sub
  use_internal_queue = false

}

[...]

```

## 기타 배포 옵션

동일한 파일에 모든 구성을 추가하여 세 가지 배포 옵션을 모두 결합할 수 있습니다.
[Terraform Module](https://github.com/wandb/terraform-google-wandb)은 표준 옵션 및 `배포 - 권장`에서 찾을 수 있는 최소 구성과 함께 결합할 수 있는 여러 옵션을 제공합니다.

## 수동 구성

W&B의 파일 스토리지 백엔드로 GCP Storage bucket을 사용하려면 다음을 만들어야 합니다.

* [PubSub Topic and Subscription]({{< relref path="#create-pubsub-topic-and-subscription" lang="ko" >}})
* [Storage Bucket]({{< relref path="#create-storage-bucket" lang="ko" >}})
* [PubSub Notification]({{< relref path="#create-pubsub-notification" lang="ko" >}})

### PubSub Topic 및 Subscription 생성

PubSub Topic 및 Subscription을 생성하려면 아래 절차를 따르십시오.

1. GCP Console 내에서 Pub/Sub 서비스로 이동합니다.
2. **Create Topic**을 선택하고 Topic 이름을 입력합니다.
3. 페이지 하단에서 **Create subscription**을 선택합니다. **Delivery Type**이 **Pull**로 설정되었는지 확인합니다.
4. **Create**를 클릭합니다.

인스턴스가 실행 중인 서비스 계정 또는 계정에 이 subscription에 대한 `pubsub.admin` 역할이 있는지 확인합니다. 자세한 내용은 https://cloud.google.com/pubsub/docs/access-control#console을 참조하십시오.

### Storage Bucket 생성

1. **Cloud Storage Buckets** 페이지로 이동합니다.
2. **Create bucket**을 선택하고 bucket 이름을 입력합니다. **Standard** [storage class](https://cloud.google.com/storage/docs/storage-classes)를 선택했는지 확인합니다.

인스턴스가 실행 중인 서비스 계정 또는 계정에 다음이 모두 있는지 확인합니다.
* 이전 단계에서 생성한 bucket에 대한 엑세스 권한
* 이 버킷에 대한 `storage.objectAdmin` 역할. 자세한 내용은 https://cloud.google.com/storage/docs/access-control/using-iam-permissions#bucket-add를 참조하십시오.

{{% alert %}}
인스턴스는 서명된 파일 URL을 생성하기 위해 GCP에서 `iam.serviceAccounts.signBlob` 권한도 필요합니다. 인스턴스가 실행 중인 서비스 계정 또는 IAM 멤버에 `Service Account Token Creator` 역할을 추가하여 권한을 활성화합니다.
{{% /alert %}}

3. CORS 엑세스를 활성화합니다. 이는 커맨드 라인에서만 수행할 수 있습니다. 먼저 다음 CORS 구성으로 JSON 파일을 만듭니다.

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

origin 값의 스키마, 호스트 및 포트가 정확히 일치해야 합니다.

4. `gcloud`가 설치되어 있고 올바른 GCP Project에 로그인했는지 확인합니다.
5. 다음을 실행합니다.

```bash
gcloud storage buckets update gs://<BUCKET_NAME> --cors-file=<CORS_CONFIG_FILE>
```

### PubSub Notification 생성
Storage Bucket에서 Pub/Sub Topic으로 알림 스트림을 생성하려면 커맨드 라인에서 다음 절차를 따르십시오.

{{% alert %}}
알림 스트림을 생성하려면 CLI를 사용해야 합니다. `gcloud`가 설치되어 있는지 확인하십시오.
{{% /alert %}}

1. GCP Project에 로그인합니다.
2. 터미널에서 다음을 실행합니다.

```bash
gcloud pubsub topics list  # 참조용으로 Topic 이름 나열
gcloud storage ls          # 참조용으로 버킷 이름 나열

# bucket 알림 생성
gcloud storage buckets notifications create gs://<BUCKET_NAME> --topic=<TOPIC_NAME>
```

[자세한 참조는 Cloud Storage 웹사이트에서 확인할 수 있습니다.](https://cloud.google.com/storage/docs/reporting-changes)

### W&B server 구성

1. 마지막으로 `http(s)://YOUR-W&B-SERVER-HOST/console/settings/system`에서 W&B `System Connections` 페이지로 이동합니다.
2. 제공업체 `Google Cloud Storage (gcs)`를 선택합니다.
3. GCS 버킷 이름을 입력합니다.

{{< img src="/images/hosting/configure_file_store_gcp.png" alt="" >}}

4. **Update settings**를 눌러 새 설정을 적용합니다.

## W&B Server 업그레이드

W&B를 업데이트하려면 여기에 설명된 단계를 따르십시오.

1. `wandb_app` 모듈의 구성에 `wandb_version`을 추가합니다. 업그레이드할 W&B 버전을 제공합니다. 예를 들어 다음 라인은 W&B 버전 `0.48.1`을 지정합니다.

  ```
  module "wandb_app" {
      source  = "wandb/wandb/kubernetes"
      version = "~>5.0"

      license       = var.license
      wandb_version = "0.58.1"
  ```

  {{% alert %}}
  또는 `wandb_version`을 `terraform.tfvars`에 추가하고 동일한 이름으로 변수를 만들고 리터럴 값을 사용하는 대신 `var.wandb_version`을 사용할 수 있습니다.
  {{% /alert %}}

2. 구성을 업데이트한 후 [배포 옵션 섹션]({{< relref path="#deployment---recommended-20-mins" lang="ko" >}})에 설명된 단계를 완료합니다.
