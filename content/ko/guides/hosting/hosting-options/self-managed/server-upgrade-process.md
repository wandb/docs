---
title: W&B 라이선스 및 버전 업데이트
description: 다양한 설치 method에서 W&B 버전 및 라이선스를 업데이트하는 가이드.
menu:
  default:
    identifier: ko-guides-hosting-hosting-options-self-managed-server-upgrade-process
    parent: self-managed
url: guides/hosting/server-upgrade-process
weight: 6
---

W&B Server 버전과 라이선스를 업데이트할 때는 W&B Server를 설치한 동일한 방법을 이용하세요. 아래 표는 다양한 배포 방법에 따라 라이선스와 버전을 업데이트하는 방법을 안내합니다:

| 릴리즈 타입    | 설명         |
| ---------------- | ------------------ |
| [Terraform]({{< relref path="#update-with-terraform" lang="ko" >}}) | W&B는 클라우드 배포를 위한 세 가지 퍼블릭 Terraform 모듈을 지원합니다: [AWS](https://registry.terraform.io/modules/wandb/wandb/aws/latest), [GCP](https://registry.terraform.io/modules/wandb/wandb/google/latest), 그리고 [Azure](https://registry.terraform.io/modules/wandb/wandb/azurerm/latest)입니다. |
| [Helm]({{< relref path="#update-with-helm" lang="ko" >}})              | 이미 구축된 Kubernetes 클러스터에 W&B를 설치하기 위해 [Helm Chart](https://github.com/wandb/helm-charts)를 사용할 수 있습니다.  |

## Terraform을 이용한 업데이트

Terraform을 이용해 라이선스와 버전을 업데이트하세요. 아래 표에서 W&B가 관리하는 Terraform 모듈을 클라우드 플랫폼별로 확인할 수 있습니다.

|클라우드 제공자| Terraform 모듈|
|-----|-----|
|AWS|[AWS Terraform 모듈](https://registry.terraform.io/modules/wandb/wandb/aws/latest)|
|GCP|[GCP Terraform 모듈](https://registry.terraform.io/modules/wandb/wandb/google/latest)|
|Azure|[Azure Terraform 모듈](https://registry.terraform.io/modules/wandb/wandb/azurerm/latest)|

1. 먼저, 사용 중인 클라우드 제공자에 적합한 W&B 유지 Terraform 모듈로 이동합니다. 위 표에서 클라우드 제공자에 맞는 Terraform 모듈을 찾으세요.
2. Terraform 설정 파일 내에서 `wandb_app` 모듈의 `wandb_version`과 `license` 항목을 다음과 같이 수정하세요:

   ```hcl
   module "wandb_app" {
       source  = "wandb/wandb/<cloud-specific-module>"
       version = "new_version"
       license       = "new_license_key" # 새 라이선스 키
       wandb_version = "new_wandb_version" # 원하는 W&B 버전
       ...
   }
   ```
3. `terraform plan`과 `terraform apply`를 사용해 Terraform 설정을 적용하세요.
   ```bash
   terraform init
   terraform apply
   ```

4. (선택 사항) 만약 `terraform.tfvars` 또는 다른 `.tfvars` 파일을 사용하는 경우,

   새 W&B 버전 및 라이선스 키로 `terraform.tfvars` 파일을 업데이트하거나 새로 만드세요.
   ```bash
   terraform plan -var-file="terraform.tfvars"
   ```
   설정을 적용하려면 Terraform 워크스페이스 디렉토리에서 다음을 실행하세요:  
   ```bash
   terraform apply -var-file="terraform.tfvars"
   ```

## Helm을 이용한 업데이트

### spec 파일로 W&B 업데이트

1. Helm chart의 `*.yaml` 설정 파일에서 `image.tag`와/또는 `license` 값을 새 버전으로 변경하세요:

   ```yaml
   license: 'new_license'
   image:
     repository: wandb/local
     tag: 'new_version'
   ```

2. 다음 코맨드로 Helm 업그레이드를 실행하세요:

   ```bash
   helm repo update
   helm upgrade --namespace=wandb --create-namespace \
     --install wandb wandb/wandb --version ${chart_version} \
     -f ${wandb_install_spec.yaml}
   ```

### 라이선스와 버전을 직접 업데이트

1. 새 라이선스 키와 이미지 태그를 환경 변수로 설정하세요:

   ```bash
   export LICENSE='new_license'
   export TAG='new_version'
   ```

2. 기존 설정에 새 값을 병합하여 Helm 릴리스를 업그레이드합니다:

   ```bash
   helm repo update
   helm upgrade --namespace=wandb --create-namespace \
     --install wandb wandb/wandb --version ${chart_version} \
     --reuse-values --set license=$LICENSE --set image.tag=$TAG
   ```

자세한 내용은 공개 저장소의 [업그레이드 가이드](https://github.com/wandb/helm-charts/blob/main/upgrade.md)를 참고하세요.

## Admin UI로 업데이트

이 방법은 일반적으로 self-managed Docker 환경에서 W&B 서버 컨테이너의 환경 변수로 설정되지 않은 라이선스만 업데이트할 수 있습니다.

1. [W&B 배포 페이지](https://deploy.wandb.ai/)에서 새 라이선스를 받아 해당 배포 환경의 조직 및 배포 ID와 일치하는지 확인합니다.
2. `<host-url>/system-settings`로 접속하여 W&B Admin UI에 엑세스하세요.
3. 라이선스 관리 섹션으로 이동하세요.
4. 새 라이선스 키를 입력하고 변경 내용을 저장하세요.