---
title: Update W&B license and version
description: 다양한 설치 방법에 따른 W&B (Weights & Biases) 버전 및 라이선스 업데이트 가이드.
menu:
  default:
    identifier: ko-guides-hosting-hosting-options-self-managed-server-upgrade-process
    parent: self-managed
url: guides/hosting/server-upgrade-process
weight: 6
---

W&B 서버를 설치한 방법과 동일한 방법으로 W&B 서버 버전과 라이선스를 업데이트합니다. 다음 표는 다양한 배포 방법에 따라 라이선스와 버전을 업데이트하는 방법을 보여줍니다.

| 릴리스 유형 | 설명 |
| --- | --- |
| [Terraform]({{< relref path="#update-with-terraform" lang="ko" >}}) | W&B는 클라우드 배포를 위해 세 개의 퍼블릭 Terraform 모듈([AWS](https://registry.terraform.io/modules/wandb/wandb/aws/latest), [GCP](https://registry.terraform.io/modules/wandb/wandb/google/latest) 및 [Azure](https://registry.terraform.io/modules/wandb/wandb/azurerm/latest))을 지원합니다. |
| [Helm]({{< relref path="#update-with-helm" lang="ko" >}}) | [Helm Chart](https://github.com/wandb/helm-charts)를 사용하여 기존 Kubernetes 클러스터에 W&B를 설치할 수 있습니다. |

## Terraform으로 업데이트

Terraform을 사용하여 라이선스와 버전을 업데이트합니다. 다음 표는 클라우드 플랫폼을 기반으로 W&B에서 관리하는 Terraform 모듈을 나열합니다.

|클라우드 공급자| Terraform 모듈|
|-----|-----|
|AWS|[AWS Terraform module](https://registry.terraform.io/modules/wandb/wandb/aws/latest)|
|GCP|[GCP Terraform module](https://registry.terraform.io/modules/wandb/wandb/google/latest)|
|Azure|[Azure Terraform module](https://registry.terraform.io/modules/wandb/wandb/azurerm/latest)|

1. 먼저 해당 클라우드 공급자에 대해 W&B에서 관리하는 Terraform 모듈로 이동합니다. 이전 표를 참조하여 클라우드 공급자를 기반으로 적절한 Terraform 모듈을 찾으십시오.
2. Terraform 구성 내에서 Terraform `wandb_app` 모듈 구성에서 `wandb_version`과 `license`를 업데이트합니다.

   ```hcl
   module "wandb_app" {
       source  = "wandb/wandb/<cloud-specific-module>"
       version = "new_version"
       license       = "new_license_key" # Your new license key
       wandb_version = "new_wandb_version" # Desired W&B version
       ...
   }
   ```
3. `terraform plan`과 `terraform apply`로 Terraform 구성을 적용합니다.
   ```bash
   terraform init
   terraform apply
   ```

4. (선택 사항) `terraform.tfvars` 또는 다른 `.tfvars` 파일을 사용하는 경우.

   새 W&B 버전과 라이선스 키로 `terraform.tfvars` 파일을 업데이트하거나 생성합니다.
   ```bash
   terraform plan -var-file="terraform.tfvars"
   ```
   구성을 적용합니다. Terraform 워크스페이스 디렉토리에서 다음을 실행합니다.
   ```bash
   terraform apply -var-file="terraform.tfvars"
   ```
## Helm으로 업데이트

### spec으로 W&B 업데이트

1. Helm chart `*.yaml` 설정 파일에서 `image.tag` 및/또는 `license` 값을 수정하여 새 버전을 지정합니다.

   ```yaml
   license: 'new_license'
   image:
     repository: wandb/local
     tag: 'new_version'
   ```

2. 다음 코맨드로 Helm 업그레이드를 실행합니다.

   ```bash
   helm repo update
   helm upgrade --namespace=wandb --create-namespace \
     --install wandb wandb/wandb --version ${chart_version} \
     -f ${wandb_install_spec.yaml}
   ```

### 라이선스 및 버전 직접 업데이트

1. 새 라이선스 키와 이미지 태그를 환경 변수로 설정합니다.

   ```bash
   export LICENSE='new_license'
   export TAG='new_version'
   ```

2. 아래 코맨드로 Helm 릴리스를 업그레이드하여 새 값을 기존 설정과 병합합니다.

   ```bash
   helm repo update
   helm upgrade --namespace=wandb --create-namespace \
     --install wandb wandb/wandb --version ${chart_version} \
     --reuse-values --set license=$LICENSE --set image.tag=$TAG
   ```

자세한 내용은 퍼블릭 저장소의 [업그레이드 가이드](https://github.com/wandb/helm-charts/blob/main/upgrade.md)를 참조하십시오.

## Admin UI로 업데이트

이 방법은 일반적으로 자체 호스팅 Docker 설치에서 W&B 서버 컨테이너의 환경 변수로 설정되지 않은 라이선스를 업데이트하는 데만 사용할 수 있습니다.

1. 업그레이드하려는 배포에 대해 올바른 organization 및 배포 ID와 일치하는지 확인하여 [W&B 배포 페이지](https://deploy.wandb.ai/)에서 새 라이선스를 받습니다.
2. `<host-url>/system-settings`에서 W&B Admin UI에 엑세스합니다.
3. 라이선스 관리 섹션으로 이동합니다.
4. 새 라이선스 키를 입력하고 변경 사항을 저장합니다.
