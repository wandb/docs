---
title: Update W&B license and version
description: W&B (Weights & Biases) 버전 및 라이센스를 다양한 설치 메소드에서 업데이트하는 가이드.
displayed_sidebar: default
---

W&B Server 버전과 라이센스를 업데이트하려면 W&B Server를 설치한 동일한 메소드를 사용하세요. 다음 표는 다양한 배포 메소드에 기반한 라이센스 및 버전 업데이트 방법을 나열합니다:

| 릴리스 유형    | 설명         |
| ---------------- | ------------------ |
| [Terraform](#update-with-terraform) | W&B는 클라우드 배포를 위한 세 가지 공개 Terraform 모듈을 지원합니다: [AWS](https://registry.terraform.io/modules/wandb/wandb/aws/latest), [GCP](https://registry.terraform.io/modules/wandb/wandb/google/latest), 그리고 [Azure](https://registry.terraform.io/modules/wandb/wandb/azurerm/latest). |
| [Helm](#update-with-helm)              | 기존 Kubernetes 클러스터에 W&B를 설치하려면 [Helm Chart](https://github.com/wandb/helm-charts)를 사용할 수 있습니다. |
| [Docker](#update-with-docker-container)     | 최신 Docker 이미지가 [W&B Docker Registry](https://hub.docker.com/r/wandb/local/tags)에 있습니다. |

## Terraform으로 업데이트하기

Terraform으로 라이센스와 버전을 업데이트합니다. 다음 표는 클라우드 플랫폼에 기반한 W&B 관리 Terraform 모듈을 나열합니다.

|클라우드 제공자| Terraform 모듈|
|-----|-----|
|AWS|[AWS Terraform 모듈](https://registry.terraform.io/modules/wandb/wandb/aws/latest)|
|GCP|[GCP Terraform 모듈](https://registry.terraform.io/modules/wandb/wandb/google/latest)|
|Azure|[Azure Terraform 모듈](https://registry.terraform.io/modules/wandb/wandb/azurerm/latest)|

1. 첫째, 사용 중인 클라우드 제공자에 맞는 W&B 유지 Terraform 모듈로 이동하세요. 적절한 Terraform 모듈을 찾으려면 앞의 표를 참조하세요.
2. Terraform 설정 내에서 Terraform `wandb_app` 모듈 설정의 `wandb_version`과 `license`를 업데이트하세요:

   ```hcl
   module "wandb_app" {
       source  = "wandb/wandb/<cloud-specific-module>"
       version = "new_version"
       license       = "new_license_key" # 새 라이센스 키
       wandb_version = "new_wandb_version" # 원하는 W&B 버전
       ...
   }
   ```
3. `terraform plan` 및 `terraform apply`로 Terraform 설정을 적용하세요.
   ```bash
   terraform init
   terraform apply
   ```

4. (선택 사항) `terraform.tfvars` 또는 기타 `.tfvars` 파일을 사용하는 경우:
   1. 새로운 W&B 버전과 라이센스 키로 `terraform.tfvars` 파일을 업데이트하거나 생성합니다.
   2. 설정을 적용합니다. Terraform 워크스페이스 디렉토리에서 실행:  
   ```bash
   terraform plan -var-file="terraform.tfvars"
   terraform apply -var-file="terraform.tfvars"
   ```

## Helm으로 업데이트하기

### spec으로 W&B 업데이트하기

1. Helm chart `*.yaml` 설정 파일의 `image.tag` 및/또는 `license` 값을 수정하여 새 버전을 지정합니다:

   ```yaml
   license: 'new_license'
   image:
     repository: wandb/local
     tag: 'new_version'
   ```

2. 다음 코맨드로 Helm 업그레이드를 실행합니다:

   ```bash
   helm repo update
   helm upgrade --namespace=wandb --create-namespace \
     --install wandb wandb/wandb --version ${chart_version} \
     -f ${wandb_install_spec.yaml}
   ```

### 라이센스와 버전을 직접 업데이트하기

1. 새 라이센스 키와 이미지 태그를 환경 변수로 설정합니다:

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

자세한 내용은 공개 레포지토리의 [업그레이드 가이드](https://github.com/wandb/helm-charts/blob/main/UPGRADE.md)를 참조하세요.

## Docker 컨테이너로 업데이트하기

1. [W&B Docker Registry](https://hub.docker.com/r/wandb/local/tags)에서 새로운 버전을 선택하세요.
2. 다음 코맨드를 사용하여 새로운 Docker 이미지 버전을 가져옵니다:

   ```bash
   docker pull wandb/local:<new_version>
   ```

3. 컨테이너 배포 및 관리를 위한 모범 사례를 따르며 새로운 이미지 버전 실행을 위해 Docker 컨테이너를 업데이트합니다.

## 관리자 UI로 업데이트하기

이 메소드는 환경 변수로 W&B 서버 컨테이너에 설정되지 않은 라이센스를 업데이트하는 경우에만 작동하며, 일반적으로 자체 호스팅 Docker 설치에서 사용됩니다.

1. [W&B 배포 페이지](https://deploy.wandb.ai/)에서 올바른 조직과 배포 ID가 일치하는 새 라이센스를 획득하세요.
2. `<host-url>/system-settings`에서 W&B 관리자 UI에 엑세스하세요.
3. 라이센스 관리 섹션으로 이동합니다.
4. 새 라이센스 키를 입력하고 변경 사항을 저장합니다.