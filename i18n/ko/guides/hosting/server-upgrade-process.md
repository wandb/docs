---
description: Guide for updating W&B (Weights & Biases) version and license across
  different installation methods.
displayed_sidebar: default
---

# 서버 업그레이드 프로세스

W&B 서버 버전 및 라이선스 정보를 업데이트할 때는 초기 설치 메소드에 맞게 프로세스를 조정하세요. 다음은 W&B를 설치하는 주요 메소드와 해당 업데이트 프로세스입니다:

| 릴리즈 타입                                               | 설명                                                                                                                                                                                                                                                                                   |
| ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Terraform](./how-to-guides#wb-production-and-development) | W&B는 클라우드 배포를 위한 3개의 공개 Terraform 모듈을 지원합니다: [AWS](https://registry.terraform.io/modules/wandb/wandb/aws/latest), [GCP](https://registry.terraform.io/modules/wandb/wandb/google/latest), [Azure](https://registry.terraform.io/modules/wandb/wandb/azurerm/latest). |
| [Helm](./how-to-guides/bare-metal#helm-chart)              | 기존 Kubernetes 클러스터에 W&B를 설치하기 위해 [Helm Chart](https://github.com/wandb/helm-charts)를 사용할 수 있습니다.                                                                                                                                                                        |
| [Docker](./how-to-guides/bare-metal#docker-deployment)     | 최신 Docker 이미지는 [W&B Docker 레지스트리](https://hub.docker.com/r/wandb/local/tags)에서 찾을 수 있습니다.                                                                                                                                                                                 |

## Terraform을 통한 업데이트

1. 업그레이드를 위해, 특정 클라우드 제공업체의 `wandb_version` 및 `license`를 조정하고, Terraform `wandb_app` 모듈 설정에서 다음을 업데이트하세요:

   ```hcl
   module "wandb_app" {
       source  = "wandb/wandb/<cloud-specific-module>"
       version = "~> new_version"

       license       = "new_license_key" # 새 라이선스 키
       wandb_version = "new_wandb_version" # 원하는 W&B 버전
       ...
   }
   ```

2. `terraform plan` 및 `terraform apply`로 Terraform 설정을 적용하세요.

`terraform.tfvars` 파일 또는 기타 `.tfvars` 파일을 사용하려는 경우:

1. **`*.tfvars` 파일 수정:** 새 W&B 버전 및 라이선스 키로 `terraform.tfvars` 파일을 업데이트하거나 생성하세요.

2. **설정 적용:** 다음을 실행하세요:  
   `terraform plan -var-file="terraform.tfvars"`  
   `terraform apply -var-file="terraform.tfvars"`  
    Terraform 워크스페이스 디렉토리에서.

## Helm을 통한 업데이트

### spec을 통한 W&B 업데이트

1. Helm 차트 `*.yaml` 설정 파일에서 `image.tag` 및/또는 `license` 값 수정하여 새 버전을 지정하세요:

   ```yaml
   license: 'new_license'
   image:
     repository: wandb/local
     tag: 'new_version'
   ```

2. 다음 명령으로 Helm 업그레이드를 실행하세요:

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

2. 아래 명령으로, 새 값을 기존 설정과 병합하여 Helm 릴리스를 업그레이드하세요:

   ```bash
   helm repo update
   helm upgrade --namespace=wandb --create-namespace \
     --install wandb wandb/wandb --version ${chart_version} \
     --reuse-values --set license=$LICENSE --set image.tag=$TAG
   ```

더 자세한 내용은 공개 저장소의 [업그레이드 가이드](https://github.com/wandb/helm-charts/blob/main/UPGRADE.md)를 참조하세요.

## Docker 컨테이너를 통한 업데이트

1. [W&B Docker 레지스트리](https://hub.docker.com/r/wandb/local/tags)에서 새 버전을 선택하세요.
2. 다음으로 새 Docker 이미지 버전을 가져오세요:

   ```bash
   docker pull wandb/local:<new_version>
   ```

3. Docker 컨테이너 배포 및 관리에 대한 모범 사례를 따르면서 새 이미지 버전으로 Docker 컨테이너를 업데이트하세요.

Docker `run` 예시 및 추가 세부 정보는 [Docker 배포](./how-to-guides/bare-metal##docker-deployment)를 참조하세요.

## 관리자 UI를 통한 업데이트

이 방법은 W&B 서버 컨테이너에서 환경 변수를 통해 설정되지 않은 라이선스를 업데이트하는 경우에만 작동합니다. 일반적으로 자체 호스팅 Docker 설치에서 해당됩니다.

1. 업그레이드하려는 배포에 맞는 올바른 조직 및 배포 ID와 일치하는 새 라이선스를 [W&B 배포 페이지](https://deploy.wandb.ai/)에서 받으세요.
2. `<host-url>/system-settings`에서 W&B 관리자 UI에 접속하세요.
3. 라이선스 관리 섹션으로 이동하세요.
4. 새 라이선스 키를 입력하고 변경 사항을 저장하세요.

## W&B 전용 클라우드 업데이트

:::note
전용 설치의 경우, W&B는 월간 기준으로 서버 버전을 업그레이드합니다. [릴리즈 프로세스](./server-release-process) 문서에서 더 많은 정보를 찾을 수 있습니다.
:::