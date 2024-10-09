---
title: Run W&B Server on Kubernetes
description: W&B 플랫폼을 Kubernetes Operator로 배포하기
displayed_sidebar: default
---

## W&B Kubernetes Operator

W&B Kubernetes Operator를 사용하여 Kubernetes에서 W&B Server 배포를 간소화하고, 관리 및 문제 해결, 확장을 더욱 쉽게 할 수 있습니다. 운영자를 W&B 인스턴스의 스마트 어시스턴트로 생각할 수 있습니다.

W&B Server 아키텍처와 설계는 AI 개발자 도구의 기능을 확장하고, 더 나은 성능과 확장성, 관리 용이성을 제공하기 위해 지속적으로 진화하고 있습니다. 이러한 진화는 컴퓨팅 서비스와 관련된 저장소, 그리고 이들 간의 연결성에도 적용됩니다. 배포 유형 전반에 걸쳐 지속적인 업데이트 및 개선을 촉진하기 위해 W&B 사용자는 Kubernetes operator를 사용합니다.

:::info
W&B는 AWS, GCP 및 Azure 공용 클라우드에서 전용 클라우드 인스턴스를 배포하고 관리하기 위해 operator를 사용합니다.
:::

Kubernetes operators에 대한 자세한 내용은 Kubernetes 문서의 [Operator 패턴](https://kubernetes.io/docs/concepts/extend-kubernetes/operator/)을 참조하십시오.

## 아키텍처 변화의 이유
역사적으로 W&B 애플리케이션은 Kubernetes 클러스터 내의 단일 배포 및 pod 또는 단일 docker 컨테이너로 배포되었습니다. W&B는 계속해서 데이터베이스와 오브젝트 스토어를 외부화할 것을 권장합니다. 데이터베이스와 오브젝트 스토어를 외부화함으로써 애플리케이션의 상태가 분리됩니다.

애플리케이션이 성장함에 따라 모놀리식 컨테이너에서 분산 시스템(마이크로서비스)으로 발전할 필요성이 명확해졌습니다. 이러한 변화는 백엔드 로직 처리와 내장형 Kubernetes 인프라 기능의 원활한 도입을 용이하게 합니다. 분산 시스템은 또한 W&B가 의존하는 추가 기능을 위한 새로운 서비스 배포를 지원합니다.

2024년 이전에는 Kubernetes 관련 변경 사항이 있을 경우 [terraform-kubernetes-wandb](https://github.com/wandb/terraform-kubernetes-wandb) Terraform 모듈을 수동으로 업데이트해야 했습니다. Terraform 모듈을 업데이트하면 클라우드 공급자 간의 호환성을 보장하고, 필요한 Terraform 변수를 구성하며, 각 백엔드 또는 Kubernetes 수준의 변경에 대해 Terraform 적용을 실행합니다.

이 프로세스는 확장성이 없었으며, W&B 지원 팀은 각 고객이 그들의 Terraform 모듈을 업그레이드하는 것을 도와야 했습니다.

해결책은 중앙 [deploy.wandb.ai](https://deploy.wandb.ai) 서버에 연결하여 주어진 릴리스 채널에 대한 최신 사양 변경을 요청하고 이를 적용하는 operator를 구현하는 것이었습니다. 업데이트는 라이선스가 유효한 한 받을 수 있습니다. [Helm](https://helm.sh/)은 W&B operator의 배포 메커니즘 및 W&B Kubernetes 스택의 모든 설정 템플릿 처리를 위한 수단으로 사용됩니다, Helm-ception입니다.

## 작동 방식
Helm 또는 소스에서 operator를 설치할 수 있습니다. 자세한 지침은 [charts/operator](https://github.com/wandb/helm-charts/tree/main/charts/operator)를 참조하십시오.

설치 프로세스는 `controller-manager`라는 배포를 생성하고, 하나의 `spec`을 사용하여 클러스터에 적용되는 `weightsandbiases.apps.wandb.com` (shortName: `wandb`)라는 [custom resource](https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/) 정의를 사용합니다:

```yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: weightsandbiases.apps.wandb.com
```

`controller-manager`는 custom resource의 spec, 릴리스 채널 및 사용자 정의 설정을 기반으로 [charts/operator-wandb](https://github.com/wandb/helm-charts/tree/main/charts/operator-wandb)를 설치합니다. 설정 사양 계층 구조는 사용자 측에 최대한의 설정 유연성을 제공하고, W&B가 새 이미지, 설정, 기능 및 Helm 업데이트를 자동으로 릴리스할 수 있게 합니다.

설정 옵션에 대한 자세한 내용은 [설정 사양 계층 구조](#configuration-specification-hierarchy) 및 [W&B Operator에 대한 설정 참조](#configuration-reference-for-wb-operator)를 참조하십시오.

## 설정 사양 계층 구조
설정 사양은 높은 수준의 사양이 낮은 수준의 사양을 덮어쓰는 계층적 모델을 따릅니다. 작동 방식은 다음과 같습니다:

- **릴리스 채널 값**: 이 기본 수준 설정은 W&B가 배포를 위해 설정한 릴리스 채널을 기반으로 기본 값과 구성을 설정합니다.
- **사용자 입력 값**: 사용자는 System Console을 통해 릴리스 채널 사양에서 제공한 기본 설정을 덮어쓸 수 있습니다.
- **Custom Resource 값**: 사용자 측에서 제공되는 가장 높은 수준의 사양입니다. 여기에 지정된 모든 값은 사용자 입력 및 릴리스 채널 사양을 모두 덮어씁니다. 설정 옵션에 대한 자세한 설명은 [설정 참조](#configuration-reference-for-wb-operator)를 참조하십시오.

이 계층적 모델은 다양한 요구를 충족할 수 있도록 유연하고 사용자 정의 가능한 구성을 보장하면서, 업그레이드 및 변경에 대한 관리 가능하고 체계적인 접근 방식을 유지합니다.

## W&B Kubernetes Operator를 사용하기 위한 요구 사항
W&B Kubernetes operator로 W&B를 배포하려면 다음 요구 사항을 충족하십시오:

* 설치 및 런타임 동안 다음 엔드포인트에 대한 출구:
    * deploy.wandb.ai
    * docker.io
    * quay.io
    * gcr.io
* Ingress 컨트롤러(예: Contour, Nginx)가 배포, 설정 및 완전히 작동 중인 Kubernetes 클러스터 버전 1.28 이상이어야 함.
* MySQL 8.0 데이터베이스를 외부에서 호스트하고 런해야 함.
* CORS 지원이 있는 오브젝트 저장소(Amazon S3, Azure Cloud Storage, Google Cloud Storage 또는 모든 S3 호환 저장소 서비스).
* 유효한 W&B Server 라이선스.

자체 관리 설치를 설정하고 구성하는 방법에 대한 자세한 설명은 [이 가이드](./self-managed/bare-metal)를 참조하십시오.

설치 방법에 따라 다음 요구 사항을 충족해야 할 수도 있습니다:
* 올바른 Kubernetes 클러스터 컨텍스트와 함께 설치 및 구성된 Kubectl.
* Helm이 설치되어 있어야 함.

# Air-gapped 설치
Air-gapped 환경에서 W&B Kubernetes Operator를 설치하는 방법에 대한 [Kubernetes 환경에서 에어갭된 W&B 배포](./operator-airgapped) 튜토리얼을 참조하십시오.

# W&B Server 애플리케이션 배포
이 섹션에서는 W&B Kubernetes operator를 배포하는 여러 가지 방법을 설명합니다.
:::note
W&B Operator는 W&B Server에 대한 기본 설치 방법이 될 것입니다. 다른 방법은 미래에 지원 중단될 것입니다.
:::

**다음 중 하나를 선택하십시오:**
- 필요한 모든 외부 서비스를 프로비저닝하고 Helm CLI를 사용하여 Kubernetes에 W&B를 배포하려면, [여기](#deploy-wb-with-helm-cli)로 이동하십시오.
- 인프라와 W&B Server를 Terraform으로 관리하길 원하면, [여기](#deploy-wb-with-helm-terraform-module)로 이동하십시오.
- W&B Cloud Terraform Modules를 활용하고 싶다면, [여기](#deploy-wb-with-wb-cloud-terraform-modules)로 이동하십시오.

## Helm CLI를 사용하여 W&B 배포하기
W&B는 W&B Kubernetes operator를 Kubernetes 클러스터에 배포하기 위한 Helm Chart를 제공합니다. 이 접근 방식은 Helm CLI나 ArgoCD와 같은 CI/CD 툴을 사용하여 W&B Server를 배포할 수 있습니다. 위에서 언급한 요구 사항이 충족되는지 확인하십시오.

Helm CLI를 사용하여 W&B Kubernetes Operator를 설치하려면 다음 단계를 수행하십시오:

1. W&B Helm 저장소 추가. W&B Helm chart는 W&B Helm 저장소에서 제공됩니다. 다음 코맨드로 저장소를 추가하십시오:
```shell
helm repo add wandb https://charts.wandb.ai
helm repo update
```
2. Kubernetes 클러스터에 Operator 설치. 다음을 복사하여 붙여넣으십시오:
```shell
helm upgrade --install operator wandb/operator -n wandb-cr --create-namespace
```
3. W&B operator custom resource를 구성하여 W&B Server 설치 트리거. W&B operator 배포를 사용자 정의하기 위해 operator.yaml 파일을 생성하고, 맞춤 설정을 지정하십시오. 자세한 내용은 [설정 참조](#configuration-reference-for-wb-operator)를 참조하십시오.

사양 YAML을 생성하고 값을 채운 후에 다음을 실행하면 operator가 설정을 적용하고 구성에 따라 W&B Server 애플리케이션을 설치합니다.

```shell
kubectl apply -f operator.yaml
```

배포가 완료될 때까지 기다리고 설치를 확인하십시오. 몇 분이 소요됩니다.

4. 설치 확인. 브라우저를 통해 새로운 설치에 엑세스하고 첫 번째 관리자 사용자 계정을 생성하십시오. 완료되면 [여기](#verify-the-installation)에서 설명된 확인 단계를 따르십시오.

## Helm Terraform Module을 사용하여 W&B 배포하기
이 방법은 특정 요구 사항에 맞춘 사용자 지정 배포를 가능하게 하며, 일관성과 반복성을 위해 Terraform의 인프라 코드 방식을 활용합니다. 공식 W&B Helm 기반 Terraform Module은 [여기](https://registry.terraform.io/modules/wandb/wandb/helm/latest)에 위치합니다.

다음 코드는 시작점으로 사용할 수 있으며, 프로덕션 수준의 배포에 필요한 모든 설정 옵션을 포함합니다.

```hcl
module "wandb" {
  source  = "wandb/wandb/helm"

  spec = {
    values = {
      global = {
        host    = "https://<HOST_URI>"
        license = "eyJhbGnUzaH...j9ZieKQ2x5GGfw"

        bucket = {
          <details depend on the provider>
        }

        mysql = {
          <redacted>
        }
      }

      ingress = {
        annotations = {
          "a" = "b"
          "x" = "y"
        }
      }
    }
  }
}
```

설정 옵션은 [설정 참조](#configuration-reference-for-wb-operator)에 설명된 것과 동일하지만, 구문은 HashiCorp Configuration Language (HCL)를 따라야 합니다. W&B custom resource 정의는 Terraform 모듈에 의해 생성될 것입니다.

Weights & Biases가 고객을 위한 “Dedicated Cloud” 설치를 배포하기 위해 Helm Terraform 모듈을 사용하는 방법을 보려면, 다음 링크를 따르십시오:
- [AWS](https://github.com/wandb/terraform-aws-wandb/blob/45e1d746f53e78e73e68f911a1f8cad5408e74b6/main.tf#L225)
- [Azure](https://github.com/wandb/terraform-azurerm-wandb/blob/170e03136b6b6fc758102d59dacda99768854045/main.tf#L155)
- [GCP](https://github.com/wandb/terraform-google-wandb/blob/49ddc3383df4cefc04337a2ae784f57ce2a2c699/main.tf#L189)

## W&B Cloud Terraform 모듈을 사용하여 W&B 배포하기
W&B는 AWS, GCP 및 Azure를 위한 Terraform 모듈을 제공합니다. 이 모듈은 Kubernetes 클러스터, 로드 밸런서, MySQL 데이터베이스 등과 같은 전체 인프라를 배포하며, W&B Server 애플리케이션도 함께 배포합니다. 공식 W&B 클라우드 전용 Terraform 모듈은 다음과 같은 버전으로 제공됩니다:

| Terraform Registry                                                  | Source Code                                      | Version |
| ------------------------------------------------------------------- | ------------------------------------------------ | ------- |
| [AWS](https://registry.terraform.io/modules/wandb/wandb/aws/latest) | https://github.com/wandb/terraform-aws-wandb     | v4.0.0+ |
| [Azure](https://github.com/wandb/terraform-azurerm-wandb)           | https://github.com/wandb/terraform-azurerm-wandb | v2.0.0+ |
| [GCP](https://github.com/wandb/terraform-google-wandb)              | https://github.com/wandb/terraform-google-wandb  | v2.0.0+ |

이 인테그레이션은 W&B Kubernetes Operator가 최소한의 설정으로 인스턴스에서 사용할 준비가 되어 있어, 클라우드 환경에서 W&B Server를 배포하고 관리하는 쉬운 길을 제공합니다.

이 모듈을 사용하는 방법에 대한 자세한 설명은 [이 섹션](./hosting-options/self-managed#deploy-wb-server-within-self-managed-cloud-accounts)을 참조하십시오.

## 설치 확인
설치를 확인하려면, W&B는 [W&B CLI](../../ref/cli/README.md)를 사용할 것을 권장합니다. verify 명령은 모든 구성 요소와 구성을 확인하는 여러 테스트를 실행합니다.

:::note
이 단계는 브라우저로 첫 번째 관리자 사용자 계정을 만든 것으로 간주합니다.
:::

설치를 확인하려면 다음 단계를 따르십시오:

1. W&B CLI 설치:
```shell
pip install wandb
```
2. W&B에 로그인:
```shell
wandb login --host=https://YOUR_DNS_DOMAIN
```

예를 들어:
```shell
wandb login --host=https://wandb.company-name.com
```

3. 설치 확인:
```shell
wandb verify
```

설치가 성공적으로 완료되고 완전히 작동하는 W&B 배포는 다음과 같은 출력을 보여줍니다:

```console
Default host selected:  https://wandb.company-name.com
Find detailed logs for this test at: /var/folders/pn/b3g3gnc11_sbsykqkm3tx5rh0000gp/T/tmpdtdjbxua/wandb
Checking if logged in...................................................✅
Checking signed URL upload..............................................✅
Checking ability to send large payloads through proxy...................✅
Checking requests to base url...........................................✅
Checking requests made over signed URLs.................................✅
Checking CORs configuration of the bucket...............................✅
Checking wandb package version is up to date............................✅
Checking logged metrics, saving and downloading a file..................✅
Checking artifact save and download workflows...........................✅
```

## W&B 관리 콘솔 엑세스
W&B Kubernetes operator는 관리 콘솔을 제공합니다. 이는 `${HOST_URI}/console`에 위치하며, 예를 들어 `https://wandb.company-name.com/console`입니다.

관리 콘솔에 로그인하는 두 가지 방법이 있습니다:

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<Tabs
  defaultValue="option1"
  values={[
    {label: 'Option 1 (Recommended)', value: 'option1'},
    {label: 'Option 2', value: 'option2'},
  ]}>
  <TabItem value="option1">

1. 브라우저에서 W&B 애플리케이션 열기 및 로그인. `${HOST_URI}/`를 통해 W&B 애플리케이션에 로그인하십시오. 예를 들어 `https://wandb.company-name.com`입니다.
2. 콘솔 엑세스. 오른쪽 상단의 아이콘을 클릭한 후 **System console**을 클릭하십시오. 관리자 권한이 있는 사용자만이 **System console** 항목을 볼 수 있습니다.

![](/images/hosting/access_system_console_via_main_app.png)

  </TabItem>
  <TabItem value="option2">

:::note
W&B는 Option 1이 작동하지 않을 경우에만 다음의 단계를 사용하여 콘솔에 엑세스할 것을 권장합니다.
:::

1. 브라우저에서 콘솔 애플리케이션 열기. 위에서 설명한 URL을 브라우저에서 열면 다음과 같은 로그인 화면이 나타납니다:
![](/images/hosting/access_system_console_directly.png)
2. 비밀번호 검색. 비밀번호는 설치 중에 생성되며 Kubernetes 비밀로 저장됩니다. 비밀번호를 검색하려면 다음 명령을 실행하십시오:
```shell
kubectl get secret wandb-password -o jsonpath='{.data.password}' | base64 -d
```
비밀번호를 클립보드에 복사합니다.
3. 콘솔에 로그인합니다. 복사한 비밀번호를 “Enter password” 텍스트 필드에 붙여넣고 로그인 버튼을 클릭합니다.

  </TabItem>
</Tabs>

## W&B Kubernetes operator 업데이트
이 섹션에서는 W&B Kubernetes operator를 업데이트하는 방법을 설명합니다.

:::note
* W&B Kubernetes operator를 업데이트해도 W&B Server 애플리케이션은 업데이트되지 않습니다.
* 이전에 W&B Kubernetes operator를 사용하지 않은 Helm chart를 사용하는 경우에는 W&B operator를 업데이트하기 전에 [여기](#migrate-self-managed-instances-to-wb-operator)의 지침을 참조하십시오.
:::

아래의 코드 조각을 복사하여 터미널에 붙여넣으십시오.

1. 먼저, [`helm repo update`](https://helm.sh/docs/helm/helm_repo_update/)로 저장소를 업데이트합니다:
```shell
helm repo update
```

2. 다음으로, [`helm upgrade`](https://helm.sh/docs/helm/helm_upgrade/)로 Helm chart를 업데이트합니다:
 
```shell
helm upgrade operator wandb/operator -n wandb-cr --reuse-values
```

## W&B Server 애플리케이션 업데이트
W&B Kubernetes operator를 사용하는 경우 더 이상 W&B Server 애플리케이션을 업데이트할 필요가 없습니다.

operator는 W&B 소프트웨어의 새 버전이 발행될 때 W&B Server 애플리케이션을 자동으로 업데이트합니다.

## W&B Operator로 자체 관리 인스턴스 마이그레이션
다음 섹션에서는 자신의 W&B Server 설치를 자체 관리에서 W&B Operator를 사용하여 전환하는 방법을 설명합니다. 마이그레이션 프로세스는 W&B Server를 설치한 방법에 따라 달라집니다:

:::note
W&B Operator는 W&B Server의 기본 설치 방법이 될 것입니다. 앞으로 W&B는 operator를 사용하지 않는 배포 메커니즘을 지원 중단할 것입니다. [고객 지원](mailto:support@wandb.com) 또는 W&B 팀에 문의하여 질문이 있는 경우에 연락하십시오.
:::

- 공식 W&B Cloud Terraform Modules를 사용한 경우, 적절한 문서로 이동하여 거기에 나와 있는 단계를 따르십시오:
  - [AWS](#migrate-to-operator-based-aws-terraform-modules)
  - [GCP](#migrate-to-operator-based-gcp-terraform-modules)
  - [Azure](#migrate-to-operator-based-azure-terraform-modules)
- [W&B Non-Operator Helm chart](https://github.com/wandb/helm-charts/tree/main/charts/wandb)를 사용한 경우, [여기](#migrate-to-operator-based-helm-chart)로 이동하십시오.
- [Terraform](https://registry.terraform.io/modules/wandb/wandb/kubernetes/latest)과 함께 Non-Operator Helm chart를 사용한 경우, [여기](#migrate-to-operator-based-terraform-helm-chart)로 이동하십시오.
- Kubernetes 리소스를 manifest(s)로 생성한 경우, [여기](#migrate-to-operator-based-helm-chart)로 이동하십시오.

### Operator 기반 AWS Terraform Modules로 마이그레이션하기

마이그레이션 프로세스에 대한 자세한 설명은 [여기](self-managed/aws-tf#migrate-to-operator-based-aws-terraform-modules)를 참조하십시오.

### Operator 기반 GCP Terraform Modules로 마이그레이션하기

질문이 있거나 도움이 필요한 경우 [고객 지원](mailto:support@wandb.com) 또는 W&B 팀에 문의하십시오.

### Operator 기반 Azure Terraform Modules로 마이그레이션하기

질문이 있거나 도움이 필요한 경우 [고객 지원](mailto:support@wandb.com) 또는 W&B 팀에 문의하십시오.

### Operator 기반 Helm chart로 마이그레이션하기

Operator 기반 Helm chart로 마이그레이션하는 단계는 다음과 같습니다:

1. 현재 W&B 설정 얻기. W&B가 operator 기반이 아닌 버전의 Helm chart로 배포되었다면, 아래와 같이 값을 내보냅니다:
```shell
helm get values wandb
```
W&B가 Kubernetes manifest로 배포된 경우, 아래와 같이 값을 내보냅니다:
```shell
kubectl get deployment wandb -o yaml
```
이제 다음 단계에 필요한 모든 설정 값을 가지고 있어야 합니다.

2. operator.yaml이라는 파일을 만듭니다. [설정 참조](#configuration-reference-for-wb-operator)에 설명된 형식을 따릅니다. 1단계의 값을 사용합니다.

3. 현재 배포를 0 pod로 스케일링합니다. 이 단계는 현재 배포를 중지합니다.
```shell
kubectl scale --replicas=0 deployment wandb
```
4. Helm chart 저장소 업데이트:
```shell
helm repo update
```
5. 새로운 Helm chart 설치:
```shell
helm upgrade --install operator wandb/operator -n wandb-cr --create-namespace
```
6. 새로운 helm chart를 구성하고 W&B 애플리케이션 배포를 트리거합니다. 새 구성을 적용합니다.
```shell
kubectl apply -f operator.yaml
```
배포가 완료될 때까지 몇 분이 소요됩니다.

7. 설치를 확인합니다. [설치 확인](#verify-the-installation)의 단계를 따라 모든 항목이 잘 작동하는지 확인하십시오.

8. 이전 설치 제거. 이전의 Helm chart를 제거하거나 manifest로 작성된 리소스를 삭제합니다.

### Operator 기반 Terraform Helm chart로 마이그레이션하기

Operator 기반 Helm chart로 마이그레이션하는 단계는 다음과 같습니다:

1. Terraform 설정 준비하기. Terraform 설정에서 이전 배포의 Terraform 코드를 [여기](#deploy-wb-with-helm-terraform-module)에 설명된 코드로 교체합니다. 동일한 변수를 설정하십시오. .tfvars 파일을 사용하는 경우, 변경하지 마십시오.
2. Terraform 실행을 수행합니다. terraform init, plan 및 apply를 실행합니다.
3. 설치를 확인합니다. [설치 확인](#verify-the-installation)의 단계를 따라 모든 항목이 잘 작동하는지 확인하십시오.
4. 이전 설치 제거. 이전의 Helm chart를 제거하거나 manifest로 작성된 리소스를 삭제합니다.

## W&B Server 설정 참조

이 섹션에서는 W&B Server 애플리케이션의 설정 옵션에 대해 설명합니다. 애플리케이션은 [WeightsAndBiases](#how-it-works)라는 custom resource 정의로 설정을 받습니다. 일부 설정 옵션은 아래 설정으로 노출되며, 일부는 환경 변수로 설정해야 합니다.

문서에는 두 개의 환경 변수 목록이 있습니다: [기본](./env-vars) 및 [고급](./iam/advanced_env_vars). 필요한 설정 옵션이 Helm Chart를 통해 노출되지 않는 경우에만 환경 변수를 사용하십시오.

프로덕션 배포를 위한 W&B Server 애플리케이션 설정 파일은 다음 내용을 요구합니다:

```yaml
apiVersion: apps.wandb.com/v1
kind: WeightsAndBiases
metadata:
  labels:
    app.kubernetes.io/name: weightsandbiases
    app.kubernetes.io/instance: wandb
  name: wandb
  namespace: default
spec:
  values:
    global:
      host: https://<HOST_URI>
      license: eyJhbGnUzaH...j9ZieKQ2x5GGfw
      bucket:
        <details depend on the provider>
      mysql:
        <redacted>
    ingress:
      annotations:
        <redacted>
```

이 YAML 파일은 W&B 배포의 원하는 상태를 정의하며, 버전, 환경 변수, 데이터베이스와 같은 외부 리소스 및 기타 필요한 설정을 포함합니다. 위의 YAML을 시작점으로 사용하고 필요한 정보를 추가하십시오.

spec 사용자 지정의 전체 목록은 Helm 저장소의 [여기](https://github.com/wandb/helm-charts/blob/main/charts/operator-wandb/values.yaml)에서 찾을 수 있습니다. 권장 접근 방식은 필요한 부분만 변경하고 나머지 기본값을 사용하는 것입니다.

### 전체 예제
다음은 GCP Kubernetes와 GCP Ingress 및 GCS(GCP 오브젝트 스토리지)를 사용하는 예제 설정입니다:

```yaml
apiVersion: apps.wandb.com/v1
kind: WeightsAndBiases
metadata:
  labels:
    app.kubernetes.io/name: weightsandbiases
    app.kubernetes.io/instance: wandb
  name: wandb
  namespace: default
spec:
  values:
    global:
      host: https://abc-wandb.sandbox-gcp.wandb.ml
      bucket:
        name: abc-wandb-moving-pipefish
        provider: gcs
      mysql:
        database: wandb_local
        host: 10.218.0.2
        name: wandb_local
        password: 8wtX6cJHizAZvYScjDzZcUarK4zZGjpV
        port: 3306
        user: wandb
      license: eyJhbGnUzaHgyQjQyQWhEU3...ZieKQ2x5GGfw
    ingress:
      annotations:
        ingress.gcp.kubernetes.io/pre-shared-cert: abc-wandb-cert-creative-puma
        kubernetes.io/ingress.class: gce
        kubernetes.io/ingress.global-static-ip-name: abc-wandb-operator-address
```

### 호스트
```yaml
 # 프로토콜 포함하여 FQDN을 제공하십시오
global:
  # 예시 호스트 이름, 본인의 것으로 교체하십시오
  host: https://abc-wandb.sandbox-gcp.wandb.ml
```

### 오브젝트 스토리지 (버킷)

**AWS**
```yaml
global:
  bucket:
    provider: "s3"
    name: ""
    kmsKey: ""
    region: ""
```

**GCP**
```yaml
global:
  bucket:
    provider: "gcs"
    name: ""
```

**Azure**
```yaml
global:
  bucket:
    provider: "az"
    name: ""
    secretKey: ""
```

**기타 제공자 (Minio, Ceph 등)**

다른 S3 호환 제공자의 경우, 다음과 같이 버킷 설정을 환경 변수로 설정합니다:
```yaml
global:
  extraEnv:
    "BUCKET": "s3://wandb:changeme@mydb.com/wandb?tls=true"
```
변수는 다음 형식의 연결 문자열을 포함합니다:

```yaml
s3://$ACCESS_KEY:$SECRET_KEY@$HOST/$BUCKET_NAME
```

트러스트된 SSL 인증서를 오브젝트 스토어에 구성한 경우, TLS로만 연결하도록 W&B에 지시할 수 있습니다. 이를 위해, url에 `tls` 쿼리 매개변수를 추가하십시오:

```yaml
s3://$ACCESS_KEY:$SECRET_KEY@$HOST/$BUCKET_NAME?tls=true
```
:::caution
이것은 SSL 인증서가 신뢰할 수 있는 경우에만 작동합니다. W&B는 자체 서명된 인증서를 지원하지 않습니다.
:::

### MySQL

```yaml
global:
   mysql:
     # 예시 값, 본인의 것으로 교체하십시오
     database: wandb_local
     host: 10.218.0.2
     name: wandb_local
     password: 8wtX6cJH...ZcUarK4zZGjpV
     port: 3306
     user: wandb
```

### 라이선스

```yaml
global:
  # 예시 라이선스, 본인의 것으로 교체하십시오
  license: eyJhbGnUzaHgyQjQy...VFnPS_KETXg1hi
```

### Ingress

Ingress 클래스를 식별하려면, 이 FAQ [항목](#how-to-identify-the-kubernetes-ingress-class)을 참조하십시오.

**TLS 없이**

```yaml
global:
# 중요: Ingress는 YAML에서 ‘global’과 동일 수준에 있으며(자식이 아님)
ingress:
  class: ""
```

**TLS와 함께**

인증서를 포함한 시크릿을 만드십시오

```console
kubectl create secret tls wandb-ingress-tls --key wandb-ingress-tls.key --cert wandb-ingress-tls.crt
```

Ingress 설정에서 시크릿을 참조하십시오
```yaml
global:
# 중요: Ingress는 YAML에서 ‘global’과 동일 수준에 있으며(자식이 아님)
ingress:
  class: ""
  annotations:
    {}
    # kubernetes.io/ingress.class: nginx
    # kubernetes.io/tls-acme: "true"
  tls: 
    - secretName: wandb-ingress-tls
      hosts:
        - <HOST_URI>
```

Nginx의 경우 다음 주석을 추가해야 할 수 있습니다:

```
ingress:
  annotations:
    nginx.ingress.kubernetes.io/proxy-body-size: 64m
```

### 사용자 지정 Kubernetes ServiceAccounts

W&B pod를 실행하는 사용자 지정 Kubernetes 서비스 계정을 지정하십시오.

배포의 일부로 서비스 계정을 생성하는 아래 스니펫은 다음 이름을 명시합니다:

```yaml
app:
  serviceAccount:
    name: custom-service-account
    create: true

parquet:
  serviceAccount:
    name: custom-service-account
    create: true

global:
  ...
```
"app" 및 "parquet" 하위 시스템은 지정된 서비스 계정 하에서 실행됩니다. 다른 하위 시스템은 기본 서비스 계정 아래에서 실행됩니다.

클러스터에 이미 서비스 계정이 있는 경우 `create: false`를 설정하십시오:

```yaml
app:
  serviceAccount:
    name: custom-service-account
    create: false

parquet:
  serviceAccount:
    name: custom-service-account
    create: false
    
global:
  ...
```

앱, parquet, 콘솔 등과 같은 다양한 하위 시스템에서 서비스 계정을 지정할 수 있습니다:

```yaml
app:
  serviceAccount:
    name: custom-service-account
    create: true

console:
  serviceAccount:
    name: custom-service-account
    create: true

global:
  ...
```

하위 시스템 간에 서비스 계정이 다를 수 있습니다:

```yaml
app:
  serviceAccount:
    name: custom-service-account
    create: false

console:
  serviceAccount:
    name: another-custom-service-account
    create: true

global:
  ...
```

### 외부 Redis

```yaml
redis:
  install: false

global:
  redis:
    host: ""
    port: 6379
    password: ""
    parameters: {}
    caCert: ""
```

대안으로 Kubernetes 시크릿에 redis 비밀번호를 저장:

```console
kubectl create secret generic redis-secret --from-literal=redis-password=supersecret
```

아래 설정에서 참조하십시오:
```yaml
redis:
  install: false

global:
  redis:
    host: redis.example
    port: 9001
    auth:
      enabled: true
      secret: redis-secret
      key: redis-password
```

### LDAP
**TLS 없이**
```yaml
global:
  ldap:
    enabled: true
    # LDAP 서버 어드레스 ( "ldap://" 또는 "ldaps://" 포함)
    host:
    # 사용자를 찾기 위한 LDAP 검색 기준
    baseDN:
    # 바인딩할 LDAP 사용자 (익명 바인딩을 사용하는 경우 생략 가능)
    bindDN:
    # 바인딩할 LDAP 비밀번호를 가지고 있는 시크릿 이름과 키 (익명 바인딩을 사용하는 경우 생략 가능)
    bindPW:
    # 이메일의 LDAP 속성과 그룹 ID 속성 이름을 쉼표로 구분된 문자열 값으로 제공하십시오.
    attributes:
    # LDAP 그룹 허용 목록
    groupAllowList:
    # LDAP TLS 활성화
    tls: false
```

**TLS를 사용하는 경우**

LDAP TLS 인증서 설정에는 인증서 콘텐츠가 사전에 생성된 config map이 필요합니다.

다음 명령을 사용하여 config map을 생성할 수 있습니다:

```console
kubectl create configmap ldap-tls-cert --from-file=certificate.crt
```

YAML에서 config map을 다음과 같이 사용하십시오:

```yaml
global:
  ldap:
    enabled: true
    # LDAP 서버 어드레스 ( "ldap://" 또는 "ldaps://" 포함)
    host:
    # 사용자를 찾기 위한 LDAP 검색 기준
    baseDN:
    # 바인딩할 LDAP 사용자 (익명 바인딩을 사용하는 경우 생략 가능)
    bindDN:
    # 바인딩할 LDAP 비밀번호를 가지고 있는 시크릿 이름과 키 (익명 바인딩을 사용하는 경우 생략 가능)
    bindPW:
    # 이메일의 LDAP 속성과 그룹 ID 속성 이름을 쉼표로 구분된 문자열 값으로 제공하십시오.
    attributes:
    # LDAP 그룹 허용 목록
    groupAllowList:
    # LDAP TLS 활성화
    tls: true
    # LDAP 서버를 위한 CA 인증서가 포함된 ConfigMap 이름과 키
    tlsCert:
      configMap:
        name: "ldap-tls-cert"
        key: "certificate.crt"
```

### OIDC SSO

```yaml
global: 
  auth:
    sessionLengthHours: 720
    oidc:
      clientId: ""
      secret: ""
      authMethod: ""
      issuer: ""
```

### SMTP

```yaml
global:
  email:
    smtp:
      host: ""
      port: 587
      user: ""
      password: ""
```

### 환경 변수

```yaml
global:
  extraEnv:
    GLOBAL_ENV: "example"
```

### 사용자 정의 인증 기관
`customCACerts`는 목록이며 많은 인증서를 포함할 수 있습니다. `customCACerts`에 지정된 인증 기관은 W&B Server 애플리케이션에만 적용됩니다.

```yaml
global:
  customCACerts:
  - |
    -----BEGIN CERTIFICATE-----
    MIIBnDCCAUKgAwIBAg.....................fucMwCgYIKoZIzj0EAwIwLDEQ
    MA4GA1UEChMHSG9tZU.....................tZUxhYiBSb290IENBMB4XDTI0
    MDQwMTA4MjgzMFoXDT.....................oNWYggsMo8O+0mWLYMAoGCCqG
    SM49BAMCA0gAMEUCIQ.....................hwuJgyQRaqMI149div72V2QIg
    P5GD+5I+02yEp58Cwxd5Bj2CvyQwTjTO4hiVl1Xd0M0=
    -----END CERTIFICATE-----
  - |
    -----BEGIN CERTIFICATE-----
    MIIBxTCCAWugAwIB.....................qaJcwCgYIKoZIzj0EAwIwLDEQ
    MA4GA1UEChMHSG9t.......................tZUxhYiBSb290IENBMB4XDTI0
    MDQwMTA4MjgzMVoX.......................UK+moK4nZYvpNpqfvz/7m5wKU
    SAAwRQIhAIzXZMW4.......................E8UFqsCcILdXjAiA7iTluM0IU
    aIgJYVqKxXt25blH/VyBRzvNhViesfkNUQ==
    -----END CERTIFICATE-----
```

## W&B Operator 설정 참조

이 섹션은 W&B Kubernetes operator (`wandb-controller-manager`)의 구성 옵션을 설명합니다. operator는 YAML 파일의 형태로 구성을 받습니다.

기본적으로, W&B Kubernetes operator는 구성 파일이 필요하지 않습니다. 필요할 경우에만 구성 파일을 생성하십시오. 예를 들어, 사용자 정의 인증 기관을 지정하거나, 에어 갭 환경에 배포해야 하는 경우에 구성 파일이 필요할 수 있습니다.

spec 사용자 정의의 전체 목록은 Helm 저장소의 [여기](https://github.com/wandb/helm-charts/blob/main/charts/operator/values.yaml)에서 찾을 수 있습니다.

### Custom CA
사용자 지정 인증 기관 (`customCACerts`)은 목록이며 여러 인증서를 포함할 수 있습니다. 추가하면, 이러한 인증 기관은 W&B Kubernetes operator (`wandb-controller-manager`)에만 적용됩니다.

```yaml
customCACerts:
- |
  -----BEGIN CERTIFICATE-----
  MIIBnDCCAUKgAwIBAg.....................fucMwCgYIKoZIzj0EAwIwLDEQ
  MA4GA1UEChMHSG9tZU.....................tZUxhYiBSb290IENBMB4XDTI0
  MDQwMTA4MjgzMFoXDT.....................oNWYggsMo8O+0mWLYMAoGCCqG
  SM49BAMCA0gAMEUCIQ.....................hwuJgyQRaqMI149div72V2QIg
  P5GD+5I+02yEp58Cwxd5Bj2CvyQwTjTO4hiVl1Xd0M0=
  -----END CERTIFICATE-----
- |
  -----BEGIN CERTIFICATE-----
  MIIBxTCCAWugAwIB.....................qaJcwCgYIKoZIzj0EAwIwLDEQ
  MA4GA1UEChMHSG9t.......................tZUxhYiBSb290IENBMB4XDTI0
  MDQwMTA4MjgzMVoX.......................UK+moK4nZYvpNpqfvz/7m5wKU
  SAAwRQIhAIzXZMW4.......................E8UFqsCcILdXjAiA7iTluM0IU
  aIgJYVqKxXt25blH/VyBRzvNhViesfkNUQ==
  -----END CERTIFICATE-----
```

## FAQ

#### W&B Operator Console 비밀번호 얻는 방법
[W&B Kubernetes Operator 관리 콘솔 엑세스](#access-the-wb-management-console)를 참조하십시오.

#### Ingress가 작동하지 않을 경우 W&B Operator Console에 엑세스하는 방법

Kubernetes 클러스터에 엑세스할 수 있는 호스트에서 다음 명령을 실행하십시오:

```console
kubectl port-forward svc/wandb-console 8082
```

`https://localhost:8082/console`로 브라우저에서 콘솔에 엑세스하십시오.

비밀번호를 얻는 방법은 [W&B Kubernetes Operator 관리 콘솔 엑세스](#access-the-wb-management-console)의 옵션 2를 참조하십시오.

#### W&B Server 로그 보기 방법

애플리케이션 pod는 **wandb-app-xxx**로 명명됩니다.

```console
kubectl get pods
kubectl logs wandb-XXXXX-XXXXX
```

#### Kubernetes ingress 클래스 식별 방법

클러스터에 설치된 ingress 클래스를 확인하려면 다음을 실행하십시오:

```console
kubectl get ingressclass
```
