---
title: Kubernetes에서 W&B 서버 실행하기
description: Kubernetes Operator로 W&B 플랫폼 배포하기
menu:
  default:
    identifier: ko-guides-hosting-hosting-options-self-managed-kubernetes-operator-_index
    parent: self-managed
url: guides/hosting/operator
weight: 2
---

## W&B Kubernetes Operator

W&B Kubernetes Operator를 사용하면 Kubernetes에서 W&B Server 배포, 관리, 문제 해결 및 확장 작업을 간소화할 수 있습니다. 오퍼레이터는 W&B 인스턴스를 위한 스마트 어시스턴트라 생각할 수 있습니다.

W&B Server의 아키텍처와 설계는 AI 개발자 툴링 능력을 확장하고, 높은 성능, 더 나은 확장성, 쉬운 관리를 위한 적합한 기본 요소를 제공하는 방향으로 계속 발전하고 있습니다. 이 변화는 컴퓨팅 서비스뿐 아니라 관련 스토리지, 그리고 이들 간의 연결성 전반에 적용됩니다. 다양한 배포 방식에서 지속적인 업데이트와 개선을 용이하게 하기 위해서, W&B는 Kubernetes 오퍼레이터를 사용합니다.

{{% alert %}}
W&B는 오퍼레이터를 활용해 AWS, GCP, Azure와 같은 퍼블릭 클라우드에서 Dedicated cloud 인스턴스의 배포 및 관리를 담당합니다.
{{% /alert %}}

Kubernetes 오퍼레이터에 대한 더 자세한 정보는 [Operator 패턴](https://kubernetes.io/docs/concepts/extend-kubernetes/operator/)을 참조하세요.

### 아키텍처 변환의 이유
과거에는 W&B 애플리케이션이 Kubernetes 클러스터 내에 단일 배포 및 pod, 또는 단일 Docker 컨테이너 형태로 배포되었습니다. W&B는 데이터베이스와 Object Store(오브젝트 저장소)를 외부로 분리해서 사용하는 것을 꾸준히 권장해왔습니다. 이렇게 외부화함으로써 애플리케이션의 상태를 분리할 수 있습니다.

애플리케이션이 성장함에 따라, 모놀리식 컨테이너에서 분산 시스템(마이크로서비스) 형태로 진화할 필요가 명확해졌습니다. 이는 백엔드 로직 처리를 더욱 용이하게 하고, Kubernetes 인프라의 기본 기능을 자연스럽게 도입하는 데 도움이 되었습니다. 분산 시스템은 추가 기능에 필수적인 새로운 서비스 배포 역시 지원합니다.

2024년 이전에는 Kubernetes 관련 변경이 있을 때마다 [terraform-kubernetes-wandb](https://github.com/wandb/terraform-kubernetes-wandb) Terraform 모듈을 직접 수동으로 업데이트해야 했습니다. Terraform 모듈 업데이트는 클라우드 제공사 간 호환성 유지, 필요한 변수 설정 및 각 백엔드 또는 Kubernetes 수준의 변경에 대응하는 Terraform apply 실행을 의미합니다.

이 방식은 확장성에 한계가 있었고, W&B Support 팀이 각 고객의 Terraform 모듈 업그레이드에 일일이 지원해야 했습니다.

해결책으로는 중앙 [deploy.wandb.ai](https://deploy.wandb.ai) 서버와 연결하는 오퍼레이터를 도입하여, 주어진 릴리즈 채널의 최신 스펙 변경을 받아 적용하도록 했습니다. 라이선스가 유효한 한 업데이트가 자동으로 적용됩니다. [Helm](https://helm.sh/)은 W&B 오퍼레이터의 배포 방식이자, 오퍼레이터가 W&B Kubernetes 스택 전반의 설정 템플릿을 관리하는 메커니즘으로도 활용됩니다(Helm-ception).

### 작동 방식
오퍼레이터는 helm 또는 소스에서 직접 설치할 수 있습니다. 자세한 설치 방법은 [charts/operator](https://github.com/wandb/helm-charts/tree/main/charts/operator)에서 확인하세요.

설치 과정에서는 `controller-manager`라는 이름의 Deployment가 생성되며, `weightsandbiases.apps.wandb.com`이라는 이름(Custom Resource Definition, 약칭 `wandb`)의 [커스텀 리소스](https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/) 정의가 클러스터에 적용됩니다. 이 커스텀 리소스는 하나의 `spec`을 받아 클러스터에 적용합니다.

```yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: weightsandbiases.apps.wandb.com
```

`controller-manager`는 커스텀 리소스의 스펙, 릴리즈 채널, 사용자 정의 설정에 따라 [charts/operator-wandb](https://github.com/wandb/helm-charts/tree/main/charts/operator-wandb)를 설치합니다. 설정 사양의 계층 구조는 사용자에게 최대한의 설정 유연성을 제공하고, W&B는 새로운 이미지, 설정, 기능, Helm 업데이트를 자동으로 릴리스할 수 있습니다.

설정 옵션 관련해서는 [설정 사양 계층]({{< relref path="#configuration-specification-hierarchy" lang="ko" >}})과 [설정 레퍼런스]({{< relref path="#configuration-reference-for-wb-operator" lang="ko" >}})를 참고하세요.

배포 환경은 각 서비스별로 개별 pod 형태로 이루어져 있으며, 각 pod 이름은 `wandb-`로 시작합니다.

### 설정 사양 계층
설정 사양은 계층 구조를 따릅니다. 상위 레벨 사양이 하위 레벨 사양을 덮어씁니다. 작동 원리는 다음과 같습니다.

- **Release Channel 값**: 이 기본 설정은 W&B가 정한 릴리즈 채널을 기준으로 기본값 및 세팅을 적용합니다.
- **사용자 입력 값**: 사용자는 System Console을 통해 Release Channel 사양의 기본값을 재정의할 수 있습니다.
- **커스텀 리소스 값**: 사용자가 직접 정의하는 최고 수준의 사양으로, 여기 지정된 값이 User Input 및 Release Channel의 모든 사양을 덮어씁니다. 더 상세한 내용은 [설정 레퍼런스]({{< relref path="#configuration-reference-for-wb-operator" lang="ko" >}})에서 확인하세요.

이 계층 구조는 다양한 필요에 맞게 유연하게 설정을 맞추는 동시에, 업그레이드나 변경에 대해 관리가 가능한 체계를 보장합니다.

### W&B Kubernetes Operator 사용 전제 조건
W&B Kubernetes Operator로 W&B를 배포하려면 다음 요건을 충족해야 합니다.

- [참조 아키텍처]({{< relref path="../ref-arch.md#infrastructure-requirements" lang="ko" >}}) 확인
- [유효한 W&B Server 라이선스 확보]({{< relref path="../#obtain-your-wb-server-license" lang="ko" >}})

자세한 셀프 매니지드 설치 방법은 [bare-metal 설치 가이드]({{< relref path="../bare-metal.md" lang="ko" >}})를 참고하세요.

설치 방식에 따라 추가로 다음 요건을 충족해야 할 수 있습니다:
* Kubectl이 설치되어 있고, 올바른 Kubernetes 클러스터 컨텍스트로 설정되어 있어야 합니다.
* Helm이 설치되어 있어야 합니다.

### 에어갭드(air-gapped) 설치
에어갭(내부망) 환경에서 W&B Kubernetes Operator 설치 방법은 [Kubernetes로 에어갭 환경에 W&B 배포하기]({{< relref path="operator-airgapped.md" lang="ko" >}}) 튜토리얼을 참고하세요.

## W&B Server 애플리케이션 배포
이 섹션에서는 W&B Kubernetes Operator를 배포하는 여러 방법을 설명합니다.
{{% alert %}}
W&B Operator는 W&B Server 설치의 기본이자 권장 방법입니다.
{{% /alert %}}

### Helm CLI로 W&B 배포
W&B는 Kubernetes 클러스터에 W&B Kubernetes Operator를 배포할 수 있도록 Helm Chart를 제공합니다. 이를 통해 Helm CLI 또는 ArgoCD와 같은 CI/CD 툴을 활용해 W&B Server를 배포할 수 있습니다. 위에서 언급한 필수 요건이 충족되었는지 확인하세요.

다음 단계에 따라 Helm CLI로 W&B Kubernetes Operator를 설치할 수 있습니다:

1. W&B Helm 저장소 추가. W&B Helm 차트는 W&B Helm 저장소에 있습니다.
    ```shell
    helm repo add wandb https://charts.wandb.ai
    helm repo update
    ```
2. Kubernetes 클러스터에 Operator 설치:
    ```shell
    helm upgrade --install operator wandb/operator -n wandb-cr --create-namespace
    ```
3. W&B operator 커스텀 리소스 설정을 통해 W&B Server 설치 트리거. Helm `values.yaml` 파일로 기본 값 오버라이드 또는 CRD(커스텀 리소스 정의)를 직접 커스터마이즈할 수 있습니다.

    - **`values.yaml` 오버라이드**(권장): 전체 [values.yaml](https://github.com/wandb/helm-charts/blob/main/charts/operator-wandb/values.yaml)에서 오버라이드하고 싶은 키만 포함해 새로운 `values.yaml` 파일을 만듭니다. 예를 들어 MySQL을 지정하려면:

      {{< prism file="/operator/values_mysql.yaml" title="values.yaml">}}{{< /prism >}}
    - **전체 CRD**: [예시 설정](https://github.com/wandb/helm-charts/blob/main/charts/operator/crds/wandb.yaml)을 복사해 새로운 `operator.yaml` 파일을 만들고 필요한 내용을 수정합니다. [설정 레퍼런스]({{< relref path="#configuration-reference-for-wb-operator" lang="ko" >}}) 참고.

      {{< prism file="/operator/wandb.yaml" title="operator.yaml">}}{{< /prism >}}

4. 커스텀 설정으로 Operator를 시작하여 W&B Server 애플리케이션의 설치, 설정 및 관리를 진행합니다.

    - `values.yaml` 오버라이드로 Operator를 시작하려면:

        ```shell
        kubectl apply -f values.yaml
        ```
    - 완전히 커스터마이즈된 CRD로 Operator를 시작하려면:
      ```shell
      kubectl apply -f operator.yaml
      ```

    배포가 완료될 때까지 잠시 기다리세요(몇 분 소요).

5. 웹 UI에서 설치를 확인하려면 첫 번째 admin 사용자 계정을 만든 후, [설치 확인]({{< relref path="#verify-the-installation" lang="ko" >}}) 절차를 따라주세요.


### Helm Terraform Module로 W&B 배포

이 방법은 인프라 코드(IaC) 방식의 Terraform을 통해 반복성과 일관성을 갖춘 맞춤형 배포를 지원합니다. 공식 W&B Helm 기반 Terraform Module은 [여기](https://registry.terraform.io/modules/wandb/wandb/helm/latest)에서 확인할 수 있습니다.

다음 코드는 프로덕션 수준 배포에 필요한 모든 기본 옵션을 포함한 예시입니다.

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

설정 옵션은 [설정 레퍼런스]({{< relref path="#configuration-reference-for-wb-operator" lang="ko" >}})와 동일하지만, 문법은 HashiCorp Configuration Language(HCL)에 맞게 작성해야 합니다. Terraform 모듈은 W&B 커스텀 리소스 정의(CRD)를 생성합니다.

W&B가 실제로 헬름 Terraform 모듈을 사용해 고객별 “Dedicated cloud” 인스턴스를 배포하는 과정은 아래 링크를 참고하세요:
- [AWS](https://github.com/wandb/terraform-aws-wandb/blob/45e1d746f53e78e73e68f911a1f8cad5408e74b6/main.tf#L225)
- [Azure](https://github.com/wandb/terraform-azurerm-wandb/blob/170e03136b6b6fc758102d59dacda99768854045/main.tf#L155)
- [GCP](https://github.com/wandb/terraform-google-wandb/blob/49ddc3383df4cefc04337a2ae784f57ce2a2c699/main.tf#L189)

### W&B Cloud Terraform modules로 W&B 배포

W&B는 AWS, GCP, Azure용 Terraform Module을 제공합니다. 이 모듈들은 Kubernetes 클러스터, 로드밸런서, MySQL 데이터베이스 등 전체 인프라와 W&B Server 애플리케이션을 한 번에 구축합니다. 오피셜 W&B 클라우드별 Terraform Module에는 이미 W&B Kubernetes Operator가 포함되어 있습니다.

| Terraform Registry                                                  | Source Code                                      | Version |
| ------------------------------------------------------------------- | ------------------------------------------------ | ------- |
| [AWS](https://registry.terraform.io/modules/wandb/wandb/aws/latest) | https://github.com/wandb/terraform-aws-wandb     | v4.0.0+ |
| [Azure](https://github.com/wandb/terraform-azurerm-wandb)           | https://github.com/wandb/terraform-azurerm-wandb | v2.0.0+ |
| [GCP](https://github.com/wandb/terraform-google-wandb)              | https://github.com/wandb/terraform-google-wandb  | v2.0.0+ |

이 통합 덕분에 최소한의 추가 설정만으로도 W&B Kubernetes Operator와 함께 W&B Server를 클라우드 환경에 손쉽게 배포하고 관리할 수 있습니다.

이들 모듈에 대한 자세한 사용법은 [셀프 매니지드 설치 섹션]({{< relref path="../#deploy-wb-server-within-self-managed-cloud-accounts" lang="ko" >}})을 참고하세요.

### 설치 확인

설치가 정상적으로 되었는지 확인하려면 [W&B CLI]({{< relref path="/ref/cli/" lang="ko" >}}) 사용을 권장합니다. verify 명령은 각종 컴포넌트와 설정 상태를 자동으로 점검합니다.

{{% alert %}}
이 단계는 브라우저에서 첫째 admin 사용자 계정이 생성된 상태임을 전제로 합니다.
{{% /alert %}}

설치 확인 절차:

1. W&B CLI 설치
    ```shell
    pip install wandb
    ```
2. W&B 로그인
    ```shell
    wandb login --host=https://YOUR_DNS_DOMAIN
    ```

    예시:
    ```shell
    wandb login --host=https://wandb.company-name.com
    ```

3. 설치 확인
    ```shell
    wandb verify
    ```

W&B가 성공적으로 배포된 경우 다음과 비슷한 출력이 나타납니다:

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

## W&B 관리 콘솔 접속
W&B Kubernetes Operator에는 관리 콘솔이 함께 제공됩니다. 주소는 `${HOST_URI}/console`, 예시로 `https://wandb.company-name.com/console` 입니다.

관리 콘솔에 로그인하는 방법은 두 가지가 있습니다:

{{< tabpane text=true >}}
{{% tab header="옵션 1 (권장)" value="option1" %}}
1. 브라우저에서 W&B 애플리케이션에 로그인합니다. `${HOST_URI}/`, 예: `https://wandb.company-name.com/`
2. 콘솔 접속: 오른쪽 상단 아이콘 클릭 → **System console** 선택. 관리자 권한이 있는 사용자만 **System console** 항목이 보입니다.

    {{< img src="/images/hosting/access_system_console_via_main_app.png" alt="System console access" >}}
{{% /tab %}}

{{% tab header="옵션 2" value="option2"%}}
{{% alert %}}
Option 1이 정상적으로 작동하지 않을 때만 아래 절차로 콘솔에 접속할 것을 권장합니다.
{{% /alert %}}

1. 브라우저에서 콘솔 애플리케이션 직접 열기. 위 URL로 접속하면 로그인 화면으로 리다이렉트 됩니다.
    {{< img src="/images/hosting/access_system_console_directly.png" alt="Direct system console access" >}}
2. Kubernetes 시크릿에서 설치 중 생성된 비밀번호를 조회합니다:
    ```shell
    kubectl get secret wandb-password -o jsonpath='{.data.password}' | base64 -d
    ```
    복사한 비밀번호를 이용하세요.
3. 콘솔에 로그인. 비밀번호를 붙여넣고 **Login**을 클릭합니다.
{{% /tab %}}
{{< /tabpane >}}

## W&B Kubernetes Operator 업데이트
이 섹션은 W&B Kubernetes Operator를 업데이트하는 방법을 안내합니다.

{{% alert %}}
* W&B Kubernetes Operator 업데이트는 W&B Server 애플리케이션을 업데이트하지 않습니다.
* 이전에 W&B Kubernetes Operator를 사용하지 않는 Helm Chart로 설치했다면 아래 Operator 업데이트 전에 [여기]({{< relref path="#migrate-self-managed-instances-to-wb-operator" lang="ko" >}})의 안내를 먼저 참고하세요.
{{% /alert %}}

아래 코드 조각을 터미널에 붙여넣어 실행하세요.

1. 먼저 repo를 최신으로 업데이트합니다[`helm repo update`](https://helm.sh/docs/helm/helm_repo_update/):
    ```shell
    helm repo update
    ```

2. Helm chart를 최신으로 업데이트합니다[`helm upgrade`](https://helm.sh/docs/helm/helm_upgrade/):
    ```shell
    helm upgrade operator wandb/operator -n wandb-cr --reuse-values
    ```

## W&B Server 애플리케이션 업데이트
W&B Kubernetes Operator를 사용하는 경우 W&B Server 애플리케이션을 따로 수동으로 업데이트할 필요가 없습니다.

오퍼레이터는 신규 버전의 W&B 소프트웨어가 릴리즈될 때마다 자동으로 W&B Server 애플리케이션을 업데이트합니다.

## 셀프 매니지드 인스턴스를 W&B Operator로 마이그레이션
이 절에서는 직접 관리하던 W&B Server 설치를 Operator 기반 관리로 전환하는 방법을 설명합니다. 마이그레이션 과정은 기존 W&B Server 설치 방법에 따라 달라집니다.

{{% alert %}}
W&B Operator는 W&B Server 설치의 기본이자 권장 방법입니다. 궁금한 점이 있으면 [고객 지원](mailto:support@wandb.com) 또는 W&B 담당 팀에 문의하세요.
{{% /alert %}}

- 공식 W&B Cloud Terraform Modules를 이용했다면 해당 문서에 따라 진행하세요:
  - [AWS]({{< relref path="#migrate-to-operator-based-aws-terraform-modules" lang="ko" >}})
  - [GCP]({{< relref path="#migrate-to-operator-based-gcp-terraform-modules" lang="ko" >}})
  - [Azure]({{< relref path="#migrate-to-operator-based-azure-terraform-modules" lang="ko" >}})
- [W&B Non-Operator Helm chart](https://github.com/wandb/helm-charts/tree/main/charts/wandb)로 설치했다면 [여기]({{< relref path="#migrate-to-operator-based-helm-chart" lang="ko" >}})로 이동하세요.
- [W&B Non-Operator Helm chart + Terraform](https://registry.terraform.io/modules/wandb/wandb/kubernetes/latest) 을 사용했다면 [여기]({{< relref path="#migrate-to-operator-based-terraform-helm-chart" lang="ko" >}})로 이동하세요.
- Kubernetes 매니페스트로 리소스를 직접 만들었다면 [여기]({{< relref path="#migrate-to-operator-based-helm-chart" lang="ko" >}})로 이동하세요.


### Operator 기반 AWS Terraform Modules로 마이그레이션

자세한 마이그레이션 절차는 [여기]({{< relref path="../install-on-public-cloud/aws-tf.md#migrate-to-operator-based-aws-terraform-modules" lang="ko" >}})에서 확인하세요.

### Operator 기반 GCP Terraform Modules로 마이그레이션

궁금한 사항이나 도움이 필요할 경우 [고객 지원](mailto:support@wandb.com) 또는 W&B 담당 팀에 문의하세요.

### Operator 기반 Azure Terraform Modules로 마이그레이션

궁금한 사항이나 도움이 필요할 경우 [고객 지원](mailto:support@wandb.com) 또는 W&B 담당 팀에 문의하세요.

### Operator 기반 Helm chart로 마이그레이션

다음 절차에 따라 Helm 차트를 Operator 기반으로 전환하세요:

1. 현재 W&B 설정값 추출. Non-Operator 버전의 Helm chart로 배포했다면 다음 명령으로 export:
    ```shell
    helm get values wandb
    ```
    Kubernetes 매니페스트로 배포했다면:
    ```shell
    kubectl get deployment wandb -o yaml
    ```
    필요한 설정값을 확보합니다. 

2. `operator.yaml` 파일 생성. [설정 레퍼런스]({{< relref path="#configuration-reference-for-wb-operator" lang="ko" >}}) 형식을 참고해 1번에서 추출한 값을 입력합니다.

3. 기존 배포를 0 pod로 scale. 즉, 현재 배포가 중지됩니다.
    ```shell
    kubectl scale --replicas=0 deployment wandb
    ```
4. Helm 차트 저장소 업데이트:
    ```shell
    helm repo update
    ```
5. 새 Helm 차트 설치:
    ```shell
    helm upgrade --install operator wandb/operator -n wandb-cr --create-namespace
    ```
6. 새 Helm 차트를 설정하고 W&B 애플리케이션 배포 트리거. 새 설정을 적용합니다.
    ```shell
    kubectl apply -f operator.yaml
    ```
    배포 완료까지 몇 분 정도 소요됩니다.

7. 설치 확인. [설치 확인]({{< relref path="#verify-the-installation" lang="ko" >}}) 가이드를 따라 모든 것이 정상 작동하는지 확인합니다.

8. 기존 설치 제거. 구 Helm 차트 삭제 혹은 매니페스트로 만든 리소스 삭제.

### Operator 기반 Terraform Helm chart로 마이그레이션

다음 절차에 따라 Helm chart 기반 Operator로 전환하세요:

1. Terraform 설정 준비. 기존 배포 코드를 [여기]({{< relref path="#deploy-wb-with-helm-terraform-module" lang="ko" >}})의 코드로 교체하고, 기존과 동일한 변수를 지정하세요. .tfvars 파일은 변경하지 않아도 됩니다.
2. Terraform 실행. `terraform init`, `plan`, `apply`를 진행.
3. 설치 확인. [설치 확인]({{< relref path="#verify-the-installation" lang="ko" >}}) 절차로 정상 동작 여부 확인.
4. 기존 설치 제거. 구 Helm 차트 삭제 또는 매니페스트 리소스 삭제.


## W&B Server 설정 레퍼런스

이 섹션에서는 W&B Server 애플리케이션의 설정 옵션을 설명합니다. 애플리케이션은 커스텀 리소스 정의인 [WeightsAndBiases]({{< relref path="#how-it-works" lang="ko" >}})로 설정을 받습니다. 일부 설정 옵션은 아래와 같이 설정에서 직접 처리하며, 나머지는 환경 변수로 지정해야 합니다.

환경 변수 목록은 [기본]({{< relref path="/guides/hosting/env-vars/" lang="ko" >}}), [고급]({{< relref path="/guides/hosting/iam/advanced_env_vars/" lang="ko" >}}) 공개되어 있습니다. 환경 변수로만 지정 가능한 항목은 Helm Chart 설정에서 노출되지 않은 경우에만 활용하세요.

프로덕션 배포를 위한 W&B Server 애플리케이션 설정 파일에는 아래와 같은 내용이 필요합니다. 이 YAML 파일은 W&B 배포의 버전, 환경 변수, 외부 자원(데이터베이스 등), 기타 필수 설정의 대상 상태를 정의합니다.

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

완전한 값 목록은 [W&B Helm 저장소](https://github.com/wandb/helm-charts/blob/main/charts/operator-wandb/values.yaml)에서 확인할 수 있으며, 오버라이드가 필요한 값만 변경하세요.

### 전체 예시
다음은 GCP Kubernetes, GCP Ingress, GCS(GCP Object storage)를 사용하는 예시 설정입니다.

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

### Host
```yaml
 # FQDN(프로토콜 포함) 설정
global:
  # 예시 호스트명. 실제 값으로 교체하세요.
  host: https://wandb.example.com
```

### 오브젝트 스토리지 (bucket)

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

**기타 공급자(Minio, Ceph 등)**

기타 S3 호환 스토리지 사용 시:
```yaml
global:
  bucket:
    # 예시 값, 실제로 교체하세요
    provider: s3
    name: storage.example.com
    kmsKey: null
    path: wandb
    region: default
    accessKey: 5WOA500...P5DK7I
    secretKey: HDKYe4Q...JAp1YyjysnX
```

AWS 외부 호스팅 S3 호환 스토리지라면 `kmsKey`를 `null`로 설정해야 합니다.

시크릿에서 `accessKey`와 `secretKey`를 참조하려면:
```yaml
global:
  bucket:
    # 예시 값, 실제로 교체하세요
    provider: s3
    name: storage.example.com
    kmsKey: null
    path: wandb
    region: default
    secret:
      secretName: bucket-secret
      accessKeyName: ACCESS_KEY
      secretKeyName: SECRET_KEY
```

### MySQL

```yaml
global:
   mysql:
     # 예시 값, 실제로 교체하세요
     host: db.example.com
     port: 3306
     database: wandb_local
     user: wandb
     password: 8wtX6cJH...ZcUarK4zZGjpV 
```

비밀번호를 시크릿에서 참조하려면:
```yaml
global:
   mysql:
     # 예시 값, 실제로 교체하세요
     host: db.example.com
     port: 3306
     database: wandb_local
     user: wandb
     passwordSecret:
       name: database-secret
       passwordKey: MYSQL_WANDB_PASSWORD
```

### 라이선스

```yaml
global:
  # 예시 라이선스. 실제 값으로 교체하세요.
  license: eyJhbGnUzaHgyQjQy...VFnPS_KETXg1hi
```

시크릿에서 `license`를 참조하려면:
```yaml
global:
  licenseSecret:
    name: license-secret
    key: CUSTOMER_WANDB_LICENSE
```

### Ingress

Ingress 클래스를 식별하려면 [FAQ]({{< relref path="#how-to-identify-the-kubernetes-ingress-class" lang="ko" >}})를 참고하세요.

**TLS 없이**

```yaml
global:
# 중요: ingress는 YAML에서 global과 같은 레벨(하위 항목이 아님)
ingress:
  class: ""
```

**TLS 포함**

다음과 같이 인증서가 포함된 시크릿 생성

```console
kubectl create secret tls wandb-ingress-tls --key wandb-ingress-tls.key --cert wandb-ingress-tls.crt
```

Ingress 설정에서 해당 시크릿 참조
```yaml
global:
# 중요: ingress는 YAML에서 global과 같은 레벨(하위 항목이 아님)
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

Nginx의 경우 다음 주석을 추가해야 할 수도 있습니다:

```
ingress:
  annotations:
    nginx.ingress.kubernetes.io/proxy-body-size: 64m
```

### 커스텀 Kubernetes ServiceAccount

W&B pod가 사용할 커스텀 service account를 지정할 수 있습니다.

아래 예는 지정한 이름의 서비스 계정을 함께 생성합니다:

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
"app", "parquet" 서브시스템이 지정한 서비스 계정으로 실행됩니다. 나머지 서브시스템은 디폴트 서비스 계정을 사용합니다.

서비스 계정이 이미 있을 경우, `create: false`로 설정:

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

app, parquet, console 등 각 서브시스템별로 서비스 계정 지정 가능:

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

서브시스템마다 서로 다른 서비스 계정 사용도 가능합니다:

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

시크릿에서 `password` 참조 예시:

```console
kubectl create secret generic redis-secret --from-literal=redis-password=supersecret
```

설정에서 참조:
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
    # "ldap://" 또는 "ldaps://" 포함 서버 어드레스
    host:
    # 사용자 검색 대상 LDAP search base
    baseDN:
    # 바인드에 사용할 LDAP 계정(익명 바인딩이 아닌 경우)
    bindDN:
    # 바인드 비밀번호를 담는 시크릿 이름 및 키(익명 바인딩이 아닌 경우)
    bindPW:
    # 이메일 및 그룹 ID 속성명, 콤마 구분
    attributes:
    # 허용 그룹 리스트
    groupAllowList:
    # LDAP TLS 사용 여부
    tls: false
```

**TLS 사용시**

LDAP TLS 인증서 세팅을 위해서는 인증서 내용이 담긴 configmap이 먼저 생성되어야 합니다.

```console
kubectl create configmap ldap-tls-cert --from-file=certificate.crt
```

아래와 같이 YAML에 사용:

```yaml
global:
  ldap:
    enabled: true
    # "ldap://" 또는 "ldaps://" 포함 서버 어드레스
    host:
    # 사용자 검색 대상 LDAP search base
    baseDN:
    # 바인드에 사용할 LDAP 계정(익명 바인딩이 아닌 경우)
    bindDN:
    # 바인드 비밀번호를 담는 시크릿 이름 및 키(익명 바인딩이 아닌 경우)
    bindPW:
    # 이메일 및 그룹 ID 속성명, 콤마 구분
    attributes:
    # 허용 그룹 리스트
    groupAllowList:
    # LDAP TLS 사용 여부
    tls: true
    # LDAP 서버용 CA 인증서가 들어 있는 ConfigMap 명과 키
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
      # IdP에서 요구하는 경우에만
      authMethod: ""
      issuer: ""
```

`authMethod`는 선택 사항입니다.

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

### 환경 변수 설정
```yaml
global:
  extraEnv:
    GLOBAL_ENV: "example"
```

### 커스텀 인증 기관(CA) 지정
`customCACerts`는 여러 개의 인증서를 받을 수 있는 리스트입니다. 해당 인증 기관은 W&B Server 애플리케이션에만 적용됩니다.

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
    MIIBxTCCAWugAwIB.......................qaJcwCgYIKoZIzj0EAwIwLDEQ
    MA4GA1UEChMHSG9t.......................tZUxhYiBSb290IENBMB4XDTI0
    MDQwMTA4MjgzMVoX.......................UK+moK4nZYvpNpqfvz/7m5wKU
    SAAwRQIhAIzXZMW4.......................E8UFqsCcILdXjAiA7iTluM0IU
    aIgJYVqKxXt25blH/VyBRzvNhViesfkNUQ==
    -----END CERTIFICATE-----
```

ConfigMap으로도 CA 인증서를 관리할 수 있습니다:
```yaml
global:
  caCertsConfigMap: custom-ca-certs
```

ConfigMap 예시:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: custom-ca-certs
data:
  ca-cert1.crt: |
    -----BEGIN CERTIFICATE-----
    ...
    -----END CERTIFICATE-----
  ca-cert2.crt: |
    -----BEGIN CERTIFICATE-----
    ...
    -----END CERTIFICATE-----
```

{{% alert %}}
ConfigMap의 각 키 이름은 반드시 `.crt`(예: `my-cert.crt`, `ca-cert1.crt`)로 끝나야 합니다. 이는 `update-ca-certificates`가 시스템 CA 저장소에 인증서를 정상적으로 추가하기 위한 요건입니다.
{{% /alert %}}

### 커스텀 보안 컨텍스트

모든 W&B 컴포넌트는 아래와 같이 커스텀 보안 컨텍스트를 지원합니다:

```yaml
pod:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1001
    runAsGroup: 0
    fsGroup: 1001
    fsGroupChangePolicy: Always
    seccompProfile:
      type: RuntimeDefault
container:
  securityContext:
    capabilities:
      drop:
        - ALL
    readOnlyRootFilesystem: false
    allowPrivilegeEscalation: false 
```

{{% alert %}}
`runAsGroup:`의 유효한 값은 `0`뿐입니다. 다른 값은 에러가 발생합니다.
{{% /alert %}}


예를 들어 애플리케이션 pod에 적용하려면 `app` 섹션을 추가하세요:

```yaml
global:
  ...
app:
  pod:
    securityContext:
      runAsNonRoot: true
      runAsUser: 1001
      runAsGroup: 0
      fsGroup: 1001
      fsGroupChangePolicy: Always
      seccompProfile:
        type: RuntimeDefault
  container:
    securityContext:
      capabilities:
        drop:
          - ALL
      readOnlyRootFilesystem: false
      allowPrivilegeEscalation: false 
```

이 개념은 `console`, `weave`, `weave-trace`, `parquet`에도 동일하게 적용됩니다.

## W&B Operator 설정 레퍼런스

이 섹션은 W&B Kubernetes Operator(`wandb-controller-manager`)의 설정 옵션을 다룹니다. 오퍼레이터는 설정 파일을 YAML 형태로 받습니다.

기본적으로 W&B Kubernetes Operator는 별도의 설정 파일 없이 사용할 수 있습니다. 필요할 경우에만 생성하세요(예: 커스텀 인증기관 지정, 에어갭 환경 배포 등).

전체 스펙 커스터마이즈 내용은 [Helm 저장소](https://github.com/wandb/helm-charts/blob/main/charts/operator/values.yaml)에서 확인하세요.

### 커스텀 CA
커스텀 인증기관(`customCACerts`)은 여러 개 인증서를 리스트 형태로 지정하며, 이는 W&B Kubernetes Operator(`wandb-controller-manager`)에만 적용됩니다.

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
  MIIBxTCCAWugAwIB.......................qaJcwCgYIKoZIzj0EAwIwLDEQ
  MA4GA1UEChMHSG9t.......................tZUxhYiBSb290IENBMB4XDTI0
  MDQwMTA4MjgzMVoX.......................UK+moK4nZYvpNpqfvz/7m5wKU
  SAAwRQIhAIzXZMW4.......................E8UFqsCcILdXjAiA7iTluM0IU
  aIgJYVqKxXt25blH/VyBRzvNhViesfkNUQ==
  -----END CERTIFICATE-----
```

ConfigMap 사용 예시:
```yaml
caCertsConfigMap: custom-ca-certs
```

ConfigMap 예시:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: custom-ca-certs
data:
  ca-cert1.crt: |
    -----BEGIN CERTIFICATE-----
    ...
    -----END CERTIFICATE-----
  ca-cert2.crt: |
    -----BEGIN CERTIFICATE-----
    ...
    -----END CERTIFICATE-----
```

{{% alert %}}
ConfigMap의 각 키 이름은 반드시 `.crt`로 끝나야 하며(예: `my-cert.crt`, `ca-cert1.crt`), 이는 `update-ca-certificates`가 각 인증서를 시스템 CA 저장소에 정상적으로 적용하기 위한 필수 요건입니다.
{{% /alert %}}

## FAQ

### 각 pod의 역할/목적은 무엇인가요?
* **`wandb-app`**: W&B의 코어(핵심) 서비스로, GraphQL API와 프론트엔드 애플리케이션이 포함됩니다. 플랫폼 주요 기능을 담당합니다.
* **`wandb-console`**: 관리 콘솔로, `/console` 경로로 접속됩니다.
* **`wandb-otel`**: OpenTelemetry 에이전트로, Kubernetes 레이어 자원에서 메트릭과 로그를 수집하여 관리 콘솔에 표시합니다.
* **`wandb-prometheus`**: Prometheus 서버로, 다양한 요소에서 메트릭을 수집해 관리 콘솔에서 보여줍니다.
* **`wandb-parquet`**: 데이터베이스 데이터를 parquet 형식으로 오브젝트 스토리지에 내보내는 백엔드 마이크로서비스입니다(별도 pod).
* **`wandb-weave`**: UI상에서 쿼리 테이블을 로드하고 기본 기능을 지원하는 또 다른 백엔드 마이크로서비스.
* **`wandb-weave-trace`**: LLM 기반 애플리케이션의 추적, 실험, 평가, 배포, 개선을 위한 프레임워크로, `wandb-app` pod에서 접근합니다.

### W&B Operator 콘솔 비밀번호 가져오기
[W&B Kubernetes Operator 관리 콘솔 접속 방법]({{< relref path="#access-the-wb-management-console" lang="ko" >}})을 참고하세요.

### Ingress가 동작하지 않을 때 W&B Operator 콘솔 접속법

Kubernetes 클러스터에 연결 가능한 호스트에서 다음 명령을 실행하세요:

```console
kubectl port-forward svc/wandb-console 8082
```

브라우저에서 `https://localhost:8082/`로 접속하면 콘솔 이용이 가능합니다.

비밀번호는 [W&B Kubernetes Operator 관리 콘솔 접속 방법]({{< relref path="#access-the-wb-management-console" lang="ko" >}})의 Option 2 안내를 참고하세요.

### W&B Server 로그 보는 법

애플리케이션 pod 이름은 **wandb-app-xxx**입니다.

```console
kubectl get pods
kubectl logs wandb-XXXXX-XXXXX
```

### Kubernetes ingress class 확인 방법

클러스터에 설치된 ingress class 목록은

```console
kubectl get ingressclass
``` 

명령으로 조회할 수 있습니다.