---
title: Run W&B Server on Kubernetes
description: Kubernetes Operator로 W&B 플랫폼 배포하기
menu:
  default:
    identifier: ko-guides-hosting-hosting-options-self-managed-kubernetes-operator-_index
    parent: self-managed
url: guides/hosting/operator
weight: 2
---

## W&B Kubernetes Operator

W&B Kubernetes Operator를 사용하여 Kubernetes에서 W&B Server 배포를 간소화하고 관리, 문제 해결 및 확장할 수 있습니다. 이 operator는 W&B 인스턴스를 위한 스마트 비서라고 생각할 수 있습니다.

W&B Server 아키텍처와 디자인은 AI 개발자 툴링 기능을 확장하고 고성능, 더 나은 확장성, 더 쉬운 관리를 위한 적절한 기본 요소를 제공하기 위해 지속적으로 발전하고 있습니다. 이러한 발전은 컴퓨팅 서비스, 관련 스토리지 및 이들 간의 연결에 적용됩니다. 모든 배포 유형에서 지속적인 업데이트와 개선을 용이하게 하기 위해 W&B 는 Kubernetes operator를 사용합니다.

{{% alert %}}
W&B 는 operator를 사용하여 AWS, GCP 및 Azure 퍼블릭 클라우드에 전용 클라우드 인스턴스를 배포하고 관리합니다.
{{% /alert %}}

Kubernetes operator에 대한 자세한 내용은 Kubernetes 설명서의 [Operator 패턴](https://kubernetes.io/docs/concepts/extend-kubernetes/operator/)을 참조하십시오.

### 아키텍처 전환 이유

과거에는 W&B 애플리케이션이 Kubernetes 클러스터 또는 단일 Docker 컨테이너 내에서 단일 배포 및 pod로 배포되었습니다. W&B 는 데이터베이스와 오브젝트 저장소를 외부화할 것을 권장해 왔으며 앞으로도 계속 권장할 것입니다. 데이터베이스와 오브젝트 저장소를 외부화하면 애플리케이션의 상태가 분리됩니다.

애플리케이션이 성장함에 따라 모놀리식 컨테이너에서 분산 시스템(마이크로서비스)으로 발전해야 할 필요성이 분명해졌습니다. 이러한 변경은 백엔드 로직 처리를 용이하게 하고 기본 제공 Kubernetes 인프라 기능을 원활하게 도입합니다. 분산 시스템은 또한 W&B 가 의존하는 추가 기능에 필수적인 새로운 서비스 배포를 지원합니다.

2024년 이전에는 Kubernetes 관련 변경 사항이 있을 때마다 [terraform-kubernetes-wandb](https://github.com/wandb/terraform-kubernetes-wandb) Terraform 모듈을 수동으로 업데이트해야 했습니다. Terraform 모듈을 업데이트하면 클라우드 공급자 간의 호환성이 보장되고 필요한 Terraform 변수가 구성되며 각 백엔드 또는 Kubernetes 수준 변경에 대해 Terraform 적용이 실행됩니다.

W&B 지원팀에서 각 고객의 Terraform 모듈 업그레이드를 지원해야 했기 때문에 이 프로세스는 확장 가능하지 않았습니다.

솔루션은 중앙 [deploy.wandb.ai](https://deploy.wandb.ai) 서버에 연결하여 특정 릴리스 채널에 대한 최신 사양 변경 사항을 요청하고 적용하는 operator를 구현하는 것이었습니다. 라이선스가 유효한 한 업데이트가 수신됩니다. [Helm](https://helm.sh/)은 W&B operator의 배포 메커니즘이자 operator가 W&B Kubernetes 스택의 모든 설정 템플릿을 처리하는 수단인 Helm-ception으로 사용됩니다.

### 작동 방식

helm 또는 소스에서 operator를 설치할 수 있습니다. 자세한 내용은 [charts/operator](https://github.com/wandb/helm-charts/tree/main/charts/operator)를 참조하십시오.

설치 프로세스는 `controller-manager`라는 배포를 생성하고 단일 `spec`을 가져와 클러스터에 적용하는 [custom resource](https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/) 정의 `weightsandbiases.apps.wandb.com`(약칭: `wandb`)를 사용합니다.

```yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: weightsandbiases.apps.wandb.com
```

`controller-manager`는 사용자 정의 리소스, 릴리스 채널 및 사용자 정의 구성의 사양을 기반으로 [charts/operator-wandb](https://github.com/wandb/helm-charts/tree/main/charts/operator-wandb)를 설치합니다. 구성 사양 계층 구조는 사용자 측에서 최대한의 구성 유연성을 가능하게 하고 W&B 가 새로운 이미지, 구성, 기능 및 Helm 업데이트를 자동으로 릴리스할 수 있도록 합니다.

구성 옵션은 [구성 사양 계층 구조]({{< relref path="#configuration-specification-hierarchy" lang="ko" >}}) 및 [구성 참조]({{< relref path="#configuration-reference-for-wb-operator" lang="ko" >}})를 참조하십시오.

### 구성 사양 계층 구조

구성 사양은 상위 수준 사양이 하위 수준 사양을 재정의하는 계층적 모델을 따릅니다. 작동 방식은 다음과 같습니다.

- **릴리스 채널 값**: 이 기본 수준 구성은 W&B 가 배포를 위해 설정한 릴리스 채널을 기반으로 기본값과 구성을 설정합니다.
- **사용자 입력 값**: 사용자는 시스템 콘솔을 통해 릴리스 채널 사양에서 제공하는 기본 설정을 재정의할 수 있습니다.
- **사용자 정의 리소스 값**: 사용자로부터 오는 가장 높은 수준의 사양입니다. 여기에 지정된 값은 사용자 입력 및 릴리스 채널 사양을 모두 재정의합니다. 구성 옵션에 대한 자세한 설명은 [구성 참조]({{< relref path="#configuration-reference-for-wb-operator" lang="ko" >}})를 참조하십시오.

이 계층적 모델은 업그레이드 및 변경에 대한 관리 가능하고 체계적인 접근 방식을 유지하면서 구성이 유연하고 다양한 요구 사항을 충족하도록 사용자 정의할 수 있도록 보장합니다.

### W&B Kubernetes Operator 사용을 위한 요구 사항

W&B Kubernetes operator로 W&B 를 배포하려면 다음 요구 사항을 충족하십시오.

[참조 아키텍처]({{< relref path="../ref-arch.md#infrastructure-requirements" lang="ko" >}})를 참조하십시오. 또한 [유효한 W&B Server 라이선스를 얻으십시오]({{< relref path="../#obtain-your-wb-server-license" lang="ko" >}}).

자체 관리 설치를 설정하고 구성하는 방법에 대한 자세한 설명은 [이]({{< relref path="../bare-metal.md" lang="ko" >}}) 가이드를 참조하십시오.

설치 방법에 따라 다음 요구 사항을 충족해야 할 수 있습니다.
* 올바른 Kubernetes 클러스터 컨텍스트로 설치 및 구성된 Kubectl.
* Helm이 설치되었습니다.

### 에어 갭 설치

에어 갭 환경에서 W&B Kubernetes Operator를 설치하는 방법에 대한 자습서는 [Kubernetes를 사용하여 에어 갭 환경에 W&B 배포]({{< relref path="./operator-airgapped.md" lang="ko" >}})를 참조하십시오.

## W&B Server 애플리케이션 배포

이 섹션에서는 W&B Kubernetes operator를 배포하는 다양한 방법에 대해 설명합니다.
{{% alert %}}
W&B Operator는 W&B Server의 기본 설치 방법이자 권장 설치 방법입니다.
{{% /alert %}}

**다음 중 하나를 선택하십시오.**
- 필요한 모든 외부 서비스를 프로비저닝했고 Helm CLI를 사용하여 Kubernetes에 W&B 를 배포하려면 [여기]({{< relref path="#deploy-wb-with-helm-cli" lang="ko" >}})를 계속하십시오.
- Terraform을 사용하여 인프라와 W&B Server를 관리하려면 [여기]({{< relref path="#deploy-wb-with-helm-terraform-module" lang="ko" >}})를 계속하십시오.
- W&B Cloud Terraform Modules를 사용하려면 [여기]({{< relref path="#deploy-wb-with-wb-cloud-terraform-modules" lang="ko" >}})를 계속하십시오.

### Helm CLI로 W&B 배포

W&B 는 W&B Kubernetes operator를 Kubernetes 클러스터에 배포하기 위한 Helm Chart를 제공합니다. 이 방법을 사용하면 Helm CLI 또는 ArgoCD와 같은 지속적인 배포 툴을 사용하여 W&B Server를 배포할 수 있습니다. 위에 언급된 요구 사항이 충족되었는지 확인하십시오.

Helm CLI로 W&B Kubernetes Operator를 설치하려면 다음 단계를 따르십시오.

1. W&B Helm 저장소를 추가합니다. W&B Helm 차트는 W&B Helm 저장소에서 사용할 수 있습니다. 다음 코맨드를 사용하여 저장소를 추가합니다.
```shell
helm repo add wandb https://charts.wandb.ai
helm repo update
```
2. Kubernetes 클러스터에 Operator를 설치합니다. 다음을 복사하여 붙여넣습니다.
```shell
helm upgrade --install operator wandb/operator -n wandb-cr --create-namespace
```
3. W&B operator custom resource를 구성하여 W&B Server 설치를 트리거합니다. 이 예제 구성을 `operator.yaml`이라는 파일에 복사하여 W&B 배포를 사용자 정의할 수 있습니다. [구성 참조]({{< relref path="#configuration-reference-for-wb-operator" lang="ko" >}})를 참조하십시오.

   ```yaml
   apiVersion: apps.wandb.com/v1
   kind: WeightsAndBiases
   metadata:
     labels:
       app.kubernetes.io/instance: wandb
       app.kubernetes.io/name: weightsandbiases
     name: wandb
     namespace: default

   spec:
     chart:
       url: http://charts.yourdomain.com
       name: operator-wandb
       version: 0.18.0

     values:
       global:
         host: https://wandb.yourdomain.com
         license: xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
         bucket:
           accessKey: xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
           secretKey: xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
           name: s3.yourdomain.com:port #Ex.: s3.yourdomain.com:9000
           path: bucket_name
           provider: s3
           region: us-east-1
         mysql:
           database: wandb
           host: mysql.home.lab
           password: password
           port: 3306
           user: wandb
         extraEnv:
           ENABLE_REGISTRY_UI: 'true'

       # Ensure it's set to use your own MySQL
       mysql:
         install: false

       app:
         image:
           repository: registry.yourdomain.com/local
           tag: 0.59.2

       console:
         image:
           repository: registry.yourdomain.com/console
           tag: 2.12.2

       ingress:
         annotations:
           nginx.ingress.kubernetes.io/proxy-body-size: 64m
         class: nginx
   ```

    W&B Server 애플리케이션을 설치하고 구성할 수 있도록 사용자 정의 구성으로 Operator를 시작합니다.

    ```shell
    kubectl apply -f operator.yaml
    ```

    배포가 완료될 때까지 기다립니다. 몇 분 정도 걸립니다.

5. 웹 UI를 사용하여 설치를 확인하려면 첫 번째 관리자 사용자 계정을 만든 다음 [설치 확인]({{< relref path="#verify-the-installation" lang="ko" >}})에 설명된 확인 단계를 따르십시오.

### Helm Terraform Module로 W&B 배포

이 방법을 사용하면 일관성과 반복성을 위해 Terraform의 infrastructure-as-code 접근 방식을 활용하여 특정 요구 사항에 맞게 사용자 정의된 배포가 가능합니다. 공식 W&B Helm 기반 Terraform Module은 [여기](https://registry.terraform.io/modules/wandb/wandb/helm/latest)에 있습니다.

다음 코드는 시작점으로 사용할 수 있으며 프로덕션 등급 배포에 필요한 모든 구성 옵션이 포함되어 있습니다.

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

구성 옵션은 [구성 참조]({{< relref path="#configuration-reference-for-wb-operator" lang="ko" >}})에 설명된 내용과 동일하지만 구문은 HashiCorp Configuration Language(HCL)를 따라야 합니다. Terraform 모듈은 W&B custom resource definition(CRD)을 생성합니다.

W&B&Biases 자체가 Helm Terraform 모듈을 사용하여 고객을 위한 "전용 클라우드" 설치를 배포하는 방법을 보려면 다음 링크를 따르십시오.
- [AWS](https://github.com/wandb/terraform-aws-wandb/blob/45e1d746f53e78e73e68f911a1f8cad5408e74b6/main.tf#L225)
- [Azure](https://github.com/wandb/terraform-azurerm-wandb/blob/170e03136b6b6fc758102d59dacda99768854045/main.tf#L155)
- [GCP](https://github.com/wandb/terraform-google-wandb/blob/49ddc3383df4cefc04337a2ae784f57ce2a2c699/main.tf#L189)

### W&B Cloud Terraform 모듈로 W&B 배포

W&B 는 AWS, GCP 및 Azure용 Terraform Modules 세트를 제공합니다. 이러한 모듈은 Kubernetes 클러스터, 로드 밸런서, MySQL 데이터베이스 등을 포함한 전체 인프라와 W&B Server 애플리케이션을 배포합니다. W&B Kubernetes Operator는 다음과 같은 버전의 공식 W&B 클라우드별 Terraform Modules와 이미 사전 구성되어 있습니다.

| Terraform Registry                                                  | Source Code                                      | Version |
| ------------------------------------------------------------------- | ------------------------------------------------ | ------- |
| [AWS](https://registry.terraform.io/modules/wandb/wandb/aws/latest) | https://github.com/wandb/terraform-aws-wandb     | v4.0.0+ |
| [Azure](https://registry.terraform.io/modules/wandb/terraform-azurerm-wandb)           | https://github.com/wandb/terraform-azurerm-wandb | v2.0.0+ |
| [GCP](https://registry.terraform.io/modules/wandb/terraform-google-wandb)              | https://github.com/wandb/terraform-google-wandb  | v2.0.0+ |

이러한 통합을 통해 W&B Kubernetes Operator를 최소한의 설정으로 인스턴스에 사용할 수 있도록 보장하여 클라우드 환경에서 W&B Server를 배포하고 관리하는 간소화된 경로를 제공합니다.

이러한 모듈을 사용하는 방법에 대한 자세한 설명은 문서의 자체 관리 설치 섹션에서 [이 섹션]({{< relref path="../#deploy-wb-server-within-self-managed-cloud-accounts" lang="ko" >}})을 참조하십시오.

### 설치 확인

설치를 확인하기 위해 W&B 는 [W&B CLI]({{< relref path="/ref/cli/" lang="ko" >}})를 사용하는 것이 좋습니다. verify 코맨드는 모든 구성 요소와 구성을 확인하는 여러 테스트를 실행합니다.

{{% alert %}}
이 단계에서는 브라우저를 사용하여 첫 번째 관리자 사용자 계정이 생성되었다고 가정합니다.
{{% /alert %}}

설치를 확인하려면 다음 단계를 따르십시오.

1. W&B CLI를 설치합니다.
    ```shell
    pip install wandb
    ```
2. W&B 에 로그인합니다.
    ```shell
    wandb login --host=https://YOUR_DNS_DOMAIN
    ```

    예를 들어 다음과 같습니다.
    ```shell
    wandb login --host=https://wandb.company-name.com
    ```

3. 설치를 확인합니다.
    ```shell
    wandb verify
    ```

설치가 성공하고 W&B 배포가 완전히 작동하면 다음 출력이 표시됩니다.

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

## W&B Management Console 엑세스

W&B Kubernetes operator에는 관리 콘솔이 함께 제공됩니다. 콘솔은 `${HOST_URI}/console`에 있습니다(예: `https://wandb.company-name.com/` console).

관리 콘솔에 로그인하는 방법은 두 가지입니다.

{{< tabpane text=true >}}
{{% tab header="옵션 1(권장)" value="option1" %}}
1. 브라우저에서 W&B 애플리케이션을 열고 로그인합니다. `${HOST_URI}/`(예: `https://wandb.company-name.com/`)로 W&B 애플리케이션에 로그인합니다.
2. 콘솔에 엑세스합니다. 오른쪽 상단 모서리에 있는 아이콘을 클릭한 다음 **시스템 콘솔**을 클릭합니다. 관리자 권한이 있는 사용자만 **시스템 콘솔** 항목을 볼 수 있습니다.

    {{< img src="/images/hosting/access_system_console_via_main_app.png" alt="" >}}
{{% /tab %}}

{{% tab header="옵션 2" value="option2"%}}
{{% alert %}}
옵션 1이 작동하지 않는 경우에만 다음 단계를 사용하여 콘솔에 엑세스하는 것이 좋습니다.
{{% /alert %}}

1. 브라우저에서 콘솔 애플리케이션을 엽니다. 위에 설명된 URL을 열면 로그인 화면으로 리디렉션됩니다.
    {{< img src="/images/hosting/access_system_console_directly.png" alt="" >}}
2. 설치에서 생성된 Kubernetes secret에서 비밀번호를 검색합니다.
    ```shell
    kubectl get secret wandb-password -o jsonpath='{.data.password}' | base64 -d
    ```
    비밀번호를 복사합니다.
3. 콘솔에 로그인합니다. 복사한 비밀번호를 붙여넣은 다음 **로그인**을 클릭합니다.
{{% /tab %}}
{{< /tabpane >}}

## W&B Kubernetes operator 업데이트

이 섹션에서는 W&B Kubernetes operator를 업데이트하는 방법에 대해 설명합니다.

{{% alert %}}
* W&B Kubernetes operator를 업데이트해도 W&B server 애플리케이션은 업데이트되지 않습니다.
* W&B operator를 업데이트하기 전에 W&B Kubernetes operator를 사용하지 않는 Helm 차트를 사용하는 경우 [여기]({{< relref path="#migrate-self-managed-instances-to-wb-operator" lang="ko" >}})의 지침을 참조하십시오.
{{% /alert %}}

아래 코드 조각을 복사하여 터미널에 붙여넣습니다.

1. 먼저 [`helm repo update`](https://helm.sh/docs/helm/helm_repo_update/)로 저장소를 업데이트합니다.
    ```shell
    helm repo update
    ```

2. 다음으로 [`helm upgrade`](https://helm.sh/docs/helm/helm_upgrade/)로 Helm 차트를 업데이트합니다.
    ```shell
    helm upgrade operator wandb/operator -n wandb-cr --reuse-values
    ```

## W&B Server 애플리케이션 업데이트

W&B Kubernetes operator를 사용하는 경우 더 이상 W&B Server 애플리케이션을 업데이트할 필요가 없습니다.

operator는 W&B 의 새 소프트웨어 버전이 릴리스되면 W&B Server 애플리케이션을 자동으로 업데이트합니다.

## 자체 관리 인스턴스를 W&B Operator로 마이그레이션

다음 섹션에서는 자체 W&B Server 설치를 자체 관리하는 것에서 W&B Operator를 사용하여 이 작업을 수행하도록 마이그레이션하는 방법에 대해 설명합니다. 마이그레이션 프로세스는 W&B Server 설치 방법에 따라 다릅니다.

{{% alert %}}
W&B Operator는 W&B Server의 기본 설치 방법이자 권장 설치 방법입니다. 질문이 있는 경우 [고객 지원](mailto:support@wandb.com) 또는 W&B 팀에 문의하십시오.
{{% /alert %}}

- 공식 W&B Cloud Terraform Modules를 사용한 경우 적절한 설명서로 이동하여 해당 단계를 따르십시오.
  - [AWS]({{< relref path="#migrate-to-operator-based-aws-terraform-modules" lang="ko" >}})
  - [GCP]({{< relref path="#migrate-to-operator-based-gcp-terraform-modules" lang="ko" >}})
  - [Azure]({{< relref path="#migrate-to-operator-based-azure-terraform-modules" lang="ko" >}})
- [W&B Non-Operator Helm 차트](https://github.com/wandb/helm-charts/tree/main/charts/wandb)를 사용한 경우 [여기]({{< relref path="#migrate-to-operator-based-helm-chart" lang="ko" >}})를 계속하십시오.
- [Terraform을 사용하는 W&B Non-Operator Helm 차트](https://registry.terraform.io/modules/wandb/wandb/kubernetes/latest)를 사용한 경우 [여기]({{< relref path="#migrate-to-operator-based-terraform-helm-chart" lang="ko" >}})를 계속하십시오.
- 매니페스트로 Kubernetes 리소스를 만든 경우 [여기]({{< relref path="#migrate-to-operator-based-helm-chart" lang="ko" >}})를 계속하십시오.

### Operator 기반 AWS Terraform Modules로 마이그레이션

마이그레이션 프로세스에 대한 자세한 설명은 [여기]({{< relref path="../install-on-public-cloud/aws-tf.md#migrate-to-operator-based-aws-terraform-modules" lang="ko" >}})를 계속하십시오.

### Operator 기반 GCP Terraform Modules로 마이그레이션

질문이 있거나 도움이 필요한 경우 [고객 지원](mailto:support@wandb.com) 또는 W&B 팀에 문의하십시오.

### Operator 기반 Azure Terraform Modules로 마이그레이션

질문이 있거나 도움이 필요한 경우 [고객 지원](mailto:support@wandb.com) 또는 W&B 팀에 문의하십시오.

### Operator 기반 Helm 차트로 마이그레이션

Operator 기반 Helm 차트로 마이그레이션하려면 다음 단계를 따르십시오.

1. 현재 W&B 구성을 가져옵니다. W&B 가 운영자가 아닌 버전의 Helm 차트로 배포된 경우 다음과 같이 값을 내보냅니다.
    ```shell
    helm get values wandb
    ```
    W&B 가 Kubernetes 매니페스트로 배포된 경우 다음과 같이 값을 내보냅니다.
    ```shell
    kubectl get deployment wandb -o yaml
    ```
    이제 다음 단계에 필요한 모든 구성 값이 있습니다.

2. `operator.yaml`이라는 파일을 만듭니다. [구성 참조]({{< relref path="#configuration-reference-for-wb-operator" lang="ko" >}})에 설명된 형식을 따르십시오. 1단계의 값을 사용하십시오.

3. 현재 배포를 0개의 pod로 확장합니다. 이 단계는 현재 배포를 중지합니다.
    ```shell
    kubectl scale --replicas=0 deployment wandb
    ```
4. Helm 차트 저장소를 업데이트합니다.
    ```shell
    helm repo update
    ```
5. 새 Helm 차트를 설치합니다.
    ```shell
    helm upgrade --install operator wandb/operator -n wandb-cr --create-namespace
    ```
6. 새 helm 차트를 구성하고 W&B 애플리케이션 배포를 트리거합니다. 새 구성을 적용합니다.
    ```shell
    kubectl apply -f operator.yaml
    ```
    배포가 완료되는 데 몇 분 정도 걸립니다.

7. 설치를 확인합니다. [설치 확인]({{< relref path="#verify-the-installation" lang="ko" >}})의 단계를 따라 모든 것이 작동하는지 확인하십시오.

8. 이전 설치를 제거합니다. 이전 helm 차트를 제거하거나 매니페스트로 생성된 리소스를 삭제합니다.

### Operator 기반 Terraform Helm 차트로 마이그레이션

Operator 기반 Helm 차트로 마이그레이션하려면 다음 단계를 따르십시오.

1. Terraform 구성을 준비합니다. Terraform 구성에서 이전 배포의 Terraform 코드를 [여기]({{< relref path="#deploy-wb-with-helm-terraform-module" lang="ko" >}})에 설명된 코드로 바꿉니다. 이전과 동일한 변수를 설정합니다. .tfvars 파일이 있는 경우 변경하지 마십시오.
2. Terraform 런을 실행합니다. terraform init, plan 및 apply를 실행합니다.
3. 설치를 확인합니다. [설치 확인]({{< relref path="#verify-the-installation" lang="ko" >}})의 단계를 따라 모든 것이 작동하는지 확인하십시오.
4. 이전 설치를 제거합니다. 이전 helm 차트를 제거하거나 매니페스트로 생성된 리소스를 삭제합니다.

## W&B Server 구성 참조

이 섹션에서는 W&B Server 애플리케이션의 구성 옵션에 대해 설명합니다. 애플리케이션은 [WeightsAndBiases]({{< relref path="#how-it-works" lang="ko" >}})라는 custom resource definition으로 구성을 수신합니다. 일부 구성 옵션은 아래 구성으로 노출되고 일부는 환경 변수로 설정해야 합니다.

이 문서에는 [기본]({{< relref path="/guides/hosting/env-vars/" lang="ko" >}}) 및 [고급]({{< relref path="/guides/hosting/iam/advanced_env_vars/" lang="ko" >}})의 두 가지 환경 변수 목록이 있습니다. 필요한 구성 옵션이 Helm Chart를 사용하여 노출되지 않은 경우에만 환경 변수를 사용하십시오.

프로덕션 배포를 위한 W&B Server 애플리케이션 구성 파일에는 다음 내용이 필요합니다. 이 YAML 파일은 버전, 환경 변수, 데이터베이스와 같은 외부 리소스 및 기타 필요한 설정을 포함하여 W&B 배포의 원하는 상태를 정의합니다.

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

[W&B Helm 저장소](https://github.com/wandb/helm-charts/blob/main/charts/operator-wandb/values.yaml)에서 전체 값 집합을 찾고 재정의해야 하는 값만 변경하십시오.

### 전체 예제
다음은 GCP Ingress 및 GCS (GCP 오브젝트 저장소)가 있는 GCP Kubernetes를 사용하는 예제 구성입니다.

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
 # 프로토콜과 함께 FQDN을 제공합니다.
global:
  # 예제 호스트 이름, 자신의 것으로 바꿉니다.
  host: https://wandb.example.com
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

**기타 공급자 (Minio, Ceph 등)**

다른 S3 호환 공급자의 경우 다음과 같이 버킷 구성을 설정합니다.
```yaml
global:
  bucket:
    # 예제 값, 자신의 것으로 바꿉니다.
    provider: s3
    name: storage.example.com
    kmsKey: null
    path: wandb
    region: default
    accessKey: 5WOA500...P5DK7I
    secretKey: HDKYe4Q...JAp1YyjysnX
```

AWS 외부에서 호스팅되는 S3 호환 스토리지의 경우 `kmsKey`는 `null`이어야 합니다.

secret에서 `accessKey` 및 `secretKey`를 참조하려면:
```yaml
global:
  bucket:
    # 예제 값, 자신의 것으로 바꿉니다.
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
     # 예제 값, 자신의 것으로 바꿉니다.
     host: db.example.com
     port: 3306
     database: wandb_local
     user: wandb
     password: 8wtX6cJH...ZcUarK4zZGjpV
```

secret에서 `password`를 참조하려면:
```yaml
global:
   mysql:
     # 예제 값, 자신의 것으로 바꿉니다.
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
  # 예제 라이선스, 자신의 것으로 바꿉니다.
  license: eyJhbGnUzaHgyQjQy...VFnPS_KETXg1hi
```

secret에서 `license`를 참조하려면:
```yaml
global:
  licenseSecret:
    name: license-secret
    key: CUSTOMER_WANDB_LICENSE
```

### Ingress

ingress 클래스를 식별하려면 이 FAQ [항목]({{< relref path="#how-to-identify-the-kubernetes-ingress-class" lang="ko" >}})을 참조하십시오.

**TLS 없음**

```yaml
global:
# 중요: Ingress는 'global'과 동일한 YAML 수준에 있습니다 (자식이 아님).
ingress:
  class: ""
```

**TLS 사용**

인증서를 포함하는 secret을 만듭니다.

```console
kubectl create secret tls wandb-ingress-tls --key wandb-ingress-tls.key --cert wandb-ingress-tls.crt
```

ingress 구성에서 secret을 참조합니다.
```yaml
global:
# 중요: Ingress는 'global'과 동일한 YAML 수준에 있습니다 (자식이 아님).
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

Nginx의 경우 다음 주석을 추가해야 할 수 있습니다.

```
ingress:
  annotations:
    nginx.ingress.kubernetes.io/proxy-body-size: 64m
```

### 사용자 정의 Kubernetes ServiceAccounts

W&B pod를 실행할 사용자 정의 Kubernetes 서비스 계정을 지정합니다.

다음 코드 조각은 지정된 이름으로 배포의 일부로 서비스 계정을 만듭니다.

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
"app" 및 "parquet" 하위 시스템은 지정된 서비스 계정에서 실행됩니다. 다른 하위 시스템은 기본 서비스 계정에서 실행됩니다.

서비스 계정이 클러스터에 이미 있는 경우 `create: false`를 설정합니다.

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

app, parquet, console 등과 같은 다른 하위 시스템에서 서비스 계정을 지정할 수 있습니다.

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

서비스 계정은 하위 시스템마다 다를 수 있습니다.

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

secret에서 `password`를 참조하려면:

```console
kubectl create secret generic redis-secret --from-literal=redis-password=supersecret
```

아래 구성에서 참조합니다.
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
**TLS 없음**
```yaml
global:
  ldap:
    enabled: true
    # "ldap://" 또는 "ldaps://"를 포함한 LDAP 서버 어드레스
    host:
    # 사용자를 찾는 데 사용할 LDAP 검색 기준
    baseDN:
    # 바인딩할 LDAP 사용자(익명 바인딩을 사용하지 않는 경우)
    bindDN:
    # 바인딩할 LDAP 비밀번호가 있는 secret 이름 및 키(익명 바인딩을 사용하지 않는 경우)
    bindPW:
    # 쉼표로 구분된 문자열 값으로 된 이메일 및 그룹 ID 속성 이름에 대한 LDAP 속성입니다.
    attributes:
    # LDAP 그룹 허용 목록
    groupAllowList:
    # LDAP TLS 활성화
    tls: false
```

**TLS 사용**

LDAP TLS 인증서 구성에는 인증서 콘텐츠로 미리 생성된 구성 맵이 필요합니다.

구성 맵을 만들려면 다음 코맨드를 사용할 수 있습니다.

```console
kubectl create configmap ldap-tls-cert --from-file=certificate.crt
```

아래 예제와 같이 YAML에서 구성 맵을 사용합니다.

```yaml
global:
  ldap:
    enabled: true
    # "ldap://" 또는 "ldaps://"를 포함한 LDAP 서버 어드레스
    host:
    # 사용자를 찾는 데 사용할 LDAP 검색 기준
    baseDN:
    # 바인딩할 LDAP 사용자(익명 바인딩을 사용하지 않는 경우)
    bindDN:
    # 바인딩할 LDAP 비밀번호가 있는 secret 이름 및 키(익명 바인딩을 사용하지 않는 경우)
    bindPW:
    # 쉼표로 구분된 문자열 값으로 된 이메일 및 그룹 ID 속성 이름에 대한 LDAP 속성입니다.
    attributes:
    # LDAP 그룹 허용 목록
    groupAllowList:
    # LDAP TLS 활성화
    tls: true
    # LDAP 서버용 CA 인증서가 있는 ConfigMap 이름 및 키
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
      # IdP가 필요한 경우에만 포함하십시오.
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

### 환경 변수
```yaml
global:
  extraEnv:
    GLOBAL_ENV: "example"
```

### 사용자 정의 인증 기관
`customCACerts`는 목록이며 많은 인증서를 사용할 수 있습니다. `customCACerts`에 지정된 인증 기관은 W&B Server 애플리케이션에만 적용됩니다.

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

CA 인증서는 ConfigMap에 저장할 수도 있습니다.
```yaml
global:
  caCertsConfigMap: custom-ca-certs
```

ConfigMap은 다음과 같아야 합니다.
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
ConfigMap을 사용하는 경우 ConfigMap의 각 키는 `.crt`로 끝나야 합니다(예: `my-cert.crt` 또는 `ca-cert1.crt`). 이 명명 규칙은 `update-ca-certificates`가 각 인증서를 구문 분석하여 시스템 CA 저장소에 추가하는 데 필요합니다.
{{% /alert %}}

### 사용자 정의 보안 컨텍스트

각 W&B 구성 요소는 다음과 같은 형식의 사용자 정의 보안 컨텍스트 구성을 지원합니다.

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
`runAsGroup:`에 대한 유효한 값은 `0`뿐입니다. 다른 값은 오류입니다.
{{% /alert %}}

예를 들어 애플리케이션 pod를 구성하려면 구성에 `app` 섹션을 추가합니다.

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

동일한 개념이 `console`, `weave`, `weave-trace` 및 `parquet`에 적용됩니다.

## W&B Operator 구성 참조

이 섹션에서는 W&B Kubernetes operator(`wandb-controller-manager`)의 구성 옵션에 대해 설명합니다. operator는 YAML 파일 형식으로 구성을 수신합니다.

기본적으로 W&B Kubernetes operator에는 구성 파일이 필요하지 않습니다. 필요한 경우 구성 파일을 만듭니다. 예를 들어 사용자 정의 인증 기관을 지정하거나 에어 갭 환경에 배포하기 위해 구성 파일이 필요할 수 있습니다.

전체 사양 사용자 정의 목록은 [Helm 저장소](https://github.com/wandb/helm-charts/blob/main/charts/operator/values.yaml)에서 찾으십시오.

### 사용자 정의 CA
사용자 정의 인증 기관(`customCACerts`)은 목록이며 많은 인증서를 사용할 수 있습니다. 추가된 이러한 인증 기관은 W&B Kubernetes operator(`wandb-controller-manager`)에만 적용됩니다.

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

CA 인증서는 ConfigMap에 저장할 수도 있습니다.
```yaml
caCertsConfigMap: custom-ca-certs
```

ConfigMap은 다음과 같아야 합니다.
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
ConfigMap의 각 키는 `.crt`로 끝나야 합니다(예: `my-cert.crt` 또는 `ca-cert1.crt`). 이 명명 규칙은 `update-ca-certificates`가 각 인증서를 구문 분석하여 시스템 CA 저장소에 추가하는 데 필요합니다.
{{% /alert %}}

## FAQ

### W&B Operator Console 비밀번호를 얻는 방법
[W&B Kubernetes Operator Management Console 엑세스]({{< relref path="#access-the-wb-management-console" lang="ko" >}})를 참조하십시오.

### Ingress가 작동하지 않는 경우 W&B Operator Console에 엑세스하는 방법

Kubernetes 클러스터에 연결할 수 있는 호스트에서 다음 코맨드를 실행합니다.

```console
kubectl port-forward svc/wandb-console 8082
```

브라우저에서 `https://localhost:8082/` 콘솔로 콘솔에 엑세스합니다.

비밀번호를 얻는 방법은 [W&B Kubernetes Operator Management Console 엑세스]({{< relref path="#access-the-wb-management-console" lang="ko" >}}) (옵션 2)를 참조하십시오.

### W&B Server 로그를 보는 방법

애플리케이션 pod의 이름은 **wandb-app-xxx**입니다.

```console
kubectl get pods
kubectl logs wandb-XXXXX-XXXXX
```

### Kubernetes ingress 클래스를 식별하는 방법

다음을 실행하여 클러스터에 설치된 ingress 클래스를 가져올 수 있습니다.

```console
kubectl get ingressclass
```
```
