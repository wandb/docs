---
title: Kubernetes operator for air-gapped instances
description: Kubernetes Operator로 W&B 플랫폼 배포 (Airgapped)
menu:
  default:
    identifier: ko-guides-hosting-hosting-options-self-managed-kubernetes-operator-operator-airgapped
    parent: kubernetes-operator
---

## 도입

본 가이드는 에어 갭 환경의 고객 관리 환경에 W&B 플랫폼을 배포하는 단계별 지침을 제공합니다.

Helm 차트와 컨테이너 이미지를 호스팅하려면 내부 저장소 또는 레지스트리를 사용하세요. Kubernetes 클러스터에 대한 적절한 엑세스 권한으로 셸 콘솔에서 모든 코맨드를 실행하세요.

Kubernetes 애플리케이션을 배포하는 데 사용하는 모든 지속적 배포 툴링에서 유사한 코맨드를 활용할 수 있습니다.

## 1단계: 필수 조건

시작하기 전에 환경이 다음 요구 사항을 충족하는지 확인하세요.

- Kubernetes 버전 >= 1.28
- Helm 버전 >= 3
- 필요한 W&B 이미지가 있는 내부 컨테이너 레지스트리에 대한 엑세스
- W&B Helm 차트에 대한 내부 Helm 저장소에 대한 엑세스

## 2단계: 내부 컨테이너 레지스트리 준비

배포를 진행하기 전에 다음 컨테이너 이미지가 내부 컨테이너 레지스트리에서 사용 가능한지 확인해야 합니다.
* [`docker.io/wandb/controller`](https://hub.docker.com/r/wandb/controller)
* [`docker.io/wandb/local`](https://hub.docker.com/r/wandb/local)
* [`docker.io/wandb/console`](https://hub.docker.com/r/wandb/console)
* [`docker.io/bitnami/redis`](https://hub.docker.com/r/bitnami/redis)
* [`docker.io/otel/opentelemetry-collector-contrib`](https://hub.docker.com/r/otel/opentelemetry-collector-contrib)
* [`quay.io/prometheus/prometheus`](https://quay.io/repository/prometheus/prometheus)
* [`quay.io/prometheus-operator/prometheus-config-reloader`](https://quay.io/repository/prometheus-operator/prometheus-config-reloader)

이러한 이미지는 W&B 구성 요소의 성공적인 배포에 매우 중요합니다. W&B는 WSM을 사용하여 컨테이너 레지스트리를 준비하는 것을 권장합니다.

조직에서 이미 내부 컨테이너 레지스트리를 사용하고 있는 경우 이미지를 추가할 수 있습니다. 그렇지 않으면 다음 섹션에 따라 WSM을 사용하여 컨테이너 저장소를 준비하세요.

[WSM 사용]({{< relref path="#list-images-and-their-versions" lang="ko" >}}) 또는 조직의 자체 프로세스를 사용하여 Operator의 요구 사항을 추적하고 이미지 업그레이드를 확인하고 다운로드해야 합니다.

### WSM 설치

다음 방법 중 하나를 사용하여 WSM을 설치합니다.

{{% alert %}}
WSM을 사용하려면 작동하는 Docker 설치가 필요합니다.
{{% /alert %}}

#### Bash
GitHub에서 직접 Bash 스크립트를 실행합니다.

```bash
curl -sSL https://raw.githubusercontent.com/wandb/wsm/main/install.sh | bash
```
스크립트는 스크립트를 실행한 폴더에 바이너리를 다운로드합니다. 다른 폴더로 이동하려면 다음을 실행합니다.

```bash
sudo mv wsm /usr/local/bin
```

#### GitHub
W&B 관리 `wandb/wsm` GitHub 저장소(`https://github.com/wandb/wsm`)에서 WSM을 다운로드하거나 복제합니다. 최신 릴리스는 `wandb/wsm` [릴리스 노트](https://github.com/wandb/wsm/releases)를 참조하세요.

### 이미지 및 해당 버전 나열

`wsm list`를 사용하여 최신 이미지 버전 목록을 가져옵니다.

```bash
wsm list
```

출력은 다음과 유사하게 표시됩니다.

```text
:package: Starting the process to list all images required for deployment...
Operator Images:
  wandb/controller:1.16.1
W&B Images:
  wandb/local:0.62.2
  docker.io/bitnami/redis:7.2.4-debian-12-r9
  quay.io/prometheus-operator/prometheus-config-reloader:v0.67.0
  quay.io/prometheus/prometheus:v2.47.0
  otel/opentelemetry-collector-contrib:0.97.0
  wandb/console:2.13.1
Here are the images required to deploy W&B. Ensure these images are available in your internal container registry and update the values.yaml accordingly.
```

### 이미지 다운로드

`wsm download`를 사용하여 최신 버전의 모든 이미지를 다운로드합니다.

```bash
wsm download
```

출력은 다음과 유사하게 표시됩니다.

```text
Downloading operator helm chart
Downloading wandb helm chart
✓ wandb/controller:1.16.1
✓ docker.io/bitnami/redis:7.2.4-debian-12-r9
✓ otel/opentelemetry-collector-contrib:0.97.0
✓ quay.io/prometheus-operator/prometheus-config-reloader:v0.67.0
✓ wandb/console:2.13.1
✓ quay.io/prometheus/prometheus:v2.47.0

  Done! Installed 7 packages.
```

WSM은 각 이미지에 대해 `.tgz` 아카이브를 `bundle` 디렉토리에 다운로드합니다.

## 3단계: 내부 Helm 차트 저장소 준비

컨테이너 이미지와 함께 다음 Helm 차트가 내부 Helm 차트 저장소에서 사용 가능한지 확인해야 합니다. 마지막 단계에서 소개된 WSM 툴은 Helm 차트도 다운로드할 수 있습니다. 또는 여기에서 다운로드하세요.

- [W&B Operator](https://github.com/wandb/helm-charts/tree/main/charts/operator)
- [W&B 플랫폼](https://github.com/wandb/helm-charts/tree/main/charts/operator-wandb)

`operator` 차트는 Controller Manager라고도 하는 W&B Operator를 배포하는 데 사용됩니다. `platform` 차트는 CRD(사용자 정의 리소스 정의)에 구성된 값을 사용하여 W&B 플랫폼을 배포하는 데 사용됩니다.

## 4단계: Helm 저장소 설정

이제 내부 저장소에서 W&B Helm 차트를 가져오도록 Helm 저장소를 구성합니다. 다음 코맨드를 실행하여 Helm 저장소를 추가하고 업데이트합니다.

```bash
helm repo add local-repo https://charts.yourdomain.com
helm repo update
```

## 5단계: Kubernetes operator 설치

controller manager라고도 하는 W&B Kubernetes operator는 W&B 플랫폼 구성 요소를 관리합니다. 에어 갭 환경에 설치하려면 내부 컨테이너 레지스트리를 사용하도록 구성해야 합니다.

이렇게 하려면 내부 컨테이너 레지스트리를 사용하도록 기본 이미지 설정을 재정의하고 예상되는 배포 유형을 나타내기 위해 `airgapped: true` 키를 설정해야 합니다. 아래와 같이 `values.yaml` 파일을 업데이트합니다.

```yaml
image:
  repository: registry.yourdomain.com/library/controller
  tag: 1.13.3
airgapped: true
```

태그를 내부 레지스트리에서 사용할 수 있는 버전으로 바꿉니다.

operator와 CRD를 설치합니다.
```bash
helm upgrade --install operator wandb/operator -n wandb --create-namespace -f values.yaml
```

지원되는 값에 대한 자세한 내용은 [Kubernetes operator GitHub 저장소](https://github.com/wandb/helm-charts/blob/main/charts/operator/values.yaml)를 참조하세요.

## 6단계: W&B 사용자 정의 리소스 구성

W&B Kubernetes operator를 설치한 후 내부 Helm 저장소와 컨테이너 레지스트리를 가리키도록 CR(사용자 정의 리소스)을 구성해야 합니다.

이 구성을 통해 Kubernetes operator는 W&B 플랫폼의 필요한 구성 요소를 배포할 때 내부 레지스트리와 저장소를 사용합니다.

이 예제 CR을 `wandb.yaml`이라는 새 파일에 복사합니다.

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
    
    # If install: true, Helm installs a MySQL database for the deployment to use. Set to `false` to use your own external MySQL deployment.
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

W&B 플랫폼을 배포하기 위해 Kubernetes Operator는 CR의 값을 사용하여 내부 저장소에서 `operator-wandb` Helm 차트를 구성합니다.

모든 태그/버전을 내부 레지스트리에서 사용할 수 있는 버전으로 바꿉니다.

이전 구성 파일을 만드는 방법에 대한 자세한 내용은 [여기]({{< relref path="/guides/hosting/hosting-options/self-managed/kubernetes-operator/#configuration-reference-for-wb-server" lang="ko" >}})에서 확인할 수 있습니다.

## 7단계: W&B 플랫폼 배포

이제 Kubernetes operator와 CR이 구성되었으므로 `wandb.yaml` 구성을 적용하여 W&B 플랫폼을 배포합니다.

```bash
kubectl apply -f wandb.yaml
```

## FAQ

배포 프로세스 중에 자주 묻는 질문(FAQ) 및 문제 해결 팁은 아래를 참조하세요.

### 다른 ingress 클래스가 있습니다. 해당 클래스를 사용할 수 있습니까?
예, `values.yaml`에서 ingress 설정을 수정하여 ingress 클래스를 구성할 수 있습니다.

### 인증서 번들에 인증서가 두 개 이상 있습니다. 작동합니까?
`values.yaml`의 `customCACerts` 섹션에서 인증서를 여러 항목으로 분할해야 합니다.

### Kubernetes operator가 무인 업데이트를 적용하지 못하도록 하는 방법. 가능합니까?
W&B 콘솔에서 자동 업데이트를 끌 수 있습니다. 지원되는 버전에 대한 질문은 W&B 팀에 문의하세요. 또한 W&B는 지난 6개월 동안 릴리스된 플랫폼 버전을 지원합니다. W&B는 주기적인 업그레이드를 수행하는 것을 권장합니다.

### 환경이 공용 저장소에 연결되어 있지 않은 경우 배포가 작동합니까?
구성에서 `airgapped`를 `true`로 설정하면 Kubernetes operator는 내부 리소스만 사용하고 공용 저장소에 연결을 시도하지 않습니다.
