---
title: 에어갭 인스턴스를 위한 Kubernetes 오퍼레이터
description: Kubernetes Operator(에어갭 환경)로 W&B 플랫폼 배포하기
menu:
  default:
    identifier: ko-guides-hosting-hosting-options-self-managed-kubernetes-operator-operator-airgapped
    parent: kubernetes-operator
---

## 소개

이 가이드는 에어갭드(air-gapped) 고객 관리 환경에서 W&B 플랫폼을 배포하는 단계별 안내를 제공합니다.

Helm 차트와 컨테이너 이미지를 호스팅할 내부 저장소 또는 레지스트리를 사용하세요. 모든 코맨드는 해당 Kubernetes 클러스터에 적절한 엑세스 권한이 있는 쉘 콘솔에서 실행해야 합니다.

Kubernetes 애플리케이션을 배포할 때 사용 중인 CI/CD 툴에서도 유사한 코맨드를 사용할 수 있습니다.

## 1단계: 사전 준비

시작 전에, 아래의 조건을 환경이 충족하는지 확인하세요:

- Kubernetes 버전 >= 1.28
- Helm 버전 >= 3
- 필수 W&B 이미지가 포함된 내부 컨테이너 레지스트리 엑세스
- W&B Helm 차트를 위한 내부 Helm 저장소 엑세스

## 2단계: 내부 컨테이너 레지스트리 준비

배포를 진행하기 전에, 아래 컨테이너 이미지들이 내부 컨테이너 레지스트리에 준비되어 있는지 확인해야 합니다:
* [`docker.io/wandb/controller`](https://hub.docker.com/r/wandb/controller)
* [`docker.io/wandb/local`](https://hub.docker.com/r/wandb/local)
* [`docker.io/wandb/console`](https://hub.docker.com/r/wandb/console)
* [`docker.io/bitnami/redis`](https://hub.docker.com/r/bitnami/redis)
* [`docker.io/otel/opentelemetry-collector-contrib`](https://hub.docker.com/r/otel/opentelemetry-collector-contrib)
* [`quay.io/prometheus/prometheus`](https://quay.io/repository/prometheus/prometheus)
* [`quay.io/prometheus-operator/prometheus-config-reloader`](https://quay.io/repository/prometheus-operator/prometheus-config-reloader)

이 이미지들은 W&B 구성요소의 성공적인 배포에 필수적입니다. W&B에서는 WSM 툴을 활용해 컨테이너 레지스트리를 준비할 것을 권장합니다.

이미 조직에서 내부 컨테이너 레지스트리를 사용 중이라면 이미지들을 추가하면 됩니다. 그렇지 않은 경우, 다음 섹션의 WSM을 활용해 컨테이너 저장소를 준비하는 방법을 따라주세요.

운영자(Operator)의 필요 사항을 추적하고 이미지 업그레이드 체크와 다운로드는 [WSM 사용]({{< relref path="#list-images-and-their-versions" lang="ko" >}}) 또는 조직의 별도 프로세스를 통해 사용자가 직접 관리해야 합니다.

### WSM 설치

WSM은 다음 방법 중 하나로 설치할 수 있습니다.

{{% alert %}}
WSM은 작동하는 Docker 설치가 필요합니다.
{{% /alert %}}

#### Bash
GitHub에서 바로 Bash 스크립트를 실행하세요:

```bash
curl -sSL https://raw.githubusercontent.com/wandb/wsm/main/install.sh | bash
```
스크립트는 실행한 폴더에 바이너리를 다운로드합니다. 다른 폴더로 이동하려면 아래와 같이 실행하세요:

```bash
sudo mv wsm /usr/local/bin
```

#### GitHub
W&B에서 관리하는 `wandb/wsm` GitHub 저장소 (`https://github.com/wandb/wsm`) 에서 다운로드하거나 클론할 수 있습니다. 최신 릴리스는 `wandb/wsm` [릴리스 노트](https://github.com/wandb/wsm/releases)에서 확인하세요.

### 이미지 목록 및 버전 확인

`wsm list` 명령어로 최신 이미지 버전 목록을 확인하세요.

```bash
wsm list
```

출력 예시는 다음과 비슷합니다:

```text
:package: 배포에 필요한 모든 이미지를 나열하는 프로세스를 시작합니다...
Operator Images:
  wandb/controller:1.16.1
W&B Images:
  wandb/local:0.62.2
  docker.io/bitnami/redis:7.2.4-debian-12-r9
  quay.io/prometheus-operator/prometheus-config-reloader:v0.67.0
  quay.io/prometheus/prometheus:v2.47.0
  otel/opentelemetry-collector-contrib:0.97.0
  wandb/console:2.13.1
W&B를 배포하는 데 필요한 이미지는 위와 같습니다. 해당 이미지들이 내부 컨테이너 레지스트리에 있는지 확인하고 values.yaml 파일을 업데이트하세요.
```

### 이미지 다운로드

최신 버전의 모든 이미지는 `wsm download`로 받을 수 있습니다.

```bash
wsm download
```

출력 예시는 다음과 비슷합니다:

```text
운영자 Helm 차트 다운로드 중
wandb Helm 차트 다운로드 중
✓ wandb/controller:1.16.1
✓ docker.io/bitnami/redis:7.2.4-debian-12-r9
✓ otel/opentelemetry-collector-contrib:0.97.0
✓ quay.io/prometheus-operator/prometheus-config-reloader:v0.67.0
✓ wandb/console:2.13.1
✓ quay.io/prometheus/prometheus:v2.47.0

  완료! 7개의 패키지가 설치되었습니다.
```

WSM은 각 이미지를 `.tgz` 아카이브로 `bundle` 디렉토리에 다운로드합니다.

## 3단계: 내부 Helm 차트 저장소 준비

컨테이너 이미지와 함께, 다음 Helm 차트가 내부 Helm Chart 저장소에 있는지 확인해야 합니다. 위 단계에서 소개한 WSM 툴로 Helm 차트도 다운로드할 수 있습니다. 아니면 아래에서 직접 받을 수도 있습니다:

- [W&B Operator](https://github.com/wandb/helm-charts/tree/main/charts/operator)
- [W&B Platform](https://github.com/wandb/helm-charts/tree/main/charts/operator-wandb)

`operator` 차트는 W&B Operator(Controller Manager라고도 함)를 배포할 때 사용합니다. `platform` 차트는 사용자가 구성한 커스텀 리소스 정의(CRD) 값에 따라 W&B Platform을 배포할 때 사용합니다.

## 4단계: Helm 저장소 설정

이제 Helm 저장소를 설정하여 내부 저장소에서 W&B Helm 차트를 가져오도록 해야 합니다. 다음 코맨드로 Helm 저장소를 추가하고 업데이트하세요:

```bash
helm repo add local-repo https://charts.yourdomain.com
helm repo update
```

## 5단계: Kubernetes Operator 설치

W&B Kubernetes Operator(Controller Manager라고도 불림)는 W&B 플랫폼 구성요소를 관리하는 역할을 합니다. 에어갭드 환경에 설치하려면,
내부 컨테이너 레지스트리를 사용하도록 설정해야 합니다.

이를 위해 기본 이미지 설정을 오버라이드하여 내부 컨테이너 레지스트리를 사용하고, `airgapped: true` 키를 설정하여 에어갭드 배포임을 명시해야 합니다. 아래와 같이 `values.yaml` 파일을 업데이트하세요:

```yaml
image:
  repository: registry.yourdomain.com/library/controller
  tag: 1.13.3
airgapped: true
```

내부 레지스트리에 있는 버전으로 tag 값을 교체하세요.

Operator와 CRD를 설치하려면:
```bash
helm upgrade --install operator wandb/operator -n wandb --create-namespace -f values.yaml
```

지원되는 모든 값에 대한 자세한 내용은 [Kubernetes operator GitHub 저장소](https://github.com/wandb/helm-charts/blob/main/charts/operator/values.yaml)에서 확인할 수 있습니다.

## 6단계: W&B Custom Resource 설정

W&B Kubernetes Operator를 설치한 뒤, Custom Resource(CR)를 구성하여 내부 Helm 저장소와 컨테이너 레지스트리를 참조하도록 해야 합니다.

이 설정을 통해 Kubernetes Operator가 W&B 플랫폼에 필요한 모든 구성요소를 배포할 때 내부 레지스트리와 저장소만을 사용하게 됩니다.

아래 예시 CR을 복사해서 `wandb.yaml` 이름의 새 파일로 저장하세요.

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
        name: s3.yourdomain.com:port #예: s3.yourdomain.com:9000
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
    
    # install: true이면 Helm이 MySQL 데이터베이스를 자동 설치합니다. 외부 MySQL을 사용하려면 false로 설정하세요.
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

Kubernetes Operator는 위의 CR 파일에 입력된 값을 사용해 내부 저장소에서 `operator-wandb` Helm 차트를 설정하고 W&B 플랫폼을 배포합니다.

반드시 tag/버전 정보를 내부 레지스트리에서 사용할 수 있는 버전으로 변경하세요.

위와 같이 설정 파일을 만드는 추가 정보는 [여기]({{< relref path="/guides/hosting/hosting-options/self-managed/kubernetes-operator/#configuration-reference-for-wb-server" lang="ko" >}})에서 확인할 수 있습니다.

## 7단계: W&B 플랫폼 배포

이제 Kubernetes Operator와 CR 설정이 완료되었으니, 아래 코맨드로 `wandb.yaml` 설정을 적용하여 W&B 플랫폼을 배포합니다:

```bash
kubectl apply -f wandb.yaml
```

## FAQ

배포 과정에서 자주 묻는 질문과 트러블슈팅 팁을 참고하세요:

### 다른 ingress class가 있습니다. 그 클래스를 사용해도 되나요?
네, `values.yaml`에서 ingress 설정을 수정하여 원하는 ingress class로 구성할 수 있습니다.

### 인증서 번들이 여러 개의 인증서를 포함합니다. 동작하나요?
`values.yaml`의 `customCACerts` 섹션에 인증서를 개별 항목들로 분리해서 입력해야 합니다.

### Kubernetes Operator가 임의로 업데이트를 적용하지 않게 할 수 있나요?
W&B 콘솔에서 자동 업데이트를 끌 수 있습니다. 지원되는 버전에 대한 문의는 W&B 팀으로 연락하세요. W&B는 **셀프 관리(Self-managed)** 인스턴스의 경우 주요 W&B Server 릴리스 후 12개월까지 지원합니다. 고객은 지원 유지 및 업그레이드를 직접 책임져야 하며, 지원이 중단된 버전을 오래 사용하지 말아야 합니다. [릴리스 정책 및 프로세스]({{< relref path="/ref/release-notes/release-policies.md" lang="ko" >}}) 를 참고하세요.

{{% alert %}}
**셀프 관리(Self-managed)** 인스턴스 사용자는 최소 분기마다 최신 버전으로 배포를 업데이트하는 것을 강력히 권장합니다. 그래야 지원과 최신 기능, 성능 개선, 보안 패치를 계속 받을 수 있습니다.
{{% /alert %}}

### 환경이 외부 저장소와 완전히 단절된 경우에도 배포가 동작하나요?
구성이 `airgapped`를 `true`로 설정하면, Kubernetes Operator는 오직 내부 리소스만 사용하고, 공개 저장소에 엑세스를 시도하지 않습니다.