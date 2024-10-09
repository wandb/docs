---
description: 에어갭으로 W&B 플랫폼을 Kubernetes Operator로 배포하기
displayed_sidebar: default
---

# 폐쇄망 인스턴스를 위한 Kubernetes 오퍼레이터

## 도입

이 가이드는 폐쇄망 고객 관리 환경에서 W&B 플랫폼을 배포하는 단계별 지침을 제공합니다.

Helm 차트와 컨테이너 이미지를 호스팅하기 위해 내부 저장소 또는 레지스트리를 사용하세요. 적절한 Kubernetes 클러스터 엑세스 권한이 있는 쉘 콘솔에서 모든 코맨드를 실행합니다.

Kubernetes 애플리케이션을 배포하는 데 사용하는 연속 전달 툴링에서도 유사한 코맨드를 사용할 수 있습니다.

## 단계 1: 사전 조건

시작하기 전에, 다음 요구 사항을 충족하는지 환경을 확인하세요:

- Kubernetes 버전 >= 1.28
- Helm 버전 >= 3
- 필요한 W&B 이미지와 내부 컨테이너 레지스트리에 대한 엑세스
- W&B Helm 차트를 위한 내부 Helm 저장소 엑세스

## 단계 2: 내부 컨테이너 레지스트리 준비

배포를 진행하기 전에, 다음 컨테이너 이미지가 내부 컨테이너 레지스트리에 있는지 확인해야 합니다.
이 이미지는 W&B 구성 요소의 성공적인 배포에 필수적입니다.

```bash
wandb/local                                             0.59.2
wandb/console                                           2.12.2
wandb/controller                                        1.13.0
otel/opentelemetry-collector-contrib                    0.97.0
bitnami/redis                                           7.2.4-debian-12-r9
quay.io/prometheus/prometheus                           v2.47.0
quay.io/prometheus-operator/prometheus-config-reloader  v0.67.0
```

## 단계 2: 내부 Helm 차트 저장소 준비

컨테이너 이미지와 함께, 다음 Helm 차트가 내부 Helm 차트 저장소에 있는지 확인해야 합니다.

- [W&B Operator](https://github.com/wandb/helm-charts/tree/main/charts/operator)
- [W&B Platform](https://github.com/wandb/helm-charts/tree/main/charts/operator-wandb)

`operator` 차트는 W&B 오퍼레이터, 또는 컨트롤러 매니저를 배포하는 데 사용됩니다. `platform` 차트는 커스텀 리소스 정의(CRD)에서 설정된 값을 사용하여 W&B 플랫폼을 배포하는 데 사용됩니다.

## 단계 3: Helm 저장소 설정

이제 Helm 저장소를 설정하여 내부 저장소에서 W&B Helm 차트를 가져옵니다. Helm 저장소를 추가하고 업데이트하기 위해 다음 코맨드를 실행하세요:

```bash
helm repo add local-repo https://charts.yourdomain.com
helm repo update
```

## 단계 4: Kubernetes 오퍼레이터 설치

W&B Kubernetes 오퍼레이터, 즉 컨트롤러 매니저는 W&B 플랫폼 구성 요소를 관리하는 책임이 있습니다. 폐쇄망 환경에서 이를 설치하려면 내부 컨테이너 레지스트리를 사용하도록 설정해야 합니다.

이를 위해 기본 이미지 설정을 내부 컨테이너 레지스트리를 사용하도록 재정의하고 배포 유형을 나타내기 위해 키 `airgapped: true`를 설정해야 합니다. `values.yaml` 파일을 아래와 같이 업데이트하세요:

```yaml
image:
  repository: registry.yourdomain.com/library/controller
  tag: 1.13.3
airgapped: true
```

지원되는 모든 값은 [공식 Kubernetes operator 저장소](https://github.com/wandb/helm-charts/blob/main/charts/operator/values.yaml)에서 찾을 수 있습니다.

## 단계 5: CustomResourceDefinitions 구성

W&B Kubernetes 오퍼레이터를 설치한 후에는 Custom Resource Definitions (CRDs)를 내부 Helm 저장소와 컨테이너 레지스트리를 가리키도록 구성해야 합니다.

이 설정은 Kubernetes 오퍼레이터가 필요한 W&B 플랫폼 구성 요소를 배포할 때 내부 레지스트리와 저장소를 사용하도록 합니다.

아래는 CRD를 구성하는 예시입니다.

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
        name: s3.yourdomain.com
        provider: s3
      mysql:
        database: wandb
        host: mysql.home.lab
        password: password
        port: 3306
        user: wandb
    
    # Ensre it's set to use your own MySQL
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

W&B 플랫폼을 배포하기 위해 Kubernetes 오퍼레이터는 내부 저장소에서 `operator-wandb` 차트를 사용하고 CRD의 값을 사용하여 Helm 차트를 구성합니다.

지원되는 모든 값은 [공식 Kubernetes operator 저장소](https://github.com/wandb/helm-charts/blob/main/charts/operator/values.yaml)에서 찾을 수 있습니다.

## 단계 6: W&B 플랫폼 배포

마지막으로, Kubernetes 오퍼레이터와 CRD를 설정한 후 다음 코맨드를 사용하여 W&B 플랫폼을 배포하세요:

```bash
kubectl apply -f wandb.yaml
```

## 문제 해결과 FAQ

배포 프로세스 중에 자주 묻는 질문(FAQ)과 문제 해결 팁은 아래를 참조하세요:

**다른 인그레스 클래스가 있습니다. 그 클래스를 사용할 수 있나요?**  
네, `values.yaml`의 인그레스 설정을 수정하여 인그레스 클래스를 구성할 수 있습니다.

**인증서 번들에 하나 이상의 인증서가 있습니다. 그것이 작동할까요?**  
인증서는 `values.yaml`의 `customCACerts` 섹션에 여러 항목으로 나누어야 합니다.

**Kubernetes 오퍼레이터가 무단 업데이트를 적용하지 않도록 하려면 어떻게 하나요? 그게 가능한가요?**  
W&B 콘솔에서 자동 업데이트를 끌 수 있습니다. 지원되는 버전에 대한 질문이 있으면 W&B 팀에 문의하세요. 또한, W&B는 최근 6개월 이내에 발표된 플랫폼 버전을 지원합니다. W&B는 주기적인 업그레이드를 권장합니다.

**환경이 공개 저장소에 연결되어 있지 않을 경우 배포가 작동할까요?**  
`airgapped: true` 설정을 활성화한 경우, Kubernetes 오퍼레이터는 공개 저장소에 접근하지 않습니다. Kubernetes 오퍼레이터는 내부 리소스를 사용하려고 시도합니다.