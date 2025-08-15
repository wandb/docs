---
title: 'Tutorial: Set up W&B Launch on Kubernetes'
menu:
  launch:
    identifier: ko-launch-set-up-launch-setup-launch-kubernetes
    parent: set-up-launch
url: /ko/guides//launch/setup-launch-kubernetes
---

W&B Launch 를 사용하여 ML 워크로드를 Kubernetes 클러스터로 푸시할 수 있습니다. 이를 통해 ML 엔지니어는 Kubernetes로 이미 관리하고 있는 리소스를 사용할 수 있는 간단한 인터페이스를 W&B 내에서 바로 이용할 수 있습니다.

W&B는 W&B가 관리하는 [공식 Launch agent 이미지](https://hub.docker.com/r/wandb/launch-agent)를 유지 관리하며, 이는 [Helm chart](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent)를 통해 클러스터에 배포할 수 있습니다.

W&B는 [Kaniko](https://github.com/GoogleContainerTools/kaniko) 빌더를 사용하여 Launch agent가 Kubernetes 클러스터에서 Docker 이미지를 빌드할 수 있도록 합니다. Launch agent용 Kaniko 설정 방법 또는 작업 빌드를 끄고 미리 빌드된 Docker 이미지만 사용하는 방법에 대한 자세한 내용은 [고급 agent 설정]({{< relref path="./setup-agent-advanced.md" lang="ko" >}})을 참조하십시오.

{{% alert %}}
Helm을 설치하고 W&B의 Launch agent Helm chart를 적용하거나 업그레이드하려면 Kubernetes 리소스를 생성, 업데이트 및 삭제할 수 있는 충분한 권한으로 클러스터에 대한 `kubectl` 엑세스 권한이 필요합니다. 일반적으로 cluster-admin 권한이 있는 사용자 또는 이와 동등한 권한이 있는 사용자 정의 역할이 필요합니다.
{{% /alert %}}

## Kubernetes용 대기열 설정

Kubernetes 대상 리소스에 대한 Launch 대기열 설정은 [Kubernetes Job spec](https://kubernetes.io/docs/concepts/workloads/controllers/job/) 또는 [Kubernetes Custom Resource spec](https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/)과 유사합니다.

Launch 대기열을 만들 때 Kubernetes 워크로드 리소스 spec의 모든 측면을 제어할 수 있습니다.

{{< tabpane text=true >}}
{{% tab "Kubernetes job spec" %}}
```yaml
spec:
  template:
    spec:
      containers:
        - env:
            - name: MY_ENV_VAR
              value: some-value
          resources:
            requests:
              cpu: 1000m
              memory: 1Gi
metadata:
  labels:
    queue: k8s-test
namespace: wandb
```
{{% /tab %}}
{{% tab "Custom resource spec" %}}
일부 유스 케이스에서는 `CustomResource` 정의를 사용하고 싶을 수 있습니다. 예를 들어, 다중 노드 분산 트레이닝을 수행하려는 경우 `CustomResource` 정의가 유용합니다. Volcano를 사용하여 다중 노드 작업으로 Launch를 사용하는 방법에 대한 튜토리얼에서 예제 애플리케이션을 참조하십시오. 또 다른 유스 케이스는 Kubeflow와 함께 W&B Launch를 사용하려는 경우일 수 있습니다.

다음 YAML 스니펫은 Kubeflow를 사용하는 샘플 Launch 대기열 설정을 보여줍니다.

```yaml
kubernetes:
  kind: PyTorchJob
  spec:
    pytorchReplicaSpecs:
      Master:
        replicas: 1
        template:
          spec:
            containers:
              - name: pytorch
                image: '${image_uri}'
                imagePullPolicy: Always
        restartPolicy: Never
      Worker:
        replicas: 2
        template:
          spec:
            containers:
              - name: pytorch
                image: '${image_uri}'
                imagePullPolicy: Always
        restartPolicy: Never
    ttlSecondsAfterFinished: 600
  metadata:
    name: '${run_id}-pytorch-job'
  apiVersion: kubeflow.org/v1
```
{{% /tab %}}
{{< /tabpane >}}

보안상의 이유로 W&B는 지정되지 않은 경우 다음 리소스를 Launch 대기열에 삽입합니다.

- `securityContext`
- `backOffLimit`
- `ttlSecondsAfterFinished`

다음 YAML 스니펫은 이러한 값들이 Launch 대기열에 어떻게 나타나는지 보여줍니다.

```yaml title="example-spec.yaml"
spec:
  template:
    `backOffLimit`: 0
    ttlSecondsAfterFinished: 60
    securityContext:
      allowPrivilegeEscalation: False,
      capabilities:
        drop:
          - ALL,
      seccompProfile:
        type: "RuntimeDefault"
```

## 대기열 만들기

Kubernetes를 컴퓨팅 리소스로 사용하는 W&B App에서 대기열을 만듭니다.

1. [Launch 페이지](https://wandb.ai/launch)로 이동합니다.
2. **대기열 만들기** 버튼을 클릭합니다.
3. 대기열을 만들려는 **Entities**를 선택합니다.
4. **이름** 필드에 대기열 이름을 입력합니다.
5. **리소스**로 **Kubernetes**를 선택합니다.
6. **설정** 필드 내에서 [이전 섹션에서 구성한]({{< relref path="#configure-a-queue-for-kubernetes" lang="ko" >}}) Kubernetes Job 워크플로우 spec 또는 Custom Resource spec을 제공합니다.

## Helm으로 Launch agent 구성

W&B에서 제공하는 [Helm chart](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent)를 사용하여 Launch agent를 Kubernetes 클러스터에 배포합니다. `values.yaml` [파일](https://github.com/wandb/helm-charts/blob/main/charts/launch-agent/values.yaml)로 Launch agent의 행동을 제어합니다.

Launch agent 구성 파일(`~/.config/wandb/launch-config.yaml`)에 일반적으로 정의되는 내용을 `values.yaml` 파일의 `launchConfig` 키 내에 지정합니다.

예를 들어, Kaniko Docker 이미지 빌더를 사용하는 EKS에서 Launch agent를 실행할 수 있도록 하는 Launch agent 구성이 있다고 가정합니다.

```yaml title="launch-config.yaml"
queues:
	- <queue name>
max_jobs: <n concurrent jobs>
environment:
	type: aws
	region: us-east-1
registry:
	type: ecr
	uri: <my-registry-uri>
builder:
	type: kaniko
	build-context-store: <s3-bucket-uri>
```

`values.yaml` 파일 내에서 다음과 같이 보일 수 있습니다.

```yaml title="values.yaml"
agent:
  labels: {}
  # W&B API 키.
  apiKey: ''
  # agent에 사용할 컨테이너 이미지.
  # 참고: 프로덕션 환경에서는 일관성을 위해 :latest 대신
  # 특정 버전으로 고정하는 것을 고려하세요
  image: wandb/launch-agent:latest
  # agent 이미지에 대한 이미지 풀 정책.
  imagePullPolicy: Always
  # agent spec에 대한 리소스 블록.
  resources:
    limits:
      cpu: 1000m
      memory: 1Gi

# Launch agent를 배포할 네임스페이스
namespace: wandb

# W&B api URL (여기에 설정하십시오)
baseUrl: https://api.wandb.ai

# Launch agent가 배포할 수 있는 추가 대상 네임스페이스
additionalTargetNamespaces:
  - default
  - wandb

# 이것은 Launch agent 구성의 리터럴 내용으로 설정해야 합니다.
launchConfig: |
  queues:
    - <queue name>
  max_jobs: <n concurrent jobs>
  environment:
    type: aws
    region: <aws-region>
  registry:
    type: ecr
    uri: <my-registry-uri>
  builder:
    type: kaniko
    build-context-store: <s3-bucket-uri>

# git 자격 증명 파일의 내용. 이것은 k8s secret에 저장됩니다
# agent 컨테이너에 마운트됩니다. 비공개 리포를 복제하려면 이것을 설정하십시오.
gitCreds: |

# wandb 서비스 계정에 대한 어노테이션. gcp에서 워크로드 아이덴티티를 설정할 때 유용합니다.
serviceAccount:
  annotations:
    iam.gke.io/gcp-service-account:
    azure.workload.identity/client-id:

# azure와 함께 kaniko를 사용하는 경우 azure 스토리지에 대한 엑세스 키로 설정합니다.
azureStorageAccessKey: ''
```

레지스트리, 환경 및 필요한 agent 권한에 대한 자세한 내용은 [고급 agent 설정]({{< relref path="./setup-agent-advanced.md" lang="ko" >}})을 참조하십시오.