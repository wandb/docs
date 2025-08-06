---
title: '튜토리얼: Kubernetes에서 W&B Launch 설정하기'
menu:
  launch:
    identifier: ko-launch-set-up-launch-setup-launch-kubernetes
    parent: set-up-launch
url: guides/launch/setup-launch-kubernetes
---

W&B Launch를 사용하면 ML workload를 Kubernetes 클러스터로 손쉽게 전송할 수 있어, ML 엔지니어가 W&B 내에서 여러분이 이미 Kubernetes로 관리하고 있는 리소스를 쉽게 활용할 수 있습니다.

W&B에서는 [공식 Launch agent 이미지](https://hub.docker.com/r/wandb/launch-agent)를 제공하며, 이 이미지는 W&B에서 관리하는 [Helm chart](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent)를 통해 클러스터에 배포할 수 있습니다.

W&B는 [Kaniko](https://github.com/GoogleContainerTools/kaniko) 빌더를 사용하여 Launch agent가 Kubernetes 클러스터 내에서 Docker 이미지를 빌드할 수 있도록 지원합니다. Launch agent용 Kaniko 설정 방법이나, job 빌드 없이 미리 빌드된 Docker 이미지만 사용하도록 설정하는 방법은 [고급 에이전트 설정]({{< relref path="./setup-agent-advanced.md" lang="ko" >}})에서 확인하세요.

{{% alert %}}
Helm을 설치하고 W&B Launch agent Helm chart를 적용 또는 업그레이드하려면, 해당 클러스터에 대해 `kubectl` 엑세스 권한이 필요하며, Kubernetes 리소스를 생성, 업데이트, 삭제할 수 있는 충분한 퍼미션이 요구됩니다. 일반적으로 cluster-admin 또는 이에 준하는 권한의 사용자 계정(혹은 커스텀 롤)이 필요합니다.
{{% /alert %}}


## Kubernetes용 큐 설정하기

Kubernetes 타겟 리소스를 위한 Launch 큐 설정은 [Kubernetes Job spec](https://kubernetes.io/docs/concepts/workloads/controllers/job/) 혹은 [Kubernetes Custom Resource spec](https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/) 중 하나와 유사한 형태를 가집니다.

Launch 큐를 만들 때 Kubernetes workload 리소스 스펙의 모든 항목을 제어할 수 있습니다.

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
특정 유스 케이스에서는 `CustomResource` 정의를 사용할 수 있습니다. 예를 들어, 다중 노드 분산 트레이닝을 하고 싶을 때 유용합니다. Volcano를 사용한 멀티노드 Launch job 튜토리얼에서 애플리케이션 예시를 참고하세요. 또 다른 유스 케이스로 Kubeflow와 함께 W&B Launch를 사용하고 싶을 때도 `CustomResource` 정의가 활용될 수 있습니다.

아래 YAML은 Kubeflow를 사용하는 Launch 큐 설정 예시입니다.

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

보안상의 이유로, 지정하지 않은 경우 W&B에서는 Launch 큐에 아래 리소스 항목들을 자동으로 추가합니다:

- `securityContext`
- `backOffLimit`
- `ttlSecondsAfterFinished`

Launch 큐에서 해당 값들이 적용되는 예시는 다음과 같습니다:

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

## 큐 생성하기

Kubernetes를 연산 리소스로 사용하는 큐를 W&B App에서 생성하려면 다음과 같이 하세요:

1. [Launch 페이지](https://wandb.ai/launch)로 이동합니다.
2. **Create Queue** 버튼을 클릭합니다.
3. 큐를 생성할 **Entity**를 선택합니다.
4. **Name** 필드에 큐 이름을 입력합니다.
5. **Kubernetes**를 **Resource**로 선택합니다.
6. **Configuration** 필드에 [이전 섹션에서 설정한]({{< relref path="#configure-a-queue-for-kubernetes" lang="ko" >}}) Kubernetes Job workflow spec 혹은 Custom Resource spec을 입력합니다.

## Helm으로 Launch agent 설정하기

W&B가 제공하는 [Helm chart](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent)를 사용하여 Launch agent를 Kubernetes 클러스터에 배포할 수 있습니다. launch agent의 행동은 `values.yaml` [파일](https://github.com/wandb/helm-charts/blob/main/charts/launch-agent/values.yaml)로 제어할 수 있습니다.

기존에 launch agent 설정 파일(`~/.config/wandb/launch-config.yaml`)에 정의했던 내용을 `values.yaml` 파일 내 `launchConfig` 키에 지정하면 됩니다.

예를 들어, 다음은 Kaniko Docker 이미지 빌더를 사용하는 Launch agent를 EKS에서 실행하도록 하는 설정 예시입니다:

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

여기서 `values.yaml` 파일에서는 다음과 같이 지정할 수 있습니다:

```yaml title="values.yaml"
agent:
  labels: {}
  # W&B API 키입니다.
  apiKey: ''
  # agent에 사용할 컨테이너 이미지입니다.
  image: wandb/launch-agent:latest
  # agent 이미지의 pull 정책입니다.
  imagePullPolicy: Always
  # agent spec의 리소스 블록입니다.
  resources:
    limits:
      cpu: 1000m
      memory: 1Gi

# launch agent를 배포할 네임스페이스입니다.
namespace: wandb

# W&B api url (여기에 본인의 값을 설정)
baseUrl: https://api.wandb.ai

# launch agent가 배포될 추가 타겟 네임스페이스
additionalTargetNamespaces:
  - default
  - wandb

# launch agent 설정 파일의 내용을 그대로 입력하세요.
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

# git credentials 파일 내용입니다. 이 내용은 k8s secret으로 저장되어 agent 컨테이너에 마운트됩니다.
# private 저장소를 클론해야 할 경우 사용합니다.
gitCreds: |

# wandb 서비스 계정용 annotation입니다. gcp에 workload identity 설정 시 유용합니다.
serviceAccount:
  annotations:
    iam.gke.io/gcp-service-account:
    azure.workload.identity/client-id:

# kaniko를 azure와 함께 쓸 경우 azure storage access 키를 설정하세요.
azureStorageAccessKey: ''
```

레지스트리, 환경, 에이전트 권한에 대해 더 자세한 사항은 [고급 에이전트 설정]({{< relref path="./setup-agent-advanced.md" lang="ko" >}})을 참고하세요.