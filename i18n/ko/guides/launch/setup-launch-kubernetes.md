---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Kubernetes 설정하기

W&B Launch를 사용하면 Kubernetes 클러스터 내에서 [Kubernetes Job](https://kubernetes.io/docs/concepts/workloads/controllers/job/) 또는 [Custom workload](https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/) 리소스로 W&B Launch 작업을 실행할 수 있습니다. 이는 특히 컴퓨팅 클러스터를 관리하기 위해 Kubernetes를 사용하고자 하며 클러스터에서 학습, 변환 또는 ML 워크플로를 실행하기 위한 간단한 인터페이스를 원하는 경우 유용합니다.

W&B는 [helm chart](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent)를 통해 클러스터에 배포할 수 있는 [공식 런치 에이전트 이미지](https://hub.docker.com/r/wandb/launch-agent)를 유지 관리하고 있습니다.

:::info
Kubernetes 클러스터 외부에서 런치 에이전트를 시작하는 것이 가능합니다. 하지만, Kubernetes Job 또는 Custom workload를 실행하는 책임이 있는 Kubernetes 클러스터 내에 런치 에이전트를 배포하는 것이 좋습니다.
:::

런치 에이전트는 현재 Kubernetes 클러스터 컨텍스트에 지정된 클러스터에 워크로드를 제출합니다.

W&B는 [Kaniko](https://github.com/GoogleContainerTools/kaniko) 빌더를 사용하여 런치 에이전트가 Kubernetes 클러스터 내에서 Docker 이미지를 빌드할 수 있도록 합니다. 런치 에이전트를 위해 Kaniko를 설정하는 방법에 대한 자세한 내용은 [고급 에이전트 설정](./setup-agent-advanced.md)을 참조하십시오.

## Kubernetes에 대한 큐 구성

Kubernetes 대상 리소스에 대한 런치 큐 구성은 Kubernetes Job 스펙 또는 Kubernetes Custom Resource 스펙 중 하나와 유사할 것입니다. 런치 큐를 생성할 때 Kubernetes 워크로드 리소스 스펙의 모든 측면을 제어할 수 있습니다.

<Tabs
defaultValue="job"
values={[
{label: 'Kubernetes Job 스펙', value: 'job'},
{label: 'Custom Resource 스펙', value: 'custom'},
]}>

<TabItem value="job">

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

</TabItem>
<TabItem value="custom">

일부 사용 사례에서는 `CustomResource` 정의를 사용하고 싶을 수 있습니다. 예를 들어, 멀티 노드 분산 학습을 수행하고자 하는 경우 `CustomResource` 정의가 유용할 수 있습니다. Volcano를 사용한 멀티 노드 작업을 사용하여 Launch와 함께 사용하는 튜토리얼을 참조하십시오. 또 다른 사용 사례는 Kubeflow와 함께 W&B Launch를 사용하고자 할 경우일 수 있습니다.

다음 YAML 스니펫은 Kubeflow를 사용하는 샘플 런치 큐 구성을 보여줍니다:

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

  </TabItem>
</Tabs>

보안상의 이유로, 다음 리소스가 명시되어 있지 않은 경우 W&B는 런치 큐에 다음 리소스를 주입할 것입니다:

- `securityContext`
- `backOffLimit`
- `ttlSecondsAfterFinished`

다음 YAML 스니펫은 런치 큐에 이 값들이 어떻게 나타날지 보여줍니다:

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

Kubernetes를 컴퓨팅 리소스로 사용하는 W&B 앱에서 큐를 생성하십시오:

1. [Launch 페이지](https://wandb.ai/launch)로 이동합니다.
2. **큐 생성** 버튼을 클릭합니다.
3. 큐를 생성하고자 하는 **엔터티**를 선택합니다.
4. **이름** 필드에 큐의 이름을 입력합니다.
5. **리소스**로 **Kubernetes**를 선택합니다.
6. **구성** 필드에 [이전 섹션에서 구성한](#configure-a-queue-for-kubernetes) Kubernetes Job 워크플로 스펙 또는 Custom Resource 스펙을 제공합니다.

## helm을 사용하여 런치 에이전트 구성하기

W&B가 제공하는 helm 차트를 사용하여 Kubernetes 클러스터에 런치 에이전트를 배포하십시오. `values.yaml` [파일](https://github.com/wandb/helm-charts/blob/main/charts/launch-agent/values.yaml)으로 런치 에이전트의 동작을 제어합니다.

런치 에이전트 구성 파일(`~/.config/wandb/launch-config.yaml`)에 일반적으로 정의될 내용을 `values.yaml` 파일의 `launchConfig` 키 내에 지정하십시오.

예를 들어, Kaniko Docker 이미지 빌더를 사용하는 EKS에서 런치 에이전트를 실행할 수 있는 런치 에이전트 구성이 있다고 가정해 보겠습니다:

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

`values.yaml` 파일에서는 다음과 같이 보일 것입니다:

```yaml title="values.yaml"
agent:
  labels: {}
  # W&B API 키.
  apiKey: ''
  # 에이전트에 사용할 컨테이너 이미지.
  image: wandb/launch-agent:latest
  # 에이전트 이미지의 이미지 풀 정책.
  imagePullPolicy: Always
  # 에이전트 스펙의 리소스 블록.
  resources:
    limits:
      cpu: 1000m
      memory: 1Gi

# 런치 에이전트를 배포할 네임스페이스
namespace: wandb

# W&B api url (여기에 설정하세요)
baseUrl: https://api.wandb.ai

# 런치 에이전트가 배포할 수 있는 추가 대상 네임스페이스
additionalTargetNamespaces:
  - default
  - wandb

# 런치 에이전트 구성의 실제 내용이어야 합니다.
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

# git 자격 증명 파일의 내용입니다. 이는 k8s 비밀로 저장되어 에이전트 컨테이너에 마운트됩니다. 비공개
# 리포지토리를 클론하려는 경우 이를 설정하세요.
gitCreds: |

# gcp에서 작업 신원을 설정할 때 유용한 wandb 서비스 계정의 주석.
serviceAccount:
  annotations:
    iam.gke.io/gcp-service-account:
    azure.workload.identity/client-id:

# kaniko를 azure와 함께 사용하는 경우 azure 저장소의 엑세스 키로 설정합니다.
azureStorageAccessKey: ''
```

레지스트리, 환경 및 필요한 에이전트 권한에 대한 자세한 정보는 [고급 에이전트 설정](./setup-agent-advanced.md)을 참조하십시오.

[helm 차트 저장소](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent)의 지침을 따라 에이전트를 배포하십시오.