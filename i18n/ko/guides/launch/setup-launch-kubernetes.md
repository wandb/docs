---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Kubernetes 설정

W&B Launch를 사용하여 ML 워크로드를 Kubernetes 클러스터로 푸시할 수 있으므로, ML 엔지니어는 Kubernetes로 관리하는 리소스를 사용하여 W&B에서 간단한 인터페이스를 사용할 수 있습니다.


W&B는 [공식 Launch 에이전트 이미지](https://hub.docker.com/r/wandb/launch-agent)를 유지 관리하며, 이를 W&B에서 유지 관리하는 [Helm 차트](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent)로 클러스터에 배포할 수 있습니다.


W&B는 Launch 에이전트가 Kubernetes 클러스터에서 Docker 이미지를 빌드할 수 있도록 [Kaniko](https://github.com/GoogleContainerTools/kaniko) 빌더를 사용합니다. Launch 에이전트를 위해 Kaniko를 설정하는 방법 또는 작업 빌딩을 비활성화하고 사전 빌드된 Docker 이미지만 사용하는 방법에 대해 자세히 알아보려면 [고급 에이전트 설정](./setup-agent-advanced.md)을 참조하세요.

## Kubernetes용 큐 설정

Kubernetes 대상 리소스에 대한 Launch 큐 설정은 [Kubernetes Job 스펙](https://kubernetes.io/docs/concepts/workloads/controllers/job/) 또는 [Kubernetes 사용자 정의 리소스 스펙](https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/)과 유사하게 보일 것입니다.


Launch 큐를 생성할 때 Kubernetes 워크로드 리소스 스펙의 모든 측면을 제어할 수 있습니다.

<Tabs
defaultValue="job"
values={[
{label: 'Kubernetes Job 스펙', value: 'job'},
{label: '사용자 정의 리소스 스펙', value: 'custom'},
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

일부 유스 케이스에서는 `CustomResource` 정의를 사용하고 싶을 수 있습니다. `CustomResource` 정의는 예를 들어, 멀티 노드 분산 트레이닝을 수행하고 싶을 때 유용합니다. Volcano를 사용하여 멀티 노드 작업을 사용하는 Launch 튜토리얼을 예시 애플리케이션으로 참조하세요. 또 다른 유스 케이스는 Kubeflow와 함께 W&B Launch를 사용하고자 하는 경우일 수 있습니다.

다음 YAML 스니펫은 Kubeflow를 사용하는 샘플 Launch 큐 설정을 보여줍니다:

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

보안상의 이유로, W&B는 다음 리소스를 Launch 큐에 지정하지 않은 경우, 해당 리소스를 삽입할 것입니다:

- `securityContext`
- `backOffLimit`
- `ttlSecondsAfterFinished`

다음 YAML 스니펫은 이러한 값이 Launch 큐에 어떻게 표시될지 보여줍니다:

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

Kubernetes를 컴퓨팅 리소스로 사용하는 W&B 앱에서 큐를 생성합니다:

1. [Launch 페이지](https://wandb.ai/launch)로 이동합니다.
2. **큐 생성** 버튼을 클릭합니다.
3. 큐를 생성하고자 하는 **엔티티**를 선택합니다.
4. **이름** 필드에 큐의 이름을 입력합니다.
5. **리소스**로 **Kubernetes**를 선택합니다.
6. **설정** 필드에, [이전 섹션에서 설정한](#configure-a-queue-for-kubernetes) Kubernetes Job 워크플로우 스펙 또는 사용자 정의 리소스 스펙을 제공합니다.

## Helm으로 Launch 에이전트 설정

W&B에서 제공하는 [Helm 차트](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent)를 사용하여 Kubernetes 클러스터에 Launch 에이전트를 배포합니다. `values.yaml` [파일](https://github.com/wandb/helm-charts/blob/main/charts/launch-agent/values.yaml)로 launch 에이전트의 행동을 제어합니다.

Launch 에이전트 설정 파일(`~/.config/wandb/launch-config.yaml`)에 일반적으로 정의되는 내용을 `values.yaml` 파일의 `launchConfig` 키 내에 지정하십시오.

예를 들어, Kaniko Docker 이미지 빌더를 사용하여 EKS에서 Launch 에이전트를 실행할 수 있도록 설정하는 Launch 에이전트 설정이 있다고 가정해 보겠습니다:

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

`values.yaml` 파일 내에서 이는 다음과 같이 보일 수 있습니다:

```yaml title="values.yaml"
agent:
  labels: {}
  # W&B API 키.
  apiKey: ''
  # 에이전트용 컨테이너 이미지.
  image: wandb/launch-agent:latest
  # 에이전트 이미지의 이미지 풀 정책.
  imagePullPolicy: Always
  # 에이전트 스펙의 리소스 블록.
  resources:
    limits:
      cpu: 1000m
      memory: 1Gi

# launch 에이전트를 배포할 네임스페이스
namespace: wandb

# W&B api url (여기에 설정하세요)
baseUrl: https://api.wandb.ai

# launch 에이전트가 배포할 수 있는 추가 대상 네임스페이스
additionalTargetNamespaces:
  - default
  - wandb

# 이것은 launch 에이전트 설정의 실제 내용을 설정해야 합니다.
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

# git 자격 증명 파일의 내용입니다. 이것은 k8s 비밀로 저장되고 에이전트 컨테이너에 마운트됩니다. 비공개
# 저장소를 복제하려면 이를 설정하세요.
gitCreds: |

# gcp에서 작업 신원을 설정할 때 유용한 wandb 서비스 계정의 주석.
serviceAccount:
  annotations:
    iam.gke.io/gcp-service-account:
    azure.workload.identity/client-id:

# kaniko를 azure와 함께 사용하는 경우 azure 저장소 엑세스 키를 설정하세요.
azureStorageAccessKey: ''
```

레지스트리, 환경 및 필요한 에이전트 권한에 대한 자세한 내용은 [고급 에이전트 설정](./setup-agent-advanced.md)을 참조하십시오.