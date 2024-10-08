---
title: Tutorial: Set up W&B Launch on Kubernetes
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

W&B Launch를 사용하여 ML 작업을 Kubernetes 클러스터로 전송할 수 있으며, ML 엔지니어는 Kubernetes로 이미 관리하고 있는 리소스를 W&B에서 간단한 인터페이스로 활용할 수 있습니다.

W&B는 [공식 Launch 에이전트 이미지](https://hub.docker.com/r/wandb/launch-agent)를 유지하고 있으며, 이를 W&B가 유지하는 [Helm 차트](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent)를 사용하여 클러스터에 배포할 수 있습니다.

W&B는 [Kaniko](https://github.com/GoogleContainerTools/kaniko) 빌더를 사용하여 Launch 에이전트가 Kubernetes 클러스터에서 Docker 이미지를 빌드할 수 있도록 합니다. Launch 에이전트에 Kaniko를 설정하는 방법 또는 작업 빌드를 비활성화하고 미리 빌드된 Docker 이미지만 사용하는 방법에 대한 자세한 내용은 [고급 에이전트 설정](./setup-agent-advanced.md)을 참조하세요.

:::note
Helm을 설치하고 W&B의 Launch 에이전트 Helm 차트를 적용하거나 업그레이드하려면, Kubernetes 리소스를 생성, 업데이트 및 삭제할 수 있는 충분한 권한을 가진 클러스터에 대한 `kubectl` 엑세스가 필요합니다. 일반적으로 cluster-admin 또는 동등한 권한을 가진 사용자 정의 역할이 필요합니다.
:::

## Kubernetes에 대한 큐 구성

Kubernetes 대상 리소스에 대한 Launch 큐 설정은 [Kubernetes Job 스펙](https://kubernetes.io/docs/concepts/workloads/controllers/job/) 또는 [Kubernetes Custom Resource 스펙](https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/)과 유사할 것입니다.

Launch 큐를 생성할 때 Kubernetes 작업 리소스 스펙의 모든 측면을 제어할 수 있습니다.

<Tabs
defaultValue="job"
values={[
{label: 'Kubernetes Job Spec', value: 'job'},
{label: 'Custom Resource Spec', value: 'custom'},
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

일부 유스 케이스에서 `CustomResource` 정의를 사용하고 싶을 수 있습니다. 예를 들어, 다중 노드 분산 트레이닝을 수행하려면 `CustomResource` 정의가 유용합니다. Volcano를 사용한 다중 노드 작업을 사용하는 Launch의 튜토리얼 예를 참고하세요. 또 다른 유스 케이스는 W&B Launch를 Kubeflow와 함께 사용할 때일 수 있습니다.

다음 YAML 스니펫은 Kubeflow를 사용하는 Launch 큐 설정의 예를 보여줍니다:

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

보안상의 이유로 W&B는 지정되지 않은 경우 다음 리소스를 Launch 큐에 삽입합니다:

- `securityContext`
- `backOffLimit`
- `ttlSecondsAfterFinished`

다음 YAML 스니펫은 Launch 큐에서 이러한 값이 어떻게 나타나는지 보여줍니다:

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

## 큐 생성

W&B 앱에서 Kubernetes를 컴퓨트 리소스로 사용하는 큐를 생성하세요:

1. [Launch 페이지](https://wandb.ai/launch)로 이동합니다.
2. **Create Queue** 버튼을 클릭합니다.
3. 큐를 생성할 **Entity**를 선택합니다.
4. **Name** 필드에 큐의 이름을 입력합니다.
5. **Resource**로 **Kubernetes**를 선택합니다.
6. **Configuration** 필드에 [이전 섹션에서 구성한 것](#configure-a-queue-for-kubernetes)과 같은 Kubernetes Job 워크플로우 스펙 또는 Custom Resource 스펙을 입력합니다.

## Helm을 사용하여 Launch 에이전트 구성

W&B가 제공하는 [Helm 차트](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent)를 사용하여 Kubernetes 클러스터에 Launch 에이전트를 배포합니다. `values.yaml` [파일](https://github.com/wandb/helm-charts/blob/main/charts/launch-agent/values.yaml)로 Launch 에이전트의 행동을 제어합니다.

일반적으로 Launch 에이전트 구성 파일(`~/.config/wandb/launch-config.yaml`)에 정의되는 내용을 `values.yaml` 파일의 `launchConfig` 키 안에 지정하세요.

예를 들어, Kaniko Docker 이미지 빌더를 사용하는 EKS에서 Launch 에이전트를 실행할 수 있는 Launch 에이전트 구성을 가지고 있다고 가정합니다:

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

`values.yaml` 파일 내에서 다음과 같이 나타날 수 있습니다:

```yaml title="values.yaml"
agent:
  labels: {}
  # W&B API key.
  apiKey: ''
  # Container image to use for the agent.
  image: wandb/launch-agent:latest
  # Image pull policy for agent image.
  imagePullPolicy: Always
  # Resources block for the agent spec.
  resources:
    limits:
      cpu: 1000m
      memory: 1Gi

# Namespace to deploy launch agent into
namespace: wandb

# W&B api url (Set yours here)
baseUrl: https://api.wandb.ai

# Additional target namespaces that the launch agent can deploy into
additionalTargetNamespaces:
  - default
  - wandb

# This should be set to the literal contents of your launch agent config.
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

# The contents of a git credentials file. This will be stored in a k8s secret
# and mounted into the agent container. Set this if you want to clone private
# repos.
gitCreds: |

# Annotations for the wandb service account. Useful when setting up workload identity on gcp.
serviceAccount:
  annotations:
    iam.gke.io/gcp-service-account:
    azure.workload.identity/client-id:

# Set to access key for azure storage if using kaniko with azure.
azureStorageAccessKey: ''
```

레지스트리, 환경 및 필수 에이전트 권한에 대한 자세한 내용은 [고급 에이전트 설정](./setup-agent-advanced.md)을 참조하세요.