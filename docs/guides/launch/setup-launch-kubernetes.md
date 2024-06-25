---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Kubernetes のセットアップ

W&B Launch を使用すると、ML エンジニアが既に Kubernetes で管理しているリソースを W&B からシンプルなインターフェースで利用して ML ワークロードを Kubernetes クラスターにプッシュできます。

W&B は [公式の Launch エージェントイメージ](https://hub.docker.com/r/wandb/launch-agent) を維持しており、これは W&B が管理する [Helm チャート](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent) を使用してクラスターにデプロイできます。

W&B は [Kaniko](https://github.com/GoogleContainerTools/kaniko) ビルダーを使用して、Launch エージェントが Kubernetes クラスターで Docker イメージをビルドできるようにします。Launch エージェント用の Kaniko のセットアップ方法、またはジョブのビルドを無効にしてプリビルトの Docker イメージのみを使用する方法については、[高度なエージェントのセットアップ](./setup-agent-advanced.md) を参照してください。

## Kubernetes 用のキューを設定する

Kubernetes ターゲットリソースに対する Launch キューの設定は、[Kubernetes Job spec](https://kubernetes.io/docs/concepts/workloads/controllers/job/) または [Kubernetes Custom Resource spec](https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/) に似たものになります。

Launch キューを作成するときに Kubernetes ワークロードリソース spec のあらゆる面を制御できます。

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

特定のユースケースでは、`CustomResource` 定義を使用することがあります。`CustomResource` 定義は、たとえばマルチノード分散トレーニングを実行したい場合に役立ちます。Volcano を使用してマルチノードジョブで Launch を使用するためのチュートリアルを参照してください。別のユースケースは、W&B Launch を Kubeflow と一緒に使用したい場合かもしれません。

以下の YAML スニペットは、Kubeflow を使用するサンプルの Launch キュー設定を示しています。

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

セキュリティ上の理由から、以下のリソースは指定されていない場合に W&B が自動的に Launch キューに挿入します。

- `securityContext`
- `backOffLimit`
- `ttlSecondsAfterFinished`

以下の YAML スニペットは、これらの値が Launch キューにどのように表示されるかを示しています。

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

## キューを作成する

Kubernetes を計算リソースとして使用するキューを W&B アプリで作成します。

1. [Launch ページ](https://wandb.ai/launch) に移動します。
2. **Create Queue** ボタンをクリックします。
3. キューを作成したい **Entity** を選択します。
4. **Name** フィールドにキューの名前を入力します。
5. **Resource** として **Kubernetes** を選択します。
6. **Configuration** フィールドに、[前のセクションで設定した](#configure-a-queue-for-kubernetes) Kubernetes ジョブワークフローやカスタムリソースの spec を入力します。

## Helm を使用して Launch エージェントを構成する

W&B が提供する [Helm チャート](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent) を使用して、Kubernetes クラスターに Launch エージェントをデプロイします。`values.yaml` [ファイル](https://github.com/wandb/helm-charts/blob/main/charts/launch-agent/values.yaml) で Launch エージェントの振る舞いを制御します。

Launch エージェントの設定ファイル（`~/.config/wandb/launch-config.yaml`）に通常記載される内容を `values.yaml` ファイル内の `launchConfig` キー内に指定します。

例えば、Kaniko Docker イメージビルダーを使用する EKS で Launch エージェントを実行するための設定があるとします。

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

`values.yaml` ファイル内では、このように見えるかもしれません。

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

レジストリ、環境、および必要なエージェント権限に関する詳細は、[高度なエージェントのセットアップ](./setup-agent-advanced.md) を参照してください。