---
title: 'Tutorial: Set up W&B Launch on Kubernetes'
menu:
  launch:
    identifier: ja-launch-set-up-launch-setup-launch-kubernetes
    parent: set-up-launch
url: guides/launch/setup-launch-kubernetes
---

W&B Launch を使用して、ML ワークロードを Kubernetes クラスターにプッシュすることができます。これにより、ML エンジニアは W&B でシンプルなインターフェースを使用して、Kubernetes ですでに管理しているリソースを使用できます。

W&B は、W&B が管理している [公式の Launch agent イメージ](https://hub.docker.com/r/wandb/launch-agent) を提供しており、[Helm chart](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent) を使用してクラスターにデプロイできます。

W&B は、Kubernetes クラスターで Launch agent が Docker イメージをビルドできるようにするために、[Kaniko](https://github.com/GoogleContainerTools/kaniko) ビルダーを使用しています。Launch agent 用に Kaniko をセットアップする方法、またはジョブビルディングをオフにして事前にビルドされた Docker イメージのみを使用する方法については、[エージェントの詳細設定]({{< relref path="./setup-agent-advanced.md" lang="ja" >}})を参照してください。

{{% alert %}}
Helm をインストールし、W&B の Launch agent Helm chart を適用またはアップグレードするには、Kubernetes リソースを作成、更新、および削除するための十分なアクセス許可を持つクラスターへの `kubectl` アクセスが必要です。通常、cluster-admin または同等の権限を持つカスタムロールを持つユーザーが必要です。
{{% /alert %}}


## Kubernetes のキューを設定する

Kubernetes ターゲットリソースの Launch キュー設定は、[Kubernetes Job spec](https://kubernetes.io/docs/concepts/workloads/controllers/job/) または [Kubernetes Custom Resource spec](https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/) のいずれかに似ているでしょう。

Launch キューを作成するときに、Kubernetes ワークロードリソース spec の任意の側面を制御できます。

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
いくつかのユースケースでは、`CustomResource` 定義を使用したい場合があります。例えば、マルチノード分散トレーニングを実行したい場合に `CustomResource` 定義が便利です。Volcano を使用したマルチノードジョブの Launch のユースケース例については、チュートリアルを参照してください。また、W&B Launch を Kubeflow と共に使用したい場合もあるかもしれません。

以下の YAML スニペットは、Kubeflow を使用する Launch キュー設定のサンプルを示しています：

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

セキュリティ上の理由から、W&B は Launch キューに以下のリソースを指定されていない場合、注入します：

- `securityContext`
- `backOffLimit`
- `ttlSecondsAfterFinished`

以下の YAML スニペットは、これらの値が Launch キューにどのように表示されるかを示しています：

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

Kubernetes を計算リソースとして使用する W&B アプリ内でキューを作成します：

1. [Launch ページ](https://wandb.ai/launch) に移動します。
2. **Create Queue** ボタンをクリックします。
3. キューを作成したい **Entity** を選択します。
4. **Name** フィールドにキューの名前を入力します。
5. **Resource** として **Kubernetes** を選択します。
6. **Configuration** フィールド内に、[前のセクションで設定した]({{< relref path="#configure-a-queue-for-kubernetes" lang="ja" >}}) Kubernetes Job ワークフロースペックまたは Custom Resource spec を入力します。

## Helm を使用して Launch agent を設定する

W&B が提供する [Helm chart](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent) を使用して、Launch agent を Kubernetes クラスターにデプロイします。`values.yaml` [ファイル](https://github.com/wandb/helm-charts/blob/main/charts/launch-agent/values.yaml) を使用して、launch agent の振る舞いを制御します。

通常は launch agent 設定ファイル (`~/.config/wandb/launch-config.yaml`) に定義されるコンテンツを、`values.yaml` ファイル内の `launchConfig` キーに指定します。

例えば、Kaniko Docker イメージビルダーを使用する EKS で Launch エージェントを実行できる Launch agent 設定があると仮定します：

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

`values.yaml` ファイル内では、次のように表示されるかもしれません：

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

レジストリ、環境、および必要なエージェント権限に関する詳細は、[エージェントの詳細設定]({{< relref path="./setup-agent-advanced.md" lang="ja" >}}) を参照してください。