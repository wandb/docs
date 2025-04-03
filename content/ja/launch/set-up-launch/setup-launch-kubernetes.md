---
title: 'Tutorial: Set up W&B Launch on Kubernetes'
menu:
  launch:
    identifier: ja-launch-set-up-launch-setup-launch-kubernetes
    parent: set-up-launch
url: guides/launch/setup-launch-kubernetes
---

W&B Launch を使用して、ML ワークロードを Kubernetes クラスターにプッシュできます。これにより、ML エンジニアは、Kubernetes で既に管理しているリソースを使用するために、W&B 内でシンプルなインターフェースを利用できます。

W&B は、W&B が管理する [公式 Launch agent イメージ](https://hub.docker.com/r/wandb/launch-agent) を保持しており、[Helm chart](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent) を使用してクラスターにデプロイできます。

W&B は [Kaniko](https://github.com/GoogleContainerTools/kaniko) ビルダーを使用して、Launch エージェントが Kubernetes クラスターで Docker イメージを構築できるようにします。Launch エージェント用に Kaniko をセットアップする方法、またはジョブの構築をオフにして、事前構築済みの Docker イメージのみを使用する方法の詳細については、[エージェントの詳細設定]({{< relref path="./setup-agent-advanced.md" lang="ja" >}}) を参照してください。

{{% alert %}}
Helm をインストールし、W&B の Launch エージェント Helm chart を適用またはアップグレードするには、Kubernetes リソースを作成、更新、および削除するための十分な権限を持つ `kubectl` アクセス権がクラスターに必要です。通常、cluster-admin または同等の権限を持つカスタムロールを持つユーザーが必要です。
{{% /alert %}}

## Kubernetes のキューを設定する

Kubernetes ターゲットリソースの Launch キュー設定は、[Kubernetes Job spec](https://kubernetes.io/docs/concepts/workloads/controllers/job/) または [Kubernetes Custom Resource spec](https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/) のいずれかに類似します。

Launch キューを作成するときに、Kubernetes ワークロードリソース spec のあらゆる側面を制御できます。

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
ユースケースによっては、`CustomResource` 定義を使用したい場合があります。たとえば、マルチノード分散トレーニングを実行する場合、`CustomResource` 定義が役立ちます。Volcano を使用したマルチノードジョブで Launch を使用するチュートリアルで、アプリケーションの例を参照してください。別のユースケースとして、W&B Launch を Kubeflow で使用したい場合もあります。

次の YAML スニペットは、Kubeflow を使用するサンプル Launch キュー設定を示しています。

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

セキュリティ上の理由から、W&B は、指定されていない場合、次のリソースを Launch キューに挿入します。

- `securityContext`
- `backOffLimit`
- `ttlSecondsAfterFinished`

次の YAML スニペットは、これらの値が Launch キューにどのように表示されるかを示しています。

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

Kubernetes をコンピューティングリソースとして使用するキューを W&B アプリケーションで作成します。

1. [Launch ページ](https://wandb.ai/launch) に移動します。
2. [**キューを作成**] ボタンをクリックします。
3. キューを作成する **Entity** を選択します。
4. [**名前**] フィールドにキューの名前を入力します。
5. [**リソース**] として **Kubernetes** を選択します。
6. [**設定**] フィールドで、[前のセクションで設定した]({{< relref path="#configure-a-queue-for-kubernetes" lang="ja" >}}) Kubernetes Job ワークフロー spec または Custom Resource spec を指定します。

## Helm で Launch エージェントを設定する

W&B が提供する [Helm chart](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent) を使用して、Launch エージェントを Kubernetes クラスターにデプロイします。`values.yaml` [ファイル](https://github.com/wandb/helm-charts/blob/main/charts/launch-agent/values.yaml) で Launch エージェントの振る舞いを制御します。

通常は Launch エージェント設定ファイル (`~/.config/wandb/launch-config.yaml`) で定義される内容を、`values.yaml` ファイルの `launchConfig` キー内に指定します。

たとえば、Kaniko Docker イメージビルダーを使用する EKS で Launch エージェントを実行できるようにする Launch エージェント設定があるとします。

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

`values.yaml` ファイル内では、次のようになります。

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

レジストリ、環境、および必要なエージェント権限の詳細については、[エージェントの詳細設定]({{< relref path="./setup-agent-advanced.md" lang="ja" >}}) を参照してください。
