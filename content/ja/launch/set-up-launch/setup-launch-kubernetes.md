---
title: 'チュートリアル: Kubernetes上でW&B ローンンチ を設定する'
menu:
  launch:
    identifier: ja-launch-set-up-launch-setup-launch-kubernetes
    parent: set-up-launch
url: /ja/guides/launch/setup-launch-kubernetes
---

W&B Launch を使用して、ML エンジニアが Kubernetes ですでに管理しているリソースを簡単に利用できるようにし、Kubernetes クラスターに ML ワークロードをプッシュできます。

W&B は、あなたのクラスターにデプロイできる公式の [Launch agent イメージ](https://hub.docker.com/r/wandb/launch-agent) を提供しており、これは W&B が管理する [Helm チャート](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent) を使用します。

W&B は [Kaniko](https://github.com/GoogleContainerTools/kaniko) ビルダーを使用して、Launch agent が Kubernetes クラスター内で Docker イメージをビルドできるようにします。Launch agent のための Kaniko のセットアップ方法や、ジョブビルディングをオフにしてプレビルドした Docker イメージのみを使用する方法については、[Advanced agent set up]({{< relref path="./setup-agent-advanced.md" lang="ja" >}}) を参照してください。

{{% alert %}}
Helm をインストールして W&B の Launch agent Helm チャートを適用またはアップグレードするには、Kubernetes リソースを作成、更新、および削除するための十分な権限を持つクラスターへの `kubectl` アクセスが必要です。通常、クラスター管理者や同等の権限を持つカスタムロールを持つユーザーが必要です。
{{% /alert %}}

## Kubernetes のキューを設定する

Kubernetes のターゲットリソースに対する Launch キューの設定は、[Kubernetes Job スペック](https://kubernetes.io/docs/concepts/workloads/controllers/job/) または [Kubernetes Custom Resource スペック](https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/) のいずれかに似ています。

Launch キューを作成する際に、Kubernetes ワークロードリソーススペックの任意の側面を制御できます。

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
一部のユースケースでは、`CustomResource` の定義を使用したいかもしれません。例えば、マルチノードの分散トレーニングを実行したい場合に `CustomResource` が便利です。Volcano を使用してマルチノードジョブを Launch で使用するためのアプリケーションの例をチュートリアルで参照してください。別のユースケースとして、W&B Launch を Kubeflow と一緒に使用したい場合があるかもしれません。

以下の YAML スニペットは、Kubeflow を使用したサンプルの Launch キュー設定を示しています：

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

セキュリティ上の理由から、W&B は、Launch キューに指定されていない場合、次のリソースを注入します：

- `securityContext`
- `backOffLimit`
- `ttlSecondsAfterFinished`

次の YAML スニペットは、これらの値が Launch キューにどのように現れるかを示しています：

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

Kubernetes を計算リソースとして使用する W&B アプリでキューを作成します：

1. [Launch page](https://wandb.ai/launch) に移動します。
2. **Create Queue** ボタンをクリックします。
3. キューを作成したい **Entity** を選択します。
4. **Name** フィールドにキューの名前を入力します。
5. **Resource** として **Kubernetes** を選択します。
6. **Configuration** フィールドに、[前のセクションで設定した]({{< relref path="#configure-a-queue-for-kubernetes" lang="ja" >}}) Kubernetes Job ワークフロースペックまたは Custom Resource スペックを入力します。

## Helm を使って Launch agent を設定する

W&B が提供する [Helm チャート](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent) を使用して、Launch agent を Kubernetes クラスターにデプロイします。`values.yaml` [ファイル](https://github.com/wandb/helm-charts/blob/main/charts/launch-agent/values.yaml)を使って、Launch agent の振る舞いを制御します。

Launch agent の設定ファイル (`~/.config/wandb/launch-config.yaml`) に通常定義されるコンテンツを `values.yaml` ファイルの `launchConfig` キーに指定します。

例えば、Kaniko Docker イメージビルダーを使用する EKS での Launch agent ルーンを可能にする Launch agent 設定があるとします：

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

`values.yaml` ファイル内では、次のようになります：

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

レジストリ、環境、および必要なエージェント権限に関する詳細は、[Advanced agent set up]({{< relref path="./setup-agent-advanced.md" lang="ja" >}}) を参照してください。