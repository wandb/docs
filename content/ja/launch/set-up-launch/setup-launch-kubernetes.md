---
title: 'チュートリアル: Kubernetes で W&B Launch をセットアップする'
menu:
  launch:
    identifier: setup-launch-kubernetes
    parent: set-up-launch
url: guides/launch/setup-launch-kubernetes
---

W&B Launch を使うことで、ML エンジニアが W&B 上のシンプルなインターフェースから、すでに Kubernetes で管理しているリソースを活用して ML ワークロードを Kubernetes クラスターに簡単に投入できます。

W&B では [公式 Launch agent イメージ](https://hub.docker.com/r/wandb/launch-agent) を提供しており、これを W&B が管理する [Helm chart](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent) を使ってクラスターにデプロイできます。

Launch エージェントが Kubernetes クラスター内で Docker イメージをビルドできるよう、W&B では [Kaniko](https://github.com/GoogleContainerTools/kaniko) ビルダーを利用しています。Launch エージェントのために Kaniko をセットアップする方法、またはジョブのビルドを停止して事前ビルド済み Docker イメージのみを使う方法については、[Advanced agent set up]({{< relref "./setup-agent-advanced.md" >}}) を参照してください。

{{% alert %}}
Helm のインストールや W&B の Launch agent Helm chart の適用・アップグレードには、十分な権限で Kubernetes クラスターに `kubectl` アクセスできる必要があります。通常は cluster-admin 権限、または同等の権限を持つカスタムロールが必要です。
{{% /alert %}}



## Kubernetes 用のキューを設定する

Kubernetes をターゲットリソースとした Launch キューの設定は、[Kubernetes Job spec](https://kubernetes.io/docs/concepts/workloads/controllers/job/) または [Kubernetes Custom Resource spec](https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/) のいずれかに似た形になります。

Launch キュー作成時に、Kubernetes ワークロードリソース spec のあらゆる側面を細かく制御できます。

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
ユースケースによっては、`CustomResource` 定義を利用したい場合があります。たとえば、マルチノード分散トレーニングを実現したい場合に `CustomResource` 定義が便利です。Volcano を使ったマルチノードジョブでの Launch の利用例は、チュートリアルをご覧ください。その他の例としては、W&B Launch を Kubeflow と組み合わせて利用する場合も考えられます。

以下は、Kubeflow を使う場合の Launch キュー設定のサンプル YAML です。

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

セキュリティ上の理由から、下記のリソースが Launch キュー内で指定されていない場合、W&B は自動的に挿入します：

- `securityContext`
- `backOffLimit`
- `ttlSecondsAfterFinished`

以下の YAML は、これらの値が Launch キュー内でどのように表示されるかを示しています：

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

Kubernetes を計算リソースとして使うキューを W&B App 上で作成するには：

1. [Launch ページ](https://wandb.ai/launch)にアクセスします。
2. **Create Queue** ボタンをクリックします。
3. キューを作成する **Entity** を選択します。
4. **Name** フィールドにキュー名を入力します。
5. **Resource** として **Kubernetes** を選択します。
6. **Configuration** フィールドに、[前のセクションで設定した]({{< relref "#configure-a-queue-for-kubernetes" >}}) Kubernetes Job ワークフロー spec または Custom Resource spec を入力します。

## Helm で Launch エージェントを設定する

W&B 提供の [Helm chart](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent) を使って、Launch エージェントを Kubernetes クラスターへデプロイできます。`values.yaml` [ファイル](https://github.com/wandb/helm-charts/blob/main/charts/launch-agent/values.yaml) で launch agent の振る舞いを制御します。

通常は launch agent の設定ファイル（`~/.config/wandb/launch-config.yaml`）で指定していた内容を、`values.yaml` ファイル内の `launchConfig` キーに記述します。

たとえば、Kaniko Docker イメージビルダーを使う EKS 上で Launch エージェントを実行する設定例：

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

この場合、`values.yaml` ファイルには次のように記述します：

```yaml title="values.yaml"
agent:
  labels: {}
  # W&B API キー
  apiKey: ''
  # エージェント用のコンテナイメージ
  image: wandb/launch-agent:latest
  # エージェントイメージのプルポリシー
  imagePullPolicy: Always
  # エージェント spec のリソース指定
  resources:
    limits:
      cpu: 1000m
      memory: 1Gi

# launch agent をデプロイする Namespace
namespace: wandb

# W&B の API URL（ご自身の環境に合わせて設定）
baseUrl: https://api.wandb.ai

# launch agent がデプロイ可能な追加ターゲット namespace
additionalTargetNamespaces:
  - default
  - wandb

# launch agent 設定ファイルの内容をそのまま記載
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

# git 認証情報ファイルの内容。k8s secret に保存され、エージェントコンテナへマウントされます。
# プライベートリポジトリをクローンする場合に設定します。
gitCreds: |

# wandb サービスアカウントのアノテーション設定。GCP の workload identity 設定時などに利用。
serviceAccount:
  annotations:
    iam.gke.io/gcp-service-account:
    azure.workload.identity/client-id:

# kaniko を azure で使う場合のストレージ用 access key を設定
azureStorageAccessKey: ''
```

レジストリ、環境、および必要なエージェント権限の詳細については [Advanced agent set up]({{< relref "./setup-agent-advanced.md" >}}) をご確認ください。