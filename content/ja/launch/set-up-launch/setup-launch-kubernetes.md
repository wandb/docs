---
title: 'チュートリアル: Kubernetes 上で W&B Launch をセットアップする'
menu:
  launch:
    identifier: ja-launch-set-up-launch-setup-launch-kubernetes
    parent: set-up-launch
url: guides/launch/setup-launch-kubernetes
---

W&B Launch を使うと、ML のワークロードを Kubernetes クラスターへ送ることができます。Kubernetes で既に管理しているリソースを、W&B 上から ML エンジニアが簡単に使えるインターフェースを提供します。

W&B は [公式の Launch エージェントイメージ](https://hub.docker.com/r/wandb/launch-agent)を提供しており、W&B がメンテナンスしている [Helm チャート](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent)でクラスターにデプロイできます。

W&B は [Kaniko](https://github.com/GoogleContainerTools/kaniko) ビルダーを使って、Launch エージェントが Kubernetes クラスター内で Docker イメージをビルドできるようにします。Launch エージェント向けに Kaniko をセットアップする方法、あるいはジョブのビルドを無効化してビルド済みの Docker イメージのみを使う方法については、[高度なエージェント設定]({{< relref path="./setup-agent-advanced.md" lang="ja" >}})を参照してください。

{{% alert %}}
Helm をインストールし、W&B の Launch エージェント用 Helm チャートを適用またはアップグレードするには、Kubernetes リソースの作成・更新・削除を行える十分な権限を持つクラスターへの `kubectl` アクセスが必要です。通常は、cluster-admin 権限のあるユーザー、または同等の権限を持つカスタムロールが必要です。
{{% /alert %}}




## Kubernetes 用のキューを設定する {#configure-a-queue-for-kubernetes}

Kubernetes のターゲットリソース向けの Launch キュー設定は、[Kubernetes Job のスペック](https://kubernetes.io/docs/concepts/workloads/controllers/job/) か [Kubernetes カスタムリソースのスペック](https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/) のいずれかに類似します。

Launch キューを作成する際、Kubernetes のワークロードリソースのスペックのあらゆる側面を制御できます。

{{< tabpane text=true >}}
{{% tab "Kubernetes Job のスペック" %}}
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
{{% tab "カスタムリソースのスペック" %}}
いくつかのユースケースでは、`CustomResource` 定義を使いたい場合があります。たとえばマルチノードの分散トレーニングを行いたいときに `CustomResource` 定義が有用です。例として Volcano を使ったマルチノードジョブで Launch を使うチュートリアルを参照してください。別のユースケースとして、W&B Launch を Kubeflow と組み合わせて使いたい場合もあるでしょう。

次の YAML スニペットは、Kubeflow を使う Launch キュー設定の例です:

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

セキュリティ上の理由から、以下の設定が未指定の場合は W&B が Launch キューに挿入します:

- `securityContext`
- `backOffLimit`
- `ttlSecondsAfterFinished`

以下の YAML は、Launch キューでこれらの値がどのように表れるかを示しています:

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

Kubernetes をコンピュートリソースとして使うキューを W&B アプリで作成します:

1. [Launch ページ](https://wandb.ai/launch) に移動します。
2. **Create Queue** ボタンをクリックします。
3. キューを作成する **Entity** を選択します。
4. **Name** フィールドにキュー名を入力します。
5. **Resource** として Kubernetes を選択します。
6. **Configuration** フィールドに、[前のセクションで設定した]({{< relref path="#configure-a-queue-for-kubernetes" lang="ja" >}}) Kubernetes Job のワークフロースペックまたはカスタムリソースのスペックを入力します。

## Helm で Launch エージェントを設定する

W&B が提供する [Helm チャート](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent)を使って、Kubernetes クラスターに Launch エージェントをデプロイします。`values.yaml` [ファイル](https://github.com/wandb/helm-charts/blob/main/charts/launch-agent/values.yaml)で Launch エージェントの振る舞いを制御します。

通常は Launch エージェントの設定ファイル（`~/.config/wandb/launch-config.yaml`）に記述する内容を、`values.yaml` の `launchConfig` キー内に記述します。

たとえば、Kaniko の Docker イメージビルダーを使う EKS 上で Launch エージェントを実行できるようにする設定があるとします:

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

`values.yaml` では次のようになります:

```yaml title="values.yaml"
agent:
  labels: {}
  # W&B の API キー。
  apiKey: ''
  # エージェントで使用するコンテナイメージ。
  image: wandb/launch-agent:latest
  # エージェントイメージの pull ポリシー。
  imagePullPolicy: Always
  # エージェントのスペックに対する Resources ブロック。
  resources:
    limits:
      cpu: 1000m
      memory: 1Gi

# Launch エージェントをデプロイする Namespace
namespace: wandb

# W&B の API URL（必要に応じて変更してください）
baseUrl: https://api.wandb.ai

# Launch エージェントがデプロイできる追加の対象 Namespace
additionalTargetNamespaces:
  - default
  - wandb

# Launch エージェントの設定ファイルの内容を、そのままここに記述します。
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

# git 認証情報ファイルの内容。これは k8s の Secret に保存され、
# エージェントのコンテナにマウントされます。プライベートリポジトリを
# クローンしたい場合に設定してください。
gitCreds: |

# wandb サービスアカウント用のアノテーション。GCP で workload identity を設定する際に便利です。
serviceAccount:
  annotations:
    iam.gke.io/gcp-service-account:
    azure.workload.identity/client-id:

# Azure で Kaniko を使う場合、Azure Storage のアクセスキーを設定します。
azureStorageAccessKey: ''
```

レジストリ、環境、エージェントに必要な権限の詳細は、[高度なエージェント設定]({{< relref path="./setup-agent-advanced.md" lang="ja" >}})を参照してください。