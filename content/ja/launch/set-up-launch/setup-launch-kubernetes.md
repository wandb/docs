---
title: 'Tutorial: Set up W&B Launch on Kubernetes'
menu:
  launch:
    identifier: ja-launch-set-up-launch-setup-launch-kubernetes
    parent: set-up-launch
url: guides/launch/setup-launch-kubernetes
---

W&B Launch を使用して ML ワークロードを Kubernetes クラスターにプッシュできます。これにより、ML エンジニアは、Kubernetes で既に管理しているリソースを使用するためのシンプルなインターフェースを W&B 内で利用できます。

W&B は、W&B が管理する [公式 Launch エージェントイメージ](https://hub.docker.com/r/wandb/launch-agent) を保持しており、[Helm chart](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent) を使用してクラスターにデプロイできます。

W&B は [Kaniko](https://github.com/GoogleContainerTools/kaniko) ビルダーを使用して、Launch エージェントが Kubernetes クラスターで Docker イメージを構築できるようにします。Launch エージェント用に Kaniko をセットアップする方法、またはジョブの構築をオフにして、構築済みの Docker イメージのみを使用する方法の詳細については、[エージェントの詳細設定]({{< relref path="./setup-agent-advanced.md" lang="ja" >}}) を参照してください。

{{% alert %}}
Helm をインストールし、W&B の Launch エージェント Helm chart を適用またはアップグレードするには、Kubernetes リソースを作成、更新、および削除するための十分な権限を持つ `kubectl` アクセスがクラスターに必要です。通常、cluster-admin または同等の権限を持つカスタムロールを持つ ユーザー が必要です。
{{% /alert %}}

## Kubernetes のキューを設定する

Kubernetes ターゲットリソースの Launch キュー設定は、[Kubernetes Job spec](https://kubernetes.io/docs/concepts/workloads/controllers/job/) または [Kubernetes Custom Resource spec](https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/) のいずれかに類似します。

Launch キューを作成する際に、Kubernetes ワークロードリソース spec のあらゆる側面を制御できます。

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
一部の ユースケース では、`CustomResource` 定義を使用したい場合があります。たとえば、マルチノード分散 トレーニング を実行したい場合に、`CustomResource` 定義が役立ちます。Volcano を使用したマルチノードジョブで Launch を使用するためのチュートリアルで、アプリケーション の例を参照してください。別の ユースケース として、W&B Launch を Kubeflow で使用したい場合などが考えられます。

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

セキュリティ上の理由から、W&B は指定されていない場合、次のリソースを Launch キューに挿入します。

- `securityContext`
- `backOffLimit`
- `ttlSecondsAfterFinished`

次の YAML スニペットは、これらの 値 が Launch キューにどのように表示されるかを示しています。

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

Kubernetes をコンピューティングリソースとして使用するキューを W&B アプリ で作成します。

1. [Launch ページ](https://wandb.ai/launch) に移動します。
2. [**キューを作成**] ボタンをクリックします。
3. キューを作成する **Entity** を選択します。
4. [**名前**] フィールドにキューの名前を入力します。
5. [**リソース**] として [**Kubernetes**] を選択します。
6. [**設定**] フィールド内で、[前のセクションで設定した]({{< relref path="#configure-a-queue-for-kubernetes" lang="ja" >}}) Kubernetes Job ワークフロー spec または Custom Resource spec を指定します。

## Helm で Launch エージェントを設定する

W&B が提供する [Helm chart](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent) を使用して、Launch エージェントを Kubernetes クラスターにデプロイします。`values.yaml` [ファイル](https://github.com/wandb/helm-charts/blob/main/charts/launch-agent/values.yaml) で Launch エージェントの 振る舞い を制御します。

通常は Launch エージェント設定ファイル (`~/.config/wandb/launch-config.yaml`) で定義されるコンテンツを、`values.yaml` ファイルの `launchConfig` キー内に指定します。

たとえば、Kaniko Docker イメージビルダーを使用する Launch エージェントを EKS で実行できるようにする Launch エージェント設定があるとします。

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
  # W&B APIキー。
  apiKey: ''
  # エージェントに使用するコンテナイメージ。
  image: wandb/launch-agent:latest
  # エージェントイメージのイメージプルポリシー。
  imagePullPolicy: Always
  # エージェント spec のリソースブロック。
  resources:
    limits:
      cpu: 1000m
      memory: 1Gi

# Launch エージェントをデプロイする名前空間
namespace: wandb

# W&B api url (ここに設定してください)
baseUrl: https://api.wandb.ai

# Launch エージェントがデプロイできる追加のターゲット名前空間
additionalTargetNamespaces:
  - default
  - wandb

# これは、Launch エージェント設定のリテラルコンテンツに設定する必要があります。
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

# git 認証情報ファイルの内容。これは k8s シークレットに保存されます
# エージェントコンテナにマウントされます。プライベートをクローンする場合は、これを設定します
# リポジトリ。
gitCreds: |

# wandb サービスアカウントのアノテーション。gcp でワークロードIDを設定する際に役立ちます。
serviceAccount:
  annotations:
    iam.gke.io/gcp-service-account:
    azure.workload.identity/client-id:

# azure で kaniko を使用している場合は、azure ストレージのアクセスキーに設定します。
azureStorageAccessKey: ''
```

レジストリ、 環境 、および必要なエージェント権限の詳細については、[エージェントの詳細設定]({{< relref path="./setup-agent-advanced.md" lang="ja" >}}) を参照してください。
