---
title: 'チュートリアル: Kubernetes で W&B Launch をセットアップする'
menu:
  launch:
    identifier: ja-launch-set-up-launch-setup-launch-kubernetes
    parent: set-up-launch
url: guides/launch/setup-launch-kubernetes
---

W&B Launch を使うことで、 ML ワークロードを Kubernetes クラスターに簡単にデプロイできます。これにより、 ML エンジニアは W&B のインターフェースから、 既に管理している Kubernetes のリソースを活用できます。

W&B では [公式の Launch agent イメージ](https://hub.docker.com/r/wandb/launch-agent) を用意しており、 [Helm chart](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent) を使ってクラスターにデプロイできます。これらは W&B がメンテナンスしています。

Launch agent では、 [Kaniko](https://github.com/GoogleContainerTools/kaniko) ビルダーを利用し、 Kubernetes クラスター内で Docker イメージのビルドを可能にしています。Launch agent で Kaniko を使用する方法や、ジョブのビルドをオフにして既存の Docker イメージのみ利用する方法の詳細は [エージェントの高度な設定]({{< relref path="./setup-agent-advanced.md" lang="ja" >}}) をご覧ください。

{{% alert %}}
Helm をインストールして W&B の Launch agent 用 Helm chart を適用・アップグレードするには、十分な権限を持つ `kubectl` アクセスが必要です。通常、 cluster-admin 権限や同等のカスタムロールを持つユーザーが必要になります。
{{% /alert %}}

## Kubernetes 用のキューを設定する

Kubernetes 用リソースをターゲットとした Launch queue の設定は、 [Kubernetes Job spec](https://kubernetes.io/docs/concepts/workloads/controllers/job/) または [Kubernetes Custom Resource spec](https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/) のいずれかに類似しています。

Launch queue の作成時に、 Kubernetes ワークロードリソース仕様のあらゆる側面を制御可能です。

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
ユースケースによっては `CustomResource` 定義の使用を検討することもできます。たとえば、マルチノード分散トレーニングを行いたい場合などです。Launch と Volcano を使ったマルチノードジョブの例（チュートリアル）もご参照ください。また、 W&B Launch を Kubeflow で利用したい場合もこれに当てはまります。

以下の YAML は、 Kubeflow を利用したサンプルの Launch queue 設定例です。

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

セキュリティ上の理由から、 以下のリソースが Launch queue に指定されていない場合、 W&B は自動的に付与します:

- `securityContext`
- `backOffLimit`
- `ttlSecondsAfterFinished`

以下の YAML は、 Launch queue 内でこれらの値がどのように現れるかの例です:

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

## キューの作成

Kubernetes を計算リソースとするキューを W&B App で作成する手順:

1. [Launch ページ](https://wandb.ai/launch) へ移動します。
2. **Create Queue** ボタンをクリックします。
3. キューを作成したい **Entity** を選択します。
4. **Name** フィールドにキューの名前を入力します。
5. **Resource** で **Kubernetes** を選択します。
6. **Configuration** フィールドには [前のセクションで設定した]({{< relref path="#configure-a-queue-for-kubernetes" lang="ja" >}}) Kubernetes Job ワークフロー仕様または Custom Resource 仕様を記入します。

## Launch agent を Helm で設定する

W&B が用意した [Helm chart](https://github.com/wandb/helm-charts/tree/main/charts/launch-agent) を利用し、 Kubernetes クラスターに Launch agent をデプロイできます。`values.yaml` [ファイル](https://github.com/wandb/helm-charts/blob/main/charts/launch-agent/values.yaml) で launch agent の振る舞いを制御します。

通常 `~/.config/wandb/launch-config.yaml` に定義する Launch agent の設定内容は、 `values.yaml` ファイル内の `launchConfig` キーに記載します。

例えば、 Kaniko の Docker イメージビルダーを使って Launch agent を EKS 上で動かす設定例です:

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

この内容を `values.yaml` で記載すると下記のようになります:

```yaml title="values.yaml"
agent:
  labels: {}
  # W&B API キー
  apiKey: ''
  # agent 用のコンテナイメージ
  image: wandb/launch-agent:latest
  # agent イメージのプルポリシー
  imagePullPolicy: Always
  # agent spec のリソース設定
  resources:
    limits:
      cpu: 1000m
      memory: 1Gi

# launch agent をデプロイする namespace
namespace: wandb

# W&B API の URL (必要に応じて設定)
baseUrl: https://api.wandb.ai

# launch agent がデプロイ可能な追加のターゲット namespace
additionalTargetNamespaces:
  - default
  - wandb

# launch agent の設定内容そのものを記載
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

# git 資格情報ファイルの内容。これは k8s シークレットに保存& agent コンテナにマウントされます。プライベートリポジトリをクローンする必要がある場合に設定してください。
gitCreds: |

# wandb サービスアカウント用のアノテーション。gcp の workload identity 設定時などに利用します。
serviceAccount:
  annotations:
    iam.gke.io/gcp-service-account:
    azure.workload.identity/client-id:

# kaniko を azure で使う場合のストレージ access key
azureStorageAccessKey: ''
```

レジストリや環境、 agent 必要権限などの詳細については [エージェントの高度な設定]({{< relref path="./setup-agent-advanced.md" lang="ja" >}}) を参照してください。