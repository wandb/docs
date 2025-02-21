---
title: Set up launch agent
menu:
  launch:
    identifier: ja-launch-set-up-launch-setup-agent-advanced
    parent: set-up-launch
url: guides/launch/setup-agent-advanced
---

# 高度なエージェント設定

このガイドは、異なる環境でコンテナイメージをビルドするための W&B Launch エージェントの設定方法に関する情報を提供します。

{{% alert %}}
ビルドは、git とコードアーティファクトジョブにのみ必要です。イメージジョブにはビルドは必要ありません。

ジョブタイプの詳細については、[Create a launch job]({{< relref path="../create-and-deploy-jobs/create-launch-job.md" lang="ja" >}}) をご覧ください。
{{% /alert %}}

## ビルダー

Launchエージェントは、[Docker](https://docs.docker.com/) または [Kaniko](https://github.com/GoogleContainerTools/kaniko) を使用してイメージをビルドすることができます。

* Kaniko: ビルドを特権コンテナとして実行することなく、Kubernetesでコンテナイメージをビルドします。
* Docker: `docker build` コマンドをローカルで実行してコンテナイメージをビルドします。

ビルダータイプは、`builder.type` キーで制御され、ランチエージェントの設定で `docker` 、`kaniko` 、またはビルドをオフにするために `noop` に設定できます。デフォルトでは、エージェントヘルムチャートは `builder.type` を `noop` に設定します。`builder` セクションの追加のキーは、ビルドプロセスの設定に使用されます。

エージェント設定でビルダーが指定されておらず、動作する `docker` CLI が見つかった場合、エージェントはデフォルトで Docker を使用します。Docker が利用できない場合、エージェントはデフォルトで `noop` を使用します。

{{% alert %}}
Kubernetes クラスターでイメージをビルドするために Kaniko を使用してください。それ以外のすべてのケースでは Docker を使用してください。
{{% /alert %}}

## コンテナレジストリへのプッシュ

ランチエージェントは、ビルドしたすべてのイメージにユニークなソースハッシュでタグを付けます。エージェントは、`builder.destination` キーで指定されたレジストリにイメージをプッシュします。

例えば、`builder.destination` キーが `my-registry.example.com/my-repository` に設定されている場合、エージェントはそのイメージを `my-registry.example.com/my-repository:<source-hash>` にタグを付けてプッシュします。イメージがすでにレジストリに存在する場合、ビルドはスキップされます。

### エージェントの設定

エージェントを Helm チャートを使ってデプロイする場合、エージェントの設定は `values.yaml` ファイルの `agentConfig` キーで提供されるべきです。

`wandb launch-agent` コマンドを使って自分でエージェントを起動する場合、`--config` フラグを使用して YAML ファイルへのパスをエージェント設定として提供できます。デフォルトでは、設定は `~/.config/wandb/launch-config.yaml` からロードされます。

ランチエージェント設定 (`launch-config.yaml`) の中で、ターゲットリソース環境とコンテナレジストリの名前をそれぞれ `environment` と `registry` キーに提供してください。

以下のタブは、環境とレジストリに基づいてランチエージェントを設定する方法を示しています。

{{< tabpane text=true >}}
{{% tab "AWS" %}}
AWS 環境の設定には region キーが必要です。地域はエージェントが実行される AWS の地域であるべきです。

```yaml title="launch-config.yaml"
environment:
  type: aws
  region: <aws-region>
builder:
  type: <kaniko|docker>
  # エージェントが画像を保存する ECR レポジトリの URI です。
  # 環境で設定した地域と一致していることを確認してください。
  destination: <account-id>.ecr.<aws-region>.amazonaws.com/<repository-name>
  # Kaniko を使用する場合、エージェントがビルドコンテキストを保存する
  # S3 バケットを指定してください。
  build-context-store: s3://<bucket-name>/<path>
```

エージェントは boto3 を使用してデフォルトの AWS クレデンシャルをロードします。デフォルトの AWS クレデンシャルの設定方法については、[boto3 ドキュメント](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)を参照してください。
{{% /tab %}}
{{% tab "GCP" %}}
Google Cloud 環境には region と project キーが必要です。`region` をエージェントが実行されるリージョンに設定し、`project` をエージェントが実行される Google Cloud プロジェクトに設定します。エージェントは Python の `google.auth.default()` を使用してデフォルトのクレデンシャルをロードします。

```yaml title="launch-config.yaml"
environment:
  type: gcp
  region: <gcp-region>
  project: <gcp-project-id>
builder:
  type: <kaniko|docker>
  # エージェントが画像を保存する Artifact Registry レポの URI および画像名。
  # 環境で設定したリージョンおよびプロジェクトと一致していることを確認してください。
  uri: <region>-docker.pkg.dev/<project-id>/<repository-name>/<image-name>
  # Kaniko を使用する場合、エージェントがビルドコンテキストを保存する
  # GCS バケットを指定してください。
  build-context-store: gs://<bucket-name>/<path>
```

エージェントに使用できるようにするためのデフォルトの GCP クレデンシャルの設定方法については、[`google-auth` ドキュメント](https://google-auth.readthedocs.io/en/latest/reference/google.auth.html#google.auth.default)をご覧ください。

{{% /tab %}}
{{% tab "Azure" %}}

Azure 環境では追加のキーは必要ありません。エージェントが起動するときに、`azure.identity.DefaultAzureCredential()` を使用してデフォルトの Azure クレデンシャルをロードします。

```yaml title="launch-config.yaml"
environment:
  type: azure
builder:
  type: <kaniko|docker>
  # Azure Container Registry レポジトリの URI です
  # エージェントが画像を保存します。
  destination: https://<registry-name>.azurecr.io/<repository-name>
  # Kaniko を使用する場合、エージェントがビルドコンテキストを保存する
  # Azure Blob Storage コンテナを指定してください。
  build-context-store: https://<storage-account-name>.blob.core.windows.net/<container-name>
```

デフォルトの Azure クレデンシャルの設定方法については、[`azure-identity` ドキュメント](https://learn.microsoft.com/en-us/python/api/azure-identity/azure.identity.defaultazurecredential?view=azure-python)をご覧ください。
{{% /tab %}}
{{< /tabpane >}}

## エージェントの権限

エージェントに必要な権限はユースケースによって異なります。

### クラウドレジストリの権限

以下は、ランチエージェントがクラウドレジストリと対話するために一般的に必要とされる権限です。

{{< tabpane text=true >}}
{{% tab "AWS" %}}
```yaml
{
  'Version': '2012-10-17',
  'Statement':
    [
      {
        'Effect': 'Allow',
        'Action':
          [
            'ecr:CreateRepository',
            'ecr:UploadLayerPart',
            'ecr:PutImage',
            'ecr:CompleteLayerUpload',
            'ecr:InitiateLayerUpload',
            'ecr:DescribeRepositories',
            'ecr:DescribeImages',
            'ecr:BatchCheckLayerAvailability',
            'ecr:BatchDeleteImage',
          ],
        'Resource': 'arn:aws:ecr:<region>:<account-id>:repository/<repository>',
      },
      {
        'Effect': 'Allow',
        'Action': 'ecr:GetAuthorizationToken',
        'Resource': '*',
      },
    ],
}
```
{{% /tab %}}
{{% tab "GCP" %}}
```js
artifactregistry.dockerimages.list;
artifactregistry.repositories.downloadArtifacts;
artifactregistry.repositories.list;
artifactregistry.repositories.uploadArtifacts;
```

{{% /tab %}}
{{% tab "Azure" %}}

Kaniko ビルダーを使用する場合は、[`AcrPush` ロール](https://learn.microsoft.com/en-us/azure/container-registry/container-registry-roles?tabs=azure-cli#acrpush)を追加してください。
{{% /tab %}}
{{< /tabpane >}}

### Kaniko のためのストレージの権限

ランチエージェントは、Kaniko ビルダーを使用する場合、クラウドストレージにプッシュする権限が必要です。Kaniko は、ビルドジョブを実行しているポッドの外部にコンテキストストアを使用します。

{{< tabpane text=true >}}
{{% tab "AWS" %}}
AWS上でKanikoビルダーに推奨されるコンテキストストアはAmazon S3です。エージェントにS3バケットへのアクセスを許可するために次のポリシーを使用できます。

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ListObjectsInBucket",
      "Effect": "Allow",
      "Action": ["s3:ListBucket"],
      "Resource": ["arn:aws:s3:::<BUCKET-NAME>"]
    },
    {
      "Sid": "AllObjectActions",
      "Effect": "Allow",
      "Action": "s3:*Object",
      "Resource": ["arn:aws:s3:::<BUCKET-NAME>/*"]
    }
  ]
}
```
{{% /tab %}}
{{% tab "GCP" %}}
GCPの際、GCSにビルドコンテキストをアップロードするためにエージェントが必要とするIAMパーミッションは次のとおりです。

```js
storage.buckets.get;
storage.objects.create;
storage.objects.delete;
storage.objects.get;
```

{{% /tab %}}
{{% tab "Azure" %}}

[Storage Blob Data Contributor](https://learn.microsoft.com/en-us/azure/role-based-access-control/built-in-roles#storage-blob-data-contributor)ロールが必要であり、エージェントがAzure Blob Storageにビルドコンテキストをアップロードするために必要とされます。
{{% /tab %}}
{{< /tabpane >}}

## Kaniko ビルドのカスタマイズ

エージェント設定の `builder.kaniko-config` キーに、Kaniko ジョブが使用する Kubernetes Job スペックを指定します。例えば次のように設定します。

```yaml title="launch-config.yaml"
builder:
  type: kaniko
  build-context-store: <my-build-context-store>
  destination: <my-image-destination>
  build-job-name: wandb-image-build
  kaniko-config:
    spec:
      template:
        spec:
          containers:
          - args:
            - "--cache=false" # Args must be in the format "key=value"
            env:
            - name: "MY_ENV_VAR"
              value: "my-env-var-value"
```

## CoreWeave へのランチエージェントのデプロイ
オプションとして、W&B Launch エージェントを CoreWeave クラウドインフラストラクチャにデプロイすることができます。CoreWeave は主に GPU 加速ワークロード用に設計されたクラウドインフラストラクチャです。

CoreWeave への Launch エージェントのデプロイ方法については、[CoreWeave ドキュメント](https://docs.coreweave.com/partners/weights-and-biases#integration) を参照してください。

{{% alert %}}
CoreWeave インフラストラクチャに Launch エージェントをデプロイするには、[CoreWeave アカウント](https://cloud.coreweave.com/login) を作成する必要があります。
{{% /alert %}}