---
title: Set up launch agent
menu:
  launch:
    identifier: ja-launch-set-up-launch-setup-agent-advanced
    parent: set-up-launch
url: guides/launch/setup-agent-advanced
---

# 高度なエージェントの設定

このガイドでは、さまざまな環境でコンテナイメージを構築するために W&B Launch エージェントをセットアップする方法について説明します。

{{% alert %}}
ビルドは、git および コード Artifacts ジョブでのみ必要です。イメージジョブはビルドを必要としません。

ジョブタイプに関する詳細については、[Launch ジョブの作成]({{< relref path="../create-and-deploy-jobs/create-launch-job.md" lang="ja" >}})を参照してください。
{{% /alert %}}

## ビルダー

Launch エージェントは、[Docker](https://docs.docker.com/) または [Kaniko](https://github.com/GoogleContainerTools/kaniko) を使用してイメージを構築できます。

*   Kaniko: 特権コンテナとしてビルドを実行せずに、Kubernetes でコンテナイメージを構築します。
*   Docker: `docker build` コマンドをローカルで実行して、コンテナイメージを構築します。

ビルダータイプは、Launch エージェント設定の `builder.type` キーで制御でき、ビルドをオフにするには `docker`、`kaniko`、または `noop` のいずれかに設定します。デフォルトでは、エージェント Helm チャートは `builder.type` を `noop` に設定します。`builder` セクションの追加キーは、ビルドプロセスを設定するために使用されます。

エージェント設定でビルダーが指定されておらず、動作する `docker` CLI が見つかった場合、エージェントはデフォルトで Docker を使用します。Docker が利用できない場合、エージェントはデフォルトで `noop` になります。

{{% alert %}}
Kubernetes クラスターでイメージを構築するには Kaniko を使用します。他のすべての場合は Docker を使用してください。
{{% /alert %}}

## コンテナレジストリへのプッシュ

Launch エージェントは、構築するすべてのイメージに一意のソースハッシュでタグを付けます。エージェントは、`builder.destination` キーで指定されたレジストリにイメージをプッシュします。

たとえば、`builder.destination` キーが `my-registry.example.com/my-repository` に設定されている場合、エージェントはイメージに `my-registry.example.com/my-repository:<source-hash>` というタグを付けてプッシュします。イメージがレジストリに存在する場合、ビルドはスキップされます。

### エージェントの設定

Helm チャートを介してエージェントをデプロイする場合、エージェント設定は `values.yaml` ファイルの `agentConfig` キーで指定する必要があります。

`wandb launch-agent` でエージェントを自分で呼び出す場合は、`--config` フラグを使用して、エージェント設定を YAML ファイルへのパスとして指定できます。デフォルトでは、設定は `~/.config/wandb/launch-config.yaml` からロードされます。

Launch エージェント設定（`launch-config.yaml`）内で、ターゲットリソース環境の名前と、`environment` および `registry` キーのコンテナレジストリの名前を指定します。

次のタブは、環境とレジストリに基づいて Launch エージェントを設定する方法を示しています。

{{< tabpane text=true >}}
{{% tab "AWS" %}}
AWS 環境設定には、`region` キーが必要です。`region` は、エージェントが実行される AWS リージョンである必要があります。

```yaml title="launch-config.yaml"
environment:
  type: aws
  region: <aws-region>
builder:
  type: <kaniko|docker>
  # エージェントがイメージを保存する ECR リポジトリの URI。
  # リージョンが環境で設定したものと一致していることを確認してください。
  destination: <account-id>.ecr.<aws-region>.amazonaws.com/<repository-name>
  # Kaniko を使用する場合は、エージェントがビルドコンテキストを保存する S3 バケットを指定します。
  build-context-store: s3://<bucket-name>/<path>
```

エージェントは `boto3` を使用してデフォルトの AWS 認証情報をロードします。デフォルトの AWS 認証情報を設定する方法の詳細については、[boto3 のドキュメント](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html) を参照してください。
{{% /tab %}}
{{% tab "GCP" %}}
Google Cloud 環境には、`region` および `project` キーが必要です。`region` をエージェントが実行されるリージョンに設定します。`project` をエージェントが実行される Google Cloud プロジェクトに設定します。エージェントは、Python で `google.auth.default()` を使用して、デフォルトの認証情報をロードします。

```yaml title="launch-config.yaml"
environment:
  type: gcp
  region: <gcp-region>
  project: <gcp-project-id>
builder:
  type: <kaniko|docker>
  # エージェントがイメージを保存する Artifact Registry リポジトリとイメージ名の URI。
  # リージョンとプロジェクトが、環境で設定したものと一致していることを確認してください。
  uri: <region>-docker.pkg.dev/<project-id>/<repository-name>/<image-name>
  # Kaniko を使用する場合は、エージェントがビルドコンテキストを保存する GCS バケットを指定します。
  build-context-store: gs://<bucket-name>/<path>
```

エージェントが利用できるように、デフォルトの GCP 認証情報を設定する方法の詳細については、[`google-auth` ドキュメント](https://google-auth.readthedocs.io/en/latest/reference/google.auth.html#google.auth.default) を参照してください。

{{% /tab %}}
{{% tab "Azure" %}}

Azure 環境は、追加のキーを必要としません。エージェントは起動時に `azure.identity.DefaultAzureCredential()` を使用して、デフォルトの Azure 認証情報をロードします。

```yaml title="launch-config.yaml"
environment:
  type: azure
builder:
  type: <kaniko|docker>
  # エージェントがイメージを保存する Azure Container Registry リポジトリの URI。
  destination: https://<registry-name>.azurecr.io/<repository-name>
  # Kaniko を使用する場合は、エージェントがビルドコンテキストを保存する Azure Blob Storage コンテナーを指定します。
  build-context-store: https://<storage-account-name>.blob.core.windows.net/<container-name>
```

デフォルトの Azure 認証情報を設定する方法の詳細については、[`azure-identity` ドキュメント](https://learn.microsoft.com/en-us/python/api/azure-identity/azure.identity.defaultazurecredential?view=azure-python) を参照してください。
{{% /tab %}}
{{< /tabpane >}}

## エージェントの権限

必要なエージェントの権限は、ユースケースによって異なります。

### クラウドレジストリの権限

以下は、Launch エージェントがクラウドレジストリとやり取りするために一般的に必要な権限です。

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

Kaniko ビルダーを使用する場合は、[`AcrPush` ロール](https://learn.microsoft.com/en-us/azure/container-registry/container-registry-roles?tabs=azure-cli#acrpush) を追加します。
{{% /tab %}}
{{< /tabpane >}}

### Kaniko のストレージ権限

エージェントが Kaniko ビルダーを使用する場合、Launch エージェントはクラウドストレージにプッシュする権限が必要です。Kaniko は、ビルドジョブを実行しているポッドの外部にあるコンテキストストアを使用します。

{{< tabpane text=true >}}
{{% tab "AWS" %}}
AWS での Kaniko ビルダーの推奨コンテキストストアは Amazon S3 です。次のポリシーを使用して、エージェントに S3 バケットへのアクセス権を付与できます。

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
GCP では、エージェントがビルドコンテキストを GCS にアップロードするために、次の IAM 権限が必要です。

```js
storage.buckets.get;
storage.objects.create;
storage.objects.delete;
storage.objects.get;
```

{{% /tab %}}
{{% tab "Azure" %}}

エージェントがビルドコンテキストを Azure Blob Storage にアップロードするには、[ストレージ BLOB データ共同作成者](https://learn.microsoft.com/en-us/azure/role-based-access-control/built-in-roles#storage-blob-data-contributor) ロールが必要です。
{{% /tab %}}
{{< /tabpane >}}

## Kaniko ビルドのカスタマイズ

エージェント設定の `builder.kaniko-config` キーで、Kaniko ジョブが使用する Kubernetes ジョブスペックを指定します。次に例を示します。

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
            - "--cache=false" # Args は "key=value" 形式である必要があります
            env:
            - name: "MY_ENV_VAR"
              value: "my-env-var-value"
```

## CoreWeave への Launch エージェントのデプロイ

オプションで、W&B Launch エージェントを CoreWeave Cloud インフラストラクチャにデプロイします。CoreWeave は、GPU アクセラレーションされたワークロード用に構築されたクラウドインフラストラクチャです。

Launch エージェントを CoreWeave にデプロイする方法については、[CoreWeave ドキュメント](https://docs.coreweave.com/partners/weights-and-biases#integration) を参照してください。

{{% alert %}}
Launch エージェントを CoreWeave インフラストラクチャにデプロイするには、[CoreWeave アカウント](https://cloud.coreweave.com/login) を作成する必要があります。
{{% /alert %}}
