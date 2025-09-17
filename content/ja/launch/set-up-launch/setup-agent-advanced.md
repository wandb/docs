---
title: Launch エージェントのセットアップ
menu:
  launch:
    identifier: ja-launch-set-up-launch-setup-agent-advanced
    parent: set-up-launch
url: guides/launch/setup-agent-advanced
---

# 高度なエージェント設定

このガイドでは、さまざまな環境でコンテナイメージをビルドするために W&B Launch エージェントを設定する方法を説明します。

{{% alert %}}
ビルドが必要なのは git と code artifact のジョブだけです。image ジョブにはビルドは不要です。

ジョブタイプの詳細は [Create a launch job]({{< relref path="../create-and-deploy-jobs/create-launch-job.md" lang="ja" >}}) を参照してください。
{{% /alert %}}

## Builders

Launch エージェントは [Docker](https://docs.docker.com/) または [Kaniko](https://github.com/GoogleContainerTools/kaniko) を使ってイメージをビルドできます。

* Kaniko: 特権コンテナとして実行することなく、Kubernetes 上でコンテナイメージをビルドします。
* Docker: ローカルで `docker build` コマンドを実行してコンテナイメージをビルドします。

builder の種類は Launch エージェントの設定ファイル内の `builder.type` キーで制御できます。`docker`、`kaniko`、またはビルドを無効化する `noop` を指定します。デフォルトでは、エージェントの Helm チャートは `builder.type` を `noop` に設定します。`builder` セクション内の追加キーでビルドプロセスを設定します。

エージェント設定で builder を指定せず、動作する `docker` CLI が見つかった場合、エージェントは既定で Docker を使用します。Docker が利用できない場合は `noop` が既定になります。

{{% alert %}}
Kubernetes クラスター内でイメージをビルドする場合は Kaniko を使用してください。その他のケースでは Docker を使用してください。
{{% /alert %}}


## コンテナレジストリへの push

Launch エージェントはビルドしたすべてのイメージに一意のソースハッシュでタグ付けします。エージェントは `builder.destination` キーで指定したレジストリへイメージを push します。

たとえば、`builder.destination` キーが `my-registry.example.com/my-repository` の場合、エージェントはイメージに `my-registry.example.com/my-repository:<source-hash>` のタグを付けて push します。イメージがレジストリに存在する場合、ビルドはスキップされます。

### エージェントの設定

Helm チャートでエージェントをデプロイする場合、エージェントの設定は `values.yaml` ファイルの `agentConfig` キーで指定します。

`wandb launch-agent` で自分でエージェントを起動する場合、`--config` フラグで YAML ファイルへのパスを渡してエージェント設定を指定できます。デフォルトでは、設定は `~/.config/wandb/launch-config.yaml` から読み込まれます。

Launch エージェントの設定ファイル (`launch-config.yaml`) では、`environment` キーにターゲットリソースの環境名を、`registry` キーにコンテナレジストリを指定します。

以下のタブでは、環境とレジストリに基づく Launch エージェントの設定方法を示します。

{{< tabpane text=true >}}
{{% tab "AWS" %}}
AWS 環境の設定には region キーが必要です。region にはエージェントを実行する AWS リージョンを指定します。 

```yaml title="launch-config.yaml"
environment:
  type: aws
  region: <aws-region>
builder:
  type: <kaniko|docker>
  # エージェントがイメージを保存する ECR リポジトリの URI。
  # environment に設定した値とリージョンが一致していることを確認してください。
  destination: <account-id>.ecr.<aws-region>.amazonaws.com/<repository-name>
  # Kaniko を使用する場合、エージェントがビルドコンテキストを保存する
  # S3 バケットを指定します。
  build-context-store: s3://<bucket-name>/<path>
```

エージェントは boto3 を使用してデフォルトの AWS 認証情報を読み込みます。デフォルトの AWS 認証情報の設定方法については [boto3 のドキュメント](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html) を参照してください。
{{% /tab %}}
{{% tab "GCP" %}}
Google Cloud 環境では region と project のキーが必要です。`region` にはエージェントを実行するリージョン、`project` にはエージェントを実行する Google Cloud プロジェクトを指定します。エージェントは Python の `google.auth.default()` を使用してデフォルトの認証情報を読み込みます。

```yaml title="launch-config.yaml"
environment:
  type: gcp
  region: <gcp-region>
  project: <gcp-project-id>
builder:
  type: <kaniko|docker>
  # エージェントがイメージを保存する Artifact Registry のリポジトリと
  # イメージ名の URI。environment に設定したリージョンとプロジェクトに
  # 一致していることを確認してください。
  uri: <region>-docker.pkg.dev/<project-id>/<repository-name>/<image-name>
  # Kaniko を使用する場合、エージェントがビルドコンテキストを保存する
  # GCS バケットを指定します。
  build-context-store: gs://<bucket-name>/<path>
```

エージェントで使用できるようにデフォルトの GCP 認証情報を設定する方法は、[`google-auth` のドキュメント](https://google-auth.readthedocs.io/en/latest/reference/google.auth.html#google.auth.default) を参照してください。

{{% /tab %}}
{{% tab "Azure" %}}

Azure 環境では追加のキーは不要です。エージェント起動時に `azure.identity.DefaultAzureCredential()` を使用してデフォルトの Azure 認証情報を読み込みます。

```yaml title="launch-config.yaml"
environment:
  type: azure
builder:
  type: <kaniko|docker>
  # エージェントがイメージを保存する Azure Container Registry のリポジトリ URI。
  destination: https://<registry-name>.azurecr.io/<repository-name>
  # Kaniko を使用する場合、エージェントがビルドコンテキストを保存する
  # Azure Blob Storage のコンテナーを指定します。
  build-context-store: https://<storage-account-name>.blob.core.windows.net/<container-name>
```

デフォルトの Azure 認証情報の設定方法は、[`azure-identity` のドキュメント](https://learn.microsoft.com/python/api/azure-identity/azure.identity.defaultazurecredential?view=azure-python) を参照してください。
{{% /tab %}}
{{< /tabpane >}}

## エージェントの権限

必要なエージェントの権限はユースケースによって異なります。

### クラウドレジストリの権限

クラウドレジストリと連携するために Launch エージェントに一般的に必要な権限は次のとおりです。

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

Kaniko ビルダーを使用する場合は、[`AcrPush` ロール](https://learn.microsoft.com/azure/role-based-access-control/built-in-roles/containers#acrpush) を追加してください。
{{% /tab %}}
{{< /tabpane >}}

### Kaniko 用のストレージ権限

Launch エージェントが Kaniko ビルダーを使用する場合、クラウドストレージへ push する権限が必要です。Kaniko は、ビルドジョブを実行する Pod の外部にあるコンテキストストアを使用します。

{{< tabpane text=true >}}
{{% tab "AWS" %}}
AWS での Kaniko ビルダーの推奨コンテキストストアは Amazon S3 です。次のポリシーを使用して、エージェントに S3 バケットへのアクセスを許可できます。

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
GCP では、GCS にビルドコンテキストをアップロードするために、エージェントに次の IAM 権限が必要です:

```js
storage.buckets.get;
storage.objects.create;
storage.objects.delete;
storage.objects.get;
```

{{% /tab %}}
{{% tab "Azure" %}}

エージェントが Azure Blob Storage にビルドコンテキストをアップロードできるようにするには、[Storage Blob Data Contributor](https://learn.microsoft.com/azure/role-based-access-control/built-in-roles#storage-blob-data-contributor) ロールが必要です。
{{% /tab %}}
{{< /tabpane >}}


## Kaniko ビルドのカスタマイズ

エージェント設定の `builder.kaniko-config` キーで、Kaniko ジョブが使用する Kubernetes Job の spec を指定します。例:

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
            - "--cache=false" # 引数は "key=value" の形式で指定する必要があります
            env:
            - name: "MY_ENV_VAR"
              value: "my-env-var-value"
```

## Launch エージェントを CoreWeave にデプロイ 
必要に応じて、W&B Launch エージェントを CoreWeave Cloud インフラストラクチャーにデプロイできます。CoreWeave は GPU アクセラレーション向けに特化して設計されたクラウドインフラストラクチャーです。

Launch エージェントを CoreWeave にデプロイする方法については、[CoreWeave のドキュメント](https://docs.coreweave.com/partners/weights-and-biases#integration) を参照してください。 

{{% alert %}}
Launch エージェントを CoreWeave のインフラストラクチャーにデプロイするには、[CoreWeave アカウント](https://cloud.coreweave.com/login) を作成する必要があります。 
{{% /alert %}}