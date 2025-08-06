---
title: ローンンチ エージェントの設定
menu:
  launch:
    identifier: ja-launch-set-up-launch-setup-agent-advanced
    parent: set-up-launch
url: guides/launch/setup-agent-advanced
---

# 高度なエージェント設定

このガイドでは、さまざまな環境で W&B Launch エージェントを使ってコンテナイメージをビルドする方法について説明します。

{{% alert %}}
ビルドは git および code artifact ジョブにのみ必要です。イメージジョブにはビルドは不要です。

ジョブタイプの詳細については [Create a launch job]({{< relref path="../create-and-deploy-jobs/create-launch-job.md" lang="ja" >}}) をご参照ください。
{{% /alert %}}

## ビルダー

Launch エージェントは [Docker](https://docs.docker.com/) と [Kaniko](https://github.com/GoogleContainerTools/kaniko) を使ってイメージをビルドできます。

* Kaniko：Kubernetes 内で特権付きコンテナとしてビルドせずにコンテナイメージを作成します。
* Docker：ローカルで `docker build` コマンドを実行してコンテナイメージを作成します。

ビルダーの種類は、launch エージェント設定ファイルの `builder.type` キーでコントロールできます。`docker`、`kaniko`、またはビルドを無効化する場合は `noop` を指定します。デフォルトでは、エージェントの Helm チャートは `builder.type` を `noop` に設定しています。`builder` セクション内の追加のキーでビルドプロセスを設定できます。

エージェント設定でビルダーを指定しない場合、かつ `docker` CLI が動作する環境では Docker がデフォルトで使用されます。Docker が利用できない場合は、エージェントは自動的に `noop` になります。

{{% alert %}}
Kubernetes クラスターでイメージをビルドするには Kaniko を使用してください。それ以外の場合は Docker を使用してください。
{{% /alert %}}

## コンテナレジストリへのプッシュ

Launch エージェントは、ビルドしたすべてのイメージに一意のソースハッシュでタグを付けます。エージェントは `builder.destination` キーで指定されたレジストリにイメージをプッシュします。

例えば、`builder.destination` キーが `my-registry.example.com/my-repository` の場合、エージェントはイメージにタグを付与し `my-registry.example.com/my-repository:<source-hash>` へプッシュします。イメージがすでにレジストリに存在する場合、ビルドはスキップされます。

### エージェントの設定

Helm チャート経由でエージェントをデプロイする場合は、`values.yaml` ファイルの `agentConfig` キーにエージェント設定を記述します。

`wandb launch-agent` コマンドで自身でエージェントを起動する場合、`--config` フラグで YAML ファイルのパスを指定してエージェント設定を渡せます。デフォルトでは `~/.config/wandb/launch-config.yaml` から設定が読み込まれます。

launch エージェント設定ファイル（`launch-config.yaml`）内では、対象となるリソース環境の名前と、コンテナレジストリを `environment` および `registry` キーで指定してください。

下記のタブでは、環境とレジストリごとに Launch エージェントの設定方法を紹介します。

{{< tabpane text=true >}}
{{% tab "AWS" %}}
AWS 環境の設定には region キーが必要です。region にはエージェントが稼働する AWS リージョンを指定してください。

```yaml title="launch-config.yaml"
environment:
  type: aws
  region: <aws-region>
builder:
  type: <kaniko|docker>
  # エージェントがイメージを保存する ECR リポジトリの URI。
  # region が環境の設定と一致していることを確認してください。
  destination: <account-id>.ecr.<aws-region>.amazonaws.com/<repository-name>
  # Kaniko を使用する場合、エージェントがビルドコンテキストを保存する S3 バケットを指定します。
  build-context-store: s3://<bucket-name>/<path>
```

エージェントは boto3 を使ってデフォルトの AWS クレデンシャルを読み込みます。デフォルトの AWS クレデンシャルの設定方法については [boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html) を参照してください。
{{% /tab %}}
{{% tab "GCP" %}}
Google Cloud 環境には region および project キーが必要です。`region` にはエージェントが動作するリージョン、`project` にはプロジェクト ID を指定してください。エージェントは Python の `google.auth.default()` を使ってデフォルトのクレデンシャルを読み込みます。

```yaml title="launch-config.yaml"
environment:
  type: gcp
  region: <gcp-region>
  project: <gcp-project-id>
builder:
  type: <kaniko|docker>
  # エージェントがイメージを保存する Artifact Registry リポジトリとイメージ名の URI。
  # region と project が環境の設定と一致していることを確認してください。
  uri: <region>-docker.pkg.dev/<project-id>/<repository-name>/<image-name>
  # Kaniko を使う場合、Build context を保存する GCS バケットを指定します。
  build-context-store: gs://<bucket-name>/<path>
```

エージェントで利用可能なデフォルト GCP クレデンシャルの設定方法については、[`google-auth` documentation](https://google-auth.readthedocs.io/en/latest/reference/google.auth.html#google.auth.default) を参照してください。

{{% /tab %}}
{{% tab "Azure" %}}

Azure 環境では特別な追加キーは必要ありません。エージェント起動時に `azure.identity.DefaultAzureCredential()` を用いてデフォルトの Azure クレデンシャルを読み込みます。

```yaml title="launch-config.yaml"
environment:
  type: azure
builder:
  type: <kaniko|docker>
  # エージェントがイメージを保存する Azure Container Registry リポジトリの URI。
  destination: https://<registry-name>.azurecr.io/<repository-name>
  # Kaniko を使用する場合、ビルドコンテキストを保存する Azure Blob Storage コンテナを指定します。
  build-context-store: https://<storage-account-name>.blob.core.windows.net/<container-name>
```

デフォルトの Azure クレデンシャルの設定方法については、[`azure-identity` documentation](https://learn.microsoft.com/python/api/azure-identity/azure.identity.defaultazurecredential?view=azure-python) を参照してください。
{{% /tab %}}
{{< /tabpane >}}

## エージェントの権限

エージェントに必要な権限はユースケースによって異なります。

### クラウドレジストリの権限

クラウドレジストリと連携するために Launch エージェントに一般的に必要な権限を以下に示します。

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

Kaniko ビルダーを使う場合は [`AcrPush` role](https://learn.microsoft.com/azure/role-based-access-control/built-in-roles/containers#acrpush) を追加してください。
{{% /tab %}}
{{< /tabpane >}}

### Kaniko 用ストレージ権限

エージェントが Kaniko ビルダーを使う場合、クラウドストレージへのプッシュ権限が必要です。Kaniko はビルドジョブが動く pod の外部にコンテキストストアを利用します。

{{< tabpane text=true >}}
{{% tab "AWS" %}}
AWS における Kaniko ビルダーの推奨コンテキストストアは Amazon S3 です。以下のポリシーで、エージェントに S3 バケットへのアクセス権限を付与できます。

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
GCP では、GCS にビルドコンテキストをアップロードするために以下の IAM 権限が必要です。

```js
storage.buckets.get;
storage.objects.create;
storage.objects.delete;
storage.objects.get;
```

{{% /tab %}}
{{% tab "Azure" %}}

Azure Blob Storage にビルドコンテキストをアップロードするには、[Storage Blob Data Contributor](https://learn.microsoft.com/azure/role-based-access-control/built-in-roles#storage-blob-data-contributor) ロールが必要です。
{{% /tab %}}
{{< /tabpane >}}

## Kaniko ビルドのカスタマイズ

Kaniko ジョブで使用する Kubernetes ジョブ spec を、エージェント設定の `builder.kaniko-config` キーで指定できます。例：

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
            - "--cache=false" # Args の形式は "key=value" で指定します
            env:
            - name: "MY_ENV_VAR"
              value: "my-env-var-value"
```

## Launch エージェントを CoreWeave へデプロイ
オプションとして、W&B Launch エージェントを CoreWeave Cloud インフラにデプロイできます。CoreWeave は GPU アクセラレート ワークロード向けに特化されたクラウドインフラストラクチャーです。

CoreWeave への Launch エージェントのデプロイ方法については、[CoreWeave ドキュメント](https://docs.coreweave.com/partners/weights-and-biases#integration) をご参照ください。

{{% alert %}}
CoreWeave インフラへ Launch エージェントをデプロイするには、[CoreWeave アカウント](https://cloud.coreweave.com/login) が必要です。
{{% /alert %}}