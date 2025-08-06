---
title: ローンチ エージェントのセットアップ
menu:
  launch:
    identifier: setup-agent-advanced
    parent: set-up-launch
url: guides/launch/setup-agent-advanced
---

# 高度なエージェント設定

このガイドでは、W&B Launch エージェントを使って異なる環境でコンテナイメージをビルドする方法について説明します。

{{% alert %}}
ビルドが必要なのは git およびコードアーティファクトジョブのみです。イメージジョブではビルドは不要です。

ジョブタイプの詳細については、[Create a launch job]({{< relref "../create-and-deploy-jobs/create-launch-job.md" >}}) をご覧ください。
{{% /alert %}}

## ビルダー

Launch エージェントは [Docker](https://docs.docker.com/) または [Kaniko](https://github.com/GoogleContainerTools/kaniko) を使ってイメージをビルドできます。

* Kaniko: Kubernetes で特権コンテナとしてビルドを実行せずにコンテナイメージを作成します。
* Docker: ローカルで `docker build` コマンドを実行してコンテナイメージを作成します。

ビルダータイプは、ローンチエージェントの設定ファイル内の `builder.type` キーで `docker`、`kaniko`、またはビルドを無効化する場合は `noop` に設定できます。デフォルトでは、エージェントのHelmチャートは `builder.type` を `noop` に設定しています。`builder` セクション内の追加のキーはビルドプロセスの設定に使用されます。

エージェント設定でビルダーが指定されておらず、また `docker` CLI が利用可能な場合、エージェントはデフォルトで Docker を使用します。Docker が利用できない場合はデフォルトで `noop` となります。

{{% alert %}}
Kubernetes クラスターでイメージをビルドする場合は Kaniko を、その他のケースでは Docker をご利用ください。
{{% /alert %}}


## コンテナレジストリへのプッシュ

Launch エージェントはビルドした全てのイメージに一意のソースハッシュをタグ付けします。エージェントは `builder.destination` キーで指定されたレジストリにイメージをプッシュします。

たとえば、`builder.destination` キーが `my-registry.example.com/my-repository` に設定されている場合、エージェントはイメージを `my-registry.example.com/my-repository:<source-hash>` としてタグ付け・プッシュします。もしそのイメージが既にレジストリに存在する場合、ビルドはスキップされます。

### エージェント設定

Helm チャートを使ってエージェントをデプロイする場合、`values.yaml` ファイル内の `agentConfig` キーにエージェントの設定を指定してください。

手動で `wandb launch-agent` を実行する場合は、`--config` フラグで YAML ファイルへのパスを指定してエージェント設定を渡せます。デフォルトでは、設定は `~/.config/wandb/launch-config.yaml` から読み込まれます。

ローンチエージェントの設定ファイル（`launch-config.yaml`）内で、ターゲットリソース環境の名前およびコンテナレジストリをそれぞれ `environment` と `registry` キーで指定してください。

下記のタブでは、ご利用の環境・レジストリごとにエージェントの設定例を紹介します。

{{< tabpane text=true >}}
{{% tab "AWS" %}}
AWS 環境の設定では、`region` キーが必須です。リージョンはエージェントが実行される AWS リージョンを指定してください。

```yaml title="launch-config.yaml"
environment:
  type: aws
  region: <aws-region>
builder:
  type: <kaniko|docker>
  # エージェントがイメージを保存する ECR リポジトリの URIです。
  # 設定したリージョンがご利用の環境と一致していることを確認してください。
  destination: <account-id>.ecr.<aws-region>.amazonaws.com/<repository-name>
  # Kaniko を使用する場合、エージェントがビルドコンテキストを保存する
  # S3 バケットを指定します。
  build-context-store: s3://<bucket-name>/<path>
```

エージェントはデフォルトの AWS 資格情報の読み込みに boto3 を使用します。デフォルトの AWS 資格情報の設定方法については [boto3 のドキュメント](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html) をご覧ください。
{{% /tab %}}
{{% tab "GCP" %}}
Google Cloud 環境では、`region` と `project` キーが必須です。`region` にはエージェントを実行するリージョンを、`project` にはエージェントを実行する Google Cloud プロジェクトを指定してください。エージェントは Python の `google.auth.default()` を使ってデフォルトの資格情報を読み込みます。

```yaml title="launch-config.yaml"
environment:
  type: gcp
  region: <gcp-region>
  project: <gcp-project-id>
builder:
  type: <kaniko|docker>
  # エージェントがイメージを保存する Artifact Registry リポジトリとイメージ名の URI。
  # region・project の値が環境設定と一致していることを確認してください。
  uri: <region>-docker.pkg.dev/<project-id>/<repository-name>/<image-name>
  # Kaniko を使う場合は、ビルドコンテキストを保存する GCS バケットを指定してください。
  build-context-store: gs://<bucket-name>/<path>
```

エージェントで利用可能な形でデフォルトの GCP 資格情報を設定する方法は、[`google-auth` のドキュメント](https://google-auth.readthedocs.io/en/latest/reference/google.auth.html#google.auth.default)をご覧ください。

{{% /tab %}}
{{% tab "Azure" %}}

Azure 環境では追加のキーは必要ありません。エージェント起動時に、`azure.identity.DefaultAzureCredential()` がデフォルトの Azure 資格情報を読み込みます。

```yaml title="launch-config.yaml"
environment:
  type: azure
builder:
  type: <kaniko|docker>
  # エージェントがイメージを保存する Azure Container Registry リポジトリの URI。
  destination: https://<registry-name>.azurecr.io/<repository-name>
  # Kaniko をご利用の場合、ビルドコンテキストを保存する Azure Blob Storage コンテナを指定ください。
  build-context-store: https://<storage-account-name>.blob.core.windows.net/<container-name>
```

デフォルトの Azure 資格情報の設定方法については、[`azure-identity` ドキュメント](https://learn.microsoft.com/python/api/azure-identity/azure.identity.defaultazurecredential?view=azure-python) をご覧ください。
{{% /tab %}}
{{< /tabpane >}}

## エージェント権限

必要なエージェントの権限はユースケースごとに異なります。

### クラウドレジストリへの権限

クラウドレジストリと連携するために Launch エージェントに一般的に必要な権限は、以下の通りです。

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

Kaniko ビルダーを使う場合は、[`AcrPush` ロール](https://learn.microsoft.com/azure/role-based-access-control/built-in-roles/containers#acrpush) を追加してください。
{{% /tab %}}
{{< /tabpane >}}

### Kaniko 用ストレージ権限

Kaniko ビルダーを使う場合、Launch エージェントにはクラウドストレージへの書き込み権限が必要です。Kaniko はビルドジョブを実行するPodの外部にコンテキストストアを利用します。

{{< tabpane text=true >}}
{{% tab "AWS" %}}
AWS では、Kaniko ビルダーの推奨コンテキストストアは Amazon S3 です。以下のポリシーでエージェントに S3 バケットへのアクセス権を付与できます。

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
GCP では、GCS へのビルドコンテキストアップロードに以下の IAM 権限が必要です。

```js
storage.buckets.get;
storage.objects.create;
storage.objects.delete;
storage.objects.get;
```

{{% /tab %}}
{{% tab "Azure" %}}

Azure Blob Storage へのビルドコンテキストのアップロードには [Storage Blob Data Contributor](https://learn.microsoft.com/azure/role-based-access-control/built-in-roles#storage-blob-data-contributor) ロールが必要です。
{{% /tab %}}
{{< /tabpane >}}


## Kaniko ビルドのカスタマイズ

エージェント設定の `builder.kaniko-config` キーで、Kaniko ジョブに使用する Kubernetes Job spec を指定できます。例:

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
            - "--cache=false" # 引数は "key=value" の形式で記述してください
            env:
            - name: "MY_ENV_VAR"
              value: "my-env-var-value"
```

## Launch エージェントを CoreWeave へデプロイする
W&B Launch エージェントを CoreWeave クラウドインフラストラクチャーにデプロイすることも可能です。CoreWeave は GPU 処理に最適化されたクラウドインフラストラクチャーです。

CoreWeave への Launch エージェントデプロイ方法は [CoreWeave のドキュメント](https://docs.coreweave.com/partners/weights-and-biases#integration) をご覧ください。

{{% alert %}}
CoreWeave インフラストラクチャーに Launch エージェントをデプロイするには、[CoreWeave アカウント](https://cloud.coreweave.com/login) を作成する必要があります。
{{% /alert %}}