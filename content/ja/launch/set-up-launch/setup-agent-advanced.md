---
title: ローンンチエージェントを設定する
menu:
  launch:
    identifier: ja-launch-set-up-launch-setup-agent-advanced
    parent: set-up-launch
url: /ja/guides/launch/setup-agent-advanced
---

# 高度なエージェント設定

このガイドでは、W&B ローンチエージェントを設定して、さまざまな環境でコンテナイメージを作成する方法について情報を提供します。

{{% alert %}}
ビルドは git およびコードアーティファクトジョブにのみ必要です。イメージジョブにはビルドは必要ありません。

ジョブタイプの詳細については、「[ローンチジョブの作成]({{< relref path="../create-and-deploy-jobs/create-launch-job.md" lang="ja" >}})」を参照してください。
{{% /alert %}}

## ビルダー

ローンチエージェントは、[Docker](https://docs.docker.com/) または [Kaniko](https://github.com/GoogleContainerTools/kaniko) を使用してイメージをビルドできます。

* Kaniko: Kubernetes で特権コンテナとしてビルドを実行せずにコンテナイメージをビルドします。
* Docker: ローカルで `docker build` コマンドを実行してコンテナイメージをビルドします。

ビルダータイプは、ローンチエージェントの設定で `builder.type` キーを使用して、`docker`、`kaniko`、またはビルドをオフにするための `noop` に制御できます。デフォルトでは、エージェントの Helm チャートは `builder.type` を `noop` に設定します。`builder` セクションの追加キーは、ビルドプロセスを設定するために使用されます。

エージェントの設定でビルダーが指定されていない場合、有効な `docker` CLI が見つかると、エージェントは自動的に Docker を使用します。Docker が利用できない場合、エージェントは `noop` をデフォルトとします。

{{% alert %}}
Kubernetes クラスターでイメージをビルドするには Kaniko を使用してください。それ以外の場合は Docker を使用してください。
{{% /alert %}}

## コンテナレジストリへのプッシュ

ローンチエージェントは、ビルドするすべてのイメージに一意のソースハッシュでタグを付けます。エージェントは、`builder.destination` キーで指定されたレジストリにイメージをプッシュします。

たとえば、`builder.destination` キーが `my-registry.example.com/my-repository` に設定されている場合、エージェントはイメージに `my-registry.example.com/my-repository:<source-hash>` というタグを付けてプッシュします。イメージがすでにレジストリに存在する場合、ビルドはスキップされます。

### エージェント設定

Helm チャートを経由してエージェントをデプロイする場合、エージェント設定は `values.yaml` ファイルの `agentConfig` キーに提供する必要があります。

自分で `wandb launch-agent` を使用してエージェントを呼び出す場合、エージェント設定を `--config` フラグを使用して YAML ファイルのパスとして提供できます。デフォルトでは、設定は `~/.config/wandb/launch-config.yaml` から読み込まれます。

ローンチエージェントの設定 (`launch-config.yaml`) 内で、ターゲットリソース環境とコンテナレジストリの名前をそれぞれ `environment` と `registry` キーに提供します。

環境とレジストリに基づいてローンチエージェントを設定する方法を、以下のタブで示します。

{{< tabpane text=true >}}
{{% tab "AWS" %}}
AWS 環境設定には地域キーが必要です。リージョンはエージェントが実行される AWS 地域であるべきです。

```yaml title="launch-config.yaml"
environment:
  type: aws
  region: <aws-region>
builder:
  type: <kaniko|docker>
  # エージェントがイメージを保存する ECR レポジトリの URI。
  # リージョンが環境に設定した内容と一致することを確認してください。
  destination: <account-id>.ecr.<aws-region>.amazonaws.com/<repository-name>
  # Kaniko を使用する場合、エージェントがビルドコンテキストを保存する S3 バケットを指定します。
  build-context-store: s3://<bucket-name>/<path>
```

エージェントは boto3 を使用してデフォルトの AWS 資格情報を読み込みます。デフォルトの AWS 資格情報の設定方法については、[boto3 ドキュメント](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html) を参照してください。
{{% /tab %}}
{{% tab "GCP" %}}
Google Cloud 環境には、region および project キーが必要です。`region` にはエージェントが実行されるリージョンを設定し、`project` にはエージェントが実行される Google Cloud プロジェクトを設定します。エージェントは Python の `google.auth.default()` を使用してデフォルトの資格情報を読み込みます。

```yaml title="launch-config.yaml"
environment:
  type: gcp
  region: <gcp-region>
  project: <gcp-project-id>
builder:
  type: <kaniko|docker>
  # エージェントがイメージを保存するアーティファクトリポジトリとイメージ名の URI。
  # リージョンとプロジェクトが環境に設定した内容と一致することを確認してください。
  uri: <region>-docker.pkg.dev/<project-id>/<repository-name>/<image-name>
  # Kaniko を使用する場合、エージェントがビルドコンテキストを保存する GCS バケットを指定します。
  build-context-store: gs://<bucket-name>/<path>
```

デフォルトの GCP 資格情報をエージェントが利用できるように設定する方法については、[`google-auth` ドキュメント](https://google-auth.readthedocs.io/en/latest/reference/google.auth.html#google.auth.default) を参照してください。

{{% /tab %}}
{{% tab "Azure" %}}

Azure 環境には追加のキーは必要ありません。エージェントが起動するときに、`azure.identity.DefaultAzureCredential()` を使用してデフォルトの Azure 資格情報を読み込みます。

```yaml title="launch-config.yaml"
environment:
  type: azure
builder:
  type: <kaniko|docker>
  # エージェントがイメージを保存する Azure コンテナレジストリレポジトリの URI。
  destination: https://<registry-name>.azurecr.io/<repository-name>
  # Kaniko を使用する場合、エージェントがビルドコンテキストを保存する Azure Blob Storage コンテナを指定します。
  build-context-store: https://<storage-account-name>.blob.core.windows.net/<container-name>
```

デフォルトの Azure 資格情報の設定方法については、[`azure-identity` ドキュメント](https://learn.microsoft.com/python/api/azure-identity/azure.identity.defaultazurecredential?view=azure-python) を参照してください。
{{% /tab %}}
{{< /tabpane >}}

## エージェント権限

エージェントの必要な権限はユースケースによって異なります。

### クラウドレジストリ権限

ローンチエージェントがクラウドレジストリと対話するために通常必要な権限は以下の通りです。

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

Kaniko ビルダーを使用する場合は、[`AcrPush` ロール](https://learn.microsoft.com/azure/container-registry/container-registry-roles?tabs=azure-cli#acrpush)を追加してください。
{{% /tab %}}
{{< /tabpane >}}

### Kaniko のためのストレージ権限

ローンチエージェントは、Kaniko ビルダーを使用している場合、クラウドストレージにプッシュする権限が必要です。Kaniko はビルドジョブを実行するポッドの外にコンテキストストアを使用します。

{{< tabpane text=true >}}
{{% tab "AWS" %}}
AWS での Kaniko ビルダーの推奨コンテキストストアは Amazon S3 です。エージェントが S3 バケットにアクセスするためのポリシーは以下の通りです：

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
GCP では、エージェントが GCS にビルドコンテキストをアップロードするために必要な IAM 権限は次の通りです：

```js
storage.buckets.get;
storage.objects.create;
storage.objects.delete;
storage.objects.get;
```

{{% /tab %}}
{{% tab "Azure" %}}

Azure Blob Storage にビルドコンテキストをアップロードするためには、[Storage Blob Data Contributor](https://learn.microsoft.com/azure/role-based-access-control/built-in-roles#storage-blob-data-contributor) ロールが必要です。
{{% /tab %}}
{{< /tabpane >}}

## Kaniko ビルドのカスタマイズ

Kaniko ジョブが使用する Kubernetes ジョブ仕様をエージェント設定の `builder.kaniko-config` キーに指定します。例えば：

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
            - "--cache=false" # 引数は "key=value" の形式でなければなりません
            env:
            - name: "MY_ENV_VAR"
              value: "my-env-var-value"
```

## Launch エージェントを CoreWeave にデプロイ
オプションとして、W&B Launch エージェントを CoreWeave クラウドインフラストラクチャにデプロイできます。CoreWeave は GPU 加速ワークロード専用に構築されたクラウドインフラストラクチャです。

CoreWeave に Launch エージェントをデプロイする方法については、[CoreWeave ドキュメント](https://docs.coreweave.com/partners/weights-and-biases#integration) を参照してください。

{{% alert %}}
Launch エージェントを CoreWeave インフラストラクチャにデプロイするには、[CoreWeave アカウント](https://cloud.coreweave.com/login) を作成する必要があります。
{{% /alert %}}