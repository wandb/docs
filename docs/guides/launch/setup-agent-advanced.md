---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Advanced agent setup

このガイドでは、W&B Launch エージェントをセットアップして、異なる環境でコンテナイメージをビルドする方法について説明します。

:::info
ビルドは git と code artifact ジョブにのみ必要です。イメージジョブにはビルドは必要ありません。

ジョブタイプの詳細については [Create a launch job](./create-launch-job.md) を参照してください。
:::

## Builders

Launch エージェントは [Docker](https://docs.docker.com/) または [Kaniko](https://github.com/GoogleContainerTools/kaniko) を使用してイメージをビルドできます。

* Kaniko: 特権コンテナとしてビルドを実行することなく、Kubernetes でコンテナイメージをビルドします。
* Docker: ローカルで `docker build` コマンドを実行してコンテナイメージをビルドします。

ビルダーのタイプは launch エージェント設定の `builder.type` キーで `docker`、`kaniko`、またはビルドを無効にする `noop` のいずれかに設定できます。デフォルトでは、エージェントの Helm チャートは `builder.type` を `noop` に設定します。`builder` セクションの追加キーはビルドプロセスの設定に使用されます。

エージェント設定にビルダーが指定されていないか、動作する `docker` CLI が見つかった場合、エージェントはデフォルトで Docker を使用します。Docker が利用できない場合、エージェントはデフォルトで `noop` を使用します。

:::tip
Kubernetes クラスターでのイメージビルドには Kaniko を使用します。それ以外の場合には Docker を使用します。
:::

## コンテナレジストリにプッシュする

launch エージェントはビルドするすべてのイメージを一意のソースハッシュでタグ付けします。エージェントは `builder.destination` キーで指定されたレジストリにイメージをプッシュします。

例えば、`builder.destination` キーが `my-registry.example.com/my-repository` に設定されている場合、エージェントはイメージを `my-registry.example.com/my-repository:<source-hash>` にタグ付けしてプッシュします。イメージがレジストリに既に存在する場合、ビルドはスキップされます。

### エージェント設定

エージェントを Helm チャート経由でデプロイする場合、エージェント設定は `values.yaml` ファイルの `agentConfig` キーで提供する必要があります。

`wandb launch-agent` を使って自分でエージェントを起動する場合、`--config` フラグを使ってエージェント設定の YAML ファイルのパスを提供できます。デフォルトでは、設定は `~/.config/wandb/launch-config.yaml` から読み込まれます。

launch エージェント設定 (`launch-config.yaml`) 内で、ターゲットリソース環境の名前とコンテナレジストリを `environment` キーおよび `registry` キーにそれぞれ指定します。

以下のタブはあなたの環境とレジストリに基づいた launch エージェントの設定方法を示します。

<Tabs
defaultValue="aws"
values={[
{label: 'Amazon Web Services', value: 'aws'},
{label: 'Google Cloud', value: 'gcp'},
{label: 'Azure', value: 'azure'},
]}>
<TabItem value="aws">

AWS環境の設定では region キーが必要です。region はエージェントが実行される AWS のリージョンである必要があります。

```yaml title="launch-config.yaml"
environment:
  type: aws
  region: <aws-region>
builder:
  type: <kaniko|docker>
  # エージェントがイメージを保存する ECR レポジトリのURI。
  # リージョンがあなたの環境で設定したものと一致することを確認してください。
  destination: <account-id>.ecr.<aws-region>.amazonaws.com/<repository-name>
  # Kanikoを使用する場合、エージェントがビルドコンテキストを保存する
  # S3バケットを指定してください。
  build-context-store: s3://<bucket-name>/<path>
```

エージェントは boto3 を使用してデフォルトの AWS 認証情報をロードします。デフォルトの AWS 認証情報の設定方法については、[boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html) を参照してください。

</TabItem>
<TabItem value="gcp">

Google Cloud 環境では region および project キーが必要です。`region` はエージェントが実行されるリージョンに設定してください。`project` はエージェントが実行される Google Cloud プロジェクトに設定してください。エージェントは Python の `google.auth.default()` を使用してデフォルトの認証情報をロードします。

```yaml title="launch-config.yaml"
environment:
  type: gcp
  region: <gcp-region>
  project: <gcp-project-id>
builder:
  type: <kaniko|docker>
  # エージェントがイメージを保存するアーティファクトレジストリリポジトリおよびイメージ名のURI。
  # リージョンとプロジェクトがあなたの環境で設定したものと一致することを確認してください。
  uri: <region>-docker.pkg.dev/<project-id>/<repository-name>/<image-name>
  # Kanikoを使用する場合、エージェントがビルドコンテキストを保存する
  # GCSバケットを指定してください。
  build-context-store: gs://<bucket-name>/<path>
```

デフォルトの GCP 認証情報がエージェントで利用可能になるように設定する方法の詳細については、[`google-auth` documentation](https://google-auth.readthedocs.io/en/latest/reference/google.auth.html#google.auth.default) を参照してください。

</TabItem>
<TabItem value="azure">

Azure 環境では追加のキーは必要ありません。エージェントが開始されると、`azure.identity.DefaultAzureCredential()` を使用してデフォルトの Azure 認証情報がロードされます。

```yaml title="launch-config.yaml"
environment:
  type: azure
builder:
  type: <kaniko|docker>
  # エージェントがイメージを保存する Azure Container Registry リポジトリのURI。
  destination: https://<registry-name>.azurecr.io/<repository-name>
  # Kanikoを使用する場合、エージェントがビルドコンテキストを保存する
  # Azure Blob Storage コンテナを指定してください。
  build-context-store: https://<storage-account-name>.blob.core.windows.net/<container-name>
```

デフォルトの Azure 認証情報の設定方法については、[`azure-identity` documentation](https://learn.microsoft.com/en-us/python/api/azure-identity/azure.identity.defaultazurecredential?view=azure-python) を参照してください。

</TabItem>
</Tabs>

## エージェントの権限

エージェントに必要な権限はユースケースによって異なります。

### クラウドレジストリの権限

以下に、launch エージェントがクラウドレジストリと対話するために通常必要な権限を示します。

<Tabs
defaultValue="aws"
values={[
{label: 'Amazon Web Services', value: 'aws'},
{label: 'Google Cloud', value: 'gcp'},
{label: 'Azure', value: 'azure'},
]}>
<TabItem value="aws">

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

</TabItem>
<TabItem value="gcp">

```js
artifactregistry.dockerimages.list;
artifactregistry.repositories.downloadArtifacts;
artifactregistry.repositories.list;
artifactregistry.repositories.uploadArtifacts;
```

</TabItem>
<TabItem value="azure">

Kaniko ビルダーを使用する場合は、[`AcrPush` role](https://learn.microsoft.com/en-us/azure/container-registry/container-registry-roles?tabs=azure-cli#acrpush) を追加してください。

</TabItem>
</Tabs>

### Kanikoのストレージ権限

launch エージェントは、Kaniko ビルダーを使用している場合にクラウドストレージにプッシュする権限が必要です。Kaniko はビルドジョブを実行するポッドの外部にコンテキストストアを使用します。

<Tabs
defaultValue="aws"
values={[
{label: 'Amazon Web Services', value: 'aws'},
{label: 'Google Cloud', value: 'gcp'},
{label: 'Azure', value: 'azure'},
]}>
<TabItem value="aws">

AWS で Kaniko ビルダーに推奨されるコンテキストストアは Amazon S3 です。以下のポリシーを使用してエージェントに S3 バケットへのアクセス権を付与できます。

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

</TabItem>
<TabItem value="gcp">

GCP では、エージェントが GCS にビルドコンテキストをアップロードするために必要な IAM 権限は次の通りです。

```js
storage.buckets.get;
storage.objects.create;
storage.objects.delete;
storage.objects.get;
```

</TabItem>
<TabItem value="azure">

エージェントが Azure Blob Storage にビルドコンテキストをアップロードするために、[Storage Blob Data Contributor](https://learn.microsoft.com/en-us/azure/role-based-access-control/built-in-roles#storage-blob-data-contributor) ロールが必要です。

</TabItem>
</Tabs>

## Kaniko ビルドのカスタマイズ

エージェント設定の `builder.kaniko-config` キーに Kaniko ジョブが使用する Kubernetes Job スペックを指定します。例えば：

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

## Launch エージェントを CoreWeave にデプロイする
任意で W&B Launch エージェントを CoreWeave クラウドインフラにデプロイします。CoreWeave は GPU 加速ワークロードのために設計されたクラウドインフラです。

Launch エージェントを CoreWeave にデプロイする方法については [CoreWeave documentation](https://docs.coreweave.com/partners/weights-and-biases#integration) を参照してください。

:::note
Launch エージェントを CoreWeave インフラにデプロイするためには [CoreWeave account](https://cloud.coreweave.com/login) を作成する必要があります。
:::