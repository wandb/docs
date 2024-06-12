---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Advanced agent setup

このガイドでは、W&B Launchエージェントを設定して、さまざまな環境でコンテナイメージをビルドする方法について説明します。

:::info
ビルドはgitおよびcode artifactジョブにのみ必要です。イメージジョブにはビルドは不要です。

ジョブタイプの詳細については、[Create a launch job](./create-launch-job.md)を参照してください。
:::

## Builders

Launchエージェントは、[Docker](https://docs.docker.com/)または[Kaniko](https://github.com/GoogleContainerTools/kaniko)を使用してイメージをビルドできます。

* Kaniko: 特権コンテナとしてビルドを実行せずにKubernetesでコンテナイメージをビルドします。
* Docker: `docker build`コマンドをローカルで実行してコンテナイメージをビルドします。

ビルダータイプは、launchエージェントの設定内の`builder.type`キーで制御できます。`docker`、`kaniko`、またはビルドを無効にするための`noop`を指定できます。デフォルトでは、エージェントのHelmチャートは`builder.type`を`noop`に設定します。`builder`セクション内の追加のキーは、ビルドプロセスを設定するために使用されます。

エージェント設定にビルダーが指定されておらず、動作する`docker` CLIが見つかった場合、エージェントはデフォルトでDockerを使用します。Dockerが利用できない場合、エージェントはデフォルトで`noop`を使用します。

:::tip
KubernetesクラスターでイメージをビルドするにはKanikoを使用します。それ以外のすべてのケースではDockerを使用します。
:::

## コンテナレジストリへのプッシュ

Launchエージェントは、ビルドしたすべてのイメージに一意のソースハッシュをタグ付けします。エージェントは、`builder.destination`キーで指定されたレジストリにイメージをプッシュします。

例えば、`builder.destination`キーが`my-registry.example.com/my-repository`に設定されている場合、エージェントはイメージを`my-registry.example.com/my-repository:<source-hash>`にタグ付けしてプッシュします。レジストリにイメージが存在する場合、ビルドはスキップされます。

### エージェント設定

エージェントをHelmチャート経由でデプロイする場合、エージェント設定は`values.yaml`ファイルの`agentConfig`キーに提供する必要があります。

`wandb launch-agent`を使用してエージェントを自分で呼び出す場合、`--config`フラグでYAMLファイルへのパスとしてエージェント設定を提供できます。デフォルトでは、設定は`~/.config/wandb/launch-config.yaml`から読み込まれます。

launchエージェント設定（`launch-config.yaml`）内で、ターゲットリソース環境の名前とコンテナレジストリをそれぞれ`environment`および`registry`キーに提供します。

以下のタブは、環境とレジストリに基づいてlaunchエージェントを設定する方法を示しています。

<Tabs
defaultValue="aws"
values={[
{label: 'Amazon Web Services', value: 'aws'},
{label: 'Google Cloud', value: 'gcp'},
{label: 'Azure', value: 'azure'},
]}>
<TabItem value="aws">

AWS環境設定にはregionキーが必要です。regionはエージェントが実行されるAWSリージョンである必要があります。

```yaml title="launch-config.yaml"
environment:
  type: aws
  region: <aws-region>
builder:
  type: <kaniko|docker>
  # エージェントがイメージを保存するECRリポジトリのURI。
  # リージョンが環境で設定したものと一致していることを確認してください。
  destination: <account-id>.ecr.<aws-region>.amazonaws.com/<repository-name>
  # Kanikoを使用する場合、エージェントがビルドコンテキストを保存するS3バケットを指定します。
  build-context-store: s3://<bucket-name>/<path>
```

エージェントはboto3を使用してデフォルトのAWSクレデンシャルを読み込みます。デフォルトのAWSクレデンシャルの設定方法については、[boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)を参照してください。

  </TabItem>
  <TabItem value="gcp">

Google Cloud環境にはregionおよびprojectキーが必要です。`region`をエージェントが実行されるリージョンに設定します。`project`をエージェントが実行されるGoogle Cloudプロジェクトに設定します。エージェントはPythonの`google.auth.default()`を使用してデフォルトのクレデンシャルを読み込みます。

```yaml title="launch-config.yaml"
environment:
  type: gcp
  region: <gcp-region>
  project: <gcp-project-id>
builder:
  type: <kaniko|docker>
  # エージェントがイメージを保存するArtifact Registryリポジトリとイメージ名のURI。
  # リージョンとプロジェクトが環境で設定したものと一致していることを確認してください。
  uri: <region>-docker.pkg.dev/<project-id>/<repository-name>/<image-name>
  # Kanikoを使用する場合、エージェントがビルドコンテキストを保存するGCSバケットを指定します。
  build-context-store: gs://<bucket-name>/<path>
```

デフォルトのGCPクレデンシャルをエージェントで利用できるように設定する方法については、[`google-auth` documentation](https://google-auth.readthedocs.io/en/latest/reference/google.auth.html#google.auth.default)を参照してください。

  </TabItem>
  <TabItem value="azure">

Azure環境には追加のキーは必要ありません。エージェントが起動すると、`azure.identity.DefaultAzureCredential()`を使用してデフォルトのAzureクレデンシャルを読み込みます。

```yaml title="launch-config.yaml"
environment:
  type: azure
builder:
  type: <kaniko|docker>
  # エージェントがイメージを保存するAzure Container RegistryリポジトリのURI。
  destination: https://<registry-name>.azurecr.io/<repository-name>
  # Kanikoを使用する場合、エージェントがビルドコンテキストを保存するAzure Blob Storageコンテナを指定します。
  build-context-store: https://<storage-account-name>.blob.core.windows.net/<container-name>
```

デフォルトのAzureクレデンシャルを設定する方法については、[`azure-identity` documentation](https://learn.microsoft.com/en-us/python/api/azure-identity/azure.identity.defaultazurecredential?view=azure-python)を参照してください。

  </TabItem>
</Tabs>

## エージェントの権限

エージェントに必要な権限はユースケースによって異なります。

### クラウドレジストリの権限

以下は、launchエージェントがクラウドレジストリと対話するために一般的に必要とされる権限です。

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

Kanikoビルダーを使用する場合は、[`AcrPush` role](https://learn.microsoft.com/en-us/azure/container-registry/container-registry-roles?tabs=azure-cli#acrpush)を追加します。

</TabItem>
</Tabs>

### Kanikoのストレージ権限

エージェントがKanikoビルダーを使用する場合、クラウドストレージにプッシュする権限が必要です。Kanikoはビルドジョブを実行するポッドの外部にコンテキストストアを使用します。

<Tabs
defaultValue="aws"
values={[
{label: 'Amazon Web Services', value: 'aws'},
{label: 'Google Cloud', value: 'gcp'},
{label: 'Azure', value: 'azure'},
]}>
<TabItem value="aws">

AWSでKanikoビルダーの推奨コンテキストストアはAmazon S3です。以下のポリシーを使用して、エージェントにS3バケットへのアクセスを許可できます。

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

GCPでは、エージェントがGCSにビルドコンテキストをアップロードするために必要なIAM権限は以下の通りです。

```js
storage.buckets.get;
storage.objects.create;
storage.objects.delete;
storage.objects.get;
```

  </TabItem>
  <TabItem value="azure">

エージェントがAzure Blob Storageにビルドコンテキストをアップロードするためには、[Storage Blob Data Contributor](https://learn.microsoft.com/en-us/azure/role-based-access-control/built-in-roles#storage-blob-data-contributor)ロールが必要です。

  </TabItem>
</Tabs>

## Kanikoビルドのカスタマイズ

エージェント設定の`builder.kaniko-config`キーに、Kanikoジョブが使用するKubernetes Job specを指定します。例えば：

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
            - "--cache=false" # Argsは"key=value"形式である必要があります
            env:
            - name: "MY_ENV_VAR"
              value: "my-env-var-value"
```

## LaunchエージェントをCoreWeaveにデプロイ
オプションで、W&B LaunchエージェントをCoreWeaveクラウドインフラストラクチャにデプロイします。CoreWeaveはGPU加速ワークロード向けに特化したクラウドインフラストラクチャです。

LaunchエージェントをCoreWeaveにデプロイする方法については、[CoreWeave documentation](https://docs.coreweave.com/partners/weights-and-biases#integration)を参照してください。

:::note
LaunchエージェントをCoreWeaveインフラストラクチャにデプロイするには、[CoreWeave account](https://cloud.coreweave.com/login)を作成する必要があります。
:::
