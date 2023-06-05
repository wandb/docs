---
description: Launch agent documentation
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# エージェントの開始

あなたのキューからジョブを実行するために、launch agentを開始します。

## 概要

**launch agent**は、1つ以上のlaunchキューをポーリングし、キューから取り出したジョブを実行する長時間実行プロセスです。`wandb launch-agent`コマンドでlaunch agentを開始し、Docker、Kubernetes、SageMakerなど、多数のコンピューティングプラットフォームでジョブを実行できます。launch agentは、[オープンソースのクライアントライブラリ](https://github.com/wandb/wandb/tree/main/wandb/sdk/launch)で完全に実装されています。

## 動作の仕組み

エージェントは、1つ以上のキューをポーリングします。launch agentがキューからアイテムを取り出すと、必要に応じて、実行するコンテナイメージをビルドし、そのコンテナイメージをキューによってターゲット指定された計算プラットフォームで実行します。

これらのコンテナビルドと実行は非同期に行われます。エージェントが実行できる最大同時ジョブ数を制御するには、`wandb launch-agent`コマンドに`-j <num-max-jobs>`を渡すか、エージェントの設定ファイルで`max_jobs`フィールドを設定します。

### ジョブをコンテナイメージにコンパイルする

ジョブがGitリポジトリ内のソースコードやコードアーティファクトを含んでいる場合、launch agentはそのコードとジョブで指定されたすべての依存関係を含むコンテナイメージをビルドします。

ジョブが`WANDB_DOCKER`環境変数経由でコンテナイメージからソースされている場合、この手順はスキップされます。

Launchでは、現在[Docker](https://docker.com)と[Kaniko](https://github.com/GoogleContainerTools/kaniko)でコンテナイメージをビルドすることができます。Kanikoは、エージェントをコンテナ内で実行するときにのみ使用し、Docker-in-Dockerのセキュリティリスクを回避してください。

### 実行を実行する

launch agentは、前の手順でビルドされたコンテナイメージ内で、またはジョブで指定されたコンテナイメージ内で実行を実行します。エージェントが実行を実行する方法は、ジョブが所属していたキューのタイプによります。
たとえば、ジョブがDockerキューにある場合、エージェントは`docker run`コマンドでローカルにrunを実行します。ジョブがKubernetesキューにある場合、エージェントはk8sクラスター上で[k8s Job](https://kubernetes.io/docs/concepts/workloads/controllers/job/)としてrunを実行し、k8s APIを介して実行します。

## エージェントの設定

ローンチエージェントは、さまざまなフラグとオプションで構成できます。 エージェントは、設定ファイルまたはコマンドラインフラグで構成できます。 設定ファイルはデフォルトで`~/.config/wandb/launch-config.yaml`に配置されていますが、`--config`フラグで設定ファイルの場所を上書きできます。 設定ファイルは、次の構造を持つYAMLファイルです：

```yaml
# W&Bエンティティ（ユーザーまたはチーム）名
entity: <entity-name>

# 並行して実行できるrunの最大数。-1 = 制限なし
max_jobs: -1

# プーリングするキューのリスト。
queues:
- default

# クラウド環境の設定。
environment:
  type: aws|gcp

# コンテナレジストリの設定。
registry:
  type: ecr|gcr

# コンテナビルド設定
builder：
  type：docker|kaniko|noop
```

`environment`、`registry`、および`builder`キーはオプションです。デフォルトでは、エージェントはクラウド環境を使用せず、ローカルのDockerレジストリとDockerビルダーを使用します。

### 環境

`environment`キーは、エージェントがその仕事をするためにアクセスする必要のあるクラウド環境を設定するために使用されます。 エージェントがクラウドリソースへのアクセスを必要としない場合、このキーは省略される必要があります。AWSまたはGCP環境の設定方法については、以下のリファレンスを参照してください。

<Tabs
  defaultValue="aws"
  values={[
    {label: 'AWS', value: 'aws'},
    {label: 'GCP', value: 'gcp'},
  ]}>

<TabItem value="aws">

AWS環境の設定では、`region`キーの設定が必要です。リージョンは、エージェントが実行されるAWSリージョンにする必要があります。 エージェントが起動すると、`boto3`を使用してデフォルトのAWS資格情報をロードします。 デフォルトのAWS資格情報の設定方法については、[boto3のドキュメント](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html#overview)を参照してください。

```yaml
environment:
  type: aws
  region: <aws-region>
```

</TabItem>

<TabItem value="gcp">
GCP環境では、`region` と `project` キーを設定する必要があります。regionは、エージェントが実行されるGCPのリージョンであるべきです。projectは、エージェントが実行されるGCPのプロジェクトであるべきです。エージェントが開始されると、`google.auth.default()` を使ってデフォルトのGCPの認証情報をロードします。デフォルトの GCP 資格情報の設定方法については、[google-auth documentation](https://google-auth.readthedocs.io/en/latest/reference/google.auth.html#google.auth.default) を参照してください。

```yaml
environment:
  type: gcp
  region: <gcp-region>
  project: <gcp-project-id>
```

</TabItem>

</Tabs>

### レジストリ

`registry` キーは、エージェントがコンテナイメージを格納するために使用するコンテナレジストリを設定するために使用されます。エージェントがコンテナレジストリへのアクセスを必要としない場合、このキーは省略されるべきです。

<Tabs
  defaultValue="ecr"
  values={[
    {label: 'AWS ECR', value: 'ecr'},
    {label: 'GCP Artifact Registry', value: 'gcr'},
  ]}>

<TabItem value="ecr">

`type: ecr` を設定して AWS Elastic Container Registry を使用します。`ecr` を使用するためには、AWS 環境を設定する必要があります。`repository` キーは必須で、エージェントがコンテナイメージを格納するために使用する ECR リポジトリの名前であるべきです。エージェントは、`environment` キーで設定されたリージョンを使用して、どのレジストリを使用するかを決定します。

```yaml
registry:
  # aws 環境設定が必要です。
  type: ecr
  # 環境で設定されたリージョンのECRリポジトリ名。
  repository: my-ecr-repo.
```
</TabItem>

<TabItem value="gcr">

`type: gcr`を設定して、GCPアーティファクトレジストリを使用します。 `gcr`を使用するには、GCP環境を設定する必要があります。`repository`と`image-name`のキーが必要です。`repository`キーは、エージェントがコンテナイメージを格納するために使用するArtifact Registryリポジトリの名前である必要があります。エージェントは、`environment`キーで設定された領域とプロジェクトを使用して、どのレジストリを使用するかを決定します。`image-name`キーは、エージェントがコンテナイメージを格納するリポジトリ内のイメージの名前である必要があります。

```yaml
registry:
  # GCP環境設定が必要です。
  type: gcr
  # 環境で設定したプロジェクト/リージョン内のアーティファクトリポジトリの名前
  repository: my-artifact-repo
  # エージェントがイメージを格納するリポジトリ内のイメージ名（タグではありません！）。
  image-name: my-image-name
```

</TabItem>

</Tabs>


### ビルダー

`builder`キーは、エージェントがコンテナイメージを構築するために使用するコンテナビルダーを設定するために使用されます。`builder`キーは必須ではなく、省略された場合、エージェントはDockerを使用してローカルでイメージを構築します。

<Tabs
  defaultValue="docker"
  values={[
    {label: 'Docker', value: 'docker'},
    {label: 'Kaniko', value: 'kaniko'},
    {label: 'Noop', value: 'noop'}
  ]}>
<TabItem value="docker">

`type: docker`を設定して、Dockerを使ってローカルでイメージをビルドします。`docker`ビルダーはデフォルトで選択されており、追加の設定は必要ありません。

```yaml
builder:
  type: docker
```

</TabItem>

<TabItem value="kaniko">

`type: kaniko`を設定し、GCPまたはAWS環境を設定して、KubernetesでKanikoを使ってコンテナイメージをビルドします。`kaniko`ビルダーは`build-context-store`キーを設定する必要があります。`build-context-store`キーは、設定された環境に応じて、エージェントがビルドコンテキストを保存するために使用するS3またはGCSのプレフィックスである必要があります。`build-job-name`を使用して、エージェントがイメージをビルドするために使用するk8sジョブの名前プレフィックスを指定できます。

```yaml
builder:
  type: kaniko
  build-context-store: s3://my-bucket/build-contexts/  # ビルドコンテキストストレージ用のs3またはgcsプレフィックス
  build-job-name: wandb-image-build  # すべてのビルドに対するk8sジョブ名プレフィックス
```

Kanikoビルドがk8sで実行されている場合、Kanikoビルドがクラウドでのブロブストレージとコンテナストレージにアクセスするために、GKEを使用している場合は[ワークロードアイデンティティ](https://cloud.google.com/kubernetes-engine/docs/how-to/workload-identity)、EKSを使用している場合は[IAMロールとサービスアカウント](https://docs.aws.amazon.com/eks/latest/userguide/iam-roles-for-service-accounts.html)を使用することが強くお勧めされています。

独自のk8sクラスターを実行している場合は、クラウド環境の資格情報が含まれたk8sシークレットを作成する必要があります。GCPにアクセス権を付与するには、このシークレットに[サービスアカウントのjson](https://cloud.google.com/iam/docs/keys-create-delete#creating)が含まれる必要があります。AWSへのアクセス権を付与するには、このシークレットに[AWS認証情報ファイル](https://docs.aws.amazon.com/sdk-for-php/v3/developer-guide/guide_credentials_profiles.html)が含まれる必要があります。Kanikoビルダーでこのシークレットを使用するには、ビルダーの設定で以下のキーを設定します。

```yaml
builder:
  type: kaniko
  build-context-store: <my-build-context-store>
  secret-name: <k8s-secret-name>
  secret-key: <secret-file-name>
```
</TabItem>

<TabItem value="noop">

`noop`ビルダーは、エージェントを事前にビルドされたコンテナイメージの実行に制限したい場合に便利です。`type: noop`を設定するだけで、エージェントがイメージをビルドするのを防ぐことができます。

```yaml
builder:
  type: noop
```

</TabItem>

</Tabs>

## エージェントを表示

特定の起動キューのページに移動し、**Agents**タブに移動して、キューに割り当てられたアクティブおよび非アクティブなエージェントを表示します。このタブ内では、以下を表示できます:

- **Agent ID**： ユニークなエージェント識別子

- **Status:** エージェントの状態。エージェントは**Killed**または**Polling**の状態を持つことができます。

- **Start date:** エージェントがアクティブになった日付。

- **Host:** エージェントがポーリングしているマシン。

![](/images/launch/queues_all_agents.png)