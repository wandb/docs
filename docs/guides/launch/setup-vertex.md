---
displayed_sidebar: default
---


# Set up Vertex AI

W&B Launch を使用して、Vertex AI トレーニングジョブとして実行するジョブを提出できます。Vertex AI トレーニングジョブを使用すると、Vertex AI プラットフォーム上で提供されたアルゴリズムまたはカスタムアルゴリズムを使用して機械学習モデルをトレーニングできます。Launch ジョブが開始されると、Vertex AI が基盤となるインフラストラクチャー、スケーリング、およびオーケストレーションを管理します。

W&B Launch は、`google-cloud-aiplatform` SDK の `CustomJob` クラスを通じて Vertex AI と連携します。`CustomJob` のパラメータは、launch queue 設定で制御できます。Vertex AI は GCP の外部のプライベートレジストリからイメージをプルするように設定できません。これにより、W&B Launch と Vertex AI を使用する場合は、コンテナイメージを GCP またはパブリックレジストリに保存する必要があります。Vertex ジョブがコンテナイメージにアクセスできるようにする方法については、Vertex AI ドキュメントを参照してください。

## Prerequisites

1. **Vertex AI API が有効になっている GCP プロジェクトを作成またはアクセスします。** API を有効にする方法については、[GCP API コンソールドキュメント](https://support.google.com/googleapi/answer/6158841?hl=en)を参照してください。
2. **Vertex で実行したいイメージを保存するための GCP Artifact Registry リポジトリを作成します。** 詳細については、[GCP Artifact Registry ドキュメント](https://cloud.google.com/artifact-registry/docs/overview)を参照してください。
3. **Vertex AI がメタデータを保存するためのステージング GCS バケットを作成します。** このバケットは、Vertex AI ワークロードと同じリージョンにある必要があります。ステージングとビルドコンテキストの両方に同じバケットを使用できます。
4. **Vertex AI ジョブを起動するための必要な権限を持つサービスアカウントを作成します。** サービスアカウントに権限を割り当てる方法については、[GCP IAM ドキュメント](https://cloud.google.com/iam/docs/creating-managing-service-accounts)を参照してください。
5. **サービスアカウントに Vertex ジョブを管理する権限を付与します**

| Permission                     | Resource Scope        | Description                                                                              |
| ------------------------------ | --------------------- | ---------------------------------------------------------------------------------------- |
| `aiplatform.customJobs.create` | Specified GCP Project | プロジェクト内で新しい機械学習ジョブを作成できるようにします。                             |
| `aiplatform.customJobs.list`   | Specified GCP Project | プロジェクト内の機械学習ジョブを一覧表示できるようにします。                               |
| `aiplatform.customJobs.get`    | Specified GCP Project | プロジェクト内の特定の機械学習ジョブに関する情報を取得できるようにします。                    |

:::info
Vertex AI ワークロードに非標準のサービスアカウントのアイデンティティを持たせたい場合は、Vertex AI ドキュメントを参照して、サービスアカウントの作成および必要な権限についての手順を確認してください。Launch queue 設定の `spec.service_account` フィールドを使用して、カスタムサービスアカウントを W&B Runs に選択できます。
:::

## Configure a queue for Vertex AI

Vertex AI リソースの queue 設定は、Vertex AI Python SDK 内の `CustomJob` コンストラクタと `CustomJob` の `run` メソッドへの入力を指定します。リソース設定は、`spec` および `run` キーの下に格納されます：

- `spec` キーには、Vertex AI Python SDK の [`CustomJob` コンストラクタ](https://cloud.google.com/ai-platform/training/docs/reference/rest/v1beta1/projects.locations.customJobs#CustomJob.FIELDS.spec) の名前付き引数の値が含まれます。
- `run` キーには、Vertex AI Python SDK 内の `CustomJob` クラスの `run` メソッドの名前付き引数の値が含まれます。

実行環境のカスタマイズは主に `spec.worker_pool_specs` リストで行われます。ワーカープールの spec は、ジョブを実行するワーカーのグループを定義します。デフォルトの設定では、加速器なしの `n1-standard-4` マシンを単一で要求します。必要に応じてマシンの種類、加速器の種類と数を変更できます。

利用可能なマシンタイプと加速器タイプの詳細については、[Vertex AI ドキュメント](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/MachineSpec)を参照してください。

## Create a queue

W&B App で Vertex AI をコンピュートリソースとする queue を作成します：

1. [Launch ページ](https://wandb.ai/launch)に移動します。
2. **Create Queue** ボタンをクリックします。
3. queue を作成したい **Entity** を選択します。
4. **Name** フィールドに queue の名前を入力します。
5. **Resource** として **GCP Vertex** を選択します。
6. **Configuration** フィールドに、前のセクションで定義した Vertex AI `CustomJob` の情報を入力します。デフォルトでは、W&B は次のような YAML および JSON リクエストボディを入力します：

```yaml
spec:
  worker_pool_specs:
    - machine_spec:
        machine_type: n1-standard-4
        accelerator_type: ACCELERATOR_TYPE_UNSPECIFIED
        accelerator_count: 0
      replica_count: 1
      container_spec:
        image_uri: ${image_uri}
  staging_bucket: <REQUIRED>
run:
  restart_job_on_worker_restart: false
```

7. queue を設定したら、**Create Queue** ボタンをクリックします。

最低限次の内容を指定する必要があります：

- `spec.worker_pool_specs` : ワーカープールの仕様の非空リスト。
- `spec.staging_bucket` : Vertex AI アセットおよびメタデータのステージングに使用される GCS バケット。

:::caution
Vertex AI ドキュメントの一部では、キーがキャメルケースになっているワーカープールの仕様が示されています (例：`workerPoolSpecs`)。Vertex AI Python SDK では、これらのキーにはスネークケースを使用します (例：`worker_pool_specs`)。

launch queue 設定のすべてのキーはスネークケースを使用する必要があります。
:::

## Configure a launch agent

launch agent は、デフォルトで `~/.config/wandb/launch-config.yaml` に配置される設定ファイルを通じて構成可能です。

```yaml
max_jobs: <n-concurrent-jobs>
queues:
  - <queue-name>
```

launch agent に Vertex AI で実行されるイメージをビルドさせたい場合は、[高度なエージェントセットアップ](./setup-agent-advanced.md)を参照してください。

## Set up agent permissions

このサービスアカウントとして認証する方法はいくつかあります。これには、Workload Identity、ダウンロードされたサービスアカウント JSON、環境変数、Google Cloud Platform コマンドラインツール、またはこれらの方法の組み合わせが使用できます。