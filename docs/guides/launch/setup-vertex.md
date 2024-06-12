---
displayed_sidebar: default
---

# Vertex AI のセットアップ

W&B Launch を使用して、Vertex AI トレーニングジョブとしてジョブを実行するために送信できます。Vertex AI トレーニングジョブを使用すると、提供されたアルゴリズムやカスタムアルゴリズムを使用して Vertex AI プラットフォーム上で機械学習モデルをトレーニングできます。Launch ジョブが開始されると、Vertex AI が基盤となるインフラストラクチャー、スケーリング、およびオーケストレーションを管理します。

W&B Launch は `google-cloud-aiplatform` SDK の `CustomJob` クラスを通じて Vertex AI と連携します。`CustomJob` のパラメーターは、launch キュー設定で制御できます。Vertex AI は GCP 外のプライベートレジストリからイメージをプルするように設定できません。つまり、Vertex AI を W&B Launch と一緒に使用する場合、コンテナイメージを GCP またはパブリックレジストリに保存する必要があります。コンテナイメージを Vertex ジョブでアクセス可能にする方法については、Vertex AI のドキュメントを参照してください。

## 前提条件

1. **Vertex AI API が有効になっている GCP プロジェクトを作成またはアクセスします。** API を有効にする方法については、[GCP API コンソールドキュメント](https://support.google.com/googleapi/answer/6158841?hl=en) を参照してください。
2. **Vertex で実行するイメージを保存するための GCP Artifact Registry リポジトリを作成します。** 詳細については、[GCP Artifact Registry ドキュメント](https://cloud.google.com/artifact-registry/docs/overview) を参照してください。
3. **Vertex AI がメタデータを保存するためのステージング GCS バケットを作成します。** このバケットは、Vertex AI ワークロードと同じリージョンにある必要があります。同じバケットをステージングおよびビルドコンテキストに使用できます。
4. **Vertex AI ジョブを起動するための必要な権限を持つサービスアカウントを作成します。** サービスアカウントに権限を割り当てる方法については、[GCP IAM ドキュメント](https://cloud.google.com/iam/docs/creating-managing-service-accounts) を参照してください。
5. **サービスアカウントに Vertex ジョブを管理する権限を付与します。**

| 権限                           | リソーススコープ       | 説明                                                                                     |
| ------------------------------ | --------------------- | ---------------------------------------------------------------------------------------- |
| `aiplatform.customJobs.create` | 指定された GCP プロジェクト | プロジェクト内で新しい機械学習ジョブを作成することを許可します。                         |
| `aiplatform.customJobs.list`   | 指定された GCP プロジェクト | プロジェクト内の機械学習ジョブをリストすることを許可します。                              |
| `aiplatform.customJobs.get`    | 指定された GCP プロジェクト | プロジェクト内の特定の機械学習ジョブに関する情報を取得することを許可します。              |

:::info
Vertex AI ワークロードが非標準のサービスアカウントのアイデンティティを引き受ける場合は、サービスアカウントの作成と必要な権限について Vertex AI ドキュメントを参照してください。launch キュー設定の `spec.service_account` フィールドを使用して、W&B runs にカスタムサービスアカウントを選択できます。
:::

## Vertex AI 用のキューを設定する

Vertex AI リソースのキュー設定は、Vertex AI Python SDK の `CustomJob` コンストラクタと `CustomJob` の `run` メソッドへの入力を指定します。リソース設定は `spec` および `run` キーの下に保存されます：

- `spec` キーには、Vertex AI Python SDK の [`CustomJob` コンストラクタ](https://cloud.google.com/ai-platform/training/docs/reference/rest/v1beta1/projects.locations.customJobs#CustomJob.FIELDS.spec) の名前付き引数の値が含まれます。
- `run` キーには、Vertex AI Python SDK の `CustomJob` クラスの `run` メソッドの名前付き引数の値が含まれます。

実行環境のカスタマイズは主に `spec.worker_pool_specs` リストで行われます。ワーカープール仕様は、ジョブを実行するワーカーのグループを定義します。デフォルトの設定では、アクセラレータなしの `n1-standard-4` マシンを1台要求します。ニーズに合わせてマシンタイプ、アクセラレータタイプ、および数を変更できます。

利用可能なマシンタイプとアクセラレータタイプの詳細については、[Vertex AI ドキュメント](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/MachineSpec) を参照してください。

## キューを作成する

Vertex AI を計算リソースとして使用するキューを W&B アプリで作成します：

1. [Launch ページ](https://wandb.ai/launch) に移動します。
2. **Create Queue** ボタンをクリックします。
3. キューを作成したい **Entity** を選択します。
4. **Name** フィールドにキューの名前を入力します。
5. **Resource** として **GCP Vertex** を選択します。
6. **Configuration** フィールドに、前のセクションで定義した Vertex AI `CustomJob` に関する情報を入力します。デフォルトでは、W&B は次のような YAML および JSON リクエストボディを自動的に入力します：

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

7. キューを設定した後、**Create Queue** ボタンをクリックします。

最低限指定する必要があるのは：

- `spec.worker_pool_specs` : ワーカープール仕様の非空リスト。
- `spec.staging_bucket` : Vertex AI アセットとメタデータをステージングするために使用される GCS バケット。

:::caution
一部の Vertex AI ドキュメントでは、すべてのキーがキャメルケースで記載されたワーカープール仕様が表示されます。例えば、`workerPoolSpecs`。Vertex AI Python SDK では、これらのキーはスネークケースで使用されます。例えば、`worker_pool_specs`。

launch キュー設定のすべてのキーはスネークケースである必要があります。
:::

## launch エージェントを設定する

launch エージェントは、デフォルトで `~/.config/wandb/launch-config.yaml` にある設定ファイルを通じて設定可能です。

```yaml
max_jobs: <n-concurrent-jobs>
queues:
  - <queue-name>
```

launch エージェントが Vertex AI で実行されるイメージをビルドするように設定する場合は、[Advanced agent set up](./setup-agent-advanced.md) を参照してください。

## エージェントの権限を設定する

このサービスアカウントとして認証する方法はいくつかあります。Workload Identity、ダウンロードされたサービスアカウント JSON、環境変数、Google Cloud Platform コマンドラインツール、またはこれらの方法の組み合わせを通じて実現できます。