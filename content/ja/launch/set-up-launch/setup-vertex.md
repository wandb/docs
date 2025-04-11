---
title: 'チュートリアル: Vertex AI で W&B Launch を設定する'
menu:
  launch:
    identifier: ja-launch-set-up-launch-setup-vertex
    parent: set-up-launch
url: guides/launch/setup-vertex
---

W&B Launch を使用して、Vertex AI トレーニングジョブとしてジョブを実行するために送信することができます。Vertex AI トレーニングジョブでは、Vertex AI プラットフォーム上の提供されたアルゴリズムまたはカスタムアルゴリズムを使用して機械学習モデルをトレーニングすることができます。ローンチジョブが開始されると、Vertex AI が基盤となるインフラストラクチャー、スケーリング、およびオーケストレーションを管理します。

W&B Launch は、`google-cloud-aiplatform` SDK の `CustomJob` クラスを介して Vertex AI と連携します。`CustomJob` のパラメータは、ローンチキュー設定で制御できます。Vertex AI は、GCP 外のプライベートレジストリからイメージをプルするように設定することはできません。つまり、Vertex AI と W&B Launch を使用したい場合、コンテナイメージを GCP またはパブリックレジストリに保存する必要があります。コンテナイメージを Vertex ジョブでアクセス可能にするための詳細は、Vertex AI ドキュメントを参照してください。

## 前提条件

1. **Vertex AI API が有効になっている GCP プロジェクトを作成またはアクセスしてください。** API を有効にする方法については、[GCP API コンソールドキュメント](https://support.google.com/googleapi/answer/6158841?hl=ja) を参照してください。
2. **Vertex で実行したいイメージを保存するための GCP Artifact Registry リポジトリを作成してください。** 詳細については、[GCP Artifact Registry ドキュメント](https://cloud.google.com/artifact-registry/docs/overview) を参照してください。
3. **Vertex AI がメタデータを保存するステージング GCS バケットを作成してください。** このバケットは、Vertex AI ワークロードと同じリージョンにある必要があります。ステージングおよびビルドコンテキストには同じバケットを使用できます。
4. **Vertex AI ジョブを立ち上げるために必要な権限を持つサービスアカウントを作成してください。** サービスアカウントに権限を割り当てる方法については、[GCP IAM ドキュメント](https://cloud.google.com/iam/docs/creating-managing-service-accounts) を参照してください。
5. **サービスアカウントに Vertex ジョブを管理する権限を付与してください。**

| パーミッション                     | リソーススコープ        | 説明                                                                              |
| ---------------------------------- | --------------------- | --------------------------------------------------------------------------------- |
| `aiplatform.customJobs.create`     | 指定された GCP プロジェクト | プロジェクト内で新しい機械学習ジョブを作成することができます。                                       |
| `aiplatform.customJobs.list`       | 指定された GCP プロジェクト | プロジェクト内の機械学習ジョブを列挙することができます。                                           |
| `aiplatform.customJobs.get`        | 指定された GCP プロジェクト | プロジェクト内の特定の機械学習ジョブの情報を取得することができます。                                         |

{{% alert %}}
Vertex AI ワークロードに標準以外のサービスアカウントのアイデンティティを引き継がせたい場合は、Vertex AI ドキュメントを参照してサービスアカウントの作成と必要な権限についての手順を確認してください。ローンチキュー設定の `spec.service_account` フィールドを使用して、W&B の run 用のカスタムサービスアカウントを選択できます。
{{% /alert %}}

## Vertex AI 用キューを設定

Vertex AI リソース用のキュー設定は、Vertex AI Python SDK の `CustomJob` コンストラクタと `run` メソッドへの入力を指定します。リソース設定は `spec` および `run` キーに格納されます。

- `spec` キーには、Vertex AI Python SDK の [`CustomJob` コンストラクタ](https://cloud.google.com/vertex-ai/docs/pipelines/customjob-component) の名前付き引数の値が含まれています。
- `run` キーには、Vertex AI Python SDK の `CustomJob` クラスの `run` メソッドの名前付き引数の値が含まれています。

実行環境のカスタマイズは主に `spec.worker_pool_specs` リストで行われます。ワーカープール仕様は、ジョブを実行する作業者グループを定義します。デフォルトの設定のワーカー仕様は、アクセラレータのない `n1-standard-4` マシンを 1 台要求します。必要に応じてマシンタイプ、アクセラレータタイプ、数を変更することができます。

利用可能なマシンタイプやアクセラレータタイプについての詳細は、[Vertex AI ドキュメント](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/MachineSpec) を参照してください。

## キューを作成

Vertex AI を計算リソースとして使用するキューを W&B アプリで作成する:

1. [Launch ページ](https://wandb.ai/launch) に移動します。
2. **Create Queue** ボタンをクリックします。
3. キューを作成したい **Entity** を選択します。
4. **Name** フィールドにキューの名前を入力します。
5. **Resource** として **GCP Vertex** を選択します。
6. **Configuration** フィールドに、前のセクションで定義した Vertex AI `CustomJob` についての情報を入力します。デフォルトで、W&B は次のような YAML および JSON のリクエストボディを自動入力します：

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

7. キューを設定したら、**Create Queue** ボタンをクリックします。

最低限指定する必要があるのは以下です：

- `spec.worker_pool_specs` : 非空のワーカープール仕様リスト。
- `spec.staging_bucket` : Vertex AI アセットとメタデータのステージングに使用する GCS バケット。

{{% alert color="secondary" %}}
一部の Vertex AI ドキュメントには、すべてのキーがキャメルケースで表示されるワーカープール仕様が示されています。例： `workerPoolSpecs`。 Vertex AI Python SDK では、これらのキーにスネークケースを使用します。例：`worker_pool_specs`。

ローンチキュー設定のすべてのキーはスネークケースを使用する必要があります。
{{% /alert %}}

## ローンチエージェントを設定

ローンチエージェントは、デフォルトでは `~/.config/wandb/launch-config.yaml` にある設定ファイルを介して設定可能です。

max_jobs: <n-concurrent-jobs>
queues:
  - <queue-name>

Vertex AI で実行されるイメージをローンチエージェントに構築してもらいたい場合は、[Advanced agent set up]({{< relref path="./setup-agent-advanced.md" lang="ja" >}}) を参照してください。

## エージェント権限を設定

このサービスアカウントとして認証する方法は複数あります。Workload Identity、ダウンロードされたサービスアカウント JSON、環境変数、Google Cloud Platform コマンドラインツール、またはこれらのメソッドの組み合わせを通じて実現できます。