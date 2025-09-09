---
title: 'チュートリアル: Vertex AI で W&B Launch をセットアップする'
menu:
  launch:
    identifier: ja-launch-set-up-launch-setup-vertex
    parent: set-up-launch
url: guides/launch/setup-vertex
---

W&B Launch を使うと、Vertex AI のトレーニング ジョブとして実行するジョブを送信できます。Vertex AI のトレーニング ジョブでは、Vertex AI プラットフォーム上で提供済みまたはカスタムのアルゴリズムを使って機械学習 モデルをトレーニングできます。Launch のジョブを開始すると、Vertex AI が基盤となるインフラストラクチャー、スケーリング、オーケストレーションを管理します。

W&B Launch は、`google-cloud-aiplatform` SDK の `CustomJob` クラスを通じて Vertex AI と連携します。`CustomJob` のパラメータは Launch のキュー 設定で制御できます。Vertex AI は、GCP の外にあるプライベート レジストリからイメージを取得するように設定できません。つまり、W&B Launch と Vertex AI を併用する場合は、コンテナ イメージを GCP 内、またはパブリック レジストリに保存する必要があります。コンテナ イメージを Vertex AI のジョブから参照可能にする方法については Vertex AI のドキュメントを参照してください。


## Prerequisites

1. Vertex AI API が有効な GCP プロジェクトを作成するか、アクセスします。API の有効化については [GCP API Console のドキュメント](https://support.google.com/googleapi/answer/6158841?hl=en) を参照してください。
2. Vertex で実行したいイメージを保存するための GCP Artifact Registry リポジトリを作成します。詳しくは [GCP Artifact Registry のドキュメント](https://cloud.google.com/artifact-registry/docs/overview) を参照してください。
3. Vertex AI がメタデータを保存するためのステージング用 GCS バケットを作成します。このバケットは、ステージング バケットとして使用するために、Vertex AI のワークロードと同じリージョンに存在する必要があります。同じバケットをステージングとビルド コンテキストの両方に使用できます。
4. Vertex AI のジョブを起動するために必要な権限を持つサービス アカウントを作成します。サービス アカウントへの権限付与については [GCP IAM のドキュメント](https://cloud.google.com/iam/docs/creating-managing-service-accounts) を参照してください。
5. サービス アカウントに Vertex のジョブを管理する権限を付与します。

| 権限                           | リソース スコープ          | 説明                                                                                     |
| ------------------------------ | -------------------------- | ---------------------------------------------------------------------------------------- |
| `aiplatform.customJobs.create` | 指定した GCP プロジェクト  | プロジェクト内で新しい機械学習 ジョブを作成できます。                                    |
| `aiplatform.customJobs.list`   | 指定した GCP プロジェクト  | プロジェクト内の機械学習 ジョブを一覧表示できます。                                      |
| `aiplatform.customJobs.get`    | 指定した GCP プロジェクト  | プロジェクト内の特定の機械学習 ジョブに関する情報を取得できます。                        |

{{% alert %}}
標準以外のサービス アカウントのアイデンティティで Vertex AI のワークロードを実行したい場合は、サービス アカウントの作成方法と必要な権限について Vertex AI のドキュメントを参照してください。Launch のキュー 設定の `spec.service_account` フィールドを使うと、W&B の Runs に対してカスタムのサービス アカウントを選択できます。
{{% /alert %}}

## Vertex AI 用のキューを設定する

Vertex AI リソース向けのキュー 設定は、Vertex AI Python SDK における `CustomJob` コンストラクタと `CustomJob` の `run` メソッドへの入力を指定します。リソースの設定は `spec` と `run` キーの下に格納されます。

- `spec` キーには、Vertex AI Python SDK の [`CustomJob` コンストラクタ](https://cloud.google.com/vertex-ai/docs/pipelines/customjob-component) の名前付き引数の値が入ります。
- `run` キーには、Vertex AI Python SDK の `CustomJob` クラスの `run` メソッドの名前付き引数の値が入ります。

実行 環境のカスタマイズは主に `spec.worker_pool_specs` リストで行います。ワーカー プールの spec は、ジョブを実行するワーカーのグループを定義します。デフォルトの設定では、アクセラレータなしの `n1-standard-4` マシン 1 台が要求されます。必要に応じて、マシン タイプ、アクセラレータの種類と数を変更できます。

利用可能なマシン タイプやアクセラレータ タイプの詳細は [Vertex AI のドキュメント](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/MachineSpec) を参照してください。

## キューを作成する

Vertex AI をコンピュート リソースとして使用するキューを W&B App で作成します。

1. [Launch ページ](https://wandb.ai/launch) に移動します。
2. **Create Queue** ボタンをクリックします。
3. 作成先の **Entity** を選択します。
4. **Name** フィールドにキュー名を入力します。
5. **Resource** として **GCP Vertex** を選択します。
6. **Configuration** フィールドに、前のセクションで定義した Vertex AI の `CustomJob` に関する情報を入力します。デフォルトでは、W&B が次のような YAML と JSON のリクエスト ボディを自動入力します。

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

7. キューを設定したら、**Create Queue** ボタンをクリックします。

最低限、次を指定する必要があります:

- `spec.worker_pool_specs` : 空でないワーカー プールの仕様のリスト。
- `spec.staging_bucket` : Vertex AI のアセットとメタデータのステージングに使用する GCS バケット。

{{% alert color="secondary" %}}
一部の Vertex AI のドキュメントでは、すべてのキーをキャメルケースで記載したワーカー プールの仕様（例: `workerPoolSpecs`）が示されています。Vertex AI Python SDK では、これらのキーはスネークケース（例: `worker_pool_specs`）を使用します。

Launch のキュー 設定内のすべてのキーはスネークケースを使用してください。
{{% /alert %}}

## Launch エージェントを設定する

Launch エージェントは、デフォルトでは `~/.config/wandb/launch-config.yaml` にある設定ファイルで構成できます。

```yaml
max_jobs: <n-concurrent-jobs>
queues:
  - <queue-name>
```

Vertex AI で実行されるイメージをエージェントにビルドさせたい場合は、[高度なエージェント設定]({{< relref path="./setup-agent-advanced.md" lang="ja" >}}) を参照してください。

## エージェントの権限を設定する

このサービス アカウントとして認証する方法はいくつかあります。Workload Identity、ダウンロードしたサービス アカウントの JSON、環境 変数、Google Cloud Platform のコマンドライン ツール、またはそれらの組み合わせで実現できます。