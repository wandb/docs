---
title: 'チュートリアル: Vertex AI で W&B Launch をセットアップする'
menu:
  launch:
    identifier: ja-launch-set-up-launch-setup-vertex
    parent: set-up-launch
url: guides/launch/setup-vertex
---

W&B Launch を使うと、Vertex AI トレーニングジョブとしてジョブを実行用に送信できます。Vertex AI トレーニングジョブでは、提供されたアルゴリズムまたはカスタムアルゴリズムを利用して、Vertex AI プラットフォーム上で機械学習モデルのトレーニングが可能です。Launch ジョブが開始されると、Vertex AI が基盤となるインフラストラクチャー、スケーリング、オーケストレーションを管理します。

W&B Launch は、`google-cloud-aiplatform` SDK の `CustomJob` クラスを通じて Vertex AI と連携します。`CustomJob` のパラメータは、ローンチのキュー設定で制御できます。Vertex AI は GCP 以外のプライベートレジストリからイメージを取得するよう設定できません。つまり、Vertex AI で W&B Launch を利用したい場合は、コンテナイメージを GCP またはパブリックレジストリ上に保存する必要があります。コンテナイメージを Vertex ジョブで利用可能にする方法については Vertex AI ドキュメントをご覧ください。

## 前提条件

1. **Vertex AI API が有効になっている GCP プロジェクトを作成またはアクセスします。** API 有効化については [GCP API Console のドキュメント](https://support.google.com/googleapi/answer/6158841?hl=ja) を参照してください。
2. **GCP Artifact Registry リポジトリを作成します**。Vertex で実行するイメージを保存するためです。詳細は [GCP Artifact Registry ドキュメント](https://cloud.google.com/artifact-registry/docs/overview) をご覧ください。
3. **Vertex AI 用のステージング用 GCS バケットを作成します**。このバケットは、Vertex AI のワークロードと同じリージョンにある必要があります。同じバケットをステージングやビルドコンテキストにも利用できます。
4. **必要な権限を持つサービスアカウントの作成**。Vertex AI ジョブの作成・実行に必要な権限を持つサービスアカウントを用意します。サービスアカウントへの権限割り当て方法は [GCP IAM ドキュメント](https://cloud.google.com/iam/docs/creating-managing-service-accounts) を参照してください。
5. **サービスアカウントに Vertex ジョブ管理権限を付与**

| 権限                             | リソーススコープ           | 説明                                                                                   |
| -------------------------------- | -------------------------- | -------------------------------------------------------------------------------------- |
| `aiplatform.customJobs.create`   | 指定した GCP プロジェクト   | プロジェクト内で新しい機械学習ジョブの作成を許可します                                 |
| `aiplatform.customJobs.list`     | 指定した GCP プロジェクト   | プロジェクト内の機械学習ジョブの一覧取得を許可します                                   |
| `aiplatform.customJobs.get`      | 指定した GCP プロジェクト   | プロジェクト内の特定の機械学習ジョブ情報の取得を許可します                            |

{{% alert %}}
Vertex AI ワークロードで標準以外のサービスアカウントを使いたい場合は、Vertex AI ドキュメントのサービスアカウント作成および権限設定ガイドを参照してください。ローンチのキュー設定の `spec.service_account` フィールドでカスタムサービスアカウントを W&B runs に指定できます。
{{% /alert %}}

## Vertex AI 用キューの設定

Vertex AI リソースのキュー設定では、Vertex AI Python SDK の `CustomJob` コンストラクタおよび `CustomJob` の `run` メソッドに渡す入力値を指定します。リソース設定は `spec` と `run` キーの下に格納されます：

- `spec` キーには、Vertex AI Python SDK の [`CustomJob` コンストラクタ](https://cloud.google.com/vertex-ai/docs/pipelines/customjob-component) に渡される引数が入ります。
- `run` キーには、Vertex AI Python SDK の `CustomJob` クラスの `run` メソッドに渡す引数が入ります。

実行環境のカスタマイズは主に `spec.worker_pool_specs` リストで行います。ワーカープール仕様で、ジョブを実行するワーカー群を定義します。デフォルト設定の worker spec では、アクセラレーターなしの `n1-standard-4` マシン 1 台を要求しています。必要に応じてマシンタイプ、アクセラレータータイプや数を変更できます。

利用可能なマシンタイプやアクセラレータータイプについては、[Vertex AI ドキュメント](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/MachineSpec) をご覧ください。

## キューの作成

W&B App で Vertex AI を計算リソースとして使うキューを作成します：

1. [Launch ページ](https://wandb.ai/launch) に移動します。
2. **Create Queue** ボタンをクリックします。
3. 任意の **Entity** を選択します。
4. **Name** フィールドにキューの名前を入力します。
5. **Resource** として **GCP Vertex** を選択します。
6. **Configuration** フィールドに、先ほど定義した Vertex AI `CustomJob` の情報を記入します。デフォルトでは、W&B によって下記のような YAML および JSON のリクエストボディが入力されています：

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

7. キューの設定後、**Create Queue** ボタンをクリックして作成します。

最低限、以下の指定が必要です：

- `spec.worker_pool_specs` ：ワーカープール仕様のリスト（非空）
- `spec.staging_bucket` ：Vertex AI 用アセットおよびメタデータ用ステージング GCS バケット

{{% alert color="secondary" %}}
Vertex AI の一部ドキュメントでは、ワーカープール仕様のキーがキャメルケース（例：`workerPoolSpecs`）で記載されています。Vertex AI Python SDK ではすべてスネークケース（例：`worker_pool_specs`）を使用します。

ローンチのキュー設定のすべてのキーはスネークケースを用いてください。
{{% /alert %}}

## ローンチエージェントの設定

ローンチエージェントは、デフォルトで `~/.config/wandb/launch-config.yaml` にある設定ファイルで調整できます。

```yaml
max_jobs: <n-concurrent-jobs>
queues:
  - <queue-name>
```

Vertex AI 上で実行するイメージのビルドをローンチエージェントに任せたい場合は、[高度なエージェントセットアップ]({{< relref path="./setup-agent-advanced.md" lang="ja" >}}) を参照してください。

## エージェント権限のセットアップ

このサービスアカウントとして認証する方法はいくつかあります。Workload Identity、サービスアカウント JSON のダウンロード、環境変数、Google Cloud Platform コマンドラインツール、またはこれらの組み合わせが利用可能です。