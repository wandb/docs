---
title: 'チュートリアル: Vertex AI で W&B Launch をセットアップする'
menu:
  launch:
    identifier: setup-vertex
    parent: set-up-launch
url: guides/launch/setup-vertex
---

W&B Launch を使って、Vertex AI のトレーニングジョブとしてジョブを実行することができます。Vertex AI のトレーニングジョブでは、Vertex AI プラットフォーム上で、用意されたアルゴリズムまたはカスタムアルゴリズムを使用して機械学習モデルのトレーニングが可能です。Launch ジョブを開始すると、Vertex AI が基盤インフラストラクチャー、スケーリング、およびオーケストレーションを自動的に管理します。

W&B Launch は、`google-cloud-aiplatform` SDK の `CustomJob` クラスを介して Vertex AI と連携します。`CustomJob` のパラメータは、Launch キューの設定で管理できます。Vertex AI は、GCP 以外のプライベートレジストリからイメージを取得するように設定できません。そのため、Vertex AI と W&B Launch を組み合わせて利用する場合は、コンテナイメージを GCP またはパブリックレジストリに保存する必要があります。コンテナイメージを Vertex ジョブで利用できるようにする方法は Vertex AI のドキュメントを参照してください。

## 前提条件

1. **Vertex AI API が有効化された GCP プロジェクトを作成またはアクセスしてください。** API 有効化については [GCP API Console ドキュメント](https://support.google.com/googleapi/answer/6158841?hl=ja) をご覧ください。
2. **GCP Artifact Registry リポジトリを作成**して、Vertex で実行したいイメージを保存します。詳細は [GCP Artifact Registry のドキュメント](https://cloud.google.com/artifact-registry/docs/overview) をご確認ください。
3. **Vertex AI がメタデータを保存するためのステージング GCS バケットを作成**します。このバケットは、Vertex AI ワークロードと同じリージョンに配置する必要があります。1 つのバケットをステージングおよびビルドコンテキストの両方で利用できます。
4. **Vertex AI ジョブを起動するために必要な権限を持つサービスアカウントを作成**します。権限の付与については [GCP IAM ドキュメント](https://cloud.google.com/iam/docs/creating-managing-service-accounts) を参照してください。
5. **サービスアカウントに Vertex ジョブ管理権限を付与**

| 権限                          | リソーススコープ      | 説明                                              |
| ---------------------------- | ----------------- | ------------------------------------------------- |
| `aiplatform.customJobs.create` | 指定した GCP プロジェクト | プロジェクト内で新しい機械学習ジョブの作成を許可       |
| `aiplatform.customJobs.list`   | 指定した GCP プロジェクト | プロジェクト内の機械学習ジョブの一覧取得を許可         |
| `aiplatform.customJobs.get`    | 指定した GCP プロジェクト | プロジェクト内の特定の機械学習ジョブ情報の取得を許可   |

{{% alert %}}
Vertex AI のワークロードが標準以外のサービスアカウントの ID で動作するようにしたい場合は、Vertex AI ドキュメントでサービスアカウントの作成と必要な権限付与方法を確認してください。Launch キュー設定の `spec.service_account` フィールドで、W&B run にカスタムサービスアカウントを指定できます。
{{% /alert %}}

## Vertex AI のキュー設定

Vertex AI リソース向けのキュー設定では、Vertex AI Python SDK の `CustomJob` コンストラクタ、ならびにその `run` メソッドに入力される内容を指定します。リソース設定は `spec` および `run` キー以下に保存されます。

- `spec` キーは、Vertex AI Python SDK の [`CustomJob` コンストラクタ](https://cloud.google.com/vertex-ai/docs/pipelines/customjob-component) で使用される名前付き引数の値を保持します。
- `run` キーは、Vertex AI Python SDK の `CustomJob` クラスの `run` メソッドの名前付き引数値を保持します。

実行環境のカスタマイズは主に `spec.worker_pool_specs` リストで行います。ワーカープールスペックは、ジョブを実行するワーカーグループを定義します。デフォルト設定では、アクセラレータなしの `n1-standard-4` マシンを 1 台要求します。用途にあわせてマシンタイプやアクセラレータータイプ・数を調整可能です。

利用可能なマシンタイプやアクセラレータータイプについては [Vertex AI ドキュメント](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/MachineSpec) をご参照ください。

## キューの作成

Vertex AI を計算リソースとして利用するキューを、W&B App 上で作成します。

1. [Launch ページ](https://wandb.ai/launch) にアクセスします。
2. **Create Queue** ボタンをクリックします。
3. キューを作成したい **Entity** を選択します。
4. **Name** フィールドにキューの名前を入力します。
5. **Resource** として **GCP Vertex** を選択します。
6. **Configuration** フィールドで、前セクションで定義した Vertex AI `CustomJob` の情報を入力します。デフォルトで、W&B により以下のような YAML および JSON のリクエストボディが設定されます。

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

7. 設定が完了したら **Create Queue** ボタンをクリックしてください。

最低限、以下は必須です。

- `spec.worker_pool_specs` : 空でないワーカープールスペックのリスト
- `spec.staging_bucket` : Vertex AI のアセットやメタデータのステージングに使う GCS バケット

{{% alert color="secondary" %}}
Vertex AI の一部ドキュメントでは、すべてのキーをキャメルケース（例: `workerPoolSpecs`）で記載していますが、Vertex AI Python SDK ではスネークケース（例: `worker_pool_specs`）を採用しています。

Launch キュー設定内のキーはすべてスネークケースで記載してください。
{{% /alert %}}

## ローンンチエージェントの設定

ローンンチエージェントは、デフォルトで `~/.config/wandb/launch-config.yaml` にある設定ファイルで設定できます。

```yaml
max_jobs: <n-concurrent-jobs>
queues:
  - <queue-name>
```

ローンンチエージェントに Vertex AI で実行するイメージをビルドさせたい場合は、[高度なエージェントセットアップ]({{< relref "./setup-agent-advanced.md" >}}) をご参照ください。

## エージェントの権限設定

このサービスアカウントとして認証する方法はいくつかあります。Workload Identity、ダウンロードしたサービスアカウント JSON、環境変数、Google Cloud Platform のコマンドラインツール、またはこれらの組み合わせで実現可能です。