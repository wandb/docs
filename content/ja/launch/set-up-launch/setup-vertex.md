---
title: 'Tutorial: Set up W&B Launch on Vertex AI'
menu:
  launch:
    identifier: ja-launch-set-up-launch-setup-vertex
    parent: set-up-launch
url: guides/launch/setup-vertex
---

W&B Launch を使用して Vertex AI トレーニング ジョブとして実行するジョブを送信できます。Vertex AI トレーニング ジョブを使用すると、Vertex AI プラットフォーム上で提供されたアルゴリズムまたはカスタム アルゴリズムを使用して機械学習モデルをトレーニングできます。ローンンチジョブが開始されると、Vertex AI は基盤となるインフラストラクチャーの管理、スケーリング、およびオーケストレーションを行います。

W&B Launch は `google-cloud-aiplatform` SDK の `CustomJob` クラスを通じて Vertex AI と連携します。`CustomJob` のパラメータは、ローンンチキュー設定で制御できます。Vertex AI は GCP 以外のプライベート レジストリからイメージをプルするように設定できません。したがって、W&B Launch で Vertex AI を使用する場合は、コンテナイメージを GCP または公開レジストリに保存する必要があります。コンテナ イメージを Vertex ジョブでアクセス可能にする方法の詳細については、Vertex AI のドキュメントを参照してください。

## 必要条件

1. **Vertex AI API が有効になっている GCP プロジェクトを作成またはアクセスします。**API を有効にする詳細については、[GCP API コンソールドキュメント](https://support.google.com/googleapi/answer/6158841?hl=ja) を参照してください。
2. **実行したいイメージを保存するための GCP Artifact Registry リポジトリを作成します。** 詳細は、[GCP Artifact Registry ドキュメント](https://cloud.google.com/artifact-registry/docs/overview) を参照してください。
3. **Vertex AI のメタデータを保存するためのステージング GCS バケットを作成します。** このバケットは、あなたの Vertex AI ワークロードと同じ地域にある必要があります。同じバケットは、ステージングおよびビルドコンテキストのために使用できます。
4. **Vertex AI ジョブを起動するための必要な権限を持つサービス アカウントを作成します。** サービス アカウントに権限を割り当てる詳細については、[GCP IAM ドキュメント](https://cloud.google.com/iam/docs/creating-managing-service-accounts) を参照してください。
5. **サービス アカウントに Vertex ジョブを管理するための権限を付与します。**

| 権限                           | リソース範囲            | 説明                                                                                       |
| ------------------------------ | --------------------- | ---------------------------------------------------------------------------------------- |
| `aiplatform.customJobs.create` | 指定された GCP プロジェクト | プロジェクト内で新しい機械学習ジョブを作成することを許可します。                          |
| `aiplatform.customJobs.list`   | 指定された GCP プロジェクト | プロジェクト内の機械学習ジョブをリスト化することを許可します。                             |
| `aiplatform.customJobs.get`    | 指定された GCP プロジェクト | プロジェクト内の特定の機械学習ジョブに関する情報を取得することを許可します。              |

{{% alert %}}
Vertex AI ワークロードが非標準のサービス アカウントを使用する場合は、サービス アカウントの作成と必要な権限について、Vertex AI ドキュメントを参照してください。ローンンチ キュー設定の `spec.service_account` フィールドを使用して、カスタム サービス アカウントを W&B の runs 用に選択できます。
{{% /alert %}}

## Vertex AI 用のキューを設定する

Vertex AI リソースのキュー設定は、Vertex AI Python SDK 内の `CustomJob` コンストラクタと `CustomJob` クラスの `run` メソッドへの入力を指定します。リソース設定は `spec` と `run` キーの下に保存されます。

- `spec` キーには、Vertex AI Python SDK の [`CustomJob` コンストラクタ](https://cloud.google.com/vertex-ai/docs/pipelines/customjob-component) の名前付き引数の値が含まれています。
- `run` キーには、Vertex AI Python SDK の `CustomJob` クラスの `run` メソッドの名前付き引数の値が含まれています。

実行環境のカスタマイズは主に `spec.worker_pool_specs` リストで行われます。ワーカー プール スペックは、ジョブを実行するワーカーグループを定義します。デフォルトの設定では、加速器なしの単一の `n1-standard-4` マシンを要求します。ニーズに応じて、マシンタイプ、加速器タイプ、およびカウントを変更できます。

利用可能なマシンタイプと加速器タイプの詳細については、[Vertex AI ドキュメント](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/MachineSpec) を参照してください。

## キューを作成する

Vertex AI を計算リソースとして使用するキューを W&B アプリで作成します:

1. [Launch page](https://wandb.ai/launch) に移動します。
2. **Create Queue** ボタンをクリックします。
3. キューを作成したい **Entity** を選択します。
4. **Name** フィールドにキューの名前を入力します。
5. **GCP Vertex** を **Resource** として選択します。
6. **Configuration** フィールドに、前のセクションで定義した Vertex AI の `CustomJob` に関する情報を提供します。デフォルトで、W&B は次のような YAML と JSON のリクエストボディを入力します:

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

最低限、以下を指定する必要があります:

- `spec.worker_pool_specs` : 空ではないワーカー プール スペックのリスト。
- `spec.staging_bucket` : Vertex AI のアセットとメタデータをステージングするための GCS バケット。

{{% alert color="secondary" %}}
いくつかの Vertex AI ドキュメントでは、すべてのキーがキャメルケースで示されているワーカー プール スペックが表示されていますが、Vertex AI Python SDK はこれらのキーをスネークケースで使用します。

ローンンチ キュー設定のすべてのキーはスネークケースを使用する必要があります。
{{% /alert %}}

## ローンンチエージェントを設定する

ローンンチエージェントは、デフォルトでは `~/.config/wandb/launch-config.yaml` にある設定ファイルを通じて設定可能です。

```yaml
max_jobs: <n-concurrent-jobs>
queues:
  - <queue-name>
```

Vertex AI で実行されるイメージをエージェントに構築させたい場合は、[Advanced agent set up]({{< relref path="./setup-agent-advanced.md" lang="ja" >}}) を参照してください。

## エージェントの権限を設定する

このサービス アカウントとして認証する方法は複数あります。Workload Identity、ダウンロードされたサービス アカウント JSON、環境変数、Google Cloud Platform コマンドライン ツール、またはこれらの方法の組み合わせで達成できます。