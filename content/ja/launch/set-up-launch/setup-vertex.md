---
title: 'Tutorial: Set up W&B Launch on Vertex AI'
menu:
  launch:
    identifier: ja-launch-set-up-launch-setup-vertex
    parent: set-up-launch
url: guides/launch/setup-vertex
---

W&B Launch を使用して、Vertex AI トレーニング ジョブとして実行するジョブを送信できます。Vertex AI トレーニング ジョブを使用すると、Vertex AI プラットフォーム上で、提供されたアルゴリズムまたはカスタム アルゴリズムを使用して、機械学習 モデルをトレーニングできます。Launch ジョブが開始されると、Vertex AI は基盤となるインフラストラクチャー、スケーリング、およびオーケストレーションを管理します。

W&B Launch は、`google-cloud-aiplatform` SDK の `CustomJob` クラスを通じて Vertex AI と連携します。`CustomJob` のパラメータは、Launch キューの設定で制御できます。Vertex AI は、GCP 外部のプライベート レジストリからイメージをプルするように構成できません。つまり、W&B Launch で Vertex AI を使用する場合は、コンテナ イメージを GCP またはパブリック レジストリに保存する必要があります。コンテナ イメージを Vertex ジョブからアクセスできるようにする方法の詳細については、Vertex AI のドキュメントを参照してください。

## 前提条件

1. **Vertex AI API が有効になっている GCP プロジェクトを作成またはアクセスします。** API を有効にする方法の詳細については、[GCP API Console のドキュメント](https://support.google.com/googleapi/answer/6158841?hl=en)を参照してください。
2. **Vertex で実行するイメージを保存するために、GCP Artifact Registry リポジトリを作成します。** 詳細については、[GCP Artifact Registry のドキュメント](https://cloud.google.com/artifact-registry/docs/overview) を参照してください。
3. **Vertex AI がメタデータを保存するための一時的な GCS バケットを作成します。** このバケットは、一時的なバケットとして使用するには、Vertex AI ワークロードと同じリージョンにある必要があることに注意してください。同じバケットを、ステージング コンテキストとビルド コンテキストの両方で使用できます。
4. **Vertex AI ジョブを起動するために必要な権限を持つサービス アカウントを作成します。** サービス アカウントに権限を割り当てる方法の詳細については、[GCP IAM のドキュメント](https://cloud.google.com/iam/docs/creating-managing-service-accounts)を参照してください。
5. **Vertex ジョブを管理する権限をサービス アカウントに付与します。**

| 権限                           | リソース スコープ         | 説明                                                                                        |
| -------------------------------- | ----------------------- | ------------------------------------------------------------------------------------------- |
| `aiplatform.customJobs.create`   | 指定された GCP プロジェクト | プロジェクト内で新しい機械学習ジョブを作成できます。                                                        |
| `aiplatform.customJobs.list`     | 指定された GCP プロジェクト | プロジェクト内の機械学習ジョブを一覧表示できます。                                                        |
| `aiplatform.customJobs.get`      | 指定された GCP プロジェクト | プロジェクト内の特定の機械学習ジョブに関する情報を取得できます。                                                    |

{{% alert %}}
Vertex AI ワークロードに標準以外のサービス アカウントの ID を引き継がせたい場合は、サービス アカウントの作成と必要な権限に関する手順について、Vertex AI のドキュメントを参照してください。Launch キュー構成の `spec.service_account` フィールドを使用して、W&B の Runs 用のカスタム サービス アカウントを選択できます。
{{% /alert %}}

## Vertex AI のキューを構成する

Vertex AI リソースのキュー構成は、Vertex AI Python SDK の `CustomJob` コンストラクターと、`CustomJob` の `run` メソッドへの入力を指定します。リソース構成は、`spec` キーと `run` キーに保存されます。

- `spec` キーには、Vertex AI Python SDK の [`CustomJob` コンストラクター](https://cloud.google.com/vertex-ai/docs/pipelines/customjob-component) の名前付き引数の値が含まれています。
- `run` キーには、Vertex AI Python SDK の `CustomJob` クラスの `run` メソッドの名前付き引数の値が含まれています。

実行環境のカスタマイズは、主に `spec.worker_pool_specs` リストで行われます。ワーカー プール仕様は、ジョブを実行するワーカーのグループを定義します。デフォルト構成のワーカー仕様は、アクセラレータなしの単一の `n1-standard-4` マシンを要求します。ニーズに合わせて、マシン タイプ、アクセラレータ タイプ、および数を変更できます。

利用可能なマシン タイプとアクセラレータ タイプの詳細については、[Vertex AI のドキュメント](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/MachineSpec)を参照してください。

## キューを作成する

Vertex AI をコンピューティング リソースとして使用するキューを W&B App で作成します。

1. [Launch ページ](https://wandb.ai/launch)に移動します。
2. [**キューを作成**] ボタンをクリックします。
3. キューを作成する **Entity** を選択します。
4. [**名前**] フィールドにキューの名前を入力します。
5. [**リソース**] として [**GCP Vertex**] を選択します。
6. [**設定**] フィールド内で、前のセクションで定義した Vertex AI `CustomJob` に関する情報を提供します。デフォルトでは、W&B は次のような YAML および JSON リクエスト本文を生成します。

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

7. キューを構成したら、[**キューを作成**] ボタンをクリックします。

少なくとも以下を指定する必要があります。

- `spec.worker_pool_specs`: ワーカー プール仕様の空でないリスト。
- `spec.staging_bucket`: Vertex AI のアセットとメタデータのステージングに使用される GCS バケット。

{{% alert color="secondary" %}}
一部の Vertex AI ドキュメントでは、ワーカー プール仕様のすべてのキーがキャメルケースで表示されます (たとえば、` workerPoolSpecs`)。Vertex AI Python SDK は、これらのキーにスネークケースを使用します (たとえば、`worker_pool_specs`)。

Launch キュー構成のすべてのキーは、スネークケースを使用する必要があります。
{{% /alert %}}

## Launch エージェントを構成する

Launch エージェントは、デフォルトで `~/.config/wandb/launch-config.yaml` にある構成ファイルから構成できます。

```yaml
max_jobs: <n-concurrent-jobs>
queues:
  - <queue-name>
```

Launch エージェントに、Vertex AI で実行されるイメージを構築させたい場合は、[エージェントの詳細設定]({{< relref path="./setup-agent-advanced.md" lang="ja" >}})を参照してください。

## エージェントの権限を設定する

このサービス アカウントとして認証する方法は複数あります。これは、Workload Identity、ダウンロードされたサービス アカウント JSON、環境変数、Google Cloud Platform コマンドライン ツール、またはこれらの方法の組み合わせによって実現できます。
