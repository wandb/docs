---
title: OpenAI Fine-Tuning
description: OpenAI モデルを W&B でファインチューンする方法
menu:
  default:
    identifier: ja-guides-integrations-openai-fine-tuning
    parent: integrations
weight: 250
---

{{< cta-button colabLink="http://wandb.me/openai-colab" >}}

OpenAI GPT-3.5 や GPT-4 モデルのファインチューニングのメトリクスと設定を W&B にログします。 W&B エコシステムを活用してファインチューニング 実験、Models、Datasets を追跡し、結果を同僚と共有できます。

{{% alert %}}
ファインチューニングできるモデルの一覧については、[OpenAI ドキュメント](https://platform.openai.com/docs/guides/fine-tuning/which-models-can-be-fine-tuned) を参照してください。
{{% /alert %}}

ファインチューニングのために W&B と OpenAI を統合する方法についての追加情報は、OpenAI ドキュメントの [Weights and Biases Integration](https://platform.openai.com/docs/guides/fine-tuning/weights-and-biases-integration) セクションを参照してください。

## OpenAI Python API のインストールまたは更新

W&B の OpenAI ファインチューニング インテグレーションは、OpenAI バージョン 1.0 以上で動作します。 最新バージョンの [OpenAI Python API](https://pypi.org/project/openai/) ライブラリに関する情報は、PyPI ドキュメントを参照してください。

OpenAI Python API をインストールするには、次のコマンドを実行してください:
```python
pip install openai
```

既に OpenAI Python API をインストールしている場合は、次のコマンドで更新できます:
```python
pip install -U openai
```

## OpenAI ファインチューニング結果を同期する

W&B を OpenAI のファインチューニング API と統合して、ファインチューニングのメトリクスと設定を W&B にログします。これには、`wandb.integration.openai.fine_tuning` モジュールの `WandbLogger` クラスを使用します。

```python
from wandb.integration.openai.fine_tuning import WandbLogger

# ファインチューニングのロジック

WandbLogger.sync(fine_tune_job_id=FINETUNE_JOB_ID)
```

{{< img src="/images/integrations/open_ai_auto_scan.png" alt="" >}}

### ファインチューニングを同期する

スクリプトからの結果を同期

```python
from wandb.integration.openai.fine_tuning import WandbLogger

# ワンラインコマンド
WandbLogger.sync()

# オプションパラメータを渡す
WandbLogger.sync(
    fine_tune_job_id=None,
    num_fine_tunes=None,
    project="OpenAI-Fine-Tune",
    entity=None,
    overwrite=False,
    model_artifact_name="model-metadata",
    model_artifact_type="model",
    **kwargs_wandb_init
)
```

### リファレンス

| 引数                      | 説明                                                                                                        |
| ------------------------ | --------------------------------------------------------------------------------------------------------- |
| fine_tune_job_id         | `client.fine_tuning.jobs.create` を使用してファインチューンジョブを作成すると取得する OpenAI ファインチューン ID です。 この引数が None（デフォルト）の場合、まだ同期されていないすべての OpenAI ファインチューン ジョブが W&B に同期されます。                                                                                     |
| openai_client            | 初期化された OpenAI クライアントを `sync` に渡します。クライアントが提供されない場合、ログは自動的にクライアントを初期化します。 デフォルトでは None です。         |
| num_fine_tunes           | ID が提供されない場合、未同期のファインチューンはすべて W&B にログされます。この引数を使用して、同期する最新のファインチューンの数を選択できます。num_fine_tunes が 5 の場合、最新のファインチューン 5 つを選択します。                                                                                                               |
| project                  | ファインチューニングのメトリクス、Models、Data などがログされる Weights and Biases プロジェクト名。 デフォルトでは、プロジェクト名は "OpenAI-Fine-Tune" です。                    |
| entity                   | W&B ユーザー名またはチーム名。実行結果を送信するエンティティです。 デフォルトでは、通常はユーザー名であるデフォルトエンティティが使用されます。 |
| overwrite                | ロギングを強制し、同一ファインチューンジョブの既存の wandb run を上書きします。デフォルトでは False です。                                                |
| wait_for_job_success     | OpenAI ファインチューニングジョブが開始されると、通常少し時間がかかります。ファインチューニングジョブが終了すると、メトリクスが W&B にすぐにログされるように、この設定は 60 秒ごとにファインチューニングジョブのステータスが `succeeded` に変わるかどうかを確認します。ファインチューニングジョブが成功したと検出されると、自動的にメトリクスが W&B に同期されます。 デフォルトで True に設定されています。                                           |
| model_artifact_name      | ログされるモデル アーティファクトの名前。デフォルトは `"model-metadata"` です。|
| model_artifact_type      | ログされるモデル アーティファクトのタイプ。デフォルトは `"model"` です。|
| \*\*kwargs_wandb_init  | 直接 [`wandb.init()`]({{< relref path="/ref/python/init.md" lang="ja" >}}) に渡される追加の引数                                     |

## データセットのバージョン管理と可視化

### バージョン管理

ファインチューニングのために OpenAI にアップロードしたトレーニングおよび検証データは、バージョン管理を容易にするために自動的に W&B Artifacts としてログされます。 以下に、この アーティファクト 内のトレーニングファイルのビューを示します。ここでは、このファイルをログした W&B run、ログされた時期、このデータセットのバージョン、メタデータ、およびトレーニングデータから学習済みモデルまでの DAG リネージを確認できます。

{{< img src="/images/integrations/openai_data_artifacts.png" alt="" >}}

### 可視化

データセットは W&B Tables として可視化され、データセットを探索、検索、および対話することができます。以下に、 W&B Tables を使用して可視化されたトレーニングサンプルをチェックしてください。

{{< img src="/images/integrations/openai_data_visualization.png" alt="" >}}

## ファインチューニング済みモデルとモデルのバージョン管理

OpenAI はファインチューニングされたモデルの ID を提供します。モデルの重みにはアアクセスできないため、`WandbLogger` はモデルのすべての詳細（ハイパーパラメーター、データファイルの ID など）と `fine_tuned_model` ID を含む `model_metadata.json` ファイルを作成し、 W&B アーティファクトとしてログします。

このモデル（メタデータ）アーティファクトは、[W&B Registry]({{< relref path="/guides/core/registry/" lang="ja" >}}) のモデルにさらにリンクすることができます。

{{< img src="/images/integrations/openai_model_metadata.png" alt="" >}}

## よくある質問

### チームとファインチューン結果を W&B で共有するにはどうすればよいですか？

以下を使用してファインチューンジョブをチームアカウントにログします:

```python
WandbLogger.sync(entity="YOUR_TEAM_NAME")
```

### 自分の runs をどのように整理できますか？

あなたの W&B runs は自動的に整理され、ジョブ タイプ、ベースモデル、学習率、トレーニングファイル名、その他のハイパーパラメーターなど、任意の設定パラメーターに基づいてフィルタリングやソートができます。

さらに、run の名前を変更したり、メモを追加したり、タグを作成してグループ化することができます。

満足したら、ワークスペースを保存し、run および保存されたアーティファクト（トレーニング/検証ファイル）からデータをインポートしてレポートを作成できます。

### ファインチューンされたモデルにアクセスするにはどうすればよいですか？

ファインチューンされたモデル ID は、アーティファクト (`model_metadata.json`) として W&B にログされます。

```python
import wandb

ft_artifact = wandb.run.use_artifact("ENTITY/PROJECT/model_metadata:VERSION")
artifact_dir = artifact.download()
```

ここで `VERSION` は次のいずれかです:

* `v2` などのバージョン番号
* `ft-xxxxxxxxx` などのファインチューン ID
* 自動的または手動で追加されたエイリアス、例えば `latest` 

ダウンロードした `model_metadata.json` ファイルを読み取ることで `fine_tuned_model` ID にアクセスできます。

### ファインチューンが正常に同期されなかった場合はどうすればよいですか？

ファインチューンが W&B に正常にログされなかった場合は、`overwrite=True` を使用し、ファインチューン ジョブ ID を渡すことができます:

```python
WandbLogger.sync(
    fine_tune_job_id="FINE_TUNE_JOB_ID",
    overwrite=True,
)
```

### W&B で自分の Datasets と Models を追跡できますか？

トレーニングと検証データは自動的に アーティファクト として W&B にログされます。ファインチューンされたモデルの ID を含むメタデータも アーティファクト としてログされます。

`wandb.Artifact`、`wandb.log` などの低レベルの wandb API を使用してパイプラインを常に制御できます。これにより、データとモデルの完全なトレーサビリティが可能になります。

{{< img src="/images/integrations/open_ai_faq_can_track.png" alt="" >}}

## リソース

* [OpenAI ファインチューニング ドキュメント](https://platform.openai.com/docs/guides/fine-tuning/) は非常に詳細で多くの有用なヒントが含まれています
* [デモ Colab](http://wandb.me/openai-colab)
* [W&B で OpenAI GPT-3.5 および GPT-4 モデルをファインチューニングする方法](http://wandb.me/openai-report) レポート