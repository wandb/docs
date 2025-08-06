---
title: OpenAI ファインチューニング
description: W&B を使って OpenAI モデルをファインチューンする方法
menu:
  default:
    identifier: openai-fine-tuning
    parent: integrations
weight: 250
---

{{< cta-button colabLink="https://wandb.me/openai-colab" >}}

OpenAI GPT-3.5 または GPT-4 モデルのファインチューニング時のメトリクスや設定を W&B にログしましょう。W&B エコシステムを活用してファインチューニングの Experiments、Models、Datasets を追跡し、結果をチームで共有できます。

{{% alert %}}
ファインチューニング可能なモデルの一覧については、[OpenAI ドキュメント](https://platform.openai.com/docs/guides/fine-tuning/which-models-can-be-fine-tuned)をご確認ください。
{{% /alert %}}

ファインチューニング時に W&B と OpenAI を連携する方法については、OpenAI ドキュメントの [W&B Integration](https://platform.openai.com/docs/guides/fine-tuning/weights-and-biases-integration) セクションも参考にしてください。

## OpenAI Python API をインストール・アップデート

W&B の OpenAI ファインチューニング連携は、OpenAI バージョン 1.0 以降で利用可能です。最新バージョンや詳細は [OpenAI Python API](https://pypi.org/project/openai/) ライブラリ（PyPI ドキュメント）をご覧ください。

OpenAI Python API をインストールするには以下を実行してください:
```python
pip install openai
```

すでにインストール済みの場合は、以下のコマンドでアップデートできます:
```python
pip install -U openai
```

## OpenAI ファインチューニング結果を同期する

W&B を OpenAI のファインチューニング API と統合し、ファインチューニングのメトリクスや設定を W&B にログしましょう。実装には `wandb.integration.openai.fine_tuning` モジュールの `WandbLogger` クラスを使用します。

```python
from wandb.integration.openai.fine_tuning import WandbLogger

# ファインチューニングの処理

WandbLogger.sync(fine_tune_job_id=FINETUNE_JOB_ID)
```

{{< img src="/images/integrations/open_ai_auto_scan.png" alt="OpenAI auto-scan feature" >}}

### ファインチューンの同期

スクリプトから結果を同期できます:

```python
from wandb.integration.openai.fine_tuning import WandbLogger

# ワンラインコマンド
WandbLogger.sync()

# オプションパラメータを渡す場合
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

### 引数リファレンス

| 引数                      | 説明                                                                                                                       |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| fine_tune_job_id         | OpenAI ファインチューンジョブ作成時に得られる Fine-Tune ID。None（デフォルト）の場合、まだ同期されていない全ての OpenAI ファインチューンジョブが W&B に同期されます。|
| openai_client            | 初期化済みの OpenAI クライアントを `sync` に渡すことができます。指定しない場合は logger 側で初期化されます（デフォルト: None）。   |
| num_fine_tunes           | ID を指定しない場合、未同期のファインチューン全てが W&B にログされます。この引数で最新 n 件を指定できます。例えば 5 なら最新の 5 件だけ同期します。|
| project                  | メトリクスやモデル、データなどをログする W&B の Project 名。デフォルトは "OpenAI-Fine-Tune" です。                 |
| entity                   | Run を送る先の W&B ユーザー名または Team 名。デフォルトは自身の entity（通常はユーザー名）です。                |
| overwrite                | 既存の同一ファインチューンジョブの wandb run を強制的に上書きしてログします（デフォルト: False）。                     |
| wait_for_job_success     | OpenAI ファインチューニングジョブ完了（`succeeded` 状態）になるまで 60 秒ごとに status を確認し、完了次第自動で W&B にメトリクスを同期します。デフォルト: True。|
| model_artifact_name      | ログするモデルアーティファクト名（デフォルト: `"model-metadata"`）。                                |
| model_artifact_type      | ログするモデルアーティファクトタイプ（デフォルト: `"model"`）。                                  |
| \*\*kwargs_wandb_init  | [`wandb.init()`]({{< relref "/ref/python/sdk/functions/init.md" >}}) に直接渡す追加引数。                      |

## データセットのバージョン管理と可視化

### バージョン管理

ファインチューニング用のトレーニングおよびバリデーションデータは、自動的に W&B Artifacts として記録・管理されます。以下は Artifacts 上でのトレーニングファイルの表示例です。どの W&B Run でいつログされたか、データセットのバージョン、メタデータ、トレーニングデータから学習済みモデルへの DAG リネージなどが確認できます。

{{< img src="/images/integrations/openai_data_artifacts.png" alt="W&B Artifacts with training datasets" >}}

### 可視化

データセットは W&B Tables で可視化でき、探索・検索・インタラクションが可能です。下記は W&B Tables で可視化したトレーニングサンプル例です。

{{< img src="/images/integrations/openai_data_visualization.png" alt="OpenAI data" >}}

## ファインチューニング済みモデルとモデルのバージョン管理

OpenAI からはファインチューニング済みモデルの ID が付与されます。モデル自体の重みにはアクセスできないため、`WandbLogger` はモデルの詳細情報（ハイパーパラメーターやデータファイル ID など）および `fine_tuned_model` ID を含む `model_metadata.json` ファイルを作成し、W&B Artifact としてログします。

このモデル（メタデータ）Artifact は [W&B Registry]({{< relref "/guides/core/registry/" >}}) に紐づけることも可能です。

{{< img src="/images/integrations/openai_model_metadata.png" alt="OpenAI model metadata" >}}

## よくある質問（FAQ）

### W&B でファインチューン結果をチームと共有するには？

以下のようにファインチューンジョブを Team アカウントにログできます:

```python
WandbLogger.sync(entity="YOUR_TEAM_NAME")
```

### Run を整理（整理・検索・グループ化など）するには？

W&B の Run は自動で整理され、ジョブタイプやベースモデル、学習率、ファイル名、各種ハイパーパラメーターなどあらゆる設定項目でフィルタ・ソートが可能です。

さらに、Run の名称変更、ノート追加、タグ作成などもできます。

満足したら Workspace の保存も可能です。保存した Workspace から Report を作成したり、Run や保存済み Artifact（トレーニング/バリデーションファイル）からデータをインポートできます。

### ファインチューニング済みモデルへアクセスするには？

ファインチューニング済みモデルの ID は `model_metadata.json` などとして W&B の Artifacts、および config に記録されます。

```python
import wandb
    
with wandb.init(project="OpenAI-Fine-Tune", entity="YOUR_TEAM_NAME") as run:
    ft_artifact = run.use_artifact("ENTITY/PROJECT/model_metadata:VERSION")
    artifact_dir = ft_artifact.download()
```

ここで `VERSION` には以下が指定できます:

* `v2` のようなバージョン番号
* `ft-xxxxxxxxx` のようなファインチューン ID
* 自動追加・手動追加も可能なエイリアス（例: `latest` など）

ダウンロードした `model_metadata.json` ファイルを読むことで `fine_tuned_model` ID にアクセスできます。

### ファインチューンが正しく同期されなかった場合は？

ファインチューンが正しく W&B に記録されなかった場合、`overwrite=True` を指定してファインチューンジョブ ID を渡せば再同期できます。

```python
WandbLogger.sync(
    fine_tune_job_id="FINE_TUNE_JOB_ID",
    overwrite=True,
)
```

### Datasets や Models の管理・追跡はできますか？

トレーニング／バリデーションデータは自動的に W&B の Artifact として記録されます。ファインチューン済みモデルの ID もメタデータとして Artifact に記録されます。

さらに、`wandb.Artifact`、`wandb.Run.log` などの低レベル wandb API を利用すれば、データやモデルのパイプライン全体を完全にトレースできます。

{{< img src="/images/integrations/open_ai_faq_can_track.png" alt="OpenAI tracking FAQ" >}}

## リソース

* [OpenAI ファインチューニング ドキュメント](https://platform.openai.com/docs/guides/fine-tuning/) ― 詳細で役立つ情報が豊富です
* [Demo Colab](https://wandb.me/openai-colab)
* [How to Fine-Tune Your OpenAI GPT-3.5 and GPT-4 Models with W&B](https://wandb.me/openai-report) レポート