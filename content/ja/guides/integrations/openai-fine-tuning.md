---
title: OpenAI Fine-Tuning
description: OpenAI モデルを W&B を使用してファインチューンする方法。
menu:
  default:
    identifier: ja-guides-integrations-openai-fine-tuning
    parent: integrations
weight: 250
---

{{< cta-button colabLink="http://wandb.me/openai-colab" >}}

OpenAI GPT-3.5 または GPT-4 モデルのファインチューニング メトリクスと設定を W&B にログとして記録します。W&B エコシステムを活用してファインチューニング実験、モデル、データセットを追跡し、同僚と結果を共有しましょう。

{{% alert %}}
ファインチューニング可能なモデルのリストについては、[OpenAI ドキュメント](https://platform.openai.com/docs/guides/fine-tuning/which-models-can-be-fine-tuned) を参照してください。
{{% /alert %}}

ファインチューニングのために OpenAI と W&B を統合する方法に関する補足情報については、OpenAI ドキュメントの [Weights and Biases Integration](https://platform.openai.com/docs/guides/fine-tuning/weights-and-biases-integration) セクションを参照してください。

## OpenAI Python API のインストールまたは更新

W&B OpenAI ファインチューニング統合は、OpenAI バージョン 1.0 以上で動作します。最新バージョンの [OpenAI Python API](https://pypi.org/project/openai/) ライブラリについては、PyPI ドキュメントを参照してください。

OpenAI Python API をインストールするには、次のコマンドを実行します:
```python
pip install openai
```

すでに OpenAI Python API がインストールされている場合は、次のコマンドで更新できます:
```python
pip install -U openai
```

## OpenAI ファインチューニング結果の同期

W&B を OpenAI のファインチューニング API と統合し、ファインチューニング メトリクスと設定を W&B にログとして記録します。これを行うには、`wandb.integration.openai.fine_tuning` モジュールから `WandbLogger` クラスを使用します。

```python
from wandb.integration.openai.fine_tuning import WandbLogger

# ファインチューニング ロジック

WandbLogger.sync(fine_tune_job_id=FINETUNE_JOB_ID)
```

{{< img src="/images/integrations/open_ai_auto_scan.png" alt="" >}}

### ファインチューンの同期

スクリプトから結果を同期します 

```python
from wandb.integration.openai.fine_tuning import WandbLogger

# 1 行コマンド
WandbLogger.sync()

# オプションのパラメータを渡す
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

### 参照

| 引数                     | 説明                                                                                                       |
| ------------------------ | --------------------------------------------------------------------------------------------------------- |
| fine_tune_job_id         | これは OpenAI のファインチューン ID であり、`client.fine_tuning.jobs.create` を使用してファインチューン ジョブを作成する際に取得します。この引数が None（デフォルト）の場合、まだ同期されていないすべての OpenAI ファインチューン ジョブが W&B に同期されます。                                                                                       |
| openai_client            | 初期化済みの OpenAI クライアントを `sync` に渡します。クライアントが提供されていない場合、ロガー自体によって初期化されます。デフォルトは None です。                |
| num_fine_tunes           | ID が提供されない場合、同期されていないすべてのファインチューンが W&B にログとして記録されます。この引数を使用すると、同期する最近のファインチューンの数を選択できます。num_fine_tunes が 5 の場合、5 つの最近のファインチューンが選択されます。                                                  |
| project                  | ファインチューンメトリクス、モデル、データ等が記録される Weights and Biases プロジェクト名。デフォルトではプロジェクト名は "OpenAI-Fine-Tune" です。 |
| entity                   | run を送信している W&B ユーザー名またはチーム名。デフォルトでは通常はユーザー名であるデフォルトのエンティティが使用されます。 |
| overwrite                | 同じファインチューン ジョブの既存の wandb run をログに強制的に記録して上書きします。デフォルトでは False です。                                                |
| wait_for_job_success     | OpenAI のファインチューニング ジョブが開始されると通常は少し時間がかかります。この設定ではファインチューン ジョブが `succeeded` にステータスが変わるまで 60 秒ごとに確認しできる限り早くメトリクスを W&B に記録できるようにします。ファインチューン ジョブが成功として検出されるとメトリクスは自動的に W&B に同期されます。デフォルトは True です。                                                    |
| model_artifact_name      | ログとして記録されるモデル アーティファクトの名前。デフォルトは `"model-metadata"` です。                    |
| model_artifact_type      | ログとして記録されるモデル アーティファクトのタイプ。デフォルトは `"model"` です。                    |
| \*\*kwargs_wandb_init  | [`wandb.init()`]({{< relref path="/ref/python/init.md" lang="ja" >}}) に直接渡される追加の引数                    |

## データセットのバージョン管理と可視化

### バージョン管理

OpenAI にファインチューニングのためにアップロードしたトレーニングおよび検証データは、バージョン管理を容易にするために W&B Artifacts として自動的にログを記録します。以下は Artifacts 内のトレーニングファイルのビューです。ここで、このファイルを記録した W&B run、ログが記録された日時、データセットのバージョン、メタデータ、および トレーニングデータからトレーニング済みモデルへの DAG リネージが表示されます。

{{< img src="/images/integrations/openai_data_artifacts.png" alt="" >}}

### 可視化

データセットは W&B Tables として可視化され、データセットを探索、検索、対話することができます。以下に W&B Tables を使用して可視化されたトレーニング サンプルをチェックしてください。

{{< img src="/images/integrations/openai_data_visualization.png" alt="" >}}

## ファインチューニング済みモデルとモデルのバージョン管理

OpenAI はファインチューニング済みモデルの ID を提供します。モデルの重みへのアクセスがないため、`WandbLogger` はモデルの詳細（ハイパーパラメーター、データファイル ID など）と `fine_tuned_model` ID を持つ `model_metadata.json` ファイルを作成し、W&B Artifact にログを記録します。

このモデル（メタデータ）アーティファクトは、[W&B Model Registry]({{< relref path="/guides/models/registry/model_registry/" lang="ja" >}}) にさらにリンクできます。

{{< img src="/images/integrations/openai_model_metadata.png" alt="" >}}

## よくある質問

### ファインチューンの結果をチームと W&B で共有するにはどうすればいいですか？

ファインチューンのジョブを次のようにチーム アカウントにログを記録します:

```python
WandbLogger.sync(entity="YOUR_TEAM_NAME")
```

### run を整理するにはどうすればいいですか？

W&B の run は自動的に整理され、ジョブタイプ、ベースモデル、学習率、トレーニングファイル名やその他のハイパーパラメータなどの任意の設定パラメータに基づいて絞り込みや並べ替えができます。

さらに、run の名前変更、メモの追加、タグの作成を行い、run をグループ化できます。

納得ができたら、ワークスペースを保存して、run や保存されたアーティファクト（トレーニング/検証ファイル）からデータをインポートして'report' として使用できます。

### ファインチューン済みモデルにアクセスするにはどうすればいいですか？

ファインチューン済みモデル ID はアーティファクト (`model_metadata.json`) として W&B にログが記録されます。

```python
import wandb

ft_artifact = wandb.run.use_artifact("ENTITY/PROJECT/model_metadata:VERSION")
artifact_dir = artifact.download()
```

ここで `VERSION` には以下のいずれかになります:

* `v2` などのバージョン番号
* `ft-xxxxxxxxx` のようなファインチューン ID
* `latest` などの自動的に追加されるエイリアスまたは手動で追加されるエイリアス

次に、ダウンロードされた `model_metadata.json` ファイルを読み取って `fine_tuned_model` ID にアクセスできます。

### ファインチューンが正常に同期されなかった場合はどうなりますか？

ファインチューンが W&B に正常にログとして記録されていない場合は、`overwrite=True` を使用してファインチューン ジョブ ID を渡すことができます：

```python
WandbLogger.sync(
    fine_tune_job_id="FINE_TUNE_JOB_ID",
    overwrite=True,
)
```

### 自分のデータセットやモデルを W&B で追跡できますか？

トレーニングおよび検証データは自動的に W&B にアーティファクトとしてログが記録されます。ファインチューン済みモデルの ID を含むメタデータもアーティファクトとして記録されます。

`wandb.Artifact`、`wandb.log` などの低レベルの wandb API を使用してパイプラインを常にコントロールできます。これにより、データとモデルの完全な追跡が可能になります。

{{< img src="/images/integrations/open_ai_faq_can_track.png" alt="" >}}

## リソース

* [OpenAI ファインチューニング ドキュメント](https://platform.openai.com/docs/guides/fine-tuning/) は非常に詳細で役立つヒントが豊富に含まれています
* [デモ Colab](http://wandb.me/openai-colab)
* [OpenAI GPT-3.5 および GPT-4 モデルを W&B でファインチューニングする方法](http://wandb.me/openai-report) レポート
