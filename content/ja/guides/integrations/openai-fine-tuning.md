---
title: OpenAI ファインチューニング
description: W&B を使って OpenAI モデルをファインチューンする方法
menu:
  default:
    identifier: ja-guides-integrations-openai-fine-tuning
    parent: integrations
weight: 250
---

{{< cta-button colabLink="https://wandb.me/openai-colab" >}}

OpenAI GPT-3.5 や GPT-4 モデルのファインチューニングメトリクスや設定を W&B に記録しましょう。W&B エコシステムを活用してファインチューニングの実験、モデル、データセットをトラッキングし、結果を同僚と共有できます。

{{% alert %}}
ファインチューニング可能なモデルの一覧は [OpenAI ドキュメント](https://platform.openai.com/docs/guides/fine-tuning/which-models-can-be-fine-tuned) をご覧ください。
{{% /alert %}}

OpenAI で W&B をファインチューニングに統合する方法については、OpenAI ドキュメントの [W&B Integration](https://platform.openai.com/docs/guides/fine-tuning/weights-and-biases-integration) セクションも併せてご参照ください。

## OpenAI Python API のインストール・アップデート

W&B の OpenAI ファインチューニング統合は OpenAI バージョン 1.0 以上で動作します。最新バージョンについては [OpenAI Python API](https://pypi.org/project/openai/) ライブラリの PyPI ドキュメントをご覧ください。

OpenAI Python API のインストール方法:
```python
pip install openai
```

すでに OpenAI Python API がインストールされている場合、以下でアップデートできます:
```python
pip install -U openai
```

## OpenAI ファインチューニング結果の同期

W&B と OpenAI のファインチューニング API を統合して、ファインチューニングのメトリクスや設定を W&B に記録しましょう。これには `wandb.integration.openai.fine_tuning` モジュールの `WandbLogger` クラスを使います。

```python
from wandb.integration.openai.fine_tuning import WandbLogger

# ファインチューニングのロジック

WandbLogger.sync(fine_tune_job_id=FINETUNE_JOB_ID)
```

{{< img src="/images/integrations/open_ai_auto_scan.png" alt="OpenAI auto-scan feature" >}}

### ファインチューン結果の同期

スクリプトから結果を同期します

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

| 引数                     | 説明                                                                                                               |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| fine_tune_job_id         | これは OpenAI ファインチューン ID で、`client.fine_tuning.jobs.create` でファインチューンジョブを作成した際に取得できます。この引数が None（デフォルト）の場合、まだ同期されていないすべての OpenAI ファインチューンジョブが W&B へ同期されます。                    |
| openai_client            | 初期化済みの OpenAI クライアントを `sync` に渡します。クライアントを指定しない場合、ロガー側で初期化されます（デフォルトは None）。|
| num_fine_tunes           | ID を指定しない場合、同期されていないすべてのファインチューン結果が W&B に記録されます。この引数で同期する最新ファインチューン数を指定できます（例: 5 を指定すると最新5件を同期）。                   |
| project                  | ファインチューニングのメトリクス、モデル、データを記録する W&B の Project 名。デフォルトでは "OpenAI-Fine-Tune"。            |
| entity                   | Run を送る先となる W&B のユーザー名またはチーム名。デフォルトは自身のエンティティ（通常はユーザー名）。          |
| overwrite                | 既存の同ファインチューンジョブの wandb run を強制的に上書き/記録します。デフォルトは False。                       |
| wait_for_job_success     | OpenAI ファインチューニングジョブ開始後、完了を自動的に検知しメトリクスを W&B に同期します（デフォルト True。60秒ごとにチェックし、`succeeded` になった時点で自動的に同期）。                                    |
| model_artifact_name      | ログされるモデルアーティファクトの名前。デフォルトは `"model-metadata"`。                      |
| model_artifact_type      | ログされるモデルアーティファクトのタイプ。デフォルトは `"model"`。                              |
| \*\*kwargs_wandb_init  | [`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init.md" lang="ja" >}}) に直接渡される追加引数。                  |

## データセットのバージョン管理と可視化

### バージョン管理

ファインチューニング用に OpenAI へアップロードしたトレーニング・バリデーションデータは、自動的に W&B Artifacts として記録されるため、バージョン管理が簡単に行えます。下記はトレーニングファイルの Artifacts 上での表示例です。どの W&B run でこのファイルが記録されたか、記録日時、データセットのバージョン、メタデータ、トレーニングデータから学習済みモデルまでの DAG リネージなどを確認できます。

{{< img src="/images/integrations/openai_data_artifacts.png" alt="W&B Artifacts with training datasets" >}}

### 可視化

データセットは W&B Tables で可視化され、データセットの探索・検索・インタラクションが可能です。下記は W&B Tables を用いて可視化されたトレーニングデータサンプルの例です。

{{< img src="/images/integrations/openai_data_visualization.png" alt="OpenAI data" >}}

## ファインチューニング済みモデルとモデルバージョン管理

OpenAI ではファインチューニングされたモデルの ID が提供されます。モデルの重み自体にはアクセスできませんが、`WandbLogger` がモデルのすべての詳細（ハイパーパラメーター、データファイルID等）を含んだ `model_metadata.json` ファイルを作成し、`fine_tuned_model` ID とともに W&B Artifact として記録します。

このモデル（メタデータ）アーティファクトは、[W&B Registry]({{< relref path="/guides/core/registry/" lang="ja" >}}) のモデルにリンクすることもできます。

{{< img src="/images/integrations/openai_model_metadata.png" alt="OpenAI model metadata" >}}

## よくある質問

### ファインチューニング結果を W&B でチームと共有するには？

ファインチューンジョブをチームアカウントに記録するには以下を実行します:

```python
WandbLogger.sync(entity="YOUR_TEAM_NAME")
```

### run の整理方法は？

W&B 上の run は自動で整理され、ジョブタイプ、ベースモデル、学習率、トレーニングファイル名、その他ハイパーパラメータなど、任意の設定項目単位でフィルタ・ソートできます。

加えて、run の名前変更、メモ追加、タグ付けによるグルーピングも可能です。

整理後はワークスペースとして保存し、run や記録済みアーティファクト（トレーニング/バリデーションファイル）からデータをインポートしてレポート作成に活用できます。

### ファインチューニング済みモデルへのアクセス方法は？

ファインチューニング済みモデルの ID は artifact（`model_metadata.json`）および設定として W&B に記録されます。

```python
import wandb
    
with wandb.init(project="OpenAI-Fine-Tune", entity="YOUR_TEAM_NAME") as run:
    ft_artifact = run.use_artifact("ENTITY/PROJECT/model_metadata:VERSION")
    artifact_dir = ft_artifact.download()
```

ここで `VERSION` には以下を指定できます：

* バージョン番号例: `v2`
* ファインチューニング ID 例: `ft-xxxxxxxxx`
* 自動追加エイリアス例: `latest` または手動で追加したもの

ダウンロードした `model_metadata.json` を読み込むことで `fine_tuned_model` の ID にアクセスできます。

### ファインチューニングがうまく同期できなかった場合は？

ファインチューンが W&B に正常に記録されていない場合は、`overwrite=True` かつファインチューンジョブ ID を渡して再同期できます:

```python
WandbLogger.sync(
    fine_tune_job_id="FINE_TUNE_JOB_ID",
    overwrite=True,
)
```

### W&B でデータセットやモデルの管理は可能？

トレーニングおよびバリデーションデータは自動的に W&B の artifact に記録されます。ファインチューニング済みモデルの ID などのメタデータも artifact として記録されます。

pipeline を細かく制御したい場合は `wandb.Artifact` や `wandb.Run.log` など、低レベルな wandb API を使うこともできます。これによりデータやモデルの完全な追跡性が保証されます。

{{< img src="/images/integrations/open_ai_faq_can_track.png" alt="OpenAI tracking FAQ" >}}

## 参考リンク

* [OpenAI ファインチューニングドキュメント](https://platform.openai.com/docs/guides/fine-tuning/)：豊富な情報と役立つヒントあり
* [デモ Colab](https://wandb.me/openai-colab)
* [How to Fine-Tune Your OpenAI GPT-3.5 and GPT-4 Models with W&B](https://wandb.me/openai-report) レポート