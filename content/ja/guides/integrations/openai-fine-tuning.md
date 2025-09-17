---
title: OpenAI の ファインチューニング
description: W&B で OpenAI のモデルをファインチューンする方法。
menu:
  default:
    identifier: ja-guides-integrations-openai-fine-tuning
    parent: integrations
weight: 250
---

{{< cta-button colabLink="https://wandb.me/openai-colab" >}}
OpenAI の GPT-3.5 や GPT-4 モデルのファインチューニングにおけるメトリクスと設定を W&B にログします。W&B のエコシステムを活用してファインチューニングの実験、Models、Datasets を追跡し、結果を同僚と共有しましょう.
{{% alert %}}
ファインチューニング可能なモデル一覧は [OpenAI documentation](https://platform.openai.com/docs/guides/fine-tuning/which-models-can-be-fine-tuned) をご覧ください。
{{% /alert %}}

OpenAI でのファインチューニングと W&B の連携方法については、OpenAI ドキュメント内の [W&B Integration](https://platform.openai.com/docs/guides/fine-tuning/weights-and-biases-integration) セクションも参照してください。


## OpenAI Python API をインストールまたはアップデートする

W&B の OpenAI ファインチューニング連携は OpenAI バージョン 1.0 以上で動作します。最新バージョンは PyPI の [OpenAI Python API](https://pypi.org/project/openai/) のドキュメントを参照してください。

OpenAI Python API をインストールするには、次を実行します:
```python
pip install openai
```

既に OpenAI Python API をインストール済みの場合は、次でアップデートできます:
```python
pip install -U openai
```


## OpenAI のファインチューニング結果を同期する

OpenAI のファインチューニング API と W&B を連携して、ファインチューニングのメトリクスと設定を W&B にログします。これには、`wandb.integration.openai.fine_tuning` モジュールの `WandbLogger` クラスを使用します。

```python
from wandb.integration.openai.fine_tuning import WandbLogger

# ファインチューニングのロジック

WandbLogger.sync(fine_tune_job_id=FINETUNE_JOB_ID)
```

{{< img src="/images/integrations/open_ai_auto_scan.png" alt="OpenAI 自動スキャン機能" >}}

### ファインチューンを同期する

スクリプトから結果を同期します

```python
from wandb.integration.openai.fine_tuning import WandbLogger

# ワンライナー
WandbLogger.sync()

# オプション引数を渡す場合
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

### Reference

| Argument                 | Description                                                                                                               |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| fine_tune_job_id         | `client.fine_tuning.jobs.create` でファインチューニングジョブを作成した際に得られる OpenAI の Fine-Tune ID です。この引数が None（デフォルト）の場合、まだ同期されていないすべての OpenAI のファインチューンジョブが W&B に同期されます。 |
| openai_client            | 初期化済みの OpenAI クライアントを `sync` に渡します。クライアントが渡されない場合は、ロガーが自動で初期化します。デフォルトは None です。                |
| num_fine_tunes           | ID を指定しない場合、未同期のファインチューンをすべて W&B にログします。この引数で同期する直近のファインチューン件数を指定できます。たとえば 5 を指定すると、最新の 5 件を対象にします。 |
| project                  | ファインチューンのメトリクス、Models、データなどをログする W&B Project 名です。デフォルトは "OpenAI-Fine-Tune" です。 |
| entity                   | run を送信する W&B のユーザー名またはチーム名です。デフォルトではあなたのデフォルト Entity（通常はユーザー名）が使用されます。 |
| overwrite                | 同一のファインチューンジョブに対応する既存の W&B run を上書きしてログします。デフォルトは False です。                                                |
| wait_for_job_success     | OpenAI のファインチューニングジョブは完了までに時間がかかることがあります。ジョブが終了し次第メトリクスを W&B にログするために、この設定を有効にすると 60 秒ごとにステータスが `succeeded` に変わるかをチェックします。成功が検知されるとメトリクスは自動的に W&B に同期されます。デフォルトは True です。 |
| model_artifact_name      | ログされる model Artifact の名前です。デフォルトは `"model-metadata"` です。                    |
| model_artifact_type      | ログされる model Artifact のタイプです。デフォルトは `"model"` です。                    |
| \*\*kwargs_wandb_init  | [`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init.md" lang="ja" >}}) に直接渡される追加の引数。                    |

## Dataset のバージョン管理と可視化

### バージョン管理

OpenAI にファインチューニング用としてアップロードしたトレーニングおよび検証データは、バージョン管理を容易にするために自動で W&B Artifacts としてログされます。以下は Artifacts 内のトレーニングファイルのビューです。どの W&B run がいつこのファイルをログしたか、このデータセットのバージョン、メタデータ、そしてトレーニングデータから学習済みモデルへ至る DAG リネージを確認できます。

{{< img src="/images/integrations/openai_data_artifacts.png" alt="W&B Artifacts によるトレーニングデータセット" >}}

### 可視化

Datasets は W&B Tables として可視化され、探索・検索・対話的な操作が可能です。以下は W&B Tables で可視化したトレーニングサンプルの例です。

{{< img src="/images/integrations/openai_data_visualization.png" alt="OpenAI のデータ" >}}


## 学習済みモデルとモデルのバージョン管理

OpenAI はファインチューニング済みモデルの ID を提供します。モデルの重みにはアクセスできないため、`WandbLogger` はハイパーパラメーターやデータファイルの ID など、モデルに関する詳細と `fine_tuned_model` の ID を含む `model_metadata.json` を作成し、W&B Artifact としてログします。

この model（メタデータ）Artifact は、[W&B Registry]({{< relref path="/guides/core/registry/" lang="ja" >}}) の model にリンクすることもできます。

{{< img src="/images/integrations/openai_model_metadata.png" alt="OpenAI モデルのメタデータ" >}}


## よくある質問

### チームでファインチューンの結果を共有するには？

次のように、ファインチューニングジョブをチームアカウントにログします:

```python
WandbLogger.sync(entity="YOUR_TEAM_NAME")
```

### run を整理するには？

あなたの W&B runs は自動で整理され、ジョブタイプ、ベースモデル、学習率、トレーニングファイル名、その他任意のハイパーパラメーターといった設定に基づいてフィルタやソートが可能です。

さらに、run 名の変更やノートの追加、タグの作成によるグルーピングもできます。

設定が固まったら Workspace を保存し、runs や保存済み Artifacts（トレーニング/検証ファイル）からデータを取り込んで Report を作成できます。

### ファインチューニング済みモデルへはどうやってアクセスできますか？

ファインチューニング済みモデルの ID は、`model_metadata.json` として W&B の Artifacts に、また設定情報としてもログされます。

```python
import wandb
    
with wandb.init(project="OpenAI-Fine-Tune", entity="YOUR_TEAM_NAME") as run:
    ft_artifact = run.use_artifact("ENTITY/PROJECT/model_metadata:VERSION")
    artifact_dir = ft_artifact.download()
```

ここで `VERSION` は次のいずれかです:

* `v2` のようなバージョン番号
* `ft-xxxxxxxxx` のようなファインチューン ID
* 自動で付与される `latest` など、または手動で付けたエイリアス

ダウンロードした `model_metadata.json` を読み込むことで、`fine_tuned_model` の ID にアクセスできます。

### ファインチューンがうまく同期されなかった場合は？

ファインチューンが W&B に正常にログされなかった場合は、`overwrite=True` を指定してファインチューンジョブ ID を渡してください:

```python
WandbLogger.sync(
    fine_tune_job_id="FINE_TUNE_JOB_ID",
    overwrite=True,
)
```

### W&B で Datasets と Models を追跡できますか？

トレーニングおよび検証データは自動的に W&B の Artifacts にログされます。ファインチューニング済みモデルの ID を含むメタデータも Artifacts としてログされます。

`wandb.Artifact`、`wandb.Run.log` などの低レベルな wandb API を使ってパイプラインを自由に制御することもできます。これにより、データとモデルの完全なトレーサビリティを実現できます。

{{< img src="/images/integrations/open_ai_faq_can_track.png" alt="OpenAI 追跡に関する FAQ" >}}

## リソース

* [OpenAI Fine-tuning のドキュメント](https://platform.openai.com/docs/guides/fine-tuning/) は非常に充実しており、有用なヒントが多数掲載されています
* [デモ Colab](https://wandb.me/openai-colab)
* [How to Fine-Tune Your OpenAI GPT-3.5 and GPT-4 Models with W&B](https://wandb.me/openai-report) レポート