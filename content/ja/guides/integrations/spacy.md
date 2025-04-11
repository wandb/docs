---
title: spaCy
menu:
  default:
    identifier: ja-guides-integrations-spacy
    parent: integrations
weight: 410
---

[spaCy](https://spacy.io) は人気のある「産業強度」のNLPライブラリで、迅速かつ高精度なモデルを手間なく利用できます。spaCy v3からは、Weights & Biasesを[`spacy train`](https://spacy.io/api/cli#train)と共に使用することで、あなたのspaCyモデルのトレーニングメトリクスを追跡し、モデルとデータセットの保存とバージョン管理も可能になりました。そして、それには設定にほんの数行追加するだけです。

## サインアップしてAPIキーを作成

APIキーは、あなたのマシンをW&Bに認証します。ユーザープロフィールからAPIキーを生成できます。

{{% alert %}}
より簡潔な方法として、[https://wandb.ai/authorize](https://wandb.ai/authorize)に直接アクセスしてAPIキーを生成することができます。表示されたAPIキーをコピーし、パスワードマネージャーなど安全な場所に保存してください。
{{% /alert %}}

1. 右上のユーザープロフィールアイコンをクリック。
1. **ユーザー設定**を選択し、**APIキー**セクションまでスクロール。
1. **表示**をクリックし、表示されたAPIキーをコピーします。APIキーを非表示にするには、ページを再読み込みしてください。

## `wandb`ライブラリをインストールしてログイン

`wandb`ライブラリをローカルにインストールし、ログインするには:

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) をあなたのAPIキーに設定します。

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

1. `wandb`ライブラリをインストールしてログインします。

    ```shell
    pip install wandb

    wandb login
    ```

{{% /tab %}}

{{% tab header="Python" value="python" %}}

```bash
pip install wandb
```
```python
import wandb
wandb.login()
```

{{% /tab %}}

{{% tab header="Python notebook" value="notebook" %}}

```notebook
!pip install wandb

import wandb
wandb.login()
```

{{% /tab %}}
{{< /tabpane >}}

## `WandbLogger`をspaCyの設定ファイルに追加

spaCyの設定ファイルは、ロギングだけでなく、GPUの割り当て、オプティマイザーの選択、データセットのパスなど、トレーニングのすべての側面を指定するために使用されます。`[training.logger]`の下に、キー `@loggers` を `値` "spacy.WandbLogger.v3" で、さらに `project_name` を指定する必要があります。

{{% alert %}}
spaCyのトレーニング設定ファイルの仕組みや、トレーニングをカスタマイズするために渡すことができるその他のオプションについては、[spaCyのドキュメント](https://spacy.io/usage/training)を参照してください。
{{% /alert %}}

```python
[training.logger]
@loggers = "spacy.WandbLogger.v3"
project_name = "my_spacy_project"
remove_config_values = ["paths.train", "paths.dev", "corpora.train.path", "corpora.dev.path"]
log_dataset_dir = "./corpus"
model_log_interval = 1000
```

| 名前                   | 説明                                                                                                                                                                                                                                                   |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `project_name`         | `str`型。W&Bプロジェクトの名前。存在しない場合は、自動的にプロジェクトが作成されます。                                                                                                    |
| `remove_config_values` | `List[str]`型。W&Bにアップロードする前に設定から除外する値のリスト。デフォルトは`[]`です。                                                                                                                                                     |
| `model_log_interval`   | `Optional int`型。デフォルトは`None`です。設定すると、[モデルのバージョン管理]({{< relref path="/guides/core/registry/" lang="ja" >}})が[Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}})とともに有効になります。モデルチェックポイントをログに記録する間隔のステップ数を渡します。デフォルトは`None`です。 |
| `log_dataset_dir`      | `Optional str`型。パスを渡すと、トレーニング開始時にデータセットはArtifactsとしてアップロードされます。デフォルトは`None`です。                                                                                                            |
| `entity`               | `Optional str`型。指定した場合、run は指定したエンティティで作成されます。                                                                                                                                                                                   |
| `run_name`             | `Optional str`型。指定された場合、run は指定された名前で作成されます。                                                                                                                                                                               |

## トレーニングを開始

`WandbLogger`をspaCyのトレーニング設定に追加したら、通常通り `spacy train` を実行できます。

{{< tabpane text=true >}}

{{% tab header="Command Line" value="cli" %}}

```python
python -m spacy train \
    config.cfg \
    --output ./output \
    --paths.train ./train \
    --paths.dev ./dev
```

{{% /tab %}}

{{% tab header="Python" value="python" %}}

```python
python -m spacy train \
    config.cfg \
    --output ./output \
    --paths.train ./train \
    --paths.dev ./dev
```

{{% /tab %}}

{{% tab header="Python notebook" value="notebook" %}}

```notebook
!python -m spacy train \
    config.cfg \
    --output ./output \
    --paths.train ./train \
    --paths.dev ./dev
```

{{% /tab %}}
{{< /tabpane >}}

トレーニングが始まると、トレーニングrun の[W&Bページ]({{< relref path="/guides/models/track/runs/" lang="ja" >}})へのリンクが出力され、このrun の実験管理[ダッシュボード]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}})にWeights & BiasesのウェブUIでアクセスできます。