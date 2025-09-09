---
title: spaCy
menu:
  default:
    identifier: ja-guides-integrations-spacy
    parent: integrations
weight: 410
---

[spaCy](https://spacy.io) は「産業レベル」の人気 NLP ライブラリです。手間を最小限に、速く正確なモデルを扱えます。spaCy v3 以降、W&B は [`spacy train`](https://spacy.io/api/cli#train) と連携して、spaCy モデルのトレーニング メトリクスを実験管理したり、モデルやデータセットを保存・バージョン管理したりできます。必要なのは設定に数行追加するだけです。

## サインアップして APIキー を作成

APIキー は、あなたのマシンを W&B に対して認証するためのものです。APIキー は ユーザープロフィール から作成できます。

{{% alert %}}
より手早く設定するには、[W&B authorization page](https://wandb.ai/authorize) に直接アクセスして APIキー を発行できます。表示された APIキー をコピーし、パスワードマネージャーなどの安全な場所に保存してください。
{{% /alert %}}

1. 右上のユーザープロフィール アイコンをクリックします。
1. **User Settings** を選び、**API Keys** セクションまでスクロールします。
1. **Reveal** をクリックします。表示された APIキー をコピーします。APIキー を隠すにはページを再読み込みします。

## `wandb` ライブラリをインストールしてログイン

ローカルに `wandb` ライブラリをインストールしてログインするには:

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) にあなたの APIキー を設定します。

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

1. `wandb` ライブラリをインストールしてログインします。



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

## spaCy の config ファイルに `WandbLogger` を追加

spaCy の config ファイルは、ログ記録だけでなく、トレーニングのあらゆる側面を指定します。GPU の割り当て、オプティマイザーの選択、データセットのパスなどです。最小限の設定として、`[training.logger]` セクションでキー `@loggers` に `"spacy.WandbLogger.v3"` を設定し、さらに `project_name` を指定します。 

{{% alert %}}
spaCy のトレーニング用 config ファイルの仕組みや、トレーニングをカスタマイズするために渡せる他のオプションについては、[spaCy のドキュメント](https://spacy.io/usage/training)を参照してください。
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
| `project_name`         | `str`。W&B の Project 名。Project が存在しない場合は自動で作成されます。                                                                                                    |
| `remove_config_values` | `List[str]`。W&B にアップロードする前に config から除外する値のリスト。デフォルトは `[]`。                                                                                                                                                     |
| `model_log_interval`   | `Optional int`。デフォルトは `None`。設定すると、[モデルのバージョン管理]({{< relref path="/guides/core/registry/" lang="ja" >}})が [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) で有効になります。モデルのチェックポイントをログに記録する間隔（ステップ数）を指定します。デフォルトは `None`。 |
| `log_dataset_dir`      | `Optional str`。パスを指定すると、トレーニング開始時にデータセットを Artifact としてアップロードします。デフォルトは `None`。                                                                                                            |
| `entity`               | `Optional str`。指定した entity に run が作成されます。                                                                                                                                                                                   |
| `run_name`             | `Optional str`。指定した名前で run が作成されます。                                                                                                                                                                               |

## トレーニングを開始

`WandbLogger` を spaCy のトレーニング用 config に追加したら、通常どおり `spacy train` を実行できます。

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

トレーニングが始まると、あなたのトレーニング run の [W&B ページ]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) へのリンクが出力されます。リンク先では、W&B の Web UI 上にあるこの run の実験管理 [ダッシュボード]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}}) を確認できます。