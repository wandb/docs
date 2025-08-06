---
title: spaCy
menu:
  default:
    identifier: spacy
    parent: integrations
weight: 410
---

[spaCy](https://spacy.io) は、人気のある "industrial-strength" NLP ライブラリです。高速かつ高精度なモデルを、シンプルに扱うことができます。spaCy v3 からは、W&B を [`spacy train`](https://spacy.io/api/cli#train) と組み合わせて、spaCy モデルのトレーニングメトリクスの記録や、モデル・データセットの保存・バージョン管理を行えるようになりました。しかも、設定ファイルをほんの少し追加するだけで簡単に利用が開始できます。

## サインアップとAPIキーの作成

APIキーは、あなたのマシンを W&B に認証するためのものです。APIキーはご自身のユーザープロフィールから生成できます。

{{% alert %}}
より簡単な方法として、[W&B 認証ページ](https://wandb.ai/authorize) に直接アクセスして APIキーを生成することもできます。表示された APIキーをコピーし、パスワードマネージャーなど安全な場所に保存してください。
{{% /alert %}}

1. 右上のユーザープロフィールアイコンをクリックします。
1. **User Settings** を選択し、**API Keys** セクションまでスクロールします。
1. **Reveal** をクリックして、表示された APIキーをコピーします。APIキーを非表示にしたい場合はページを再読み込みしてください。

## `wandb` ライブラリのインストールとログイン

ローカル環境に `wandb` ライブラリをインストールし、ログインしましょう。

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` の[環境変数]({{< relref "/guides/models/track/environment-variables.md" >}}) に、あなたの APIキーを設定します。

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

1. `wandb` ライブラリをインストールし、ログインします。


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

spaCy の config ファイルでは、ロギングだけでなく、GPU 割り当てやオプティマイザー選択、データセットパスなど、トレーニングに関するあらゆる設定を行います。最低限としては、`[training.logger]` の下で `@loggers` キーに `"spacy.WandbLogger.v3"` を、そして `project_name` を追加してください。

{{% alert %}}
spaCy のトレーニング設定ファイルの詳細や、トレーニングをカスタマイズするためのその他オプションについては、[spaCy のドキュメント](https://spacy.io/usage/training) をご参照ください。
{{% /alert %}}

```python
[training.logger]
@loggers = "spacy.WandbLogger.v3"
project_name = "my_spacy_project"
remove_config_values = ["paths.train", "paths.dev", "corpora.train.path", "corpora.dev.path"]
log_dataset_dir = "./corpus"
model_log_interval = 1000
```

| 名前                   | 説明                                                                                                                                                                                                                                                    |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `project_name`         | `str`。W&B Project の名前です。Project がまだ存在しない場合、自動的に作成されます。                                                                            |
| `remove_config_values` | `List[str]` 。W&B にアップロードする前に config から除外したい設定値のリストです。デフォルトは `[]`。                                                                                                   |
| `model_log_interval`   | `Optional int`。デフォルトは `None`。設定すると [model versioning]({{< relref "/guides/core/registry/" >}}) が [Artifacts]({{< relref "/guides/core/artifacts/" >}}) で有効になります。モデルチェックポイントを記録する間隔（ステップ数）を指定してください。デフォルトは `None`。|
| `log_dataset_dir`      | `Optional str`。パスを設定すると、データセットをトレーニング開始時に Artifact としてアップロードします。デフォルトは `None`。                                                                   |
| `entity`               | `Optional str`。指定すると、run がその entity で作成されます。                                                                                                                                                  |
| `run_name`             | `Optional str`。指定すると、その run に名前が付けられます。                                                                                                                                                |

## トレーニングの開始

`WandbLogger` を spaCy のトレーニング設定ファイルに追加したら、いつも通り `spacy train` を実行できます。

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

トレーニングが始まると、あなたのトレーニング run の [W&B ページ]({{< relref "/guides/models/track/runs/" >}}) へのリンクが表示されます。このリンクから、この run の実験管理 [ダッシュボード]({{< relref "/guides/models/track/workspaces.md" >}})（W&B の Web UI 内）にアクセスできます。