---
title: spaCy
menu:
  default:
    identifier: ja-guides-integrations-spacy
    parent: integrations
weight: 410
---

[spaCy](https://spacy.io) は、人気のある「産業グレード」のNLPライブラリで、手間をかけずに高速かつ高精度なモデルを提供します。spaCy v3 からは、W&B を [`spacy train`](https://spacy.io/api/cli#train) コマンドと組み合わせて、spaCy モデルのトレーニングメトリクスの追跡や、モデル・データセットの保存とバージョン管理ができるようになりました。設定ファイルに数行追加するだけで利用できます。

## サインアップして APIキー を作成する

APIキー は、お使いのマシンを W&B と認証するために使用されます。ユーザープロフィールから APIキー を生成できます。

{{% alert %}}
より簡単な方法として、[W&B 認証ページ](https://wandb.ai/authorize) に直接アクセスして APIキー を生成できます。表示された APIキー をコピーし、パスワード管理ツールなど安全な場所に保存してください。
{{% /alert %}}

1. 画面右上のユーザープロフィールアイコンをクリックします。
1. **User Settings** を選択し、**API Keys** セクションまでスクロールします。
1. **Reveal** をクリックして表示された APIキー をコピーします。APIキー を非表示にするには、ページをリロードしてください。

## `wandb` ライブラリをインストールしてログインする

`wandb` ライブラリをローカルにインストールし、ログインする手順は以下の通りです。

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) に APIキー を設定します。

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

## spaCy の config ファイルに `WandbLogger` を追加する

spaCy の config ファイルでは、ロギングだけでなく、GPU割り当てやオプティマイザーの選択、データセットのパスなど、トレーニングに関わる全ての設定を指定します。基本的には、`[training.logger]` セクションで、キー `@loggers` に `"spacy.WandbLogger.v3"` を、加えて `project_name` を設定すれば十分です。

{{% alert %}}
spaCy のトレーニング設定ファイルの仕組みや、他にも指定できるオプションについては [spaCy の公式ドキュメント](https://spacy.io/usage/training) をご覧ください。
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
| `project_name`         | `str`。W&B Project の名前です。まだ存在しない場合は自動で作成されます。                                                                                                    |
| `remove_config_values` | `List[str]`。W&B にアップロードする前に config から除外する値のリスト。デフォルトは `[]`。                                                                                                                                                     |
| `model_log_interval`   | `Optional int`。デフォルトは `None`。設定した場合は [model versioning]({{< relref path="/guides/core/registry/" lang="ja" >}}) が [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) とともに有効になります。モデルのチェックポイントをログする間隔（ステップ数）を指定します。デフォルトは `None`。 |
| `log_dataset_dir`      | `Optional str`。パスを指定すると、トレーニング開始時にデータセットが Artifact としてアップロードされます。デフォルトは `None`。                                                                                                            |
| `entity`               | `Optional str`。指定した場合、その Entity に run が作成されます。                                                                                                                                                                                   |
| `run_name`             | `Optional str`。指定した名前で run を作成します。                                                                                                                                                                               |

## トレーニングを開始する

`WandbLogger` を spaCy のトレーニング config に追加したら、通常通り `spacy train` を実行できます。

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

トレーニングが開始されると、トレーニング run の [W&B ページ]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) へのリンクが表示され、W&B のウェブUIからこの run の実験管理 [ダッシュボード]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}}) へアクセスできます。