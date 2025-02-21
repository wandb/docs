---
title: spaCy
menu:
  default:
    identifier: ja-guides-integrations-spacy
    parent: integrations
weight: 410
---

[spaCy](https://spacy.io) は、"産業レベル" の NLP ライブラリとして人気があります。高速かつ正確なモデルを最小限の手間で利用できます。spaCy v3 以降、Weights & Biases を [`spacy train`](https://spacy.io/api/cli#train) で使用して、spaCy モデルのトレーニングメトリクスを追跡したり、モデルやデータセットを保存してバージョン管理したりできるようになりました。必要な作業は、設定に数行追加するだけです。

## サインアップして API キーを作成する

API キーは、お使いのマシンを W&B に対して認証します。API キーは、ユーザープロフィールから生成できます。

{{% alert %}}
より効率的な方法として、[https://wandb.ai/authorize](https://wandb.ai/authorize) に直接アクセスして API キーを生成することもできます。表示された API キーをコピーして、パスワードマネージャーなどの安全な場所に保存してください。
{{% /alert %}}

1. 右上隅にあるユーザープロフィールアイコンをクリックします。
2. **User Settings** を選択し、**API Keys** セクションまでスクロールします。
3. **Reveal** をクリックします。表示された API キーをコピーします。API キーを非表示にするには、ページをリロードしてください。

## `wandb` ライブラリをインストールしてログインする

`wandb` ライブラリをローカルにインストールしてログインするには:

{{< tabpane text=true >}}
{{% tab header="コマンドライン" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) を API キーに設定します。

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

2. `wandb` ライブラリをインストールしてログインします。

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

## `WandbLogger` を spaCy 設定ファイルに追加する

spaCy の設定ファイルは、ロギングだけでなく、GPU の割り当て、オプティマイザーの選択、データセットのパスなど、トレーニングのあらゆる側面を指定するために使用されます。最小限、`[training.logger]` の下に、`@loggers` キーに `"spacy.WandbLogger.v3"` の値と、`project_name` を指定する必要があります。

{{% alert %}}
spaCy のトレーニング設定ファイルの仕組みや、トレーニングをカスタマイズするために渡すことができるその他のオプションの詳細については、[spaCy のドキュメント](https://spacy.io/usage/training) を参照してください。
{{% /alert %}}

```python
[training.logger]
@loggers = "spacy.WandbLogger.v3"
project_name = "my_spacy_project"
remove_config_values = ["paths.train", "paths.dev", "corpora.train.path", "corpora.dev.path"]
log_dataset_dir = "./corpus"
model_log_interval = 1000
```

| Name                   | Description                                                                                                                                                                                                                                                   |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `project_name`         | `str`. W&B の Project の名前。Project がまだ存在しない場合は、自動的に作成されます。                                                                                                                                                                                             |
| `remove_config_values` | `List[str]` 。設定から除外する値のリストで、W&B にアップロードする前に除外されます。デフォルトは `[]` です。                                                                                                                                                     |
| `model_log_interval`   | `Optional int`。デフォルトは `None` です。設定すると、[Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) による [モデルのバージョン管理]({{< relref path="/guides/models/registry/model_registry/" lang="ja" >}}) が有効になります。モデルのチェックポイントのログ記録の間隔をステップ数で渡します。デフォルトは `None` です。 |
| `log_dataset_dir`      | `Optional str`。パスを渡すと、トレーニングの開始時にデータセットが Artifacts としてアップロードされます。デフォルトは `None` です。                                                                                                                                                                       |
| `entity`               | `Optional str` 。渡された場合、run は指定された entity に作成されます。                                                                                                                                                                                   |
| `run_name`             | `Optional str` 。指定された場合、run は指定された名前で作成されます。                                                                                                                                                                               |

## トレーニングを開始する

`WandbLogger` を spaCy のトレーニング設定に追加したら、通常どおり `spacy train` を実行できます。

{{< tabpane text=true >}}

{{% tab header="コマンドライン" value="cli" %}}

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

トレーニングが開始されると、トレーニング run の [W&B page]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) へのリンクが出力されます。このリンクをクリックすると、Weights & Biases Web UI でこの run の 実験管理 [ダッシュボード]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}}) に移動します。
