---
title: spaCy
menu:
  default:
    identifier: ja-guides-integrations-spacy
    parent: integrations
weight: 410
---

[spaCy](https://spacy.io) は、人気のある「産業用強度」の NLP ライブラリであり、高速かつ正確なモデルを最小限の手間で実現します。spaCy v3 以降、Weights & Biases を [`spacy train`](https://spacy.io/api/cli#train) と共に使用して、spaCy モデルのトレーニング メトリクスを追跡したり、モデルとデータセットを保存およびバージョン管理したりできるようになりました。必要な作業は、設定に数行追加するだけです。

## サインアップして APIキー を作成する

APIキー は、お使いのマシンを W&B に対して認証します。APIキー は、[ユーザープロフィール](https://wandb.ai/settings)から生成できます。

{{% alert %}}
より効率的なアプローチとして、[https://wandb.ai/authorize](https://wandb.ai/authorize) に直接アクセスして APIキー を生成できます。表示された APIキー をコピーして、パスワードマネージャーなどの安全な場所に保存してください。
{{% /alert %}}

1. 右上隅にあるユーザープロフィールアイコンをクリックします。
2. **ユーザー設定** を選択し、**APIキー** セクションまでスクロールします。
3. **公開** をクリックします。表示された APIキー をコピーします。APIキー を非表示にするには、ページをリロードしてください。

## `wandb` ライブラリをインストールしてログインする

ローカルに `wandb` ライブラリをインストールしてログインするには:

{{< tabpane text=true >}}
{{% tab header="コマンドライン" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) を APIキー に設定します。

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

spaCy の設定ファイルは、ロギングだけでなく、トレーニングのあらゆる側面（GPU 割り当て、オプティマイザー の選択、データセット のパスなど）を指定するために使用されます。最小限の構成として、`[training.logger]` の下に、キー `@loggers` に値 `"spacy.WandbLogger.v3"` と `project_name` を指定する必要があります。

{{% alert %}}
spaCy のトレーニング設定ファイルの仕組みと、トレーニングをカスタマイズするために渡すことができるその他のオプションの詳細については、[spaCy のドキュメント](https://spacy.io/usage/training) を参照してください。
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
| `project_name`         | `str` 。W&B の Project の名前。まだ存在しない場合、Project は自動的に作成されます。                                                                                                                                                                                                |
| `remove_config_values` | `List[str]` 。W&B にアップロードする前に、設定から除外する値のリスト。デフォルトは `[]` です。                                                                                                                                                                                                        |
| `model_log_interval`   | `Optional int`。デフォルトは `None`。設定すると、[モデルのバージョン管理]({{< relref path="/guides/core/registry/" lang="ja" >}})が [Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) で有効になります。モデルのチェックポイントのロギング間隔までのステップ数を渡します。デフォルトは `None` です。 |
| `log_dataset_dir`      | `Optional str`。パスを渡すと、トレーニングの開始時にデータセット が Artifacts としてアップロードされます。デフォルトは `None` です。                                                                                                                                                               |
| `entity`               | `Optional str` 。渡された場合、run は指定された entity に作成されます                                                                                                                                                                                                                       |
| `run_name`             | `Optional str` 。指定された場合、run は指定された名前で作成されます。                                                                                                                                                                                                                         |

## トレーニングを開始する

`WandbLogger` を spaCy トレーニング設定に追加したら、通常どおり `spacy train` を実行できます。

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

トレーニングが開始されると、トレーニング run の [W&B ページ]({{< relref path="/guides/models/track/runs/" lang="ja" >}})へのリンクが出力されます。このリンクをクリックすると、Weights & Biases Web UI で、この run の 実験管理 [ダッシュボード]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}})に移動します。
