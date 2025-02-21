---
title: spaCy
menu:
  default:
    identifier: ja-guides-integrations-spacy
    parent: integrations
weight: 410
---

[spaCy](https://spacy.io) は、人気のある「産業用強度」のNLPライブラリで、高速で正確なモデルを最小限の手間で提供します。spaCy v3から、Weights & Biasesは[`spacy train`](https://spacy.io/api/cli#train)と連携して、spaCyモデルのトレーニングメトリクスを管理したり、ModelsやDatasetsを保存およびバージョン管理することができます。そして、設定ファイルにいくつかの行を追加するだけで済みます。

## 登録とAPIキーの作成

APIキーは、あなたのマシンをW&Bに認証するためのものです。ユーザープロフィールからAPIキーを生成できます。

{{% alert %}}
より効率的な方法として、[https://wandb.ai/authorize](https://wandb.ai/authorize) に直接アクセスすることで、APIキーを生成できます。表示されたAPIキーをコピーし、パスワードマネージャーのような安全な場所に保存してください。
{{% /alert %}}

1. 右上のユーザーアイコンをクリック。
1. **ユーザー設定**を選択し、**API キー**セクションにスクロール。
1. **表示**をクリック。表示されたAPIキーをコピーします。APIキーを非表示にするには、ページをリロード。

## `wandb`ライブラリをインストールしてログイン

ローカルに`wandb`ライブラリをインストールし、ログインするには:

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` 環境変数をAPIキーに設定します。[環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}})を参照。

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

1. `wandb`ライブラリをインストールし、ログインします。

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

## `WandbLogger` をspaCy設定ファイルに追加

spaCyの設定ファイルはトレーニングのあらゆる側面を指定するためのものです。ロギングだけでなく、GPUの割り当て、オプティマイザーの選択、データセットのパスなども指定します。最低限、`[training.logger]` の下に、キー `@loggers` を `値` `"spacy.WandbLogger.v3"` として指定し、加えて `project_name` を設定してください。

{{% alert %}}
spaCyのトレーニング設定ファイルの詳細およびトレーニングのカスタマイズに渡すことができる他のオプションについては、[spaCyのドキュメント](https://spacy.io/usage/training)を参照してください。
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
| `project_name`         | `str`。W&B Projectの名前。プロジェクトは存在しない場合自動的に作成されます。                                                                                                    |
| `remove_config_values` | `List[str]`。ConfigからW&Bにアップロードされる前に除外する値のリスト。デフォルトは`[]`。                                                                                                                                                     |
| `model_log_interval`   | `Optional int`。デフォルトは`None`。設定されている場合、[モデルのバージョン管理]({{< relref path="/guides/models/registry/model_registry/" lang="ja" >}})は[Artifacts]({{< relref path="/guides/core/artifacts/" lang="ja" >}})で有効になります。モデルのチェックポイントをログする間隔のステップ数を指定します。デフォルトは`None`。 |
| `log_dataset_dir`      | `Optional str`。パスが指定された場合、データセットはトレーニングの開始時にArtifactとしてアップロードされます。デフォルトは`None`。                                                                                                            |
| `entity`               | `Optional str`。指定された場合、このentityでrunが作成されます。                                                                                                                                                                                   |
| `run_name`             | `Optional str`。指定された場合、runはその名前で作成されます。                                                                                                                                                                               |

## トレーニングの開始

`WandbLogger`をspaCyトレーニング設定に追加したら、通常通り`spacy train`を実行できます。

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

トレーニングが始まると、トレーニングrunの [W&Bページ]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) へのリンクが出力されます。これは、このrunの実験管理[ダッシュボード]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}}) へと導くリンクです。これにより、Weights & Biasesのweb UIでrunを確認できます。