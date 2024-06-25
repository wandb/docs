---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# spaCy

[spaCy](https://spacy.io) は、人気のある「工業用強度」のNLPライブラリです。高速で高精度なモデルを、最小限の手間で提供します。spaCy v3以降では、Weights and Biases を [`spacy train`](https://spacy.io/api/cli#train) に使用して、spaCyモデルのトレーニングメトリクスを追跡したり、モデルやデータセットを保存およびバージョン管理したりすることができます。必要なのは設定ファイルに数行追加するだけです！

## 始めましょう: モデルを追跡して保存する

### 1. `wandb` ライブラリをインストールしてログイン

<Tabs
  defaultValue="cli"
  values={[
    {label: 'Command Line', value: 'cli'},
    {label: 'Notebook', value: 'notebook'},
  ]}>
  <TabItem value="cli">

```python
pip install wandb
wandb login
```

  </TabItem>
  <TabItem value="notebook">

```python
!pip install wandb

import wandb
wandb.login()
```

  </TabItem>
</Tabs>

### 2) `WandbLogger` をspaCyの設定ファイルに追加

spaCyの設定ファイルは、ログの記録だけでなく、GPUの割り当て、オプティマイザーの選択、データセットのパスなど、トレーニングのすべての側面を指定するために使用されます。最小限でも、`[training.logger]` に `@loggers` キーと 値 `"spacy.WandbLogger.v3"`、さらに `project_name` を指定する必要があります。

:::info
spaCyのトレーニング設定ファイルの詳細や、トレーニングをカスタマイズするために渡すことができるその他のオプションについては、[spaCyのドキュメント](https://spacy.io/usage/training) を参考にしてください。
:::

```python
[training.logger]
@loggers = "spacy.WandbLogger.v3"
project_name = "my_spacy_project"
remove_config_values = ["paths.train", "paths.dev", "corpora.train.path", "corpora.dev.path"]
log_dataset_dir = "./corpus"
model_log_interval = 1000
```

| 名前                   | 説明                                                                                                                                                                                                                                                 |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `project_name`         | `str`。Weights & Biases の [project](../app/pages/project-page.md) の名前です。プロジェクトはまだ存在しない場合、自動的に作成されます。                                                                                                    |
| `remove_config_values` | `List[str]` 。W&Bにアップロードする前に設定から除外する値のリスト。デフォルトは `[]` 。                                                                                                                                                   |
| `model_log_interval`   | `Optional int`。デフォルトは `None` 。設定すると、[model versioning](../model_registry/intro.md) と [Artifacts](../artifacts/intro.md) が有効になります。モデルチェックポイントを記録する間隔のステップ数を指定します。デフォルトは `None` 。   |
| `log_dataset_dir`      | `Optional str`。パスを渡すと、データセットがトレーニングの開始時にArtifactとしてアップロードされます。デフォルトは `None` 。                                                                                                    |
| `entity`               | `Optional str`。指定した場合、run は指定された entity に作成されます。                                                                                                                                                                                |
| `run_name`             | `Optional str`。指定した場合、run は指定された名前で作成されます。                                                                                                                                                                                    |

### 3) トレーニングを開始

`WandbLogger` をspaCyのトレーニング設定に追加したら、通常通り `spacy train` を実行できます。

<Tabs
  defaultValue="cli"
  values={[
    {label: 'Command Line', value: 'cli'},
    {label: 'Notebook', value: 'notebook'},
  ]}>
  <TabItem value="cli">

```python
python -m spacy train \
    config.cfg \
    --output ./output \
    --paths.train ./train \
    --paths.dev ./dev
```

  </TabItem>
  <TabItem value="notebook">

```python
!python -m spacy train \
    config.cfg \
    --output ./output \
    --paths.train ./train \
    --paths.dev ./dev
```

  </TabItem>
</Tabs>

トレーニングが始まると、トレーニングrunの [W&Bページ](../app/pages/run-page.md) へのリンクが表示され、このrunの実験管理ダッシュボードにアクセスすることができます。