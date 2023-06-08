---
displayed_sidebar: ja
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# spaCy

[spaCy](https://spacy.io)は、高速で正確なモデルを簡単に扱える「産業強度」のNLPライブラリです。spaCy v3以降では、Weights and Biasesを[`spacy train`](https://spacy.io/api/cli#train)と組み合わせて使用することで、spaCyモデルのトレーニングメトリクスを追跡したり、モデルとデータセットを保存・バージョン管理することができます。そして、設定に数行を追加するだけで実現できます！

## はじめに：モデルのトラッキングと保存

### 1. `wandb` ライブラリをインストールしてログイン

<Tabs
  defaultValue="cli"
  values={[
    {label: 'コマンドライン', value: 'cli'},
    {label: 'ノートブック', value: 'notebook'},
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

### 2) spaCy設定ファイルに`WandbLogger`を追加

spaCyの設定ファイルは、ロギングだけでなく、トレーニングのすべての側面を指定するために使用されます。GPU割り当て、オプティマイザーの選択、データセットのパスなどです。最小限、`[training.logger]`の下で、キー`@loggers`に値`"spacy.WandbLogger.v3"`と、`project_name`を指定する必要があります。

:::info
spaCyトレーニング設定ファイルの機能や、トレーニングをカスタマイズするために渡すことができる他のオプションについては、[spaCyのドキュメント](https://spacy.io/usage/training)をご覧ください。
:::

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
| `project_name`         | `str`. Weights & Biasesの[プロジェクト](../app/pages/project-page.md)の名前。プロジェクトがまだ存在しない場合は自動的に作成されます。                                                                                                    |
| `remove_config_values` | `List[str]` . 設定から除外する値のリスト。初期値は `[]`です。                                                                                                                                                     |
| `model_log_interval`   | `Optional int`. 初期値は`None`です。設定された場合、[モデルのバージョン管理](../models/intro.md)と[アーティファクト](../artifacts/intro.md)が有効化されます。モデルチェックポイントのロギング間隔を設定してください。初期値は`None`です。 |
| `log_dataset_dir`      | `Optional str`. パスが指定されている場合、トレーニング開始時にデータセットがアーティファクトとしてアップロードされます。初期値は`None`です。                                                                                                            |
| `entity`               | `Optional str` . 指定された場合、特定のエンティティでrunが作成されます。                                                                                                                                                                                   |
| `run_name`             | `Optional str` . 指定された場合、指定された名前でrunが作成されます。                                                                                                                                                                               |
### 3) トレーニングを開始する

`WandbLogger`をspaCyトレーニング設定に追加したら、通常通り`spacy train`を実行できます。

<Tabs
  defaultValue="cli"
  values={[
    {label: 'コマンドライン', value: 'cli'},
    {label: 'ノートブック', value: 'notebook'},
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

トレーニングが始まると、トレーニングrunの[W&Bページ](../app/pages/run-page.md)へのリンクが出力され、このリンクからWeights & Biases Web UIの実験トラッキング[ダッシュボード](../track/app.md)にアクセスできるようになります。