---
description: Sweeps の設定ファイルを作成する方法を学びましょう。
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Sweep configuration structure

<head>
  <title>ハイパーパラメータチューニングのためのスウィープ設定を定義する。</title>
</head>

W&B Sweepは、ハイパーパラメータの値を探索するための戦略と、それを評価するコードを組み合わせたものです。戦略はすべてのオプションを試す簡単なものから、ベイズ最適化やHyperband（[BOHB](https://arxiv.org/abs/1807.01774)）のような複雑なものまで幅広く対応しています。

Sweep設定は、[Python辞書](https://docs.python.org/3/tutorial/datastructures.html#dictionaries)または[YAML](https://yaml.org/)ファイルで定義します。どちらの方法を選ぶかは、どのようにSweepを管理したいかによります。

:::info
スウィープ設定をYAMLファイルで定義すると、コマンドラインからスウィープエージェントを初期化して開始することができます。PythonスクリプトやJupyterノートブック内で完全にスウィープを初期化して開始したい場合は、Python辞書でスウィープを定義します。
:::

次のガイドでは、スウィープ設定をどのようにフォーマットするかについて説明します。トップレベルのスウィープ設定キーの包括的なリストについては、[Sweep configuration options](./sweep-config-keys.md)を参照してください。

## 基本構造

両方のスウィープ設定フォーマットオプション（YAMLとPython辞書）は、キーと値のペアおよびネストされた構造を使用します。

スウィープ設定内のトップレベルキーを使用して、スウィープの検索の特性を定義します。例えば、スウィープの名前（[`name`](./sweep-config-keys.md#name)キー）、検索するパラメーター（[`parameters`](./sweep-config-keys.md#parameters)キー）、パラメーター空間の検索方法（[`method`](./sweep-config-keys.md#method)キー）などです。

例えば、以下のコードスニペットは、同じスウィープ設定をYAMLファイル内とPython辞書内で定義しています。スウィープ設定には、`program`、`name`、`method`、`metric`、`parameters`の五つのトップレベルキーが指定されています。

<Tabs
  defaultValue="cli"
  values={[    
    {label: 'CLI', value: 'cli'},
    {label: 'Python script or Jupyter notebook', value: 'script'},
  ]}>
  <TabItem value="script">

PythonスクリプトやJupyterノートブックでトレーニングアルゴリズムを定義する場合は、Python辞書データ構造でスウィープを定義します。

以下のコードスニペットは、`sweep_configuration`という変数にスウィープ設定を保持しています：

```python title="train.py"
sweep_configuration = {
    "name": "sweepdemo",
    "method": "bayes",
    "metric": {"goal": "minimize", "name": "validation_loss"},
    "parameters": {
        "learning_rate": {"min": 0.0001, "max": 0.1},
        "batch_size": {"values": [16, 32, 64]},
        "epochs": {"values": [5, 10, 15]},
        "optimizer": {"values": ["adam", "sgd"]},
    },
}
```
  </TabItem>
  <TabItem value="cli">
スウィープをコマンドライン（CLI）からインタラクティブに管理したい場合は、YAMLファイルにスウィープ設定を定義します。

```yaml title="config.yaml"
program: train.py
name: sweepdemo
method: bayes
metric:
  goal: minimize
  name: validation_loss
parameters:
  learning_rate:
    min: 0.0001
    max: 0.1
  batch_size:
    values: [16, 32, 64]
  epochs:
    values: [5, 10, 15]
  optimizer:
    values: ["adam", "sgd"]
```
  </TabItem>
</Tabs>

トップレベルの`parameters`キー内には、`learning_rate`、`batch_size`、`epoch`、`optimizer`というキーがネストされています。指定するネストされた各キーについては、1つ以上の値、分布、確率などを提供できます。詳細については、[Sweep configuration options](./sweep-config-keys.md)の[parameters](./sweep-config-keys.md#parameters)セクションを参照してください。


## 二重にネストされたパラメータ

スウィープ設定はネストされたパラメータをサポートしています。ネストされたパラメータを区別するには、トップレベルのパラメータ名の下に追加の `parameters` キーを使用します。スウィープ設定は多層ネストをサポートしています。

ベイズやランダムハイパーパラメータ検索を使用する場合は、ランダム変数の確率分布を指定します。各ハイパーパラメータについて：

1. スウィープ設定にトップレベルの `parameters` キーを作成します。
2. `parameters` キー内に次の内容をネストします：
   1. 最適化したいハイパーパラメータの名前を指定します。 
   2. `distribution` キーのために使用したい分布を指定します。`distribution` キー値ペアをハイパーパラメータ名の下にネストします。
   3. 探索する1つ以上の値を指定します。値（または値の複数）は、`distribution` キーに対応するものとします。  
      1. （オプション） トップレベルのパラメータ名の下に追加の `parameters` キーを使用して、ネストされたパラメータを区別します。

:::caution
スウィープ設定で定義されたネストされたパラメータは、W&B run 設定で指定されたキーを上書きします。

例えば、`train.py` Pythonスクリプトで次の設定を使用してW&B runを初期化したとします（1-2行）。次に、辞書型のスウィープ設定` sweep_configuration `を定義します（4-13行）。そして、スウィープ設定辞書を`sweep`メソッドに渡してスウィープ設定を初期化します（16行）。

```python title="train.py" showLineNumbers
def main():
    run = wandb.init(config={"nested_param": {"manual_key": 1}})


sweep_configuration = {
    "top_level_param": 0,
    "nested_param": {
        "learning_rate": 0.01,
        "double_nested_param": {"x": 0.9, "y": 0.8},
    },
}

# 設定を渡してスウィープを初期化
sweep_id = wandb.sweep(sweep=sweep_configuration, project="<project>")

# スウィープジョブを開始
wandb.agent(sweep_id, function=main, count=4)
```
W&B runが初期化されるときに渡される`nested_param.manual_key`（2行目）はアクセスできません。`run.config`はスウィープ設定辞書で定義されたキー値ペア（4-13行）しか持ちません。
:::

## スウィープ設定テンプレート

以下のテンプレートは、パラメータを設定し、検索制約を指定する方法を示しています。`hyperparameter_name`をハイパーパラメータの名前に、`<>`で囲まれた値を任意の値に置き換えます。

```yaml title="config.yaml"
program: <insert>
method: <insert>
parameter:
  hyperparameter_name0:
    value: 0  
  hyperparameter_name1: 
    values: [0, 0, 0]
  hyperparameter_name: 
    distribution: <insert>
    value: <insert>
  hyperparameter_name2:  
    distribution: <insert>
    min: <insert>
    max: <insert>
    q: <insert>
  hyperparameter_name3: 
    distribution: <insert>
    values:
      - <list_of_values>
      - <list_of_values>
      - <list_of_values>
early_terminate:
  type: hyperband
  s: 0
  eta: 0
  max_iter: 0
command:
- ${Command macro}
- ${Command macro}
- ${Command macro}
- ${Command macro}      
```

## スウィープ設定例

<Tabs
  defaultValue="cli"
  values={[
    {label: 'CLI', value: 'cli'},
    {label: 'Python script or Jupyter notebook', value: 'notebook'},
  ]}>
  <TabItem value="cli">

```yaml title="config.yaml" 
program: train.py
method: random
metric:
  goal: minimize
  name: loss
parameters:
  batch_size:
    distribution: q_log_uniform_values
    max: 256 
    min: 32
    q: 8
  dropout: 
    values: [0.3, 0.4, 0.5]
 epochs:
    value: 1
  fc_layer_size: 
    values: [128, 256, 512]
  learning_rate:
    distribution: uniform
    max: 0.1
    min: 0
  optimizer:
    values: ["adam", "sgd"]
```

  </TabItem>
  <TabItem value="notebook">

```python title="train.py" 
sweep_config = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "loss"},
    "parameters": {
        "batch_size": {
            "distribution": "q_log_uniform_values",
            "max": 256,
            "min": 32,
            "q": 8,
        },
        "dropout": {"values": [0.3, 0.4, 0.5]},
        "epochs": {"value": 1},
        "fc_layer_size": {"values": [128, 256, 512]},
        "learning_rate": {"distribution": "uniform", "max": 0.1, "min": 0},
        "optimizer": {"values": ["adam", "sgd"]},
    },
}
```

  </TabItem>
</Tabs>

### ベイズハイパーバンドの例
```yaml
program: train.py
method: bayes
metric:
  goal: minimize
  name: val_loss
parameters:
  dropout:
    values: [0.15, 0.2, 0.25, 0.3, 0.4]
  hidden_layer_size:
    values: [96, 128, 148]
  layer_1_size:
    values: [10, 12, 14, 16, 18, 20]
  layer_2_size:
    values: [24, 28, 32, 36, 40, 44]
  learn_rate:
    values: [0.001, 0.01, 0.003]
  decay:
    values: [1e-5, 1e-6, 1e-7]
  momentum:
    values: [0.8, 0.9, 0.95]
  epochs:
    value: 27
early_terminate:
  type: hyperband
  s: 2
  eta: 3
  max_iter: 27
```

次のタブは、`early_terminate`のために最小または最大のイテレーション数を指定する方法を示しています：

<Tabs
  defaultValue="min_iter"
  values={[
    {label: 'Minimum number of iterations specified', value: 'min_iter'},
    {label: 'Maximum number of iterations specified', value: 'max_iter'},
  ]}>
  <TabItem value="min_iter">

```yaml
early_terminate:
  type: hyperband
  min_iter: 3
```

この例の括弧は、 `[3, 3*eta, 3*eta*eta, 3*eta*eta*eta]` となり、 `[3, 9, 27, 81]` となります。
  </TabItem>
  <TabItem value="max_iter">

```yaml
early_terminate:
  type: hyperband
  max_iter: 27
  s: 2
```

この例の括弧は、 `[27/eta, 27/eta/eta]` となり、 `[9, 3]` となります。
  </TabItem>
</Tabs>

### コマンドの例
```yaml
program: main.py
metric:
  name: val_loss
  goal: minimize

method: bayes
parameters:
  optimizer.config.learning_rate:
    min: !!float 1e-5
    max: 0.1
  experiment:
    values: [expt001, expt002]
  optimizer:
    values: [sgd, adagrad, adam]

command:
- ${env}
- ${interpreter}
- ${program}
- ${args_no_hyphens}
```

<Tabs
  defaultValue="unix"
  values={[
    {label: 'Unix', value: 'unix'},
    {label: 'Windows', value: 'windows'},
  ]}>
  <TabItem value="unix">

```bash
/usr/bin/env python train.py --param1=value1 --param2=value2
```
  </TabItem>
  <TabItem value="windows">

```bash
python train.py --param1=value1 --param2=value2
```
  </TabItem>
</Tabs>

次のタブは、一般的なコマンドマクロの指定方法を示しています：

<Tabs
  defaultValue="python"
  values={[
    {label: 'Set python interpreter', value: 'python'},
    {label: 'Add extra parameters', value: 'parameters'},
    {label: 'Omit arguments', value: 'omit'},
    {label: 'Hydra', value: 'hydra'}
  ]}>
  <TabItem value="python">

`{$interpreter}`マクロを削除し、Pythonインタープリタをハードコードするために値を明示的に提供します。例えば、次のコードスニペットはその方法を示しています：

```yaml
command:
  - ${env}
  - python3
  - ${program}
  - ${args}
```
  </TabItem>
  <TabItem value="parameters">

次の例は、スウィープ設定パラメータで指定されていない追加のコマンドライン引数を追加する方法を示しています：

```
command:
  - ${env}
  - ${interpreter}
  - ${program}
  - "--config"
  - "your-training-config.json"
  - ${args}
```

  </TabItem>
  <TabItem value="omit">

プログラムが引数解析を使用しない場合は、引数の受け渡しを完全に回避し、`wandb.init`がスウィープパラメータを`wandb.config`に自動的に取り入れるのを活用できます：

```
command:
  - ${env}
  - ${interpreter}
  - ${program}
```
  </TabItem>
  <TabItem value="hydra">

[Hydra](https://hydra.cc)のようなツールが期待する引数の渡し方にコマンドを変更できます。詳細については、[Hydra with W&B](../integrations/other/hydra.md)を参照してください。

```
command:
  - ${env}
  - ${interpreter}
  - ${program}
  - ${args_no_hyphens}
```
  </TabItem>
</Tabs>