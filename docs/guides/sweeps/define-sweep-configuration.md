---
description: スイープの設定ファイルを作成する方法を学びます。
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# スイープ設定の構造

<head>
  <title>ハイパーパラメータチューニングのためのスイープ設定を定義する</title>
</head>

W&B スイープは、ハイパーパラメータの値を探索する戦略と、それらを評価するコードを組み合わせたものです。戦略は単純なすべてのオプションを試すものから、ベイズ最適化とハイパーバンド（[BOHB](https://arxiv.org/abs/1807.01774))のような複雑なものまで多岐にわたります。

スイープ設定を [Pythonの辞書](https://docs.python.org/3/tutorial/datastructures.html#dictionaries) または [YAML](https://yaml.org/) ファイルのいずれかで定義します。どの形式で定義するかは、スイープをどのように管理したいかによります。

:::info
スイープをコマンドラインから初期化し、スイープエージェントを開始したい場合は、スイープ設定を YAML ファイルで定義します。Python スクリプトや Jupyter ノートブック内でスイープを初期化し、スイープを開始したい場合は、Pythonの辞書でスイープを定義します。
:::

以下のガイドでは、スイープ設定のフォーマット方法について説明します。詳しいスイープ設定キーの一覧は [スイープ設定オプション](./sweep-config-keys.md) を参照してください。

## 基本構造

YAMLとPythonの辞書形式のどちらのスイープ設定も、キーと値のペアやネストされた構造を利用します。

スイープ設定内のトップレベルキーを使用して、スイープ検索の特性を定義します。例えば、スイープの名前（[`name`](./sweep-config-keys.md#name) キー）、検索するパラメーター（[`parameters`](./sweep-config-keys.md#parameters) キー）、パラメータ空間を検索する方法（[`method`](./sweep-config-keys.md#method) キー）などです。

例として、以下のコードスニペットは、YAMLファイルとPythonの辞書内で定義された同じスイープ設定を示しています。スイープ設定内には、`program`, `name`, `method`, `metric`, `parameters` の5つのトップレベルキーが指定されています。

<Tabs
  defaultValue="cli"
  values={[    
    {label: 'CLI', value: 'cli'},
    {label: 'Python script or Jupyter notebook', value: 'script'},
  ]}>
  <TabItem value="script">

トレーニングアルゴリズムをPythonスクリプトやJupyter ノートブックで定義する場合は、スイープをPythonの辞書データ構造で定義します。

以下のコードスニペットは、変数 `sweep_configuration` にスイープ設定を格納しています：

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
スイープをコマンドライン（CLI）から対話的に管理したい場合は、スイープ設定を YAML ファイルで定義します

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

トップレベルの `parameters` キーの内側には、以下のキーがネストされています：`learning_rate`, `batch_size`, `epoch`, `optimizer`。各ネストされたキーに対して、1つ以上の値や分布、確率などを指定できます。詳しくは、[スイープ設定オプション](./sweep-config-keys.md) の [parameters](./sweep-config-keys.md#parameters) セクションを参照してください。

## 二重にネストされたパラメータ

スイープ設定はネストされたパラメータをサポートしています。ネストされたパラメータを区別するには、トップレベルのパラメータ名の下に追加の `parameters` キーを使用します。スイープ設定は多層ネストをサポートしています。

ベイズまたはランダムのハイパーパラメータ検索を使用する場合は、ランダム変数の確率分布を指定します。各ハイパーパラメータについて：

1. スイープ設定にトップレベルの `parameters` キーを作成します。
2. `parameters` キー内に以下をネストします：
   1. 最適化したいハイパーパラメータの名前を指定します。
   2. `distribution` キーを使って使用したい分布を指定します。`distribution` キー値ペアをハイパーパラメータ名の下にネストします。
   3. 探索する1つ以上の値を指定します。値は分布キーと一致している必要があります。  
      1. （オプション）ネストされたパラメータを区別するために、トップレベルパラメータ名の下に追加の parameters キーを使用します。

:::caution
スイープ設定で定義されたネストされたパラメータは、W&B run 設定で指定されたキーを上書きします。

例えば、`train.py` Pythonスクリプト内で次のようにW&B run を初期化する設定を行った場合（1行目と2行目参照）。次に、辞書として定義されたスイープ設定を `sweep_configuration` として定義します（4行目から13行目参照）。その後、スイープ設定辞書を `wandb.sweep` に渡してスイープ設定を初期化します（16行目参照）。

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

# 設定を渡してスイープを初期化。
sweep_id = wandb.sweep(sweep=sweep_configuration, project="<project>")

# スイープジョブを開始。
wandb.agent(sweep_id, function=main, count=4)
```
W&B runが初期化されるときに渡される `nested_param.manual_key`（2行目）は、アクセスできません。`run.config` にはスイープ設定辞書に定義されたキー値ペアのみが存在します（4行目から13行目まで）。
:::

## スイープ設定テンプレート

以下のテンプレートは、パラメータを設定し、検索制約を指定する方法を示しています。`hyperparameter_name` をハイパーパラメータの名前に置き換え、`<>` で囲まれた値を適宜置き換えてください。

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

## スイープ設定の例

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

以下のタブは、`early_terminate` の最小または最大イテレーション数を指定する方法を示しています：

<Tabs
  defaultValue="min_iter"
  values={[
    {label: '最小イテレーション数の指定', value: 'min_iter'},
    {label: '最大イテレーション数の指定', value: 'max_iter'},
  ]}>
  <TabItem value="min_iter">

```yaml
early_terminate:
  type: hyperband
  min_iter: 3
```

この例のブラケットは `[3, 3*eta, 3*eta*eta, 3*eta*eta*eta]` で、`[3, 9, 27, 81]` となります。
  </TabItem>
  <TabItem value="max_iter">

```yaml
early_terminate:
  type: hyperband
  max_iter: 27
  s: 2
```

この例のブラケットは `[27/eta, 27/eta/eta]` で、`[9, 3]` となります。
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

次のタブは、一般的なコマンドマクロを指定する方法を示しています:

<Tabs
  defaultValue="python"
  values={[
    {label: 'Pythonインタープリタの設定', value: 'python'},
    {label: '追加パラメータの追加', value: 'parameters'},
    {label: '引数の省略', value: 'omit'},
    {label: 'Hydra', value: 'hydra'}
  ]}>
  <TabItem value="python">

`{$interpreter}` マクロを削除し、Pythonインタープリタの値を明示的に提供します。例えば、以下のコードスニペットでは、その方法を示しています：

```yaml
command:
  - ${env}
  - python3
  - ${program}
  - ${args}
```
  </TabItem>
  <TabItem value="parameters">

以下は、スイープ設定パラメータでは指定されていない追加のコマンドライン引数を追加する方法を示しています：

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

プログラムが引数解析を使用しない場合、引数をすべて渡さないようにし、`wandb.init` がスイープパラメータを `wandb.config` に自動的に取得する利点を活用できます：

```
command:
  - ${env}
  - ${interpreter}
  - ${program}
```
  </TabItem>
  <TabItem value="hydra">

コマンドを変更して、[Hydra](https://hydra.cc) のようなツールが期待する方法で引数を渡すことができます。詳細は [HydraとW&B](../integrations/other/hydra.md)を参照してください。

