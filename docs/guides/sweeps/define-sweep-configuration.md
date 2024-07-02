---
description: スイープの設定ファイルの作成方法を学びましょう。
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Sweep configuration structure

<head>
  <title>ハイパーパラメータチューニングのためのスイープ設定を定義する。</title>
</head>

W&B Sweepは、ハイパーパラメータの値を探索するための戦略と、それらを評価するコードを組み合わせたものです。戦略はすべてのオプションを試すほどシンプルなものから、ベイズ最適化やHyperband ([BOHB](https://arxiv.org/abs/1807.01774))のように複雑なものまであります。

[sweep configuration](./sweep-config-keys.md) のオプションについては、[トップレベルのスイープ設定キーの完全なリスト](./sweep-config-keys.md) を参照してください。

## 基本構造

両方のスイープ設定フォーマットオプション（YAML と Python 辞書）は、キーと値のペア、およびネストされた構造を利用します。

スイープ設定内のトップレベルキーを使用して、sweep の検索の品質（例: スイープの名前 ([`name`](./sweep-config-keys.md#name) キー)、検索するパラメーター ([`parameters`](./sweep-config-keys.md#parameters) キー)、パラメータスペースを検索する方法論 ([`method`](./sweep-config-keys.md#method) キー)など）を定義します。

例えば、次のコードスニペットは、同じスイープ設定が YAML ファイルと Python 辞書内で定義されていることを示しています。スイープ設定内には以下の5つのトップレベルキーが指定されています： `program`, `name`, `method`, `metric`, `parameters`.

<Tabs
  defaultValue="cli"
  values={[    
    {label: 'CLI', value: 'cli'},
    {label: 'Python script or Jupyter notebook', value: 'script'},
  ]}>
  <TabItem value="script">

PythonスクリプトやJupyterノートブックでトレーニングアルゴリズムを定義する場合、Python辞書形式のデータ構造にスイープを定義します。

次のコードスニペットは、`sweep_configuration`という変数にスイープ設定を格納しています：

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
コマンドライン（CLI）からインタラクティブにスイープを管理したい場合は、YAMLファイルにスイープ設定を定義します。

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

トップレベルの `parameters` キー内には、以下のキーがネストされています：`learning_rate`, `batch_size`, `epoch`, `optimizer`。指定する各ネストされたキーについて、1つ以上の値、分布、確率などを提供できます。詳細は、[Sweeps 設定オプション](./sweep-config-keys.md#parameters)内の[parameters](./sweep-config-keys.md#parameters)セクションを参照してください。

## 二重ネストされたパラメータ

スイープ設定はネストされたパラメータをサポートしています。ネストされたパラメータを区別するには、トップレベルのパラメータ名の下に追加の `parameters` キーを使用します。スイープ設定は複数レベルのネストをサポートします。

ベイズまたはランダムハイパーパラメータ検索を使用する場合は、ランダム変数の確率分布を指定します。各ハイパーパラメータについて：

1. スイープ設定にトップレベルの `parameters` キーを作成します。
2. `parameters` キー内に次の要素をネストします：
   1. 最適化したいハイパーパラメータの名前を指定します。
   2. `distribution` キーのために使用する分布を指定します。ハイパーパラメータ名の下に `distribution` キーとその値をネストします。
   3. 探索する値を1つ以上指定します。値（または値）の指定は分布キーと一致する必要があります。
      1. （オプション）ネストされたパラメータを区別するために、トップレベルのパラメータ名の下に追加の `parameters` キーを使用します。

:::caution
スイープ設定に定義されたネストされたパラメータは、W&B run 設定で指定されたキーを上書きします。

例えば、次のコードスニペットでは、`train.py` Pythonスクリプト内で次の設定を使って W&B run を初期化したとします（1-2行目）。続いて、辞書 `sweep_configuration` にスイープ設定を定義します（4-13行目）。その後、スイープ設定辞書を `wandb.sweep` に渡してスイープ設定を初期化します（16行目）。


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

# Initialize sweep by passing in config.
sweep_id = wandb.sweep(sweep=sweep_configuration, project="<project>")

# Start sweep job.
wandb.agent(sweep_id, function=main, count=4)
```
W&B run が初期化されたときに渡される `nested_param.manual_key`（2行目）はアクセスできません。`run.config` にはスイープ設定辞書（4-13行目）で定義されたキー値ペアのみが含まれます。
:::

## スイープ設定テンプレート

以下のテンプレートは、パラメータを設定し検索制約を指定する方法を示しています。`hyperparameter_name`をハイパーパラメータ名に置き換え、`<>`で囲まれた値を適切な値に置き換えてください。

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

## スイープ設定例

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

### ベイズハイパーバンド例

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

以下のタブは`early_terminate` の最少または最大反復回数を指定する方法を示しています：

<Tabs
  defaultValue="min_iter"
  values={[
    {label: '最低限の反復回数を指定', value: 'min_iter'},
    {label: '最大反復回数を指定', value: 'max_iter'},
  ]}>
  <TabItem value="min_iter">

```yaml
early_terminate:
  type: hyperband
  min_iter: 3
```

この例の括弧は `[3, 3*eta, 3*eta*eta, 3*eta*eta*eta]` です。これは `[3, 9, 27, 81]` になります。
  </TabItem>
  <TabItem value="max_iter">

```yaml
early_terminate:
  type: hyperband
  max_iter: 27
  s: 2
```

この例の括弧は `[27/eta, 27/eta/eta]` です。これは `[9, 3]` になります。
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

以下のタブは、一般的なコマンドマクロを指定する方法を示しています：

<Tabs
  defaultValue="python"
  values={[
    {label: 'Pythonインタープリターを設定', value: 'python'},
    {label: '追加のパラメータを追加', value: 'parameters'},
    {label: '引数を省略', value: 'omit'},
    {label: 'Hydra', value: 'hydra'}
  ]}>
  <TabItem value="python">

`{$interpreter}` マクロを削除し、Pythonインタープリターをハードコードするために値を明示的に指定します。例えば、次のコードスニペットはその方法を示しています：

```yaml
command:
  - ${env}
  - python3
  - ${program}
  - ${args}
```
  </TabItem>
  <TabItem value="parameters">

以下は、スイープ設定のパラメータによって指定されていない追加のコマンドライン引数を追加する方法を示しています：

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

プログラムが引数解析を利用しない場合、引数を全く渡さずに、`wandb.init` を利用してスイープパラメータを自動的に `wandb.config` に取り込むことができます：

```
command:
  - ${env}
  - ${interpreter}
  - ${program}
```
  </TabItem>
  <TabItem value="hydra">

コマンドを変更して、[Hydra](https://hydra.cc) のようなツールが期待する方法で引数を渡すことができます。詳細は、[HydraとW&B](../integrations/other/hydra.md) を参照してください。

```
command:
  - ${env}
  - ${interpreter}
  - ${program}
  - ${args_no_hyphens}
```
  </TabItem>
</Tabs>