---
title: sweep configuration を定義する
description: スイープの設定ファイルを作成する方法を学びましょう。
menu:
  default:
    identifier: ja-guides-models-sweeps-define-sweep-configuration-_index
    parent: sweeps
url: /ja/guides/sweeps/define-sweep-configuration
weight: 3
---

A W&B Sweep は、ハイパーパラメータの値を探索するための戦略と、それを評価するコードを組み合わせたものです。この戦略はすべてのオプションを試すというシンプルなものから、ベイズ最適化やハイパーバンド（[BOHB](https://arxiv.org/abs/1807.01774)）のように複雑なものまであります。

Sweep configuration を [Python 辞書](https://docs.python.org/3/tutorial/datastructures.html#dictionaries) または [YAML](https://yaml.org/) ファイルで定義します。Sweep configuration をどのように定義するかは、あなたが sweep をどのように管理したいかによって異なります。

{{% alert %}}
スイープを初期化し、コマンドラインからスイープエージェントを開始したい場合は、YAMLファイルでスイープ設定を定義します。スイープを初期化し、完全にPythonスクリプトまたはJupyterノートブック内でスイープを開始する場合は、Python辞書でスイープを定義します。
{{% /alert %}}

以下のガイドでは、sweep configuration のフォーマット方法について説明します。[Sweep configuration options]({{< relref path="./sweep-config-keys.md" lang="ja" >}}) で、トップレベルの sweep configuration キーの包括的なリストをご覧ください。

## 基本構造

両方のスイープ設定フォーマットオプション (YAML および Python 辞書) は、キーと値のペアおよびネストされた構造を利用します。

スイープ設定内でトップレベルキーを使用して、sweep 検索の特性を定義します。たとえば、スイープの名前（[`name`]({{< relref path="./sweep-config-keys.md" lang="ja" >}}) キー）、検索するパラメータ（[`parameters`]({{< relref path="./sweep-config-keys.md#parameters" lang="ja" >}}) キー）、パラメータ空間を検索する方法（[`method`]({{< relref path="./sweep-config-keys.md#method" lang="ja" >}}) キー）、その他があります。

例えば、以下のコードスニペットは、YAML ファイルと Python 辞書の両方で定義された同じスイープ設定を示しています。スイープ設定内には、`program`、`name`、`method`、`metric`、および `parameters` という5つのトップレベルキーが指定されています。

{{< tabpane text=true >}}
  {{% tab header="CLI" %}}
スイープをコマンドライン (CLI) からインタラクティブに管理したい場合、YAML ファイルでスイープ設定を定義します。

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
  {{% /tab %}}
  {{% tab header="Python スクリプトまたは Jupyter ノートブック" %}}
Python スクリプトまたは Jupyter ノートブックでトレーニングアルゴリズムを定義する場合は、Python 辞書データ構造でスイープを定義します。

以下のコードスニペットは、`sweep_configuration` という変数名でスイープ設定を格納します：

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
  {{% /tab %}}
{{< /tabpane >}}

トップレベルの `parameters` キーの中に、以下のキーがネストされています：`learning_rate`、`batch_size`、`epoch`、および `optimizer`。指定したネストされたキーごとに、1つ以上の値、分布、確率などを提供できます。詳細については、[Sweep configuration options]({{< relref path="./sweep-config-keys.md" lang="ja" >}}) の [parameters]({{< relref path="./sweep-config-keys.md#parameters" lang="ja" >}}) セクションを参照してください。

## 二重ネストパラメータ

Sweep configurations はネストされたパラメータをサポートします。ネストされたパラメータを区切るには、トップレベルのパラメータ名の下に追加の `parameters` キーを使用します。スイープ設定は多層ネストをサポートします。

ベイズまたはランダムなハイパーパラメータ検索を使用する場合、確率分布を指定します。各ハイパーパラメータについて：

1. スイープ設定でトップレベル `parameters` キーを作成します。
2. `parameters` キーの中に次のものをネストします：
   1. 最適化したいハイパーパラメータの名前を指定します。
   2. `distribution` キーのために使用したい分布を指定します。ハイパーパラメータ名の下に `distribution` キーと値のペアをネストします。
   3. 探索する1つ以上の値を指定します。その値（または値のリスト）は分布キーと整合している必要があります。
      1. (オプション) トップレベルのパラメータ名の下に追加のパラメータキーを使用して、ネストされたパラメータを区切ります。

{{% alert color="secondary" %}}
スイープ設定で定義されたネストされたパラメータは、W&B run 設定で指定されたキーを上書きします。

例として、次の設定で `train.py` Python スクリプト（行1-2で確認可能）で W&B run を初期化するとします。次に、`sweep_configuration`（行4-13）の辞書でスイープ設定を定義します。その後、スイープ設定辞書を `wandb.sweep` に渡してスイープ設定を初期化します（行16を確認）。

```python title="train.py" 
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
W&B run が初期化されたときに渡された `nested_param.manual_key` はアクセスできません。`run.config` は、スイープ設定辞書で定義されたキーと値のペアのみを持っています。
{{% /alert %}}

## Sweep configuration テンプレート

次のテンプレートには、パラメータを構成し、検索制約を指定する方法を示しています。`hyperparameter_name` をあなたのハイパーパラメータの名前と、`<>` 内の任意の値で置き換えます。

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

## Sweep configuration の例

{{< tabpane text=true >}}
  {{% tab header="CLI" %}}

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

  {{% /tab %}}
  {{% tab header="Python スクリプトまたは Jupyter ノートブック" %}}

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

  {{% /tab %}}
{{< /tabpane >}}

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

以下のタブで `early_terminate` の最小または最大のイテレーション回数を指定する方法を示します：

{{< tabpane text=true >}}
  {{% tab header="最大のイテレーション回数" %}}

この例のブラケットは `[3, 3*eta, 3*eta*eta, 3*eta*eta*eta]` で、結果として `[3, 9, 27, 81]` になります。

```yaml
early_terminate:
  type: hyperband
  min_iter: 3
```

  {{% /tab %}}
  {{% tab header="最小のイテレーション回数" %}}

この例のブラケットは `[27/eta, 27/eta/eta]` で、`[9, 3]` になります。

```yaml
early_terminate:
  type: hyperband
  max_iter: 27
  s: 2
```

  {{% /tab %}}
{{< /tabpane >}}

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

{{< tabpane text=true >}}
  {{% tab header="Unix" %}}

```bash
/usr/bin/env python train.py --param1=value1 --param2=value2
```  

  {{% /tab %}}
  {{% tab header="Windows" %}}

```bash
python train.py --param1=value1 --param2=value2

```  
  {{% /tab %}}
{{< /tabpane >}}

以下のタブで一般的なコマンドマクロを指定する方法を示します：

{{< tabpane text=true >}}
  {{% tab header="Python インタープリタの設定" %}}

`{$interpreter}` マクロを削除し、値を明示的に提供して Python インタプリタをハードコードします。例えば、以下のコードスニペットはその方法を示しています：

```yaml
command:
  - ${env}
  - python3
  - ${program}
  - ${args}
```

  {{% /tab %}}
  {{% tab header="追加のパラメータを追加" %}}

次の例では、sweep configuration のパラメータで指定されていないコマンドライン引数を追加する方法を示します：

```yaml
command:
  - ${env}
  - ${interpreter}
  - ${program}
  - "--config"
  - "your-training-config.json"
  - ${args}
```
  
  {{% /tab %}}
  {{% tab header="引数を省略" %}}

あなたのプログラムが引数パースを使用しない場合、すべての引数を渡すのを避け、`wandb.init` がスイープパラメータを自動的に `wandb.config` に取り込むことを利用できます：

```yaml
command:
  - ${env}
  - ${interpreter}
  - ${program}
```  

  {{% /tab %}}
  {{% tab header="Hydra" %}}

ツールのように引数を渡すコマンドを変更できます [Hydra](https://hydra.cc) が期待する方法です。詳細については、[Hydra with W&B]({{< relref path="/guides/integrations/hydra.md" lang="ja" >}}) を参照してください。

```yaml
command:
  - ${env}
  - ${interpreter}
  - ${program}
  - ${args_no_hyphens}
```

  {{% /tab %}}
{{< /tabpane >}}