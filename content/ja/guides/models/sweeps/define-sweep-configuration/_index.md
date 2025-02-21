---
title: Define a sweep configuration
description: sweep の 設定ファイルを作成する方法を学びます。
menu:
  default:
    identifier: ja-guides-models-sweeps-define-sweep-configuration-_index
    parent: sweeps
url: guides/sweeps/define-sweep-configuration
weight: 3
---

W&B の Sweep は、ハイパーパラメータの値を探索するための戦略と、それらを評価するコードを組み合わせたものです。この戦略は、すべてのオプションを試すという単純なものから、ベイズ最適化や Hyperband ( [BOHB](https://arxiv.org/abs/1807.01774) ) のように複雑なものまであります。

[Python の 辞書](https://docs.python.org/3/tutorial/datastructures.html#dictionaries) または [YAML](https://yaml.org/) ファイルで sweep configuration を定義します。sweep configuration の定義方法は、sweep の管理方法によって異なります。

{{% alert %}}
コマンドラインから sweep を初期化して sweep agent を開始する場合は、YAML ファイルで sweep configuration を定義します。Python スクリプトまたは Jupyter notebook 内で sweep を初期化して開始する場合は、Python の辞書で sweep を定義します。
{{% /alert %}}

以下のガイドでは、sweep configuration のフォーマット方法について説明します。トップレベルの sweep configuration キーの包括的なリストについては、[Sweep configuration options]({{< relref path="./sweep-config-keys.md" lang="ja" >}}) を参照してください。

## 基本構造

sweep configuration のフォーマットオプション（YAML と Python の辞書）はどちらも、キーと値のペアとネストされた構造を利用します。

sweep configuration 内のトップレベルのキーを使用して、sweep の名前 ( [`name`]({{< relref path="./sweep-config-keys.md" lang="ja" >}}) キー)、検索するパラメータ ( [`parameters`]({{< relref path="./sweep-config-keys.md#parameters" lang="ja" >}}) キー)、パラメータ空間を検索する方法 ( [`method`]({{< relref path="./sweep-config-keys.md#method" lang="ja" >}}) キー) など、sweep 検索の特性を定義します。

たとえば、以下のコードスニペットは、YAML ファイル内と Python の辞書内で定義された同じ sweep configuration を示しています。sweep configuration 内には、`program`、`name`、`method`、`metric`、`parameters` という 5 つのトップレベルのキーが指定されています。

{{< tabpane  text=true >}}
  {{% tab header="CLI" %}}
コマンドライン（CLI）からインタラクティブに Sweeps を管理する場合は、YAML ファイルで sweep configuration を定義します。

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
  {{% tab header="Python script or Jupyter notebook" %}}
Python スクリプトまたは Jupyter notebook でトレーニングアルゴリズムを定義する場合は、Python の辞書データ構造で sweep を定義します。

次のコードスニペットは、sweep configuration を `sweep_configuration` という名前の変数に格納します。

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

トップレベルの `parameters` キー内には、`learning_rate`、`batch_size`、`epoch`、`optimizer` のキーがネストされています。指定するネストされたキーごとに、1 つまたは複数の値、分布、確率などを指定できます。詳細については、[Sweep configuration options]({{< relref path="./sweep-config-keys.md" lang="ja" >}}) の [parameters]({{< relref path="./sweep-config-keys.md#parameters" lang="ja" >}}) セクションを参照してください。

## 二重にネストされたパラメータ

Sweep configuration は、ネストされたパラメータをサポートしています。ネストされたパラメータを区切るには、トップレベルのパラメータ名の下に追加の `parameters` キーを使用します。Sweep configuration は、マルチレベルのネスティングをサポートしています。

ベイズまたはランダムなハイパーパラメータ検索を使用する場合は、ランダム変数の確率分布を指定します。ハイパーパラメータごとに、次の手順に従います。

1. sweep config でトップレベルの `parameters` キーを作成します。
2. `parameters` キーの中に、以下をネストします。
   1. 最適化するハイパーパラメータの名前を指定します。
   2. `distribution` キーに使用する分布を指定します。ハイパーパラメータ名の下に `distribution` キーと値のペアをネストします。
   3. 探索する 1 つまたは複数の値を指定します。値は、distribution キーとインラインである必要があります。
      1. (オプション) トップレベルのパラメータ名の下に追加の parameters キーを使用して、ネストされたパラメータを区切ります。

{{% alert color="secondary" %}}
sweep configuration で定義されたネストされたパラメータは、W&B run configuration で指定されたキーを上書きします。

たとえば、`train.py` Python スクリプトで、次の設定で W&B run を初期化するとします (1 ～ 2 行目)。次に、`sweep_configuration` という辞書で sweep configuration を定義します (4 ～ 13 行目)。次に、sweep config 辞書を `wandb.sweep` に渡して、sweep config を初期化します (16 行目)。

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

# config を渡して sweep を初期化します。
sweep_id = wandb.sweep(sweep=sweep_configuration, project="<project>")

# sweep ジョブを開始します。
wandb.agent(sweep_id, function=main, count=4)
```

W&B run の初期化時に渡される `nested_param.manual_key` (2 行目) にはアクセスできません。`run.config` は、sweep configuration 辞書 (4 ～ 13 行目) で定義されているキーと値のペアのみを保持します。
{{% /alert %}}

## Sweep configuration テンプレート

次のテンプレートは、パラメータを設定し、検索制約を指定する方法を示しています。`<>` で囲まれた `hyperparameter_name` をハイパーパラメータの名前に置き換えます。

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
  {{% tab header="Python script or Jupyter notebook" %}}

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

### ベイズ Hyperband の例

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

次のタブは、`early_terminate` の最小または最大反復回数を指定する方法を示しています。

{{% tabpane text=true %}}
  {{% tab header="反復の最大回数を指定" %}}

```yaml
early_terminate:
  type: hyperband
  min_iter: 3
```

この例のブラケットは、`[3, 3*eta, 3*eta*eta, 3*eta*eta*eta]` で、これは `[3, 9, 27, 81]` と同じです。
  {{% /tab %}}
  {{% tab header="反復の最小回数を指定" %}}

```yaml
early_terminate:
  type: hyperband
  max_iter: 27
  s: 2
```

この例のブラケットは、`[27/eta, 27/eta/eta]` で、これは `[9, 3]` と同じです。

  {{% /tab %}}
{{% /tabpane %}}

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

次のタブは、一般的なコマンドマクロを指定する方法を示しています。

{{< tabpane text=true >}}
  {{% tab header="Python インタープリターの設定" %}}

`{$interpreter}` マクロを削除し、値を明示的に指定して、Python インタープリターをハードコードします。たとえば、次のコードスニペットは、これを行う方法を示しています。

```yaml
command:
  - ${env}
  - python3
  - ${program}
  - ${args}
```

  {{% /tab %}}
  {{% tab header="追加のパラメータの追加" %}}

次は、sweep configuration パラメータで指定されていない追加のコマンドライン引数を追加する方法を示しています。

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
  {{% tab header="引数の省略" %}}

プログラムが引数の解析を使用しない場合は、引数をすべて渡すことを避け、`wandb.init` が sweep パラメータを `wandb.config` に自動的に取り込むことを利用できます。

```yaml
command:
  - ${env}
  - ${interpreter}
  - ${program}
```  

  {{% /tab %}}
  {{% tab header="Hydra" %}}

[Hydra](https://hydra.cc) などのツールが想定する方法で引数を渡すようにコマンドを変更できます。詳細については、[W&B と Hydra]({{< relref path="/guides/integrations/hydra.md" lang="ja" >}}) を参照してください。

```yaml
command:
  - ${env}
  - ${interpreter}
  - ${program}
  - ${args_no_hyphens}
```

  {{% /tab %}}
{{< /tabpane >}}
