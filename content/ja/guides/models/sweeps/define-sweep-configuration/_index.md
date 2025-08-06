---
title: sweep configuration を定義する
description: スイープの設定ファイルを作成する方法について学びましょう。
menu:
  default:
    identifier: define-sweep-configuration
    parent: sweeps
url: guides/sweeps/define-sweep-configuration
weight: 3
---

W&B Sweep は、ハイパーパラメータ値を探索するための戦略と、それらを評価するコードを組み合わせたものです。戦略は、すべてのオプションを試すシンプルなものから、ベイズ最適化や Hyperband（[BOHB](https://arxiv.org/abs/1807.01774)）のような複雑なものまで選択できます。

Sweep の設定は [Python 辞書](https://docs.python.org/3/tutorial/datastructures.html#dictionaries) または [YAML](https://yaml.org/) ファイルのいずれかで定義します。どちらで定義するかは、Sweep の運用方法に依存します。

{{% alert %}}
コマンドラインから Sweep を初期化し、sweep agent を開始したい場合は YAML ファイルで設定を定義してください。Python スクリプトやノートブック内で Sweep の初期化から開始まですべて完結させたい場合は Python 辞書で Sweep を定義してください。
{{% /alert %}}

以下のガイドでは Sweep 設定ファイルのフォーマット方法について説明します。トップレベルの Sweep 設定キーの詳細は [Sweep configuration options]({{< relref "./sweep-config-keys.md" >}}) を参照してください。

## 基本構造

YAML・Python 辞書どちらの Sweep 設定フォーマットも、キーと値のペア、そしてネスト構造を利用します。

Sweep 設定のトップレベルキーを使って、Sweep 検索の名前（[`name`]({{< relref "./sweep-config-keys.md" >}}) キー）、検索対象パラメータ（[`parameters`]({{< relref "./sweep-config-keys.md#parameters" >}}) キー）、ハイパーパラメータサーチの手法（[`method`]({{< relref "./sweep-config-keys.md#method" >}}) キー）などを明示します。

例えば、次のコードスニペットは同じ Sweep 設定を YAML ファイルと Python 辞書の両方で定義したものです。Sweep 設定には `program`、`name`、`method`、`metric`、`parameters` の５つのトップレベルキーが指定されています。

{{< tabpane  text=true >}}
  {{% tab header="CLI" %}}
Sweep をコマンドライン（CLI）から対話的に管理したい場合は YAML ファイルで設定しましょう。

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
  {{% tab header="Python script or notebook" %}}
トレーニングアルゴリズムを Python スクリプトやノートブックで定義する場合は、Sweep を Python 辞書データ構造で定義します。

次のコードスニペットでは、Sweep 設定を `sweep_configuration` という変数に格納しています。

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

トップレベルの `parameters` キーの中に、`learning_rate`、`batch_size`、`epoch`、`optimizer` というネストされたキーが含まれます。それぞれのネストされたキーごとに、値や分布、確率などを指定できます。詳細は [parameters]({{< relref "./sweep-config-keys.md#parameters" >}})の項や [Sweep configuration options]({{< relref "./sweep-config-keys.md" >}}) をご覧ください。

## 二重ネストパラメータ

Sweep 設定では、ネストされたパラメータ（入れ子パラメータ）もサポートしています。ネストさせたい場合は、トップレベルのパラメータ名称の下に追加で `parameters` キーを設けます。Sweep の設定は多階層のネストも可能です。

ベイズ・ランダムサーチを利用する場合は、乱数変数の確率分布も設定できます。各ハイパーパラメータの設定手順は次の通りです：

1. Sweep 設定のトップレベルに `parameters` キーを作成します。
2. `parameters` キー内で次をネストします：
   1. 最適化したいハイパーパラメータ名を指定します。
   2. `distribution` キーでは使用する分布を指定し、ハイパーパラメータ名の下にネストします。
   3. 探索したい値を1つまたは複数指定します。値は distribution キーと同じ階層で書きます。
      1. （オプション）さらに、トップレベルパラメータ名の下に追加の parameters キーを設けて、ネストパラメータを表現することもできます。

{{% alert color="secondary" %}}
Sweep 設定内で定義されたネストパラメータは、W&B run 設定で指定したキーを上書きします。

例えば、`train.py` Python スクリプト内（1-2行目）で以下のような設定で W&B run を初期化したとします。そして次に `sweep_configuration` という辞書で Sweep 設定を定義し（4-13行目）、最後にその設定辞書を `wandb.sweep` に渡して Sweep を初期化します（16行目）。


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

# Sweep を初期化し、設定を渡します。
sweep_id = wandb.sweep(sweep=sweep_configuration, project="<project>")

# Sweep ジョブの開始
wandb.agent(sweep_id, function=main, count=4)
```
W&B run 初期化時に渡した `nested_param.manual_key` はアクセスできません。`wandb.Run.config` には Sweep 設定辞書で定義されたキーと値のみが保持されます。
{{% /alert %}}


## Sweep 設定テンプレート

以下のテンプレートは、パラメータ設定や探索制約条件の指定方法を示しています。`hyperparameter_name` は任意のハイパーパラメータ名、`<>` で囲まれた部分は任意の値に置き換えてください。

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

科学的記法で数値を表したい場合は、YAML の `!!float` 演算子（値を浮動小数点値にキャスト）を追加してください。例：`min: !!float 1e-5`。詳細は [Command example]({{< relref "#command-example" >}}) を参照してください。

## Sweep 設定例

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
  {{% tab header="Python script or notebook" %}}

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

### Bayes hyperband の例

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

以下のタブでは、`early_terminate` の反復回数（最大・最小回数）の指定方法を示します。

{{< tabpane  text=true >}}
  {{% tab header="Maximum number of iterations" %}}

この例ではブラケットは `[3, 3*eta, 3*eta*eta, 3*eta*eta*eta]`、すなわち `[3, 9, 27, 81]` です。

```yaml
early_terminate:
  type: hyperband
  min_iter: 3
```

  {{% /tab %}}
  {{% tab header="Minimum number of iterations" %}}

この例ではブラケットは `[27/eta, 27/eta/eta]`、つまり `[9, 3]` です。

```yaml
early_terminate:
  type: hyperband
  max_iter: 27
  s: 2
```

  {{% /tab %}}
{{< /tabpane >}}

### マクロとカスタムコマンド引数の例

コマンドライン引数が複雑になる場合は、Environment 変数や Python インタプリタ、追加の引数を渡すためにマクロを利用できます。[W&B ではあらかじめ用意されたマクロ]({{< relref "./sweep-config-keys.md#command-macros" >}}) と、Sweep 設定で自由に指定できるカスタムコマンドライン引数の両方をサポートしています。

以下の Sweep 設定（`sweep.yaml`）では、コマンド実行時に Python スクリプト（`run.py`）と `${env}` `${interpreter}` `${program}` マクロが、それぞれ実行時の適切な値に置換されます。

また、`--batch_size=${batch_size}`、`--test=True`、`--optimizer=${optimizer}` などの引数では、パラメータの値をマクロ展開した値で渡しています。

```yaml title="sweep.yaml"
program: run.py
method: random
metric:
  name: validation_loss
parameters:
  learning_rate:
    min: 0.0001
    max: 0.1
command:
  - ${env}
  - ${interpreter}
  - ${program}
  - "--batch_size=${batch_size}"
  - "--optimizer=${optimizer}"
  - "--test=True"
```
対応する Python スクリプト（`run.py`）では、`argparse` モジュールを使ってコマンドライン引数を受け取れます。

```python title="run.py"
# run.py
import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int)
parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd'], required=True)
parser.add_argument('--test', type=str2bool, default=False)
args = parser.parse_args()

# W&B Run の初期化
with wandb.init('test-project') as run:
    run.log({'validation_loss':1})
```

Sweep 設定ファイルで使えるマクロの一覧は [Command macros]({{< relref "./sweep-config-keys.md#command-macros" >}})、[Sweep configuration options]({{< relref "./sweep-config-keys.md" >}}) のセクションを参照ください。

#### ブーリアン（真偽値）引数

`argparse` モジュールはデフォルトでブーリアン引数をサポートしていません。ブーリアン引数を定義する方法としては、[`action`](https://docs.python.org/3/library/argparse.html#action) パラメータを指定するか、文字列をブーリアン型に変換するカスタム関数を作成します。

例えば、以下のコードスニペットのように、`ArgumentParser` に `store_true` または `store_false` を渡すことでブーリアン引数を定義できます。

```python
import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')
args = parser.parse_args()

args.test  # --test が渡された場合は True、それ以外は False になります
```

また、ブーリアン値の文字列表現をブーリアン型に変換するカスタム関数を定義することもできます。例えば、次のコードスニペットでは、`str2bool` 関数によって文字列がブーリアン型に変換されます。

```python
def str2bool(v: str) -> bool:
  """文字列をブーリアン値に変換する関数です。
  argparse はデフォルトでブーリアン引数をサポートしていないため、この関数を使います。
  """
  if isinstance(v, bool):
      return v
  return v.lower() in ('yes', 'true', 't', '1')
```