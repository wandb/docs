---
title: sweep configuration を定義する
description: スイープの設定ファイルを作成する方法について学びましょう。
menu:
  default:
    identifier: ja-guides-models-sweeps-define-sweep-configuration-_index
    parent: sweeps
url: guides/sweeps/define-sweep-configuration
weight: 3
---

W&B Sweep は、ハイパーパラメーター値の探索戦略とその評価を行うコードを組み合わせます。戦略は、全てのオプションを試すだけのシンプルなものから、ベイズ最適化やハイパーバンド（[BOHB](https://arxiv.org/abs/1807.01774)）のような複雑なものまで対応しています。

スイープ設定は [Python 辞書](https://docs.python.org/3/tutorial/datastructures.html#dictionaries) または [YAML](https://yaml.org/) ファイルで定義できます。どちらを選ぶかは、Sweep の管理方法次第です。

{{% alert %}}
コマンドラインから Sweep の初期化と sweep agent の起動をしたい場合は、Sweep 設定を YAML ファイルで定義してください。Python スクリプトやノートブック内ですべてを完結させる場合は、Python 辞書で定義してください。
{{% /alert %}}

このガイドでは、Sweep 設定のフォーマット方法を説明します。トップレベルの sweep 設定キーの一覧は、[Sweep 設定オプション]({{< relref path="./sweep-config-keys.md" lang="ja" >}}) を参照してください。

## 基本構造

YAML と Python 辞書、いずれの sweep 設定フォーマットでもキーと値のペアやネスト構造を利用します。

sweep 設定のトップレベルキーには、Sweep の検索名（[`name`]({{< relref path="./sweep-config-keys.md" lang="ja" >}}) キー）、探索するパラメータ（[`parameters`]({{< relref path="./sweep-config-keys.md#parameters" lang="ja" >}}) キー）、パラメータ探索の手法（[`method`]({{< relref path="./sweep-config-keys.md#method" lang="ja" >}}) キー）など、Sweep に関する特性を定義します。

例えば、以下のコードスニペットは同じ sweep 設定を YAML ファイルと Python 辞書の両方で定義した例です。それぞれの例では、`program`, `name`, `method`, `metric`, `parameters` の5つのトップレベルキーを指定しています。

{{< tabpane  text=true >}}
  {{% tab header="CLI" %}}
コマンドライン（CLI）でインタラクティブに Sweeps を管理する場合は、YAML ファイルで Sweep 設定を定義します。

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
トレーニングアルゴリズムを Python スクリプトやノートブックで記述する場合は、Python 辞書のデータ構造で Sweep を定義します。

次のコードスニペットは、`sweep_configuration` という変数に Sweep 設定を格納しています。

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

`parameters` キーの下には、`learning_rate`, `batch_size`, `epoch`, `optimizer` といったキーがネストされています。それぞれのネストしたキーごとに、一つまたは複数の値や、分布、確率などを指定できます。詳細は、[Sweep 設定オプション]({{< relref path="./sweep-config-keys.md" lang="ja" >}}) の [parameters]({{< relref path="./sweep-config-keys.md#parameters" lang="ja" >}}) セクションを参照してください。

## 多重ネストされたパラメータ

Sweep 設定では、ネストしたパラメータをサポートしています。パラメータにネストを付与したい場合、トップレベルのパラメータ名の下にもう一つ `parameters` キーを追加してください。Sweep の設定は多段階のネストも可能です。

ベイズやランダムなハイパーパラメーター探索を行う場合、ランダム変数に対して確率分布を指定できます。各ハイパーパラメーターごとに次のように設定します。

1. sweep 設定でトップレベルの `parameters` キーを作成します。
2. `parameters` キー内に以下をネストします:
   1. 最適化したいハイパーパラメーター名を指定します。
   2. 使用したい分布を `distribution` キーで指定し、その分布のキー・値ペアをパラメータ名の下にネストします。
   3. 探索したい値を1つ以上指定します。値は `distribution` キーのレベルに記述します。  
      1. （オプション）トップレベルパラメータ名の下にもう一つ `parameters` キーを追加し、よりネストされたパラメータを定義できます。

{{% alert color="secondary" %}}
Sweep 設定で定義したネストしたパラメータは、W&B Run の config で指定したキーを上書きします。

例えば、`train.py` の Python スクリプトで以下のように W&B Run を初期化したとします（1〜2行目参照）。次に、`sweep_configuration` という辞書で Sweep 設定を定義します（4〜13行目）。最後に、この Sweep 設定辞書を `wandb.sweep` に渡して Sweep を初期化します（16行目）。

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

# 設定を渡して Sweep を初期化
sweep_id = wandb.sweep(sweep=sweep_configuration, project="<project>")

# sweep ジョブを開始
wandb.agent(sweep_id, function=main, count=4)
```
Run 初期化時に渡した `nested_param.manual_key` は利用できません。`wandb.Run.config` には Sweep 設定辞書に定義されたキー・値ペアのみが格納されます。
{{% /alert %}}

## Sweep 設定テンプレート

以下のテンプレートは、パラメータの設定や探索制約の指定方法を示しています。`hyperparameter_name` をハイパーパラメーター名に、`<>` 内の値を任意の値に置き換えてください。

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

数値を科学的記法で表現したい場合、YAML の `!!float` 演算子を利用して値を浮動小数点にキャストできます（例: `min: !!float 1e-5`）。詳細は [コマンド例]({{< relref path="#command-example" lang="ja" >}}) を参照してください。

## Sweep 設定の例

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

以下のタブでは、`early_terminate` における最小または最大イテレーション数の指定例を示します。

{{< tabpane  text=true >}}
  {{% tab header="Maximum number of iterations" %}}

この例のブラケットは: `[3, 3*eta, 3*eta*eta, 3*eta*eta*eta]` で、計算すると `[3, 9, 27, 81]` となります。

```yaml
early_terminate:
  type: hyperband
  min_iter: 3
```

  {{% /tab %}}
  {{% tab header="Minimum number of iterations" %}}

この例のブラケットは `[27/eta, 27/eta/eta]` となり、`[9, 3]` となります。

```yaml
early_terminate:
  type: hyperband
  max_iter: 27
  s: 2
```

  {{% /tab %}}
{{< /tabpane >}}

### マクロとカスタムコマンド引数の例

より複雑なコマンドライン引数が必要な場合、マクロを利用して環境変数や Python インタプリタ、追加の引数などを渡すことができます。[W&B はあらかじめ定義されたマクロ]({{< relref path="./sweep-config-keys.md#command-macros" lang="ja" >}})や、Sweep 設定内で任意に指定できるカスタムコマンドライン引数をサポートしています。

例えば、次の Sweep 設定（`sweep.yaml`）では、コマンドとして Python スクリプト（`run.py`）を実行し、`${env}`, `${interpreter}`, `${program}` というマクロが Sweep 実行時に適切な値に置き換えられます。

`--batch_size=${batch_size}`、`--test=True`、`--optimizer=${optimizer}` は、Sweep 設定で定義された `batch_size`, `test`, `optimizer` の値をコマンド引数として渡すカスタムマクロです。

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
紐づく Python スクリプト（`run.py`）側では、`argparse` モジュールでこれらのコマンドライン引数をパースできます。

```python title="run.py"
# run.py  
import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int)
parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd'], required=True)
parser.add_argument('--test', type=str2bool, default=False)
args = parser.parse_args()

# W&B Run を初期化
with wandb.init('test-project') as run:
    run.log({'validation_loss':1})
```

利用可能なマクロ一覧については、[Sweep 設定オプション]({{< relref path="./sweep-config-keys.md" lang="ja" >}})の [Command macros]({{< relref path="./sweep-config-keys.md#command-macros" lang="ja" >}}) セクションもご覧ください。

#### ブール引数

`argparse` モジュールではデフォルトでブール引数をサポートしていません。ブール値の引数を定義したい場合、[`action`](https://docs.python.org/3/library/argparse.html#action) パラメータを使うか、文字列をブール値に変換するカスタム関数を使う方法があります。

例えば、次のコードスニペットのように、`ArgumentParser` に `store_true` または `store_false` のどちらかを指定してブール引数を定義できます。

```python
import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')
args = parser.parse_args()

args.test  # --test が指定された場合 True、ない場合は False となります
```

また、文字列によるブール値表現をブール型へ変換するカスタム関数を定義することも可能です。例えば以下のコードは文字列をブール値へ変換する `str2bool` 関数の例です。

```python
def str2bool(v: str) -> bool:
  """文字列をブール値に変換します。
  argparse ではデフォルトでブール引数をサポートしていないため、この関数が必要です。
  """
  if isinstance(v, bool):
      return v
  return v.lower() in ('yes', 'true', 't', '1')
```