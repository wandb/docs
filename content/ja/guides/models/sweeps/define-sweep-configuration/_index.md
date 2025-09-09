---
title: sweep configuration を定義する
description: Sweeps の 設定ファイルを作成する方法を学びましょう。
menu:
  default:
    identifier: ja-guides-models-sweeps-define-sweep-configuration-_index
    parent: sweeps
url: guides/sweeps/define-sweep-configuration
weight: 3
---

W&B Sweep は、ハイパーパラメーターの 値 を探索する戦略と、それらを評価する コード を組み合わせたものです。戦略は、全探索のように単純なものから、ベイズ最適化 と Hyperband（[BOHB](https://arxiv.org/abs/1807.01774)）のように複雑なものまであります。

sweep configuration は [Python dictionary](https://docs.python.org/3/tutorial/datastructures.html#dictionaries) か [YAML](https://yaml.org/) ファイルで定義できます。どちらで定義するかは、sweep をどのように管理したいかによって決まります。

{{% alert %}}
コマンドラインから sweep を初期化して sweep agent を開始したい場合は、sweep configuration を YAML ファイルで定義してください。Python スクリプトまたは ノートブック の中だけで sweep を初期化して開始する場合は、Python の dictionary で sweep を定義してください。
{{% /alert %}}

以下のガイドでは、sweep configuration の書式について説明します。トップレベルの sweep configuration キーの包括的な一覧は、[Sweep configuration のオプション]({{< relref path="./sweep-config-keys.md" lang="ja" >}}) を参照してください。

## 基本構造

YAML と Python dictionary のどちらの形式も、キーと 値 のペアおよびネスト構造を使います。

sweep configuration では、トップレベルの キー を使って、sweep 検索の特性（たとえば sweep の名前（[`name`]({{< relref path="./sweep-config-keys.md" lang="ja" >}}) キー）、探索するパラメータ（[`parameters`]({{< relref path="./sweep-config-keys.md#parameters" lang="ja" >}}) キー）、パラメータ空間の探索手法（[`method`]({{< relref path="./sweep-config-keys.md#method" lang="ja" >}}) キー）など）を定義します。

たとえば、以下の コードスニペット は、同一の sweep configuration を YAML ファイルと Python dictionary の両方で定義した例です。ここでは `program`、`name`、`method`、`metric`、`parameters` の 5 つのトップレベル キー を指定しています。

{{< tabpane  text=true >}}
  {{% tab header="CLI" %}}
コマンドライン（CLI）から対話的に sweeps を管理したい場合は、YAML ファイルで sweep configuration を定義します。

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
  {{% tab header="Python スクリプトまたは ノートブック" %}}
トレーニング アルゴリズムを Python スクリプトまたは ノートブック で記述する場合は、Python の dictionary データ構造で sweep を定義します。

以下の コードスニペット は、変数 `sweep_configuration` に sweep configuration を格納しています:

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

トップレベルの `parameters` キーの下には、`learning_rate`、`batch_size`、`epoch`、`optimizer` の各キーがネストされています。指定した各ネスト キーに対して、1 つ以上の 値、分布、確率などを与えることができます。詳しくは、[Sweep configuration のオプション]({{< relref path="./sweep-config-keys.md" lang="ja" >}}) の [parameters]({{< relref path="./sweep-config-keys.md#parameters" lang="ja" >}}) セクションを参照してください。

## 二重ネストのパラメータ

sweep configuration はネストしたパラメータをサポートします。ネストしたパラメータを表すには、トップレベルのパラメータ名の下にもう 1 つの `parameters` キーを使います。複数レベルのネストに対応しています。

ベイズ または ランダム のハイパーパラメーター探索を行う場合は、乱数変数の確率分布を指定してください。各ハイパーパラメーターについて:

1. sweep config にトップレベルの `parameters` キーを作成します。
2. `parameters` キーの中に次をネストします:
   1. 最適化したいハイパーパラメーター名を指定します。
   2. 使用する分布を `distribution` キーで指定します。`distribution` のキーと 値 のペアは、そのハイパーパラメーター名の下にネストします。
   3. 探索する 値 を 1 つ以上指定します。これらの 値 は distribution キーと同じ階層に記述します。  
      1. （オプション）トップレベルのパラメータ名の下にさらに `parameters` キーを追加して、ネストしたパラメータを表現します。










{{% alert color="secondary" %}}
sweep configuration で定義されたネスト パラメータは、W&B run の設定で指定されたキーを上書きします。

たとえば、`train.py` という Python スクリプトで（1–2 行目を参照）次の設定で W&B run を初期化したとします。次に、`sweep_configuration` という辞書で sweep configuration を定義します（4–13 行目を参照）。そして、その辞書を `wandb.sweep` に渡して sweep config を初期化します（16 行目を参照）。


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
W&B run の初期化時に渡した `nested_param.manual_key` は参照できません。`wandb.Run.config` には、sweep configuration の辞書で定義されたキーと 値 のペアのみが含まれます。
{{% /alert %}}

## Sweep configuration のテンプレート

以下のテンプレートでは、パラメータの設定方法と検索制約の指定方法を示します。`hyperparameter_name` は対象のハイパーパラメーター名に、`<>` で囲まれた 値 は適切な 値 に置き換えてください。

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

数値を指数表記で表現するには、YAML の `!!float` 演算子を付けて 値 を浮動小数点数にキャストします。例: `min: !!float 1e-5`。詳しくは [Command の例]({{< relref path="#command-example" lang="ja" >}}) を参照してください。

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
  {{% tab header="Python スクリプトまたは ノートブック" %}}

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

以下のタブでは、`early_terminate` に対して反復回数の最小または最大を指定する方法を示します:

{{< tabpane  text=true >}}
  {{% tab header="最大反復回数" %}}

この例のブラケットは `[3, 3*eta, 3*eta*eta, 3*eta*eta*eta]` で、`[3, 9, 27, 81]` になります。  

```yaml
early_terminate:
  type: hyperband
  min_iter: 3
```

  {{% /tab %}}
  {{% tab header="最小反復回数" %}}

この例のブラケットは `[27/eta, 27/eta/eta]` で、`[9, 3]` になります。 

```yaml
early_terminate:
  type: hyperband
  max_iter: 27
  s: 2
```

  {{% /tab %}}
{{< /tabpane >}}

### マクロとカスタム コマンド引数の例

より複雑なコマンドライン引数には、マクロを使って 環境変数、Python インタープリタ、追加の引数を渡せます。W&B は [あらかじめ定義されたマクロ]({{< relref path="./sweep-config-keys.md#command-macros" lang="ja" >}}) と、sweep configuration で指定できるカスタム コマンドライン引数をサポートします。

たとえば、次の sweep configuration（`sweep.yaml`）は、Python スクリプト（`run.py`）を実行する command を定義し、sweep 実行時に `${env}`、`${interpreter}`、`${program}` のマクロが適切な 値 に置き換えられます。

`--batch_size=${batch_size}`、`--test=True`、`--optimizer=${optimizer}` の引数は、sweep configuration で定義した `batch_size`、`test`、`optimizer` パラメータの 値 を渡すためのカスタム マクロです。

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
対応する Python スクリプト（`run.py`）は、`argparse` モジュールでこれらのコマンドライン引数をパースできます。 

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

使用可能な事前定義マクロの一覧は、[Sweep configuration のオプション]({{< relref path="./sweep-config-keys.md" lang="ja" >}}) の [Command macros]({{< relref path="./sweep-config-keys.md#command-macros" lang="ja" >}}) セクションを参照してください。

#### ブール引数

`argparse` モジュールはデフォルトではブール引数をサポートしません。ブール引数を定義するには、[`action`](https://docs.python.org/3/library/argparse.html#action) パラメータを使用するか、ブール 値 の文字列表現をブール型に変換するカスタム関数を使用します。

たとえば、次の コードスニペット では、ブール引数を定義しています。`ArgumentParser` に `store_true` または `store_false` を引数として渡します。 

```python
import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')
args = parser.parse_args()

args.test  # --test が指定された場合は True、指定されなければ False になります
```

また、ブール 値 の文字列表現をブール型に変換するカスタム関数を定義することもできます。たとえば、次の コードスニペット は、文字列をブール 値 に変換する `str2bool` 関数を定義しています。 

```python
def str2bool(v: str) -> bool:
  """Convert a string to a boolean. This is required because
  argparse does not support boolean arguments by default.
  """
  if isinstance(v, bool):
      return v
  return v.lower() in ('yes', 'true', 't', '1')
```