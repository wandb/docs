---
description: スイープの設定ファイルの作成方法を学びます。
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# スイープ構成の定義

<head>
  <title>ハイパーパラメータチューニングのためのスイープ構成を定義します。</title>
</head>

Weights & Biasesスイープは、ハイパーパラメーターの値を探索する戦略と、それらを評価するコードを組み合わせたものです。戦略は、すべてのオプションを試すだけのシンプルなものから、ベイズ最適化やHyperband（[BOHB](https://arxiv.org/abs/1807.01774)）のように複雑なものまであります。

スイープ構成の形で戦略を定義します。構成は以下のいずれかで指定してください：

1. Jupyter NotebookまたはPythonスクリプトを使用している場合は、Pythonのネストされたディクショナリデータ構造。
2. コマンドライン（CLI）を使用している場合は、YAMLファイル。

以下のコードスニペットは、Jupyter NotebookやPythonスクリプト内、またはYAMLファイル内でスイープ構成を定義する方法の例を示しています。構成キーは、後続のセクションで詳細に定義されています。


<Tabs
  defaultValue="script"
  values={[
    {label: 'PythonスクリプトまたはJupyterノートブック', value: 'script'},
    {label: 'YAML', value: 'yaml'},
  ]}>
  <TabItem value="script">
  JupyterノートブックやPythonスクリプト内で、スイープをPythonディクショナリデータ構造として定義します。

```python
sweep_configuration = {
    'method': 'random',
    'name': 'sweep',
    'metric': {
        'goal': 'minimize', 
        'name': 'validation_loss'
		},
    'parameters': {
        'batch_size': {'values': [16, 32, 64]},
        'epochs': {'values': [5, 10, 15]},
        'lr': {'max': 0.1, 'min': 0.0001}
     }
}
```
  </TabItem>
  <TabItem value="yaml">
  YAML でキーを持つマップを作成し、そのキーの値にさらにキーを持たせることができます。

```yaml
program: train.py
method: bayes
metric:
  name: validation_loss
  goal: minimize
parameters:
  learning_rate:
    min: 0.0001
    max: 0.1
  optimizer:
    values: ["adam", "sgd"]
```
  </TabItem>
</Tabs>

:::注意
1. PythonスクリプトやJupyterノートブック内で、スイープが最適化するために定義した正確なメトリック名をログ（`wandb.log`）に記録してください。
2. W&Bスイープエージェントを開始した後は、スイープ構成を変更することはできません。
:::

例えば、W&Bスイープでトレーニング中の検証精度を最大化したい場合、Pythonスクリプト内で検証精度を`val_loss`という変数に格納します。YAML構成ファイルでは、以下のように定義します。

```yaml
metric:
  goal: maximize
  name: val_loss
```

PythonスクリプトやJupyterノートブックで`val_loss`変数（この例では）をW&Bにログに記録する必要があります。

```python
wandb.log({
        'val_loss': validation_loss
      })
```

### スイープ構成の構造

スイープ構成はネストされており、キーはさらにキーを値として持つことができます。以下にトップレベルのキーを一覧し、簡単に説明した後に、次のセクションで詳しく述べます。

| トップレベルキー   | 説明                                                                                                                               |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `program`         | （必須）実行するトレーニングスクリプト。                                                                                         |
| `method`          | （必須）[検索戦略](./define-sweep-configuration.md#configuration-keys)を指定します。                                                  |
| `parameters`      | （必須）[パラメーター](define-sweep-configuration.md#parameters)の検索範囲を指定します。                                              |
| `name`            | W&B UIで表示されるスイープの名前。                                                                                               |
| `description`     | スイープのテキスト説明。                                                                                                         |
| `metric`          | 最適化するメトリックを指定します（特定の検索戦略および停止基準でのみ使用されます）。                                                  |
| `early_terminate` | [早期停止基準](./define-sweep-configuration.md#early_terminate)を指定します。                                                       |
| `command`         | トレーニングスクリプトに引数を渡すための[コマンド構造](./define-sweep-configuration.md#command)を指定します。                      |
| `project`         | このスイープのプロジェクトを指定します。                                                                                         |
| `entity`          | このスイープのエンティティを指定します。                                                                                         |
| `run_cap` | スイープ内の最大ラン数を指定します。                                                                                           |
### 検索タイプのメソッド

以下のリストは、ハイパーパラメーター検索方法を説明しています。検索戦略は `method` で指定します。

* **`grid`**  - 各ハイパーパラメーターの値のすべての組み合わせを繰り返します。計算コストが高いことがあります。
* **`random`**  - 提供された分布に基づいて、各反復でランダムなセットのハイパーパラメーター値を選択します。
* **`bayes`** - ハイパーパラメーターの関数としてメトリックスコアの確率モデルを作成し、メトリックを改善する可能性が高いパラメーターを選択します。ベイジアンハイパーパラメーター検索方法は、ガウスプロセスを使用してパラメーターとモデルメトリックの関係をモデル化し、改善の確率を最適化するためのパラメーターを選択します。この戦略では、`metric`キーを指定する必要があります。連続パラメーターの数が少ない場合にはうまく機能するが、スケールは悪い。


<Tabs
  defaultValue="random"
  values={[
    {label: 'ランダム検索', value: 'random'},
    {label: 'グリッド検索', value: 'grid'},
    {label: 'ベイズ検索', value: 'bayes'},
  ]}>
  <TabItem value="random">

  ```yaml  
  method: random
  ```

  </TabItem>
  <TabItem value="grid">

  ```yaml  
  method: grid
  ```

  </TabItem>
  <TabItem value="bayes">


```yaml
  method: bayes
  metric:
    name: val_loss
    goal: minimize
  ```

  </TabItem>
</Tabs>

:::警告
ランダムおよびベイズ探索は、コマンドライン、Pythonスクリプト内、または[UI](./sweeps-ui.md)からプロセスを停止するまで永遠に実行されます。連続した探索空間の中で検索を行う場合、グリッド検索も永遠に実行されます。
:::

## 設定キー

### `method`

スイープ構成の`method`キーで、検索戦略を指定します。

| `method` | 説明                                               |
| -------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `grid`   | グリッド検索は、パラメータ値のすべての組み合わせを繰り返し検索します。                                                         |
| `random` | ランダム検索は、各反復でパラメータ値のランダムなセットを選択します。        |
| `bayes`  | ベイズハイパーパラメータ検索方法では、ガウス過程を使用して、パラメータとモデル指標の関係をモデル化し、改善の確率を最適化するパラメータを選択します。この戦略では、`metric`キーを指定する必要があります。 |

### `parameters`

スイープの間に探索するハイパーパラメーターを記述します。それぞれのハイパーパラメーターについて、名前と可能な値を定数リスト（任意の`method`に対して）として指定するか、`random`または`bayes`用に`distribution`を指定します。
| 値              | 説明                                    |
| --------------- | ------------------------------------------------------------------- |
| `values`        | このハイパーパラメーターに対して有効なすべての値を指定します。`grid`と互換性があります。                                         |
| `value`         | このハイパーパラメーターに対して単一の有効な値を指定します。`grid`と互換性があります。                                                                                 |
| `distribution`  | (`str`) 下の分布表から分布を選択します。指定されていない場合、`values`が設定されている場合は`categorical`、`max`と`min`が整数に設定されている場合は`int_uniform`、`max`と`min`が小数に設定されている場合は`uniform`、`value`が設定されている場合は`constant`がデフォルトになります。 |
| `probabilities` | `random`を使用する際に、`values`の各要素を選択する確率を指定します。                                           |
| `min`, `max`    | (`int`or `float`) 最大値と最小値。`int`の場合、`int_uniform`分布のハイパーパラメーター用。`float`の場合、`uniform`分布のハイパーパラメーター用。                     |
| `mu`            | (`float`) `normal`または`lognormal`分布のハイパーパラメータに対する平均パラメータ。                                             |
| `sigma`         | (`float`) `normal`または`lognormal`分布のハイパーパラメータに対する標準偏差パラメータ。                                      |
| `q`             | (`float`) 量子化パラメータのステップサイズ。                                                                |
| `parameters`    | ルートレベルパラメータ内に他のパラメータをネストします。                                                |

#### 例

<Tabs
  defaultValue="single"
  values={[
    {label: '単一の値', value: 'single'},
    {label: '複数の値', value: 'multiple'},
    {label: '確率', value: 'probabilities'},
    {label: '分布', value: 'distribution'},
    {label: 'ネスト', value: 'nested'},
  ]}>
  <TabItem value="single">

  ```yaml
  parameter_name:
    value: 1.618
  ```

</TabItem>
  <TabItem value="multiple">

  ```yaml
  パラメータ名:
  値:
    - 8
    - 6
    - 7
    - 5
    - 3
    - 0
    - 9
  ```
  </TabItem>
  <TabItem value="probabilities">

  ```yaml
  パラメータ名:
    値: [1, 2, 3, 4, 5]
    確率: [0.1, 0.2, 0.1, 0.25, 0.35]
  ```

  </TabItem>
  <TabItem value="distribution">

  ```yaml
  パラメータ名:
    分布: 正規分布
    平均: 100
    標準偏差: 10
  ```
</TabItem>
  <TabItem value="nested">

  ```yaml
  optimizer:
      parameters:
          learning_rate:
              values: [0.01, 0.001]
          momentum:
              value: 0.9
  ```

  </TabItem>
</Tabs>

#### `distribution`

ランダム (`random`) またはベイズ (`bayes`) の探索メソッドを選択した場合に、値の分布方法を指定します。

| 値                    | 説明            |
| ------------------------ | ------------------------------------ |
| `constant`               | 定数分布。`value` を指定する必要があります。                         |
| `categorical`            | カテゴリ分布。`values` を指定する必要があります。                     |
| `int_uniform`            | 整数に対する離散一様分布。`max` と `min` を整数で指定する必要があります。     |
| `uniform`                | 連続一様分布。`max` と `min` を浮動小数点で指定する必要があります。      |
| `q_uniform`              | 量子化された一様分布。`round(X / q) * q` を返し、X は一様分布です。`q` はデフォルトで `1`。|
| `log_uniform`            | 対数一様分布。`exp(min)` と `exp(max)` の間にある値 `X` を返し、自然対数が `min` と `max` の間で一様に分布します。   |
| `log_uniform_values`     | 対数一様分布。`log(X)` が `log(min)` と `log(max)` の間で一様に分布するような、`min` と `max` の間にある値 `X` を返します。     |
| `q_log_uniform`          | 量子化された対数一様分布。`round(X / q) * q` を返し、`X` は `log_uniform` です。`q` はデフォルトで `1`。       |
| `q_log_uniform_values`   | 量子化された対数一様分布。`round(X / q) * q` を返し、`X` は `log_uniform_values` です。`q` はデフォルトで `1`。     |
| `inv_log_uniform`        | 逆対数一様分布。`log(1/X)` が `min` と `max` の間で一様に分布するような値 `X` を返します。           |
| `inv_log_uniform_values` | 逆対数一様分布。`log(1/X)` が `log(1/max)` と `log(1/min)` の間で一様に分布するような値 `X` を返します。    |
| `normal`                 | 正規分布。平均 `mu`（デフォルトは `0`）と標準偏差 `sigma`（デフォルトは `1`）で正規分布する値を返します。|
| `q_normal`               | 量子化された正規分布。`round(X / q) * q` を返し、`X` は `normal` です。`q` はデフォルトで `1`。      |
| `log_normal`             | 対数正規分布。自然対数 `log(X)` が平均 `mu`（デフォルトは `0`）と標準偏差 `sigma`（デフォルトは `1`）で正規分布する値 `X` を返します。|
| `q_log_normal`  | 量子化された対数正規分布。`round(X / q) * q` を返し、`X` は `log_normal` です。`q` はデフォルトで `1`。             |
#### 例

<Tabs
  defaultValue="constant"
  values={[
    {label: 'constant', value: 'constant'},
    {label: 'categorical', value: 'categorical'},
    {label: 'uniform', value: 'uniform'},
    {label: 'q_uniform', value: 'q_uniform'}
  ]}>
  <TabItem value="constant">

```yaml
parameter_name:
  distribution: constant
  value: 2.71828
```
  </TabItem>
  <TabItem value="categorical">

```yaml
parameter_name:
  distribution: categorical
  values:
      - elu
      - relu
      - gelu
      - selu
      - relu
      - prelu
      - lrelu
      - rrelu
      - relu6
```
  </TabItem>
  <TabItem value="uniform">

```yaml
parameter_name:
  distribution: uniform
  min: 0
  max: 1
```
  </TabItem>
  <TabItem value="q_uniform">

```yaml
parameter_name:
  distribution: q_uniform
  min: 0
  max: 256
  q: 1
```
  </TabItem>
</Tabs>

### `metric`

最適化するメトリックを記述します。このメトリックは、トレーニングスクリプトによってW&Bに明示的にログされる必要があります。

| キー      | 説明                                                                                                                                                                                                                                                         |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `name`   | 最適化するメトリックの名前。                                                                                                                                                                                                                                 |
| `goal`   | `minimize` もしくは `maximize`（デフォルトは `minimize`）。                                                                                                                                                                                                  |
| `target` | 最適化するメトリックの目標値。スイープ内の任意のrunがその目標値を達成した場合、スイープの状態は `finished` に設定されます。これは、アクティブなrunを持つすべてのエージェントがジョブを終了することを意味しますが、スイープには新しいrunが開始されません。 |

例えば、モデルの検証ロスを最小化したい場合は以下のようになります。

```python
# valid_lossとして検証損失を返すモデルトレーニングコード
wandb.log({"val_loss" : valid_loss})
```

#### 例

<Tabs
  defaultValue="maximize"
  values={[
    {label: 'Maximize', value: 'maximize'},
    {label: 'Minimize', value: 'minimize'},
    {label: 'Target', value: 'target'},
  ]}>
  <TabItem value="maximize">

```yaml
metric:
  name: val_acc
  goal: maximize
```
  </TabItem>
  <TabItem value="minimize">

```yaml
metric:
  name: val_loss
  goal: minimize
```
  </TabItem>
  <TabItem value="target">

```yaml
metric:
  name: val_acc
  goal: maximize
  target: 0.95
```
  </TabItem>
</Tabs>

:::caution
最適化するメトリクスはトップレベルのメトリクスである必要があります。
:::


スイープのメトリクスをサブディレクトリ内にログしないでください。先程のコード例では、検証ロス（`"loss": val_loss`）をログしたいと思っています。まず、それを辞書に定義します。しかし、`wandb.log`に渡される辞書では、トラッキングするキーと値のペアが指定されていません。

```
val_metrics = {
        "loss": val_loss, 
        "acc": val_acc
        }

# 不正確。辞書のキーと値のペアが提供されていません。
wandb.log({"val_loss", val_metrics})
```

代わりに、メトリクスをトップレベルでログしてください。例えば、辞書を作成した後、辞書を`wandb.log`メソッドに渡す際にキーと値のペアを指定します。

```
val_metrics = {
        "loss": val_loss, 
        "acc": val_acc
        }

wandb.log({"val_loss", val_metrics["loss"]})
```

### `early_terminate`

早期終了は、性能が低い実行を停止することでハイパーパラメータの検索を高速化するオプション機能です。早期停止がトリガされると、エージェントは現在の実行を停止し、次のハイパーパラメーターセットを試みます。

| キー  | 説明                            |
| ---- | ------------------------------ |
| `type` | 停止アルゴリズムを指定する         |

以下の停止アルゴリズムをサポートしています。

| `type`      | 説明                                                   |
| ----------- | ------------------------------------------------------------- |
| `hyperband` | [ハイパーバンド法](https://arxiv.org/abs/1603.06560) を使用する |

#### `hyperband`

[Hyperband](https://arxiv.org/abs/1603.06560) 停止は、プログラムを停止するか、いわゆる "ブラケット" と呼ばれる1つ以上のプリセット・イテレーションカウントで続行するかどうかを評価します。実行がブラケットに達すると、そのメトリクス値は、これまでに報告されたすべてのメトリクス値と比較され、値が高すぎる場合 (目標が最小化の場合) または低すぎる場合 (目標が最大化の場合) [W&B Run](https://docs.wandb.ai/ref/python/run) が終了します。

ブラケットは、ログされたイテレーションの数に基づいています。ブラケットの数は、最適化するメトリックをログする回数に対応します。イテレーションは、ステップ、エポック、またはその間のものに対応することができます。ステップカウンターの数値は、ブラケットの計算には使用されません。

:::caution
ブラケットスケジュールを作成するには、`min_iter` または `max_iter` のいずれかを指定してください。
:::

| キー        | 説明                                                    |
| ---------- | -------------------------------------------------------------- |
| `min_iter` | 最初のブラケットのイテレーションを指定する                     |
| `max_iter` | 最大イテレーション数を指定します。                             |
| `s`        | ブラケットの総数を指定します（`max_iter` に必要）              |
| `eta`      | ブラケット乗数スケジュールを指定します（デフォルト: `3`）。   |
:::info
ハイパーバンド早期終了器は、数分ごとにどの[W&B Runs](https://docs.wandb.ai/ref/python/run)を終了させるかをチェックします。ランまたは反復が短い場合、終了ランのタイムスタンプが指定されたブラケットと異なる場合があります。
:::

#### 例

<Tabs
  defaultValue="min_iter"
  values={[
    {label: 'Hyperband（min_iter）', value: 'min_iter'},
    {label: 'Hyperband（max_iter）', value: 'max_iter'},
  ]}>
  <TabItem value="min_iter">

```yaml
early_terminate:
  type: hyperband
  min_iter: 3
```

この例のブラケットは `[3, 3*eta, 3*eta*eta, 3*eta*eta*eta]` で、`[3, 9, 27, 81]` と等しいです。
  </TabItem>
  <TabItem value="max_iter">

```yaml
early_terminate:
  type: hyperband
  max_iter: 27
  s: 2
```
この例のカッコは、`[27/eta, 27/eta/eta]`であり、`[9, 3]`に等しいです。
  </TabItem>
</Tabs>

### `command` <a href="#command" id="command"></a>

<!-- [`wandb agent`](../../ref/cli/wandb-agent.md) で作成されたエージェントは、デフォルトで以下の形式のコマンドを受信します: -->

<Tabs
  defaultValue="unix"
  values={[
    {label: 'UNIX', value: 'unix'},
    {label: 'Windows', value: 'windows'},
  ]}>
  <TabItem value="unix">

```python
/usr/bin/env python train.py --param1=value1 --param2=value2
```
  </TabItem>
  <TabItem value="windows">

```python
python train.py --param1=value1 --param2=value2
```

  </TabItem>
</Tabs>

:::info
UNIXシステムでは、`/usr/bin/env`が環境に基づいて適切なPythonインタープリタが選択されることを保証します。
:::
`command`キーの下に値を指定することで、フォーマットと内容を変更することができます。ファイル名などの固定されたコマンドの要素は、直接含めることができます（以下の例を参照）。

コマンドの可変要素については、以下のマクロをサポートしています。

| コマンドマクロ                   | 説明                                                                                                                                           |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `${env}`                     | UNIXシステムでは `/usr/bin/env`、Windowsでは省略されます。                                                                                        |
| `${interpreter}`             | `python`に展開されます。                                                                                                                           |
| `${program}`                 | スイープ構成の`program`キーで指定されたトレーニングスクリプトファイル名。                                                                              |
| `${args}`                    | `--param1=value1 --param2=value2`の形式でハイパーパラメータとその値を指定します。                                                                 |
| `${args_no_boolean_flags}`   | `--param1=value1`の形式でハイパーパラメータとその値を指定しますが、ブールパラメータは`--boolean_flag_param`の形式で`True`、省略されて`False`。 |
| `${args_no_hyphens}`         | `param1=value1 param2=value2`の形式で、ハイパーパラメータとその値。                                                                                 |
| `${args_json}`               | JSONでエンコードされたハイパーパラメータとその値。                                                                                                |
| `${args_json_file}`          | JSONでエンコードされたハイパーパラメータとその値が含まれるファイルへのパス。                                                                          |
| `${envvar}`                  | 環境変数を渡す方法。`${envvar: MYENVVAR}` __MYENVVAR環境変数の値に展開されます。__                                                                      |

デフォルトのコマンドフォーマットは以下のように定義されています。

```yaml
command:
  - ${env}
  - ${interpreter}
  - ${program}
  - ${args}
```

#### 例

<Tabs
  defaultValue="python"
  values={[
    {label: 'Pythonインタプリタを設定', value: 'python'},
    {label: '追加パラメータ', value: 'parameters'},
    {label: '引数を省略', value: 'omit'},
    {label: 'Hydra', value: 'hydra'}
  ]}>
  <TabItem value="python">

`{$interpreter}`マクロを削除し、Pythonインタプリタをハードコードするために値を明示的に指定します。例えば、以下のコードスニペットはその方法を示しています。

```yaml
command:
  - ${env}
  - python3
  - ${program}
  - ${args}
```
  </TabItem>
  <TabItem value="parameters">

スイープ設定パラメーターで指定されていない追加のコマンドライン引数を追加するには：

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

プログラムが引数解析を使用していない場合、引数をまったく渡さずに、`wandb.init`が自動的にスイープパラメータを`wandb.config`に取り込む機能を利用できます。

```
command:
  - ${env}
  - ${interpreter}
  - ${program}
```
  </TabItem>
  <TabItem value="hydra">
コマンドを変更して、[Hydra](https://hydra.cc)のようなツールが期待する方法で引数を渡すことができます。詳細は[HydraとW＆B](../integrations/other/hydra.md)を参照してください。



```
command:

  - ${env}

  - ${interpreter}

  - ${program}

  - ${args_no_hyphens}

```

  </TabItem>

</Tabs>