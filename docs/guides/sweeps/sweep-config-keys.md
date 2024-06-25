---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Sweep configuration options

sweep の設定は、ネストされたキーと値のペアで構成されます。sweep 設定のトップレベルキーを使用して、sweep 検索のパラメータ ([`parameter`](./sweep-config-keys.md#parameters) キー)、パラメータ空間を検索する方法論 ([`method`](./sweep-config-keys.md#method) キー) などの特性を定義します。

以下の表は、トップレベルの sweep 設定キーとその簡単な説明を示しています。それぞれのキーに関する詳細は該当セクションを参照してください。

| トップレベルキー  | 説明                                                                                         |
| ----------------- | ---------------------------------------------------------------------------------------------- |
| `program`         | (必須) 実行するトレーニングスクリプト。                                                      |
| `entity`          | この sweep のエンティティを指定。                                                             |
| `project`         | この sweep のプロジェクトを指定。                                                             |
| `description`     | sweep のテキスト記述。                                                                        |
| `name`            | W&B UI に表示される sweep の名前。                                                            |
| [`method`](#method) | (必須) [サーチストラテジー](./define-sweep-configuration.md#configuration-keys) を指定。     |
| [`metric`](#metric) | 最適化するメトリックを指定（一部のサーチストラテジーや停止基準でのみ使用）。                |
| [`parameters`](#parameters) | (必須) 探索する [パラメータ](define-sweep-configuration.md#parameters) 範囲を指定。  |
| [`early_terminate`](#early_terminate) | [早期停止基準](./define-sweep-configuration.md#early_terminate) を指定。                  |
| [`command`](#command)         | トレーニングスクリプトを呼び出し、引数を渡すための [コマンド構造](./define-sweep-configuration.md#command) を指定。 |
| `run_cap`        | sweep 内の最大 run 数を指定。                                                                      |

sweep 設定の構造についての詳細は、[Sweep configuration](./sweep-config-keys.md) を参照してください。

## `metric`

トップレベルの sweep 設定キー `metric` を使用して、最適化するメトリックの名前、目標、およびターゲットを指定します。

| キー  | 説明                                               |
| -------- | -------------------------------------------------- |
| `name`   | 最適化するメトリックの名前。                       |
| `goal`   | `minimize` または `maximize` のいずれか（デフォルトは `minimize`）。 |
| `target` | 最適化するメトリックの目標値。この目標値に到達した時点で新しい run を作成しません。現在実行中の run が目標値に到達した場合、エージェントは run が完了するまで待機し、その後さらに新しい run を作成しなくなります。 |

## `parameters`
YAML ファイルや Python スクリプトで、`parameters` をトップレベルキーとして指定します。`parameters` キーの中に、最適化したいハイパーパラメータの名前を記述します。一般的なハイパーパラメータには、学習率、バッチサイズ、エポック、オプティマイザーなどがあります。それぞれのハイパーパラメータについて一つ以上の探索制約を指定します。

以下の表は、サポートされているハイパーパラメータの検索制約を示しています。ハイパーパラメータとユースケースに基づいて、下記のいずれかの検索制約を使用して sweep agent に探索場所（分布の場合）または探索内容（`value`、`values` など）を指定します。

| 検索制約         | 説明                                                                                               |
| --------------- | -------------------------------------------------------------------------------------------------- |
| `values`        | このハイパーパラメータの全有効値を指定。`grid` との互換性あり。                                      |
| `value`         | このハイパーパラメータの単一の有効値を指定。`grid` との互換性あり。                                 |
| `distribution`  | 確率[分布](#distribution-options-for-random-and-bayesian-search) を指定。デフォルト値については次のノートを参照。 |
| `probabilities` | `random` を使用する際の各要素の選択確率を指定。                                                    |
| `min`, `max`    | (`int` または `float`) 最大値および最小値。`max` および `min` を整数として指定する場合は `int_uniform` で、浮動小数点数として指定する場合は `uniform` で分布します。 |
| `mu`            | (`float`) `normal` または `lognormal` で分布するハイパーパラメータの平均値。                       |
| `sigma`         | (`float`) `normal` または `lognormal` で分布するハイパーパラメータの標準偏差。                     |
| `q`             | (`float`) 量子化されたハイパーパラメータの量子化ステップサイズ。                                   |
| `parameters`    | ルートレベルのパラメータ内に他のパラメータをネスト。                                              |

:::info
[分布](#distribution-options-for-random-and-bayesian-search) が指定されていない場合、W&B は次の条件に基づいて分布を設定します:
* `values` を指定した場合は `categorical`
* `max` と `min` を整数で指定した場合は `int_uniform`
* `max` と `min` を浮動小数点数で指定した場合は `uniform`
* `value` を設定した場合は `constant`
:::

## `method`
トップレベルキー `method` でハイパーパラメータ検索戦略を指定します。ハイパーパラメータ検索戦略は3つあります: グリッド検索、ランダム検索、およびベイズ探索。
#### グリッド検索
すべてのハイパーパラメータの値の組み合わせを反復処理します。グリッド検索は、各反復で使用するハイパーパラメータの値のセットについて無作為な決定を行います。グリッド検索は計算コストが高くなる可能性があります。

連続探索空間を探索する場合、グリッド検索は無限に実行されます。

#### ランダム検索
各反復で、分布に基づいた無作為のハイパーパラメータ値のセットを選択します。ランダム検索は、コマンドライン、Python スクリプト、または [the W&B App UI](./sweeps-ui.md) からプロセスを停止しない限り、無限に実行されます。

ランダム検索を選択した場合は、metric キーで分布空間を指定します (`method: random`)。

#### ベイズ探索
[ランダム](#random-search) および [グリッド](#grid-search) 検索と対照的に、ベイズモデルは情報を元に決定を行います。ベイズ最適化は、代理関数で値をテストした後、目的関数を評価する反復プロセスを通じて使用する値を決定するために確率モデルを使用します。ベイズ探索は少数の連続パラメータにはうまく機能しますが、スケールが困難です。ベイズ探索に関する詳細は [Bayesian Optimization Primer paper](https://static.sigopt.com/b/20a144d208ef255d3b981ce419667ec25d8412e2/static/pdf/SigOpt_Bayesian_Optimization_Primer.pdf) を参照してください。

ベイズ探索は、コマンドライン、Python スクリプト、または [the W&B App UI](./sweeps-ui.md) からプロセスを停止しない限り、無限に実行されます。

### ランダムおよびベイズ探索のための分布オプション
`parameter` キー内にハイパーパラメータの名前をネストします。次に `distribution` キーを指定し、値の分布を指定します。

以下の表は、W&B がサポートする分布を示しています。

| `distribution` キーの値  | 説明                                           |
| ------------------------ | ---------------------------------------------- |
| `constant`               | 定数分布。使用する定数値 (`value`) を指定する必要があります。                    |
| `categorical`            | カテゴリ分布。このハイパーパラメータの有効なすべての値 (`values`) を指定する必要があります。 |
| `int_uniform`            | 整数の離散一様分布。`max` と `min` を整数として指定する必要があります。       |
| `uniform`                | 連続一様分布。`max` と `min` を浮動小数点数として指定する必要があります。       |
| `q_uniform`              | 量子化一様分布。`round(X / q) * q` を返します。ここで X は一様分布です。`q` のデフォルトは `1`。 |
| `log_uniform`            | 対数一様分布。自然対数が `min` から `max` の間で一様に分布しているように、`exp(min)` と `exp(max)` の間の値 `X` を返します。   |
| `log_uniform_values`     | 対数一様分布。`log(min)` と `log(max)` の間で `log(X)` が一様に分布しているように `min` と `max` の間の値 `X` を返します。     |
| `q_log_uniform`          | 量子化対数一様分布。`round(X / q) * q` を返します。ここで `X` は `log_uniform`。`q` のデフォルトは `1`。 |
| `q_log_uniform_values`   | 量子化対数一様分布。`round(X / q) * q` を返します。ここで `X` は `log_uniform_values`。`q` のデフォルトは `1`。  |
| `inv_log_uniform`        | 逆対数一様分布。`log(1/X)` が `min` から `max` の間で一様に分布している値 `X` を返します。          |
| `inv_log_uniform_values` | 逆対数一様分布。`log(1/min)` と `log(1/max)` の間で `log(1/X)` が一様に分布している値 `X` を返します。      |
| `normal`                 | 正規分布。平均 `mu`（デフォルト `0`）および標準偏差 `sigma`（デフォルト `1`）の正規分布の値を返します。 |
| `q_normal`               | 量子化正規分布。`round(X / q) * q` を返します。ここで `X` は `normal`。`q` のデフォルトは `1`。  |
| `log_normal`             | 対数正規分布。平均 `mu`（デフォルト `0`）および標準偏差 `sigma`（デフォルト `1`）で `log(X)` が正規分布している値 `X` を返します。 |
| `q_log_normal`           | 量子化対数正規分布。`round(X / q) * q` を返します。ここで `X` は `log_normal`。`q` のデフォルトは `1`。 |

## `early_terminate`

早期終了 (`early_terminate`) を使用して、パフォーマンスが悪い run を停止します。早期終了が発生した場合、W&B は現在の run を停止し、新しいハイパーパラメータ値のセットで新しい run を作成します。

:::note
`early_terminate` を使用する場合は、停止アルゴリズムを指定する必要があります。`early_terminate` 内に `type` キーをネストして指定します。
:::

### 停止アルゴリズム

:::info
W&B は現在、[Hyperband](https://arxiv.org/abs/1603.06560) 停止アルゴリズムをサポートしています。
:::

[Hyperband](https://arxiv.org/abs/1603.06560) ハイパーパラメータ最適化では、プログラムを停止するか続行するかを、*brackets* と呼ばれる一つ以上の設定された繰り返し数で判断します。

W&B run が bracket に達すると、sweep はその run のメトリックを以前に報告されたすべてのメトリック値と比較します。目標が最小化である場合、run のメトリック値が高すぎると run を終了します。目標が最大化である場合、run のメトリック値が低すぎると run を終了します。

bracket はログされた繰り返し回数に基づいています。bracket の数は最適化するメトリックをログした回数に対応しています。繰り返しはステップ、エポック、その他の中間に対応する場合があります。ステップカウンターの数値は bracket 計算に使用されません。

:::info
スケジュールを作成するには `min_iter` または `max_iter` を指定します。
:::

| キー          | 説明                                                   |
| ------------- | ------------------------------------------------------ |
| `min_iter`    | 最初の bracket のイテレーションを指定                   |
| `max_iter`    | 最大イテレーション数を指定                              |
| `s`           | 総 bracket 数を指定 (`max_iter` に必須)                 |
| `eta`         | bracket の乗数スケジュールを指定 (デフォルト: `3`)       |
| `strict`      | 'strict' モードを有効にして run を積極的に削減します。デフォルトで無効です。                         |

:::info
Hyperband は数分ごとに [W&B runs](../../ref/python/run.md) を終了するかどうかを確認します。run またはイテレーションが短い場合、指定された brackets と終了のタイムスタンプが異なる可能性があります。
:::

## `command` 

トップレベルキー `command` を使用して、コマンドの形式および内容をネストされた値で修正します。固定コンポーネント（ファイル名など）を直接含めることができます。

:::info
Unix システムでは、`/usr/bin/env` を使用して環境に基づいた正しい Python インタープリターを選択します。
:::

W&B は、コマンドの変数コンポーネントに次のマクロをサポートしています:

| コマンドマクロ              | 説明                                                                                                                                                    |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `${env}`                   | Unix システムでは `/usr/bin/env`、Windows では省略されます。                                                                                             |
| `${interpreter}`           | `python` に展開。                                                                                                                                         |
| `${program}`               | sweep 設定の `program` キーで指定されたトレーニングスクリプトファイル名。                                                                                 |
| `${args}`                  | ハイパーパラメータとその値 (`--param1=value1 --param2=value2` の形式)。                                                                                   |
| `${args_no_boolean_flags}` | ハイパーパラメータとその値 (`--param1=value1` の形式) ただし、ブール値パラメータは `True` の場合 `--boolean_flag_param` の形式で、`False` の場合は省略されます。 |
| `${args_no_hyphens}`       | ハイパーパラメータとその値 (`param1=value1 param2=value2` の形式)。                                                                                         |
| `${args_json}`             | ハイパーパラメータとその値を JSON としてエンコード。                                                                                                       |
| `${args_json_file}`        | ハイパーパラメータとその値をエンコードした JSON を含むファイルへのパス。                                                                                   |
| `${envvar}`                | 環境変数を渡す方法。 `${envvar:MYENVVAR}` __ は MYENVVAR 環境変数の値に展開されます。 __                                                                 |