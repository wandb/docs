---
title: Sweep configuration オプション
menu:
  default:
    identifier: ja-guides-models-sweeps-define-sweep-configuration-sweep-config-keys
    parent: define-a-sweep-configuration
---

スイープ設定は、ネストされたキーと値のペアで構成されます。スイープ設定内のトップレベルのキーを使用して、スイープ検索の特性を定義します。例えば、検索するパラメータ（[`parameter`]({{< relref path="./sweep-config-keys.md#parameters" lang="ja" >}}) キー）、パラメータ空間を検索するための方法論（[`method`]({{< relref path="./sweep-config-keys.md#method" lang="ja" >}}) キー）などがあります。

以下のテーブルはトップレベルのスイープ設定キーとその簡単な説明を示しています。各キーについての詳細情報は、該当するセクションを参照してください。

| トップレベルキー | 説明 |
| -------------- | ----------- |
| `program` | （必須）実行するトレーニングスクリプト |
| `entity` | このスイープのエンティティ |
| `project` | このスイープのプロジェクト |
| `description` | スイープのテキスト説明 |
| `name` | W&B UIに表示されるスイープの名前。 |
| [`method`]({{< relref path="#method" lang="ja" >}}) | （必須）検索戦略 |
| [`metric`]({{< relref path="#metric" lang="ja" >}}) | 最適化するメトリック（特定の検索戦略と停止基準でのみ使用） |
| [`parameters`]({{< relref path="#parameters" lang="ja" >}}) | （必須）検索するパラメータの範囲 |
| [`early_terminate`]({{< relref path="#early_terminate" lang="ja" >}}) | 任意の早期停止基準 |
| [`command`]({{< relref path="#command" lang="ja" >}}) | トレーニングスクリプトに引数を渡して呼び出すためのコマンド構造 |
| `run_cap` | このスイープの最大 run 数 |

スイープ設定の構造については、[スイープ設定]({{< relref path="./sweep-config-keys.md" lang="ja" >}})の構造を参照してください。

## `metric`

`metric` トップレベルスイープ設定キーを使用して、最適化するメトリックの名前、目標、そして対象のメトリックを指定します。

|キー | 説明 |
| -------- | --------------------------------------------------------- |
| `name`   | 最適化するメトリックの名前。                          |
| `goal`   | `minimize` または `maximize` のいずれか（デフォルトは `minimize`）。  |
| `target` | 最適化するメトリックの目標値。このスイープは、指定した目標値に run が到達した場合や到達する場合、新しい run を作成しません。アクティブなエージェントが run を実行中の場合（runがターゲットに到達した場合）、エージェントが新しい run を作成するのを停止する前に、run が完了するのを待ちます。 |

## `parameters`
YAML ファイルまたは Python スクリプト内で、`parameters` をトップレベルキーとして指定します。`parameters` キーの中に、最適化したいハイパーパラメータの名前を提供します。一般的なハイパーパラメーターには、学習率、バッチサイズ、エポック数、オプティマイザーなどがあります。あなたのスイープ設定で定義された各ハイパーパラメータに対して、1つ以上の検索制約を指定します。

以下のテーブルは、サポートされているハイパーパラメータ検索制約を示しています。ハイパーパラメータとユースケースに基づいて、以下のサーチ制約のいずれかを使用して、スイープエージェントに検索する場所（分布の場合）または何を（`value`、`values`など）検索または使用するかを指示します。

| 検索制約 | 説明   |
| --------------- | ------------------------------------------------------------------------------ |
| `values`        | このハイパーパラメータのすべての有効な値を指定します。`grid`と互換性があります。    |
| `value`         | このハイパーパラメータの単一の有効な値を指定します。`grid`と互換性があります。  |
| `distribution`  | 確率 [分布]({{< relref path="#distribution-options-for-random-and-bayesian-search" lang="ja" >}}) を指定します。この表の後の注記ではデフォルト値に関する情報について説明しています。 |
| `probabilities` | `random`を使用する際に、`values`のそれぞれの要素を選択する確率を指定します。  |
| `min`, `max`    | （`int`または`float`）最大値と最小値。`int`の場合、`int_uniform` で分布されたハイパーパラメータ用。`float`の場合、`uniform`で分布されたハイパーパラメータ用。 |
| `mu`            | ( `float` ) `normal` または `lognormal` で分布されたハイパーパラメータの平均パラメータ。 |
| `sigma`         | ( `float` ) `normal` または `lognormal` で分布されたハイパーパラメータの標準偏差パラメータ。 |
| `q`             | ( `float` ) 量子化されたハイパーパラメーターの量子化ステップサイズ。     |
| `parameters`    | ルートレベルのパラメーター内に他のパラメーターをネストします。    |

{{% alert %}}
W&B は、[distribution]({{< relref path="#distribution-options-for-random-and-bayesian-search" lang="ja" >}}) が指定されていない場合、以下の条件に基づいて以下の分布を設定します：
* `categorical`：`values`が指定された場合
* `int_uniform`：`max`と`min`が整数として指定された場合
* `uniform`：`max`と`min`が浮動小数点数として指定された場合
* `constant`：`value`にセットを提供した場合
{{% /alert %}}

## `method`
`method`キーを使用して、ハイパーパラメータ検索戦略を指定します。選択できるハイパーパラメーター検索戦略は、グリッド検索、ランダム検索、およびベイズ探索です。
#### グリッド検索
ハイパーパラメータのすべての組み合わせを反復します。グリッド検索は、各反復で使用するハイパーパラメータ値のセットに対して無知な決定を下します。グリッド検索は計算的に高コストになる可能性があります。

グリッド検索は、連続的な検索空間内を検索している場合、永遠に実行されます。

#### ランダム検索
分布に基づいて、各反復でランダムかつ無知なハイパーパラメータ値のセットを選択します。ランダム検索は、コマンドラインやあなたの python スクリプト、または [W&B アプリUI]({{< relref path="../sweeps-ui.md" lang="ja" >}}) でプロセスを停止しない限り、永遠に実行されます。

ランダム(`method: random`)検索を選択した場合、`metric`キーで分布空間を指定します。

#### ベイズ探索
[ランダム検索]({{< relref path="#random-search" lang="ja" >}})と[グリッド検索]({{< relref path="#grid-search" lang="ja" >}})とは対照的に、ベイズモデルを使用して情報に基づく決定を行います。ベイズ最適化は、確率モデルを使用して、代理関数の値をテストする反復プロセスを経て、どの値を使用するかを決定します。ベイズ探索は、少数の連続的なパラメータに対して効果的ですが、スケールがうまくいかないことがあります。ベイズ探索に関する詳細情報は、[ベイズ最適化の入門書](https://web.archive.org/web/20240209053347/https://static.sigopt.com/b/20a144d208ef255d3b981ce419667ec25d8412e2/static/pdf/SigOpt_Bayesian_Optimization_Primer.pdf)を参照してください。

ベイズ探索は、コマンドラインやあなたの python スクリプト、または [W&B アプリUI]({{< relref path="../sweeps-ui.md" lang="ja" >}}) でプロセスを停止しない限り、永遠に実行されます。

### ランダムおよびベイズ探索の分布オプション
`parameter` キー内で、ハイパーパラメーターの名前をネストします。次に、`distribution`キーを指定し、値の分布を指定します。

以下のテーブルでは、W&B がサポートする分布を示しています。

| `distribution`キーの値  | 説明            |
| ------------------------ | ------------------------------------ |
| `constant`               | 定数分布。使用する定数値（`value`）を指定する必要があります。                    |
| `categorical`            | カテゴリ分布。このハイパーパラメータのすべての有効な値（`values`）を指定する必要があります。 |
| `int_uniform`            | 整数上の離散一様分布。`max` と `min` を整数として指定する必要があります。     |
| `uniform`                | 連続一様分布。`max` と `min` を浮動小数点数として指定する必要があります。      |
| `q_uniform`              | 量子化一様分布。`X` が一様である場合、`round(X / q) * q` を返します。`q` はデフォルトで `1`。|
| `log_uniform`            | 対数一様分布。`exp(min)` と `exp(max)` の間で `X` を返し、自然対数が `min` と `max` の間で一様に分布。 |
| `log_uniform_values`     | 対数一様分布。`min` と `max` の間で `X` を返し、`log(`X`)` が `log(min)` と `log(max)` の間で一様に分布。     |
| `q_log_uniform`          | 量子化対数一様分布。`X` が `log_uniform` である場合、`round(X / q) * q` を返します。`q` はデフォルトで `1`。 |
| `q_log_uniform_values`   | 量子化対数一様分布。`X` が `log_uniform_values` である場合、`round(X / q) * q` を返します。`q` はデフォルトで `1`。  |
| `inv_log_uniform`        | 逆対数一様分布。`X` を返し、`log(1/X)` が `min` と `max` の間で一様に分布。 |
| `inv_log_uniform_values` | 逆対数一様分布。`X` を返し、`log(1/X)` が `log(1/max)` と `log(1/min)` の間で一様に分布。    |
| `normal`                 | 正規分布。返される値は平均 `mu`（デフォルト `0`）と標準偏差 `sigma`（デフォルト `1`）で通常に分布。|
| `q_normal`               | 量子化正規分布。`X` が `normal` である場合、`round(X / q) * q` を返します。`q` はデフォルトで `1`。  |
| `log_normal`             | 対数正規分布。`X` の自然対数 `log(X)` が平均 `mu`（デフォルト `0`）と標準偏差 `sigma`（デフォルト `1`）で通常に分布する値 `X` を返します。 |
| `q_log_normal`  | 量子化対数正規分布。`X` が `log_normal` である場合、`round(X / q) * q` を返します。`q` はデフォルトで `1`。 |

## `early_terminate`

実行のパフォーマンスが悪い場合に停止させるために早期終了（`early_terminate`）を使用します。早期終了が発生した場合、W&B は現在の run を停止し、新しいハイパーパラメータの値のセットで新しい run を作成します。

{{% alert %}}
`early_terminate` を使用する場合、停止アルゴリズムを指定する必要があります。スイープ設定内で `early_terminate` 内に `type` キーをネストします。
{{% /alert %}}

### 停止アルゴリズム

{{% alert %}}
W&B は現在 [Hyperband](https://arxiv.org/abs/1603.06560) 停止アルゴリズムをサポートしています。
{{% /alert %}}

[Hyperband](https://arxiv.org/abs/1603.06560) ハイパーパラメータ最適化は、プログラムが停止すべきか、先に進むべきかを、*brackets*と呼ばれるあらかじめ設定されたイテレーション数で評価します。

W&B run が bracket に到達したとき、sweep はその run のメトリックを過去に報告されたすべてのメトリック値と比較します。run のメトリック値が高すぎる場合（目標が最小化の場合）、または run のメトリックが低すぎる場合（目標が最大化の場合）、sweep は run を終了します。

ベースの反復数に基づいて bracket が設定されます。bracket の数は、最適化するメトリックをログした回数に対応します。反復はステップ、エポック、またはその中間に対応することができます。ステップカウンタの数値は bracket 計算に使用されません。

{{% alert %}}
bracket スケジュールを作成するには、`min_iter` または `max_iter` のいずれかを指定してください。
{{% /alert %}}

| キー        | 説明                                                    |
| ---------- | -------------------------------------------------------------- |
| `min_iter` | 最初の bracket の反復を指定                                    |
| `max_iter` | 最大反復数を指定。                      |
| `s`        | bracket の合計数を指定（ `max_iter` に必要） |
| `eta`      | bracket 倍数スケジュールを指定（デフォルト： `3`）。        |
| `strict`   | より厳格にオリジナルの Hyperband 論文に従って run を厳しく削減する「strict」モードを有効にします。デフォルトでは false。 |

{{% alert %}}
Hyperband は数分ごとに終了する [W&B run]({{< relref path="/ref/python/run.md" lang="ja" >}}) を確認します。終了時刻は、run やイテレーションが短い場合、指定された bracket とは異なることがあります。
{{% /alert %}}

## `command`

`command` キー内のネストされた値を使用して、形式と内容を修正できます。ファイル名などの固定コンポーネントを直接含めることができます。

{{% alert %}}
Unix システムでは、`/usr/bin/env` は環境に基づいて OS が正しい Python インタープリターを選択することを保証します。
{{% /alert %}}

W&B は、コマンドの可変コンポーネントのために次のマクロをサポートしています：

| コマンドマクロ              | 説明                                                                                                                                                           |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `${env}`                   | Unix システムでは `/usr/bin/env`、Windows では省略されます。                                                                                                                   |
| `${interpreter}`           | `python` に展開されます。                                                                                                                                                  |
| `${program}`               | スイープ設定 `program` キーで指定されたトレーニングスクリプトファイル名。                                                                                          |
| `${args}`                  | `--param1=value1 --param2=value2` の形式でのハイパーパラメーターとその値。                                                                                       |
| `${args_no_boolean_flags}` | ハイパーパラメータとその値が `--param1=value1` の形式であるが、ブールパラメータは `True` の場合を `--boolean_flag_param` の形にし、`False` の場合は省略します。 |
| `${args_no_hyphens}`       | `param1=value1 param2=value2` の形式でのハイパーパラメータとその値。                                                                                           |
| `${args_json}`             | JSON としてエンコードされたハイパーパラメーターとその値。                                                                                                                     |
| `${args_json_file}`        | JSON としてエンコードされたハイパーパラメータとその値を含むファイルへのパス。                                                                                   |
| `${envvar}`                | 環境変数を渡す方法。`${envvar:MYENVVAR}` __ は MYENVVAR 環境変数の値に展開されます。 __                                               |