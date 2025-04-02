---
title: Sweep configuration options
menu:
  default:
    identifier: ja-guides-models-sweeps-define-sweep-configuration-sweep-config-keys
    parent: define-a-sweep-configuration
---

sweep configurationは、ネストされたキーと値のペアで構成されています。sweep configuration内のトップレベルキーを使用して、検索するパラメータ（[`parameter`]({{< relref path="./sweep-config-keys.md#parameters" lang="ja" >}}) キー）、パラメータ空間を検索する方法（[`method`]({{< relref path="./sweep-config-keys.md#method" lang="ja" >}}) キー）など、sweep検索の品質を定義します。

以下の表は、トップレベルのsweep configurationキーと簡単な説明を示しています。各キーの詳細については、それぞれのセクションを参照してください。

| トップレベルキー | 説明 |
| -------------- | ----------- |
| `program` | （必須）実行するトレーニングスクリプト |
| `entity` | このsweepのエンティティ |
| `project` | このsweepのプロジェクト |
| `description` | sweepのテキストによる説明 |
| `name` | sweepの名前。W&B UIに表示されます。 |
| [`method`]({{< relref path="#method" lang="ja" >}}) | （必須）検索戦略 |
| [`metric`]({{< relref path="#metric" lang="ja" >}}) | 最適化するメトリック（特定の検索戦略と停止基準でのみ使用されます） |
| [`parameters`]({{< relref path="#parameters" lang="ja" >}}) | （必須）検索するパラメータ範囲 |
| [`early_terminate`]({{< relref path="#early_terminate" lang="ja" >}}) | 早期停止基準 |
| [`command`]({{< relref path="#command" lang="ja" >}}) | トレーニングスクリプトを呼び出し、引数を渡すためのコマンド構造 |
| `run_cap` | このsweepのrunの最大数 |

sweep configurationの構造化方法の詳細については、[Sweep configuration]({{< relref path="./sweep-config-keys.md" lang="ja" >}})の構造を参照してください。

## `metric`

`metric` トップレベルのsweep configurationキーを使用して、最適化する名前、目標、およびターゲットメトリックを指定します。

| キー | 説明 |
| -------- | --------------------------------------------------------- |
| `name`   | 最適化するメトリックの名前。 |
| `goal`   | `minimize` または `maximize`（デフォルトは `minimize`）。 |
| `target` | 最適化しているメトリックの目標値。指定した目標値にrunが到達した場合、sweepは新しいrunを作成しません。（runがターゲットに到達すると）runを実行しているアクティブなエージェントは、エージェントが新しいrunの作成を停止するまでrunの完了を待ちます。 |

## `parameters`
YAMLファイルまたはPythonスクリプトで、`parameters`をトップレベルキーとして指定します。`parameters`キーの中で、最適化するハイパーパラメータの名前を指定します。一般的なハイパーパラメータには、学習率、バッチサイズ、エポック、オプティマイザーなどがあります。sweep configurationで定義するハイパーパラメータごとに、1つまたは複数の検索制約を指定します。

次の表は、サポートされているハイパーパラメータ検索制約を示しています。ハイパーパラメータとユースケースに基づいて、以下の検索制約のいずれかを使用して、sweep agentに検索場所（分布の場合）または検索または使用するもの（`value`、`values`など）を指示します。

| 検索制約 | 説明   |
| --------------- | ------------------------------------------------------------------------------ |
| `values`        | このハイパーパラメータの有効な値をすべて指定します。`grid`と互換性があります。 |
| `value`         | このハイパーパラメータの単一の有効な値を指定します。`grid`と互換性があります。 |
| `distribution`  | 確率[分布]({{< relref path="#distribution-options-for-random-and-bayesian-search" lang="ja" >}})を指定します。デフォルト値については、この表の後の注記を参照してください。 |
| `probabilities` | `random`を使用する場合に、`values`の各要素を選択する確率を指定します。 |
| `min`、`max`    | （`int`または `float`）最大値と最小値。`int`の場合、`int_uniform`分散ハイパーパラメータ用。`float`の場合、`uniform`分散ハイパーパラメータ用。 |
| `mu`            | （`float`）`normal` - または `lognormal` - 分散ハイパーパラメータの平均パラメータ。 |
| `sigma`         | （`float`）`normal` - または `lognormal` - 分散ハイパーパラメータの標準偏差パラメータ。 |
| `q`             | （`float`）量子化されたハイパーパラメータの量子化ステップサイズ。 |
| `parameters`    | ルートレベルのパラメータ内に他のパラメータをネストします。 |

{{% alert %}}
[分布]({{< relref path="#distribution-options-for-random-and-bayesian-search" lang="ja" >}})が指定されていない場合、W&Bは次の条件に基づいて次の分布を設定します。
* `values`を指定した場合は`categorical`
* `max`と`min`を整数として指定した場合は`int_uniform`
* `max`と`min`をfloatとして指定した場合は`uniform`
* `value`にセットを提供した場合は`constant`
{{% /alert %}}

## `method`
`method`キーでハイパーパラメータ検索戦略を指定します。選択できるハイパーパラメータ検索戦略は、グリッド、ランダム、ベイズ探索の3つです。
#### グリッド検索
ハイパーパラメータ値のすべての組み合わせを反復処理します。グリッド検索は、各反復で使用するハイパーパラメータ値のセットについて、情報に基づかない決定を行います。グリッド検索は、計算コストが高くなる可能性があります。

グリッド検索は、連続検索空間内で検索している場合、永久に実行されます。

#### ランダム検索
分布に基づいて、各反復でランダムで情報に基づかないハイパーパラメータ値のセットを選択します。コマンドライン、Pythonスクリプト内、または[W&B App UI]({{< relref path="../sweeps-ui.md" lang="ja" >}})からプロセスを停止しない限り、ランダム検索は永久に実行されます。

ランダム（`method: random`）検索を選択した場合は、メトリックキーで分布空間を指定します。

#### ベイズ探索
[ランダム]({{< relref path="#random-search" lang="ja" >}})および[グリッド]({{< relref path="#grid-search" lang="ja" >}})検索とは対照的に、ベイズモデルは情報に基づいた決定を行います。ベイズ最適化は、確率モデルを使用して、目的関数を評価する前に代用関数で値をテストする反復プロセスを通じて、使用する値を決定します。ベイズ探索は、連続パラメータの数が少ない場合にはうまく機能しますが、スケールは劣ります。ベイズ探索の詳細については、[ベイズ最適化入門論文](https://web.archive.org/web/20240209053347/https://static.sigopt.com/b/20a144d208ef255d3b981ce419667ec25d8412e2/static/pdf/SigOpt_Bayesian_Optimization_Primer.pdf)を参照してください。

ベイズ探索は、コマンドライン、Pythonスクリプト内、または[W&B App UI]({{< relref path="../sweeps-ui.md" lang="ja" >}})からプロセスを停止しない限り、永久に実行されます。

### ランダムおよびベイズ探索の分布オプション
`parameter`キー内で、ハイパーパラメータの名前をネストします。次に、`distribution`キーを指定し、値の分布を指定します。

次の表は、W&Bがサポートする分布を示しています。

| `distribution`キーの値  | 説明 |
| ------------------------ | ------------------------------------ |
| `constant`               | 定数分布。使用する定数（`value`）を指定する必要があります。 |
| `categorical`            | カテゴリ分布。このハイパーパラメータの有効な値（`values`）をすべて指定する必要があります。 |
| `int_uniform`            | 整数に対する離散一様分布。`max`と`min`を整数として指定する必要があります。 |
| `uniform`                | 連続一様分布。`max`と`min`をfloatとして指定する必要があります。 |
| `q_uniform`              | 量子化された一様分布。`round(X / q) * q`を返します。ここで、Xは一様です。`q`のデフォルトは`1`です。 |
| `log_uniform`            | 対数一様分布。`exp(min)`と`exp(max)`の間の値`X`を返します。自然対数は`min`と`max`の間で均等に分布します。 |
| `log_uniform_values`     | 対数一様分布。`min`と`max`の間の値`X`を返します。`log(`X`)`は`log(min)`と`log(max)`の間で均等に分布します。 |
| `q_log_uniform`          | 量子化された対数一様分布。`round(X / q) * q`を返します。ここで、`X`は`log_uniform`です。`q`のデフォルトは`1`です。 |
| `q_log_uniform_values`   | 量子化された対数一様分布。`round(X / q) * q`を返します。ここで、`X`は`log_uniform_values`です。`q`のデフォルトは`1`です。 |
| `inv_log_uniform`        | 逆対数一様分布。`X`を返します。ここで、`log(1/X)`は`min`と`max`の間で均等に分布します。 |
| `inv_log_uniform_values` | 逆対数一様分布。`X`を返します。ここで、`log(1/X)`は`log(1/max)`と`log(1/min)`の間で均等に分布します。 |
| `normal`                 | 正規分布。平均`mu`（デフォルト`0`）および標準偏差`sigma`（デフォルト`1`）で正規分布した値を返します。 |
| `q_normal`               | 量子化された正規分布。`round(X / q) * q`を返します。ここで、`X`は`normal`です。Qのデフォルトは1です。 |
| `log_normal`             | 対数正規分布。自然対数`log(X)`が平均`mu`（デフォルト`0`）および標準偏差`sigma`（デフォルト`1`）で正規分布するように、値`X`を返します。 |
| `q_log_normal`  | 量子化された対数正規分布。`round(X / q) * q`を返します。ここで、`X`は`log_normal`です。`q`のデフォルトは`1`です。 |

## `early_terminate`

パフォーマンスの低いrunを停止するには、早期終了（`early_terminate`）を使用します。早期終了が発生した場合、W&Bは新しいハイパーパラメータ値のセットで新しいrunを作成する前に、現在のrunを停止します。

{{% alert %}}
`early_terminate`を使用する場合は、停止アルゴリズムを指定する必要があります。sweep configuration内の`early_terminate`内に`type`キーをネストします。
{{% /alert %}}

### 停止アルゴリズム

{{% alert %}}
現在、W&Bは[Hyperband](https://arxiv.org/abs/1603.06560)停止アルゴリズムをサポートしています。
{{% /alert %}}

[Hyperband](https://arxiv.org/abs/1603.06560)ハイパーパラメータ最適化は、プログラムを停止するか、事前設定された1つ以上の反復回数（*ブラケット*と呼ばれる）で続行するかを評価します。

W&Bのrunがブラケットに到達すると、sweepはそのrunのメトリックを以前に報告されたすべてのメトリック値と比較します。runのメトリック値が高すぎる場合（目標が最小化の場合）、またはrunのメトリックが低すぎる場合（目標が最大化の場合）、sweepはrunを終了します。

ブラケットは、ログに記録された反復回数に基づいています。ブラケットの数は、最適化するメトリックをログに記録する回数に対応します。反復は、ステップ、エポック、またはその間の何かに対応できます。ステップカウンターの数値は、ブラケットの計算には使用されません。

{{% alert %}}
ブラケットスケジュールを作成するには、`min_iter`または`max_iter`のいずれかを指定します。
{{% /alert %}}

| キー | 説明 |
| ---------- | -------------------------------------------------------------- |
| `min_iter` | 最初のブラケットの反復を指定します |
| `max_iter` | 最大反復回数を指定します。 |
| `s` | ブラケットの総数を指定します（`max_iter`に必要）。 |
| `eta` | ブラケット乗数スケジュールを指定します（デフォルト：`3`）。 |
| `strict`   | 元のHyperband論文に厳密に従って、runをより積極的にプルーニングする「strict」モードを有効にします。デフォルトはfalseです。 |

{{% alert %}}
Hyperbandは、数分ごとに終了する[W&B run]({{< relref path="/ref/python/run.md" lang="ja" >}})を確認します。runまたは反復が短い場合、終了runのタイムスタンプは、指定されたブラケットと異なる場合があります。
{{% /alert %}}

## `command`

`command`キー内のネストされた値を使用して、形式と内容を変更します。ファイル名などの固定コンポーネントを直接含めることができます。

{{% alert %}}
Unixシステムでは、`/usr/bin/env`は、OSが環境に基づいて正しいPythonインタープリターを選択するようにします。
{{% /alert %}}

W&Bは、コマンドの可変コンポーネントに対して次のマクロをサポートしています。

| コマンドマクロ | 説明 |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `${env}`                   | Unixシステムでは`/usr/bin/env`、Windowsでは省略。 |
| `${interpreter}`           | `python`に展開されます。 |
| `${program}`               | sweep configurationの`program`キーで指定されたトレーニングスクリプトのファイル名。 |
| `${args}`                  | `--param1=value1 --param2=value2`の形式のハイパーパラメータとその値。 |
| `${args_no_boolean_flags}` | `--param1=value1`の形式のハイパーパラメータとその値。ただし、ブールパラメータは`True`の場合は`--boolean_flag_param`の形式になり、`False`の場合は省略されます。 |
| `${args_no_hyphens}`       | `param1=value1 param2=value2`の形式のハイパーパラメータとその値。 |
| `${args_json}`             | JSONとしてエンコードされたハイパーパラメータとその値。 |
| `${args_json_file}`        | JSONとしてエンコードされたハイパーパラメータとその値を含むファイルへのパス。 |
| `${envvar}`                | 環境変数を渡す方法。`${envvar:MYENVVAR}` __はMYENVVAR環境変数の値に展開されます。__ |
