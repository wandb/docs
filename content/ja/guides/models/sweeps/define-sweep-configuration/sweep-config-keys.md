---
title: Sweep configuration options
menu:
  default:
    identifier: ja-guides-models-sweeps-define-sweep-configuration-sweep-config-keys
    parent: define-a-sweep-configuration
---

sweep の設定は、ネストされたキーと値のペアで構成されます。 sweep 設定内でトップレベルのキーを使用して、検索するパラメーター ([`parameter`]({{< relref path="./sweep-config-keys.md#parameters" lang="ja" >}}) キー)、パラメーター空間を検索する方法 ([`method`]({{< relref path="./sweep-config-keys.md#method" lang="ja" >}}) キー) など、sweep 検索の特性を定義します。

次の表に、トップレベルの sweep 設定キーと簡単な説明を示します。 各キーの詳細については、それぞれのセクションを参照してください。

| トップレベルのキー | 説明 |
| -------------- | ----------- |
| `program` | (必須) 実行するトレーニングスクリプト |
| `entity` | この sweep のエンティティ |
| `project` | この sweep のプロジェクト |
| `description` | sweep のテキストによる説明 |
| `name` | sweep の名前。W&B UI に表示されます。 |
| [`method`]({{< relref path="#method" lang="ja" >}}) | (必須) 検索戦略 |
| [`metric`]({{< relref path="#metric" lang="ja" >}}) | 最適化するメトリック (特定の検索戦略および停止基準でのみ使用) |
| [`parameters`]({{< relref path="#parameters" lang="ja" >}}) | (必須) 検索するパラメーターの範囲 |
| [`early_terminate`]({{< relref path="#early_terminate" lang="ja" >}}) | 早期停止の基準 |
| [`command`]({{< relref path="#command" lang="ja" >}}) | トレーニングスクリプトを呼び出して引数を渡すためのコマンド構造 |
| `run_cap` | この sweep の run の最大数 |

sweep の設定方法の詳細については、[Sweep configuration]({{< relref path="./sweep-config-keys.md" lang="ja" >}}) の構造を参照してください。

## `metric`

`metric` トップレベルの sweep 設定キーを使用して、最適化する名前、目標、およびターゲットメトリックを指定します。

| キー | 説明 |
| -------- | --------------------------------------------------------- |
| `name`   | 最適化するメトリックの名前。 |
| `goal`   | `minimize` または `maximize` (デフォルトは `minimize`)。 |
| `target` | 最適化するメトリックの目標値。 指定した目標値に run が達した場合、sweep は新しい run を作成しません。 run の実行中に (run が目標に達すると) アクティブな エージェント は、 エージェント が新しい run の作成を停止するまで run が完了するのを待ちます。 |

## `parameters`
YAML ファイルまたは Python スクリプトで、`parameters` をトップレベルのキーとして指定します。 `parameters` キー内で、最適化する ハイパーパラメーター の名前を指定します。 一般的な ハイパーパラメーター には、学習率、バッチサイズ、エポック、オプティマイザー などがあります。 sweep 設定で定義する各 ハイパーパラメーター に対して、1 つ以上の検索制約を指定します。

次の表に、サポートされている ハイパーパラメーター の検索制約を示します。 ハイパーパラメーター と ユースケース に基づいて、以下の検索制約のいずれかを使用して、検索または使用する場所 (分布の場合) または内容 (`value`、`values` など) を sweep agent に指示します。

| 検索制約 | 説明 |
| --------------- | ------------------------------------------------------------------------------ |
| `values` | この ハイパーパラメーター のすべての有効な値を指定します。 `grid` と互換性があります。 |
| `value` | この ハイパーパラメーター の単一の有効な値を指定します。 `grid` と互換性があります。 |
| `distribution` | 確率 [分布]({{< relref path="#distribution-options-for-random-and-bayesian-search" lang="ja" >}}) を指定します。 デフォルト値については、この表の後の注記を参照してください。 |
| `probabilities` | `random` を使用する場合に、`values` の各要素を選択する確率を指定します。 |
| `min`、`max` | (`int` または `float`) 最大値と最小値。 `int` の場合は、`int_uniform` 分布の ハイパーパラメーター 。 `float` の場合は、`uniform` 分布の ハイパーパラメーター 。 |
| `mu` | (`float`) `normal` または `lognormal` 分布の ハイパーパラメーター の平均パラメーター。 |
| `sigma` | (`float`) `normal` または `lognormal` 分布の ハイパーパラメーター の標準偏差パラメーター。 |
| `q` | (`float`) 量子化された ハイパーパラメーター の量子化ステップサイズ。 |
| `parameters` | 他の パラメータ をルートレベルの パラメータ の内側にネストします。 |

{{% alert %}}
W&B は、[分布]({{< relref path="#distribution-options-for-random-and-bayesian-search" lang="ja" >}}) が指定されていない場合、次の条件に基づいて次の分布を設定します。
* `values` を指定した場合は `categorical`
* `max` と `min` を整数として指定した場合は `int_uniform`
* `max` と `min` を浮動小数点数として指定した場合は `uniform`
* `value` にセットを提供した場合は `constant`
{{% /alert %}}

## `method`
`method` キーを使用して、 ハイパーパラメーター 検索戦略を指定します。 選択できる ハイパーパラメーター 検索戦略は、グリッド検索、ランダム検索、ベイズ探索の 3 つです。
#### グリッド検索
ハイパーパラメーター 値のすべての組み合わせを反復処理します。 グリッド検索では、各反復で使用する ハイパーパラメーター 値のセットについて、情報に基づかない決定を行います。 グリッド検索は、計算コストが高くなる可能性があります。

グリッド検索は、連続した検索空間内で検索している場合、永久に実行されます。

#### ランダム検索
分布に基づいて、各反復で情報に基づかないランダムな ハイパーパラメーター 値のセットを選択します。 コマンドライン、Python スクリプト内、または [W&B アプリ UI]({{< relref path="../sweeps-ui.md" lang="ja" >}}) からプロセスを停止しない限り、ランダム検索は永久に実行されます。

ランダム (`method: random`) 検索を選択する場合は、メトリックキーを使用して分布空間を指定します。

#### ベイズ探索
[ランダム]({{< relref path="#random-search" lang="ja" >}}) および [グリッド]({{< relref path="#grid-search" lang="ja" >}}) 検索とは対照的に、ベイズモデルは情報に基づいた決定を行います。 ベイズ最適化は、確率モデルを使用して、目的関数を評価する前に代替関数で値をテストする反復プロセスを通じて使用する値を決定します。 ベイズ探索は、連続パラメーターの数が少ない場合には適していますが、スケールが不十分です。 ベイズ探索の詳細については、[ベイズ最適化入門論文](https://web.archive.org/web/20240209053347/https://static.sigopt.com/b/20a144d208ef255d3b981ce419667ec25d8412e2/static/pdf/SigOpt_Bayesian_Optimization_Primer.pdf) を参照してください。

ベイズ探索は、コマンドライン、Python スクリプト内、または [W&B アプリ UI]({{< relref path="../sweeps-ui.md" lang="ja" >}}) からプロセスを停止しない限り、永久に実行されます。

### ランダム検索とベイズ探索の分布オプション
`parameter` キー内で、 ハイパーパラメーター の名前をネストします。 次に、`distribution` キーを指定し、値の分布を指定します。

次の表に、W&B がサポートする分布を示します。

| `distribution` キーの値 | 説明 |
| ------------------------ | ------------------------------------ |
| `constant` | 定数分布。 使用する定数値 (`value`) を指定する必要があります。 |
| `categorical` | カテゴリ分布。 この ハイパーパラメーター のすべての有効な値 (`values`) を指定する必要があります。 |
| `int_uniform` | 整数に対する離散一様分布。 `max` と `min` を整数として指定する必要があります。 |
| `uniform` | 連続一様分布。 `max` と `min` を浮動小数点数として指定する必要があります。 |
| `q_uniform` | 量子化された一様分布。 `round(X / q) * q` を返します。ここで、X は一様です。 `q` のデフォルトは `1` です。 |
| `log_uniform` | 対数一様分布。 自然対数が `min` と `max` の間で一様に分布するように、`exp(min)` と `exp(max)` の間の値 `X` を返します。 |
| `log_uniform_values` | 対数一様分布。 `log(X)` が `log(min)` と `log(max)` の間で一様に分布するように、`min` と `max` の間の値 `X` を返します。 |
| `q_log_uniform` | 量子化された対数一様。 `round(X / q) * q` を返します。ここで、`X` は `log_uniform` です。 `q` のデフォルトは `1` です。 |
| `q_log_uniform_values` | 量子化された対数一様。 `round(X / q) * q` を返します。ここで、`X` は `log_uniform_values` です。 `q` のデフォルトは `1` です。 |
| `inv_log_uniform` | 逆対数一様分布。 `log(1/X)` が `min` と `max` の間で一様に分布する `X` を返します。 |
| `inv_log_uniform_values` | 逆対数一様分布。 `log(1/X)` が `log(1/max)` と `log(1/min)` の間で一様に分布する `X` を返します。 |
| `normal` | 正規分布。 平均 `mu` (デフォルト `0`) および標準偏差 `sigma` (デフォルト `1`) で正規分布する値を返します。 |
| `q_normal` | 量子化された正規分布。 `round(X / q) * q` を返します。ここで、`X` は `normal` です。 Q のデフォルトは 1 です。 |
| `log_normal` | 対数正規分布。 自然対数 `log(X)` が平均 `mu` (デフォルト `0`) および標準偏差 `sigma` (デフォルト `1`) で正規分布するように、値 `X` を返します。 |
| `q_log_normal` | 量子化された対数正規分布。 `round(X / q) * q` を返します。ここで、`X` は `log_normal` です。 `q` のデフォルトは `1` です。 |

## `early_terminate`

早期終了 (`early_terminate`) を使用して、パフォーマンスの低い run を停止します。 早期終了が発生した場合、W&B は新しい ハイパーパラメーター 値のセットで新しい run を作成する前に、現在の run を停止します。

{{% alert %}}
`early_terminate` を使用する場合は、停止アルゴリズムを指定する必要があります。 sweep 設定内の `early_terminate` 内に `type` キーをネストします。
{{% /alert %}}

### 停止アルゴリズム

{{% alert %}}
W&B は現在、[Hyperband](https://arxiv.org/abs/1603.06560) 停止アルゴリズムをサポートしています。
{{% /alert %}}

[Hyperband](https://arxiv.org/abs/1603.06560) ハイパーパラメーター 最適化は、プログラムを停止するか、*ブラケット* と呼ばれる 1 つまたは複数の事前設定された反復回数で続行する必要があるかを評価します。

W&B の run がブラケットに到達すると、sweep はその run のメトリックを以前に報告されたすべてのメトリック値と比較します。 run のメトリック値が高すぎる場合 (目標が最小化の場合)、または run のメトリックが低すぎる場合 (目標が最大化の場合)、sweep は run を終了します。

ブラケットは、記録された反復回数に基づいています。 ブラケットの数は、最適化しているメトリックを記録する回数に対応します。 反復は、ステップ、エポック、またはその中間にあるものに対応できます。 ステップカウンターの数値は、ブラケットの計算には使用されません。

{{% alert %}}
ブラケットスケジュールを作成するには、`min_iter` または `max_iter` のいずれかを指定します。
{{% /alert %}}

| キー | 説明 |
| ---------- | -------------------------------------------------------------- |
| `min_iter` | 最初のブラケットの反復を指定します |
| `max_iter` | 最大反復回数を指定します。 |
| `s` | ブラケットの総数を指定します (`max_iter` に必須) |
| `eta` | ブラケットの乗数スケジュールを指定します (デフォルト: `3`)。 |
| `strict` | より厳密に元の Hyperband ペーパーに従って、run を積極的にプルーニングする「厳密」モードを有効にします。 デフォルトは false です。 |

{{% alert %}}
Hyperband は、数分ごとに終了する [W&B runs]({{< relref path="/ref/python/run.md" lang="ja" >}}) をチェックします。 run または反復が短い場合、終了 run のタイムスタンプは、指定されたブラケットと異なる場合があります。
{{% /alert %}}

## `command`

`command` キー内のネストされた値を使用して、形式と内容を変更します。 ファイル名などの固定コンポーネントを直接含めることができます。

{{% alert %}}
Unix システムでは、`/usr/bin/env` は、OS が環境に基づいて正しい Python インタープリターを選択するようにします。
{{% /alert %}}

W&B は、コマンドの可変コンポーネントに対して次のマクロをサポートしています。

| コマンドマクロ | 説明 |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `${env}` | Unix システムでは `/usr/bin/env`。Windows では省略されます。 |
| `${interpreter}` | `python` に展開されます。 |
| `${program}` | sweep 設定の `program` キーで指定されたトレーニングスクリプトのファイル名。 |
| `${args}` | `--param1=value1 --param2=value2` の形式の ハイパーパラメーター とその値。 |
| `${args_no_boolean_flags}` | `--param1=value1` の形式の ハイパーパラメーター とその値。ただし、ブールパラメーターは、`True` の場合は `--boolean_flag_param` の形式になり、`False` の場合は省略されます。 |
| `${args_no_hyphens}` | `param1=value1 param2=value2` の形式の ハイパーパラメーター とその値。 |
| `${args_json}` | JSON としてエンコードされた ハイパーパラメーター とその値。 |
| `${args_json_file}` | JSON としてエンコードされた ハイパーパラメーター とその値を含むファイルへのパス。 |
| `${envvar}` | 環境変数を渡す方法。 `${envvar:MYENVVAR}` __ は、MYENVVAR 環境変数の値に展開されます。__ |
