---
title: sweep 設定オプション
menu:
  default:
    identifier: sweep-config-keys
    parent: define-a-sweep-configuration
---

sweep 設定は、ネストされたキーと値のペアで構成されます。sweep 設定内のトップレベルのキーを使用して、検索したいパラメータ（[`parameter`]({{< relref "./sweep-config-keys.md#parameters" >}}) キー）、パラメータ空間の探索手法（[`method`]({{< relref "./sweep-config-keys.md#method" >}}) キー）など、sweep 検索に関わる要素を定義します。

以下のテーブルは、sweep 設定で利用できるトップレベルのキーと、その簡単な説明です。各キーの詳細については、該当セクションを参照してください。

| トップレベルキー        | 説明                           |
|----------------------|------------------------------|
| `program`            | （必須）実行するトレーニングスクリプト            |
| `entity`             | この sweep に紐付く Entity        |
| `project`            | この sweep に紐付く Project        |
| `description`        | sweep の概要テキスト                |
| `name`               | sweep の名前。W&B UI で表示されます        |
| [`method`]({{< relref "#method" >}})   | （必須）探索手法                 |
| [`metric`]({{< relref "#metric" >}})   | 最適化する指標（特定の探索手法や停止基準で利用） |
| [`parameters`]({{< relref "#parameters" >}}) | （必須）探索対象パラメータの範囲              |
| [`early_terminate`]({{< relref "#early_terminate" >}}) | アーリーストッピングの条件              |
| [`command`]({{< relref "#command" >}}) | トレーニングスクリプトへの引数や呼び出し用コマンド |
| `run_cap`             | sweep で実行する run の最大数                 |

sweep 設定の構造については [Sweep configuration]({{< relref "./sweep-config-keys.md" >}}) をご覧ください。


## `metric`

`sweep` 設定のトップレベルキー `metric` を使うことで、最適化する指標の名前・ゴール・ターゲット値を指定できます。

| キー       | 説明 |
|------------|----------------------------------------------------------|
| `name`     | 最適化する指標名                                       |
| `goal`     | `minimize` か `maximize`（デフォルトは `minimize`）     |
| `target`   | 最適化する指標の目標値。run がこの値に到達すると新たな run を作りません。該当 run が終了するまで新しい run の作成は停止されます。 |


## `parameters`
YAML ファイルまたは Python スクリプト内で、`parameters` をトップレベルキーとして指定します。その下に、最適化したいハイパーパラメーター名を記述してください。よく使うハイパーパラメーターには、learning rate、バッチサイズ、エポック数、オプティマイザーなどがあります。sweep 設定で各ハイパーパラメーターに対して、1 つ以上の探索条件を記述できます。

下記のテーブルは、対応しているハイパーパラメーター探索条件の一覧です。ハイパーパラメーターやユースケースに応じて、以下の各探索条件（分布 or `value`,`values` など）を指定し、sweep agent に探索範囲などを伝えます。

|探索条件       | 説明 |
|---------------|------------------------------------------------------------------------|
| `values`      | このハイパーパラメーターで利用可能なすべての値を指定。`grid` に互換  |
| `value`       | このハイパーパラメーターで利用可能な単一の値を指定。`grid` に互換    |
| `distribution`| 探索で使う確率 [分布]({{< relref "#distribution-options-for-random-and-bayesian-search" >}}) を指定。デフォルト値はこの表の後に記載しています。|
| `probabilities` | `random` 探索時に各 `values` の選択確率を指定            |
| `min`, `max`  | (`int` または `float`) 最大・最小値。`int` は `int_uniform` 分布、`float` は `uniform` 分布で使用 |
| `mu`          | (`float`) `normal` または `lognormal` 分布の平均            |
| `sigma`       | (`float`) `normal` または `lognormal` 分布の標準偏差         |
| `q`           | (`float`) 量子化ハイパーパラメーターのステップ幅                  |
| `parameters`  | ルートレベルパラメータに他のパラメータをネストする                 |

{{% alert %}}
[distribution]({{< relref "#distribution-options-for-random-and-bayesian-search" >}}) が指定されていない場合、W&B は以下の条件で分布を自動で決定します:
* `values` を指定した場合は `categorical`
* `max`, `min` を整数で指定すると `int_uniform`
* `max`, `min` を小数で指定すると `uniform`
* `value` をセットした場合は `constant`
{{% /alert %}}


## `method`
`method` キーでハイパーパラメーター探索のアルゴリズムを指定します。選択できる探索手法は、グリッド検索（grid）、ランダム検索（random）、ベイズ探索（Bayesian search）の 3 種類です。

#### Grid search
すべてのハイパーパラメーターの組み合わせを順番に試します。グリッド検索はハイパーパラメーターセットを無作為に選ぶことなく、すべて走査するため計算コストが高くなることがあります。

連続値を持つ探索空間の場合、グリッド検索は永遠に終わりません。

#### Random search
各イテレーションで確率分布に従って無作為にハイパーパラメーター値のセットを選択します。random 探索は、コマンドラインや Python スクリプト、[W&B App]({{< relref "../sweeps-ui.md" >}}) からプロセスを手動で停止しない限り、無限に実行されます。

`method: random` のときは、分布の範囲を `metric` キーで指定できます。

#### Bayesian search
[ランダム検索]({{< relref "#random-search" >}}) や [グリッド検索]({{< relref "#grid-search" >}}) と異なり、ベイズモデルは実際の実行結果に基づいて探索先を決定します。ベイズ最適化では、確率的モデル（サロゲート関数）を使ってどの値を試すか判断し、本来の目的関数の評価を繰り返します。ベイズ探索は連続パラメータが少ない場合に効果的ですが、大きなスケールには不向きです。詳細は [Bayesian Optimization Primer paper](https://web.archive.org/web/20240209053347/https://static.sigopt.com/b/20a144d208ef255d3b981ce419667ec25d8412e2/static/pdf/SigOpt_Bayesian_Optimization_Primer.pdf) をご覧ください。

ベイズ探索も、コマンドラインや Python スクリプト、[W&B App]({{< relref "../sweeps-ui.md" >}}) からプロセスを止めない限り、永遠に続きます。

### ランダム・ベイズ探索用の分布オプション
`parameter` キーの中にハイパーパラメーター名を記述し、続けて `distribution` キーで値の分布を指定します。

次のテーブルは W&B がサポートする分布の一覧です。

| `distribution`キーの値   | 説明         |
|----------------------|-----------------------------------|
| `constant`               | 一定の値を返す分布。`value` で値を指定 |
| `categorical`            | 離散的な分布。`values` ですべての値を指定 |
| `int_uniform`            | 整数上での一様分布。`max`/`min` を整数で指定 |
| `uniform`                | 連続値の一様分布。`max`/`min` を小数で指定 |
| `q_uniform`              | 量子化一様分布。`round(X / q) * q`（Xは一様分布）。`q`のデフォルトは1 |
| `log_uniform`            | 対数一様分布。`exp(min)` から `exp(max)` の間の値を返し、自然対数が一様分布となる |
| `log_uniform_values`     | 対数一様分布。`min` から `max` の間で `log(X)` が `log(min)` から `log(max)` に一様分布 |
| `q_log_uniform`          | 量子化対数一様分布。`round(X / q) * q`（Xは `log_uniform`）。`q`のデフォルトは1 |
| `q_log_uniform_values`   | 量子化対数一様分布。`round(X / q) * q`（Xは `log_uniform_values`）。`q`のデフォルトは1 |
| `inv_log_uniform`        | 逆対数一様分布。`log(1/X)` が `min` から `max` で一様分布するようなXを返す |
| `inv_log_uniform_values` | 逆対数一様分布。`log(1/X)` が `log(1/max)` から `log(1/min)` の間で一様分布 |
| `normal`                 | 正規分布。返り値は平均 `mu`（デフォルト `0`）、標準偏差 `sigma`（デフォルト `1`）で正規分布 |
| `q_normal`               | 量子化正規分布。`round(X / q) * q`（Xは正規分布）。Qのデフォルトは1 |
| `log_normal`             | 対数正規分布。`log(X)` が平均 `mu`（デフォルト `0`）、標準偏差 `sigma`（デフォルト `1`）で正規分布となるXを返す |
| `q_log_normal`  | 量子化対数正規分布。`round(X / q) * q`（Xは `log_normal`）。`q`のデフォルトは1 |


## `early_terminate`

`early_terminate`（アーリーストッピング）を使用すると、成績が良くない run を途中で停止できます。アーリーストッピングが発動した場合、W&B はその run を早期終了し、次のハイパーパラメーターで新しい run を作成します。

{{% alert %}}
`early_terminate` を利用する場合、停止アルゴリズム（stopping algorithm）が必須です。`early_terminate` 内に `type` キーをネストしてください。
{{% /alert %}}


### 停止アルゴリズム

{{% alert %}}
W&B は現時点で [Hyperband](https://arxiv.org/abs/1603.06560) 停止アルゴリズムに対応しています。
{{% /alert %}}

[Hyperband](https://arxiv.org/abs/1603.06560) ハイパーパラメーター最適化は、プログラムを特定のイテレーション（*bracket* と呼ばれる）毎に継続・停止すべきか評価します。

W&B run が bracket に到達すると、その run の指標が過去の記録と比較されます。指標（metric）が高すぎる（最小化の場合）または低すぎる（最大化の場合）はその run が終了します。

bracket の数はロギングしたイテレーション数を元に決定されます。最適化対象の metric を記録した回数が bracket 数に対応します。イテレーションカウント（stepなど）の数値自体は bracket の判定に使われません。

{{% alert %}}
bracket スケジュールを作成するには `min_iter` または `max_iter` のいずれかを指定してください。
{{% /alert %}}


| キー        | 説明                                               |
|-------------|----------------------------------------------------|
| `min_iter`  | 最初の bracket とするイテレーション数を指定             |
| `max_iter`  | 最大イテレーション数を指定                            |
| `s`         | bracket の合計数を指定（`max_iter` の場合必須）            |
| `eta`       | bracket の分割スケジュール倍率（デフォルトは `3`）         |
| `strict`    | Hyperband 論文通りの厳格な動作モードを有効化。run を積極的に間引きます（デフォルト: false）|


{{% alert %}}
Hyperband は数分に 1 回、[runs]({{< relref "/ref/python/sdk/classes/run.md" >}}) の終了可否をチェックします。run またはイテレーションが非常に短いと、スケジュールした bracket とは終了タイミングがズレる場合があります。
{{% /alert %}}

## `command`

`command` キーの中でネストした値を使うことで、コマンドのフォーマットや内容を柔軟に調整できます。ファイル名などの固定要素も直接記載可能です。

{{% alert %}}
Unix システムでは `/usr/bin/env` を使うことで、環境に応じて適切な Python インタプリタを選択できます。
{{% /alert %}}

W&B ではコマンド内で変数的に利用できる以下のマクロが用意されています:

| コマンドマクロ                   | 説明                                                                                                                                               |
|------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| `${env}`                     | Unix システムで `/usr/bin/env`、Windows では省略されます                                                            |
| `${interpreter}`             | `python` に展開                                                                                                                                |
| `${program}`                 | sweep 設定の `program` キーで指定したトレーニングスクリプト名                                                                  |
| `${args}`                    | `--param1=value1 --param2=value2` の形式でハイパーパラメーターと値を展開                                                         |
| `${args_no_boolean_flags}`   | ブーリアンパラメータは `--boolean_flag_param`（True の場合のみ）を使い、それ以外は `--param1=value1` 形式                          |
| `${args_no_hyphens}`         | `param1=value1 param2=value2` 形式でハイパーパラメーターを展開                                                                |
| `${args_json}`               | JSON 形式でハイパーパラメーターと値をエンコード                                                                                      |
| `${args_json_file}`          | ハイパーパラメーターと値を JSON でエンコードしたファイルへのパス                                                                             |
| `${envvar}`                  | 環境変数を渡すために利用。たとえば `${envvar:MYENVVAR}` は MYENVVAR 環境変数の値に展開されます                                       |