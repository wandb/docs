---
title: sweep configuration の オプション
menu:
  default:
    identifier: ja-guides-models-sweeps-define-sweep-configuration-sweep-config-keys
    parent: define-a-sweep-configuration
---

sweep configuration は、入れ子になった キー と 値 のペアで構成されます。sweep configuration のトップレベル キーを使って、探索するパラメータ（[`parameter`]({{< relref path="./sweep-config-keys.md#parameters" lang="ja" >}}) キー）や、パラメータ空間の探索に使う手法（[`method`]({{< relref path="./sweep-config-keys.md#method" lang="ja" >}}) キー）など、sweep 検索の属性を定義します。

以下の表は、トップレベルの sweep configuration キーと、その概要です。各キーの詳細は、それぞれのセクションを参照してください。


| Top-level keys | Description |
| -------------- | ----------- |
| `program` | （必須）実行するトレーニングスクリプト |
| `entity` | この sweep の Entity |
| `project` | この sweep の Project |
| `description` | sweep のテキストによる説明 |
| `name` | W&B の UI に表示される sweep の名前 |
| [`method`]({{< relref path="#method" lang="ja" >}}) | （必須）検索戦略 |
| [`metric`]({{< relref path="#metric" lang="ja" >}}) | 最適化する指標（特定の検索戦略や停止条件でのみ使用） |
| [`parameters`]({{< relref path="#parameters" lang="ja" >}}) | （必須）探索するパラメータの範囲 |
| [`early_terminate`]({{< relref path="#early_terminate" lang="ja" >}}) | 任意の早期停止条件 |
| [`command`]({{< relref path="#command" lang="ja" >}}) | トレーニングスクリプトの呼び出しと引数受け渡しのためのコマンド構造 |
| `run_cap` | この sweep の最大 run 数 |

sweep configuration の構造については、[Sweep configuration]({{< relref path="./sweep-config-keys.md" lang="ja" >}}) を参照してください。




## `metric`

トップレベルの `metric` キーで、最適化する指標の名前、目標（goal）、ターゲットを指定します。

|Key | Description |
| -------- | --------------------------------------------------------- |
| `name`   | 最適化する指標の名前。                           |
| `goal`   | `minimize` または `maximize`（デフォルトは `minimize`）。  |
| `target` | 最適化する指標の目標 値。run が指定したターゲット 値 に到達した場合、sweep は新しい run を作成しません。実行中のエージェントに run がある場合（run がターゲットに到達したとき）、その run の完了を待ってから、エージェントは新規 run の作成を停止します。 |




## `parameters`
YAML ファイルまたは Python スクリプトで、トップレベルのキーとして `parameters` を指定します。`parameters` キーの中に、最適化したいハイパーパラメーターの名前を記述します。よく使われるハイパーパラメーターには、学習率、バッチサイズ、エポック、オプティマイザー などがあります。sweep configuration で定義する各ハイパーパラメーターに対して、1 つ以上の検索制約を指定します。

以下の表は、サポートされているハイパーパラメーターの検索制約です。ハイパーパラメーターとユースケースに応じて、以下のいずれかの検索制約を使って、sweep agent に分布（どこを探索するか）や `value`・`values` など（何を探索・使用するか）を指示します。


| Search constraint | Description   |
| --------------- | ------------------------------------------------------------------------------ |
| `values`        | このハイパーパラメーターの取りうるすべての 値 を指定します。`grid` と互換。    |
| `value`         | このハイパーパラメーターの単一の 値 を指定します。`grid` と互換。  |
| `distribution`  | 確率[分布]({{< relref path="#distribution-options-for-random-and-bayesian-search" lang="ja" >}})を指定します。デフォルト値の注意事項は、この表の後のノートを参照してください。 |
| `probabilities` | `random` を使うとき、`values` の各要素を選択する確率を指定します。  |
| `min`, `max`    |（`int` または `float`）最小値と最大値。`int` の場合は `int_uniform` 分布のハイパーパラメーター、`float` の場合は `uniform` 分布のハイパーパラメーターに使用します。 |
| `mu`            |（`float`）`normal` または `lognormal` 分布の平均（mean）パラメータ。 |
| `sigma`         |（`float`）`normal` または `lognormal` 分布の標準偏差パラメータ。 |
| `q`             |（`float`）量子化されたハイパーパラメーターの量子化ステップ幅。     |
| `parameters`    | ルート レベルのパラメータの内側に、他のパラメータを入れ子にします。    |


{{% alert %}}
[distribution]({{< relref path="#distribution-options-for-random-and-bayesian-search" lang="ja" >}}) を指定しない場合、W&B は次の条件に基づいて分布を自動設定します:
* `values` を指定した場合は `categorical`
* `max` と `min` を整数で指定した場合は `int_uniform`
* `max` と `min` を浮動小数で指定した場合は `uniform`
* `value` に単一値を指定した場合は `constant`
{{% /alert %}}

## `method`
`method` キーでハイパーパラメーターの検索戦略を指定します。選べる検索戦略は、grid、random、ベイズ探索 の 3 種類です。
#### Grid search
ハイパーパラメーター 値 のすべての組み合わせを反復します。Grid search は各反復で使うハイパーパラメーター 値 の選択に関して、事前知識を用いない手法です。計算コストが高くなる場合があります。     

連続的な探索空間を探索している場合、Grid search は永遠に実行されます。

#### Random search
分布に基づいて、各反復でランダム（無作為）にハイパーパラメーター 値 の集合を選びます。Random search は、コマンドライン、Python スクリプト内、または [the W&B App]({{< relref path="../sweeps-ui.md" lang="ja" >}}) から停止しない限り、永遠に実行されます。

random（`method: random`）を選ぶ場合は、metric キーで分布空間を指定します。

#### Bayesian search
[random]({{< relref path="#random-search" lang="ja" >}}) や [grid]({{< relref path="#grid-search" lang="ja" >}}) と異なり、ベイズ モデルは情報に基づいた意思決定を行います。ベイズ最適化は、代理関数上で 値 を試す反復的なプロセスを通じて、どの 値 を使うかを確率モデルで決定してから、目的関数を評価します。ベイズ探索は、少数の連続パラメータに対しては有効ですが、スケールは良くありません。ベイズ探索の詳細は、[Bayesian Optimization Primer paper](https://web.archive.org/web/20240209053347/https://static.sigopt.com/b/20a144d208ef255d3b981ce419667ec25d8412e2/static/pdf/SigOpt_Bayesian_Optimization_Primer.pdf) を参照してください。




ベイズ探索は、コマンドライン、Python スクリプト内、または [the W&B App]({{< relref path="../sweeps-ui.md" lang="ja" >}}) から停止しない限り、永遠に実行されます。

### Distribution options for random and Bayesian search
`parameter` キーの中に、ハイパーパラメーターの名前を入れ子にします。次に、`distribution` キーで、その 値 に対して使用する分布を指定します。

以下の表は、W&B がサポートする分布です。

| Value for `distribution` key  | Description            |
| ------------------------ | ------------------------------------ |
| `constant`               | 一様な定数分布。使用する定数 値（`value`）を指定する必要があります。                    |
| `categorical`            | カテゴリカル分布。このハイパーパラメーターに対するすべての有効な 値（`values`）を指定する必要があります。 |
| `int_uniform`            | 整数上の離散一様分布。`max` と `min` を整数で指定する必要があります。     |
| `uniform`                | 連続一様分布。`max` と `min` を浮動小数で指定する必要があります。      |
| `q_uniform`              | 量子化された一様分布。一様分布の `X` に対して `round(X / q) * q` を返します。`q` のデフォルトは `1`。|
| `log_uniform`            | 対数一様分布。`exp(min)` と `exp(max)` の間の 値 `X` を返し、自然対数が `min` と `max` の間で一様に分布します。   |
| `log_uniform_values`     | 対数一様分布。`min` と `max` の間の 値 `X` を返し、`log(`X`)` が `log(min)` と `log(max)` の間で一様に分布します。     |
| `q_log_uniform`          | 量子化された対数一様分布。`log_uniform` の `X` に対して `round(X / q) * q` を返します。`q` のデフォルトは `1`。 |
| `q_log_uniform_values`   | 量子化された対数一様分布。`log_uniform_values` の `X` に対して `round(X / q) * q` を返します。`q` のデフォルトは `1`。  |
| `inv_log_uniform`        | 逆対数一様分布。`log(1/X)` が `min` と `max` の間で一様に分布するような `X` を返します。 |
| `inv_log_uniform_values` | 逆対数一様分布。`log(1/X)` が `log(1/max)` と `log(1/min)` の間で一様に分布するような `X` を返します。    |
| `normal`                 | 正規分布。返り値は平均 `mu`（デフォルト `0`）と標準偏差 `sigma`（デフォルト `1`）の正規分布に従います。|
| `q_normal`               | 量子化された正規分布。`normal` の `X` に対して `round(X / q) * q` を返します。Q の既定値は 1 です。  |
| `log_normal`             | 対数正規分布。自然対数 `log(X)` が平均 `mu`（デフォルト `0`）、標準偏差 `sigma`（デフォルト `1`）の正規分布に従うような 値 `X` を返します。 |
| `q_log_normal`  | 量子化された対数正規分布。`log_normal` の `X` に対して `round(X / q) * q` を返します。`q` のデフォルトは `1`。 |



## `early_terminate`

`early_terminate`（早期終了）を使って、成績の悪い run を停止できます。早期終了が発生した場合、W&B は新しいハイパーパラメーターの組み合わせで run を作成する前に、現在の run を停止します。

{{% alert %}}
`early_terminate` を使う場合は、停止アルゴリズムを必ず指定してください。sweep configuration の `early_terminate` の中に `type` キーを入れ子にします。
{{% /alert %}}


### Stopping algorithm

{{% alert %}}
W&B は現在、[Hyperband](https://arxiv.org/abs/1603.06560) の停止アルゴリズムをサポートしています。 
{{% /alert %}}

[Hyperband](https://arxiv.org/abs/1603.06560) によるハイパーパラメーター最適化では、あらかじめ設定した 1 回以上の反復回数で、そのプログラムを停止すべきか継続すべきかを評価します。これらの反復回数は、 *ブラケット* と呼ばれます。

W&B の run がブラケットに到達すると、sweep はその run の metric を、これまでに報告されたすべての metric 値と比較します。目的が最小化のときに metric が高すぎる場合、または目的が最大化のときに metric が低すぎる場合、sweep はその run を終了します。

ブラケットは、ログされた反復回数に基づきます。ブラケットの数は、最適化対象の metric をログする回数に対応します。反復は step、epoch、またはその中間の何かに対応しても構いません。step カウンターの数値自体は、ブラケットの計算には使用されません。

{{% alert %}}
ブラケットのスケジュールを作るには、`min_iter` か `max_iter` のいずれかを指定してください。
{{% /alert %}}


| Key        | Description                                                    |
| ---------- | -------------------------------------------------------------- |
| `min_iter` | 最初のブラケットに対応する反復回数を指定します。                    |
| `max_iter` | 反復回数の上限を指定します。                      |
| `s`        | ブラケットの総数を指定します（`max_iter` を使う場合は必須）。 |
| `eta`      | ブラケットの乗数スケジュールを指定します（デフォルト: `3`）。        |
| `strict`   | 'strict' モードを有効にして、より攻撃的に run を刈り込み、元の Hyperband 論文により近い動作にします。デフォルトは false。 |



{{% alert %}}
Hyperband は数分おきに、どの [runs]({{< relref path="/ref/python/sdk/classes/run.md" lang="ja" >}}) を終了するかを確認します。run や反復が短い場合、終了タイムスタンプは指定したブラケットからずれることがあります。
{{% /alert %}}

## `command` 




`command` キーの中に 値 を入れ子にして、書式や内容を変更できます。ファイル名などの固定要素を直接含めることができます。

{{% alert %}}
Unix 系システムでは、`/usr/bin/env` により、環境に基づいて OS が正しい Python インタープリタを選択します。
{{% /alert %}}

W&B は、コマンドの可変要素に対して次のマクロをサポートします。

| Command macro              | Description                                                                                                                                                           |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `${env}`                   | Unix 系システムでは `/usr/bin/env`、Windows では省略。                                                                                                                   |
| `${interpreter}`           | `python` に展開されます。                                                                                                                                                  |
| `${program}`               | sweep configuration の `program` キーで指定したトレーニングスクリプトのファイル名。                                                                                          |
| `${args}`                  | `--param1=value1 --param2=value2` の形式のハイパーパラメーターとその 値。                                                                                       |
| `${args_no_boolean_flags}` | `--param1=value1` の形式のハイパーパラメーターとその 値。ただし、boolean パラメータは `True` のとき `--boolean_flag_param` の形、`False` のときは省略されます。 |
| `${args_no_hyphens}`       | `param1=value1 param2=value2` の形式のハイパーパラメーターとその 値。                                                                                           |
| `${args_json}`             | JSON にエンコードされたハイパーパラメーターとその 値。                                                                                                                     |
| `${args_json_file}`        | JSON にエンコードされたハイパーパラメーターとその 値 を含むファイルへのパス。                                                                                   |
| `${envvar}`                | 環境変数を渡す方法。`${envvar:MYENVVAR}` __ は MYENVVAR 環境変数の値に展開されます。 __                                               |