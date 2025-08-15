---
title: sweep 設定オプション
menu:
  default:
    identifier: ja-guides-models-sweeps-define-sweep-configuration-sweep-config-keys
    parent: define-a-sweep-configuration
---

スイープ設定は、ネストされたキーと値のペアで構成されています。スイープ設定の中でトップレベルのキーを利用して、検索するパラメータ（[`parameter`]({{< relref path="./sweep-config-keys.md#parameters" lang="ja" >}}) キー）、パラメータ空間の探索方法（[`method`]({{< relref path="./sweep-config-keys.md#method" lang="ja" >}}) キー）など、スイープの特徴を定義します。

以下の表は、主要なスイープ設定キーとその簡単な説明です。各キーについての詳細は、それぞれのセクションをご覧ください。

| トップレベルキー | 説明 |
| -------------- | ----------- |
| `program` | （必須）実行するトレーニングスクリプト |
| `entity` | このスイープに関連付ける Entity |
| `project` | このスイープに関連付ける Project |
| `description` | スイープの説明テキスト |
| `name` | スイープ名。W&B UI で表示されます。 |
| [`method`]({{< relref path="#method" lang="ja" >}}) | （必須）探索戦略 |
| [`metric`]({{< relref path="#metric" lang="ja" >}}) | 最適化するメトリクス（特定の探索戦略や停止基準で利用） |
| [`parameters`]({{< relref path="#parameters" lang="ja" >}}) | （必須）探索するパラメータの範囲 |
| [`early_terminate`]({{< relref path="#early_terminate" lang="ja" >}}) | アーリーストッピングの基準 |
| [`command`]({{< relref path="#command" lang="ja" >}}) | トレーニングスクリプト呼び出し用のコマンド構造 |
| `run_cap` | このスイープで作成する最大 run 数 |

スイープ設定の構造や詳細は [スイープ設定]({{< relref path="./sweep-config-keys.md" lang="ja" >}}) をご覧ください。

## `metric`

スイープ設定のトップレベルキー `metric` を使って、最適化対象のメトリクス名やゴール、ターゲットとなるメトリクス値を指定します。

| キー | 説明 |
| -------- | --------------------------------------------------------- |
| `name`   | 最適化対象のメトリクス名 |
| `goal`   | `minimize` か `maximize` のいずれか（デフォルトは `minimize`） |
| `target` | 目標値。run がこの値に到達した場合、新しい run は作成されません。run 中のエージェントは run の完了まで待ち、新規 run の作成を停止します。 |

## `parameters`
YAML ファイルや Python スクリプト内で、トップレベルキーとして `parameters` を指定します。その下に最適化したいハイパーパラメータ名を記述します。一般的なハイパーパラメータには learning rate（学習率）、batch size（バッチサイズ）、epochs（エポック数）、オプティマイザーなどがあります。各ハイパーパラメーターにつき、1つ以上の検索制約を設定します。

以下の表は利用可能なハイパーパラメータ探索制約の一覧です。パラメータやユースケースに応じて、エージェントに探索空間や使用する値・範囲を指定してください。

| 検索制約 | 説明 |
| --------------- | ------------------------------------------------------------------------------ |
| `values`        | このハイパーパラメータで許容される全ての値。`grid` と互換。    |
| `value`         | このハイパーパラメータの単一値。`grid` と互換。  |
| `distribution`  | 確率 [分布]({{< relref path="#distribution-options-for-random-and-bayesian-search" lang="ja" >}}) を指定。デフォルト値については次の注記参照。|
| `probabilities` | `random` 使用時に、`values` の各要素が選ばれる確率。  |
| `min`, `max`    | (`int` または `float`) の最大・最小値。整数なら `int_uniform` 分布、浮動小数点なら `uniform` 分布。 |
| `mu`            | (`float`) `normal` または `lognormal` 分布の平均。 |
| `sigma`         | (`float`) `normal` または `lognormal` 分布の標準偏差。|
| `q`             | (`float`) 量子化ハイパーパラメータのステップ幅。     |
| `parameters`    | ルートパラメータ内に他のパラメータをネスト可能。    |

{{% alert %}}
W&B では [distribution]({{< relref path="#distribution-options-for-random-and-bayesian-search" lang="ja" >}}) を指定しない場合、以下の条件で分布が自動設定されます:
* `values` がある場合は `categorical`
* `min` と `max` を整数で指定した場合は `int_uniform`
* `min` と `max` を浮動小数で指定した場合は `uniform`
* `value` を指定した場合は `constant`
{{% /alert %}}

## `method`
`method` キーでハイパーパラメータ検索戦略を指定します。選べる戦略はグリッド検索・ランダム検索・ベイズ探索の3つです。
#### グリッド検索
全てのハイパーパラメータ値の組み合わせを総当たりで試します。グリッド検索は各イテレーションで使用する値をランダムではなくすべて列挙します。計算コストが高くなる場合があります。

連続空間を検索対象とした場合、グリッド検索は無限に実行されます。

#### ランダム検索
探索する値の分布に基づき、毎回ランダムにハイパーパラメータ値セットを選びます。ランダム検索は、コマンドラインや Python スクリプト、または [W&B App]({{< relref path="../sweeps-ui.md" lang="ja" >}}) からプロセスを停止しない限り、永遠に実行され続けます。

`method: random` を選んだ場合は、`metric` キーなどで分布空間を指定してください。

#### ベイズ探索
[ランダム]({{< relref path="#random-search" lang="ja" >}}) や [グリッド]({{< relref path="#grid-search" lang="ja" >}}) 検索と異なり、ベイズモデルは、事前知識や過去結果を踏まえた決定を行います。ベイズ最適化は、代用関数で値をテストしながら、目的関数の評価前に次の値を選択します。連続パラメータが少ない場合によく機能しますが、大規模なパラメータ空間ではスケールしにくいです。詳細は [Bayesian Optimization Primer 論文](https://web.archive.org/web/20240209053347/https://static.sigopt.com/b/20a144d208ef255d3b981ce419667ec25d8412e2/static/pdf/SigOpt_Bayesian_Optimization_Primer.pdf) をご覧ください。

ベイズ探索も、コマンドラインや Python スクリプト、[W&B App]({{< relref path="../sweeps-ui.md" lang="ja" >}}) から停止しない限り、無限に実行されます。

### ランダム・ベイズ探索向け分布オプション
`parameter` キー内にハイパーパラメータ名を記述し、その下で `distribution` キーを指定して分布を明示します。

次表は、W&B でサポートしている分布の内容です。

| `distribution` キー値           | 説明            |
| ------------------------ | ------------------------------------ |
| `constant`               | 定数分布。`value` で値を指定。                    |
| `categorical`            | 離散カテゴリ分布。`values` ですべての候補値を指定。 |
| `int_uniform`            | 整数の一様分布。`min`, `max` を整数で指定。     |
| `uniform`                | 実数の一様分布。`min`, `max` を浮動小数で指定。    |
| `q_uniform`              | 量子化一様分布。`round(X / q) * q`（Xは一様分布）`q`はデフォルト1。|
| `log_uniform`            | 対数一様分布。`exp(min)`〜`exp(max)` の範囲で、logが均一分布。|
| `log_uniform_values`     | 対数一様分布。`min`〜`max` の範囲で `log(X)` が一様分布。|
| `q_log_uniform`          | 量子化対数一様分布。`X`は`log_uniform`分布の値。`q`はデフォルト1。 |
| `q_log_uniform_values`   | 量子化対数一様分布。`X`は`log_uniform_values`分布の値。`q`はデフォルト1。  |
| `inv_log_uniform`        | 逆対数一様分布。`log(1/X)` が `min`〜`max` で一様分布。|
| `inv_log_uniform_values` | 逆対数一様分布。`log(1/X)` が `log(1/max)`〜`log(1/min)` で一様分布。|
| `normal`                 | 正規分布。平均 `mu`（デフォルト0）、標準偏差`sigma`（デフォルト1）。|
| `q_normal`               | 量子化正規分布。`X`は`normal`分布。`q`はデフォルト1。|
| `log_normal`             | 対数正規分布。`log(X)` が平均`mu`・標準偏差`sigma`の正規分布。|
| `q_log_normal`  | 量子化対数正規分布。`X` は `log_normal`。`q` デフォルト1。 |

## `early_terminate`

`early_terminate`（アーリーストッピング設定）を使うと、低パフォーマンスな run を早期に停止できます。指定した条件で早期停止が発生すると W&B は現在の run を終了し、新しいハイパーパラメータ値で新規 run を作成します。

{{% alert %}}
`early_terminate` を有効利用するには、停止アルゴリズムを明示的に指定してください。スイープ設定内の `early_terminate` 配下に `type` キーをネストします。
{{% /alert %}}

### 停止アルゴリズム

{{% alert %}}
W&B では現時点で [Hyperband](https://arxiv.org/abs/1603.06560) 停止アルゴリズムのみサポートしています。
{{% /alert %}}

[Hyperband](https://arxiv.org/abs/1603.06560) ハイパーパラメータ最適化は、*ブラケット* と呼ばれるいくつかの事前設定イテレーションカウントごとに、プログラムを停止すべきか継続すべきか判定します。

W&B run がブラケットに到達すると、その run のメトリクスとこれまでのメトリクス履歴を比較します。目標が最小化の場合に値が高すぎる（もしくは最大化のときに値が低すぎる）と、その run は停止されます。

ブラケットはログ取得イテレーション数に基づきます。ブラケット数は、最適化しているメトリクスのログ回数と一致します。イテレーションはステップ、エポック、その他任意のカウンタに対応できますが、ブラケット計算にはステップカウンターの数値自体は使用されません。

{{% alert %}}
ブラケット方式のスケジュールを作るには `min_iter` または `max_iter` のいずれかを指定してください。
{{% /alert %}}

| キー        | 説明                                                    |
| ---------- | -------------------------------------------------------------- |
| `min_iter` | 最初のブラケットのイテレーション数を指定                    |
| `max_iter` | 最大イテレーション数を指定                      |
| `s`        | ブラケットの合計数を指定（`max_iter` 時は必須） |
| `eta`      | ブラケット乗法スケジュールを指定（デフォルト: `3`）       |
| `strict`   | オリジナル Hyperband 論文に忠実な run を積極的にプルーニングする 'strict' モード。デフォルトは false。 |

{{% alert %}}
Hyperband は、いくつかの分ごとに [runs]({{< relref path="/ref/python/sdk/classes/run.md" lang="ja" >}}) の終了判定をします。そのため run またはイテレーションが短い場合、終了タイミングがブラケット指定からずれる場合があります。
{{% /alert %}}

## `command` 

`command` キーの内部で値をネストすることで、書式や内容を柔軟にカスタマイズできます。ファイル名などの固定要素を直接含めることも可能です。

{{% alert %}}
Unix系システムでは `/usr/bin/env` によって、環境に合わせて正しい Python インタプリタが選択されます。
{{% /alert %}}

W&B ではコマンドの可変部分向けに以下のマクロをサポートしています。

| コマンドマクロ                | 説明                                                                                                                                                      |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `${env}`                   | Unix 系では `/usr/bin/env`、Windows では省略。                                                                                                                    |
| `${interpreter}`           | `python` に展開されます。                                                                                                                                    |
| `${program}`               | スイープ設定 `program` キーで指定したトレーニングスクリプトのファイル名。                                                                                   |
| `${args}`                  | ハイパーパラメータとその値（例: `--param1=value1 --param2=value2`）。                                                                                        |
| `${args_no_boolean_flags}` | ハイパーパラメータとその値（例: `--param1=value1`）。ブーリアンパラメータは True の場合だけ `--boolean_flag_param`、False の時は非表示。                             |
| `${args_no_hyphens}`       | ハイパーパラメータとその値（例: `param1=value1 param2=value2`）。                                                                                                 |
| `${args_json}`             | ハイパーパラメータと値を JSON 形式でエンコード。                                                                                                               |
| `${args_json_file}`        | ハイパーパラメータと値が JSON でエンコードされたファイルのパス。                                                                                                |
| `${envvar}`                | 環境変数の値を渡す方法。`${envvar:MYENVVAR}` __は、MYENVVAR 環境変数の値に展開されます。__                                               |