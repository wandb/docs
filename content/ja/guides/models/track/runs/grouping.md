---
title: Group runs into experiments
description: トレーニング と評価の run を、より大規模な Experiments にグループ化します。
menu:
  default:
    identifier: ja-guides-models-track-runs-grouping
    parent: what-are-runs
---

個々のジョブをグループ化して実験をまとめるには、一意の **group** 名を **wandb.init()** に渡します。

## ユースケース

1. **分散トレーニング:** 実験が、より大きな全体の一部として捉えられるべき個別のトレーニング スクリプトや評価スクリプトに分割されている場合は、グループ化を使用します。
2. **複数のプロセス**: 複数のより小さなプロセスをグループ化して、1つの実験としてまとめます。
3. **K-分割交差検証**: 異なる乱数シードを持つrunをグループ化して、より大規模な実験を把握します。Sweepsとグループ化を用いたk-分割交差検証の[例](https://github.com/wandb/examples/tree/master/examples/wandb-sweeps/sweeps-cross-validation)をご覧ください。

グループ化を設定するには、次の3つの方法があります。

### 1. スクリプトでグループを設定する

オプションの group と job_type を wandb.init() に渡します。これにより、個々のrunを含む実験ごとに専用のグループページが作成されます。例：`wandb.init(group="experiment_1", job_type="eval")`

### 2. グループ環境変数を設定する

`WANDB_RUN_GROUP` を使用して、runのグループを環境変数として指定します。詳細については、[**環境変数**]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) に関するドキュメントをご覧ください。**Group** は、**Project** 内で一意であり、グループ内のすべてのrunで共有される必要があります。`wandb.util.generate_id()` を使用して、すべてのプロセスで使用する一意の8文字の文字列を生成できます。たとえば、`os.environ["WANDB_RUN_GROUP"] = "experiment-" + wandb.util.generate_id()` のようにします。

### 3. UIでグループ化を切り替える

任意の設定列で動的にグループ化できます。たとえば、`wandb.config` を使用してバッチサイズまたは学習率をログに記録する場合、Webアプリケーションでこれらのハイパーパラメーターを動的にグループ化できます。

## グループ化による分散トレーニング

`wandb.init()` でグループ化を設定すると、UIでrunがデフォルトでグループ化されます。これは、テーブルの上部にある **Group** ボタンをクリックしてオン/オフを切り替えることができます。グループ化を設定した [サンプルコード](http://wandb.me/grouping) から生成された [プロジェクトの例](https://wandb.ai/carey/group-demo?workspace=user-carey) を示します。サイドバーの各「Group」行をクリックすると、その実験専用のグループページに移動できます。

{{< img src="/images/track/distributed_training_wgrouping_1.png" alt="" >}}

上記のプロジェクトページから、左側のサイドバーにある **Group** をクリックして、[このページ](https://wandb.ai/carey/group-demo/groups/exp_5?workspace=user-carey) のような専用ページに移動できます。

{{< img src="/images/track/distributed_training_wgrouping_2.png" alt="" >}}

## UI での動的なグループ化

任意の列（例えば、ハイパーパラメーター）でrunをグループ化できます。その様子を以下に示します。

* **サイドバー**: Runはエポック数でグループ化されています。
* **グラフ**: 各線はグループの平均値を表し、網掛けは分散を示します。この振る舞いはグラフ設定で変更できます。

{{< img src="/images/track/demo_grouping.png" alt="" >}}

## グループ化をオフにする

グループ化ボタンをクリックして、グループフィールドをいつでもクリアできます。これにより、テーブルとグラフはグループ化されていない状態に戻ります。

{{< img src="/images/track/demo_no_grouping.png" alt="" >}}

## グラフ設定のグループ化

グラフの右上隅にある編集ボタンをクリックし、**Advanced** タブを選択して、線と網掛けを変更します。各グループの線の平均値、最小値、または最大値を選択できます。網掛けの場合、網掛けをオフにしたり、最小値と最大値、標準偏差、および標準誤差を表示したりできます。

{{< img src="/images/track/demo_grouping_options_for_line_plots.gif" alt="" >}}
