---
title: Group runs into experiments
description: トレーニング と評価 の run をグループ化して、より大規模な Experiments にまとめます。
menu:
  default:
    identifier: ja-guides-models-track-runs-grouping
    parent: what-are-runs
---

個々のジョブを実験にグループ化するには、一意の **group** 名を **wandb.init()** に渡します。

## ユースケース

1. **分散トレーニング:** 実験が、より大きな全体の一部と見なされるべき個別のトレーニング および 評価スクリプトに分割されている場合は、グループ化を使用します。
2. **複数のプロセス:** 複数の小規模プロセスをまとめて1つの実験にグループ化します。
3. **K 分割交差検証:** 異なるランダムシードを持つ run をグループ化して、より大規模な実験を確認します。以下に、sweeps とグループ化を使用した K 分割交差検証の [例](https://github.com/wandb/examples/tree/master/examples/wandb-sweeps/sweeps-cross-validation) を示します。

グループ化を設定するには、次の3つの方法があります。

### 1. スクリプトでグループを設定する

オプションの group と job_type を wandb.init() に渡します。これにより、個々の run が含まれる実験専用のグループページが表示されます。例：`wandb.init(group="experiment_1", job_type="eval")`

### 2. グループ環境変数を設定する

`WANDB_RUN_GROUP` を使用して、run のグループを環境変数として指定します。詳細については、[**環境変数**]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) に関するドキュメントを確認してください。**Group** は、project 内で一意であり、グループ内のすべての run で共有されている必要があります。`wandb.util.generate_id()` を使用して、すべてのプロセスで使用する一意の 8 文字の文字列を生成できます。たとえば、`os.environ["WANDB_RUN_GROUP"] = "experiment-" + wandb.util.generate_id()` のようにします。

### 3. UI でグループ化を切り替える

任意の config 列で動的にグループ化できます。たとえば、`wandb.config` を使用してバッチサイズまたは学習率を記録する場合、Web アプリでこれらのハイパーパラメーターによって動的にグループ化できます。

## グループ化による分散トレーニング

`wandb.init()` でグループ化を設定すると、UI で run がデフォルトでグループ化されます。これは、テーブルの上部にある [**グループ**] ボタンをクリックしてオン/オフを切り替えることができます。[サンプルコード](http://wandb.me/grouping) から生成された [プロジェクトの例](https://wandb.ai/carey/group-demo?workspace=user-carey) を示します。サイドバーの各「グループ」行をクリックすると、その実験専用のグループページに移動できます。

{{< img src="/images/track/distributed_training_wgrouping_1.png" alt="" >}}

上記の project ページから、左側のサイドバーにある **Group** をクリックして、[このような](https://wandb.ai/carey/group-demo/groups/exp_5?workspace=user-carey) 専用ページに移動できます。

{{< img src="/images/track/distributed_training_wgrouping_2.png" alt="" >}}

## UI で動的にグループ化する

任意の列 (たとえば、ハイパーパラメーター) で run をグループ化できます。次に例を示します。

* **サイドバー**: run はエポック数でグループ化されています。
* **グラフ**: 各線はグループの平均を表し、網掛けは分散を示します。この振る舞いは、グラフの設定で変更できます。

{{< img src="/images/track/demo_grouping.png" alt="" >}}

## グループ化をオフにする

グループ化ボタンをクリックして、いつでもグループフィールドをクリアすると、テーブルとグラフがグループ化されていない状態に戻ります。

{{< img src="/images/track/demo_no_grouping.png" alt="" >}}

## グラフ設定のグループ化

グラフの右上隅にある編集ボタンをクリックし、[**詳細設定**] タブを選択して、線と網掛けを変更します。各グループの線の平均値、最小値、または最大値を選択できます。網掛けの場合、網掛けをオフにして、最小値と最大値、標準偏差、および標準誤差を表示できます。

{{< img src="/images/track/demo_grouping_options_for_line_plots.gif" alt="" >}}
