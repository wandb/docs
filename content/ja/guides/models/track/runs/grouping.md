---
title: run を実験にまとめる
description: トレーニングと評価 run をグループ化して大規模な Experiments を構成する
menu:
  default:
    identifier: ja-guides-models-track-runs-grouping
    parent: what-are-runs
---

個々のジョブを実験としてグループ化するには、一意の**グループ**名を**wandb.init()**に渡します。

## ユースケース

1. **分散トレーニング**: 実験が異なるトレーニングや評価スクリプトに分割されている場合、グループ化を使用してそれらを一つの大きな全体として見ることができます。
2. **複数のプロセス**: 複数の小さなプロセスを一つの実験としてグループ化します。
3. **K-分割交差検証**: 異なるランダムシードを持つrunをグループ化して、大きな実験を見ます。こちらがスイープとグループ化を使用した[K-分割交差検証の例です](https://github.com/wandb/examples/tree/master/examples/wandb-sweeps/sweeps-cross-validation)。

グループ化を設定する方法は3つあります:

### 1. スクリプトでグループを設定する

オプションで `group` と `job_type` を wandb.init() に渡します。これにより、各実験に対して専用のグループページが作成され、個々のrunが含まれます。例: `wandb.init(group="experiment_1", job_type="eval")`

### 2. グループ環境変数を設定する

`WANDB_RUN_GROUP` を使用して、runのグループを環境変数として指定します。詳細は[**Environment Variables**]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}})をご覧ください。**Group**はプロジェクト内で一意である必要があり、グループ内のすべてのrunで共有されます。`wandb.util.generate_id()` を使用して、すべてのプロセスで使用するための一意の8文字の文字列を生成することができます。例: `os.environ["WANDB_RUN_GROUP"] = "experiment-" + wandb.util.generate_id()`

### 3. UIでグループ化を切り替える

任意の設定列で動的にグループ化できます。例として、`wandb.config` を使用してバッチサイズまたは学習率をログすると、それらのハイパーパラメーターでWebアプリ内で動的にグループ化できます。

## グループ化を伴う分散トレーニング

`wandb.init()`でグループ化を設定したと仮定すると、UIではデフォルトでrunがグループ化されます。テーブルの上部にある**Group**ボタンをクリックして、これをオン・オフすることができます。こちらはグループ化を設定した[サンプルコード](http://wandb.me/grouping)から生成された[例のプロジェクト](https://wandb.ai/carey/group-demo?workspace=user-carey)です。サイドバーの各「グループ」行をクリックすると、その実験の専用グループページにアクセスできます。

{{< img src="/images/track/distributed_training_wgrouping_1.png" alt="" >}}

上記のプロジェクトページから、左サイドバーで**Group**をクリックすると、[このような専用ページ](https://wandb.ai/carey/group-demo/groups/exp_5?workspace=user-carey)にアクセスできます。

{{< img src="/images/track/distributed_training_wgrouping_2.png" alt="" >}}

## UIでの動的なグループ化

任意の列でrunをグループ化できます。例として、ハイパーパラメーターでグループ化することができます。これがどのように見えるかの例です：

* **サイドバー**: runがエポック数でグループ化されています。
* **グラフ**: 各線はグループの平均を表し、陰影は分散を示します。この振る舞いはグラフ設定で変更できます。

{{< img src="/images/track/demo_grouping.png" alt="" >}}

## グループ化をオフにする

グループ化ボタンをクリックし、グループフィールドをいつでもクリアすることで、テーブルとグラフをグループ化されていない状態に戻します。

{{< img src="/images/track/demo_no_grouping.png" alt="" >}}

## グループ化グラフの設定

グラフの右上にある編集ボタンをクリックし、**Advanced**タブを選択して線と陰影を変更します。各グループの線には平均、最小、最大値を選択できます。陰影については無効にしたり、最小と最大、標準偏差、標準誤差を表示することができます。

{{< img src="/images/track/demo_grouping_options_for_line_plots.gif" alt="" >}}