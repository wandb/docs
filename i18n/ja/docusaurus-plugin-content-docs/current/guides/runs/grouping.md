---
description: Group training and evaluation runs into larger experiments
displayed_sidebar: default
---

# Runのグループ化

<head>
  <title>W&B Runのグループ化</title>
</head>


**wandb.init()** に一意の **group** 名を渡すことで、個々のジョブを実験にグループ化します。

## ユースケース

1. **分散トレーニング**: 実験が複数の部分に分割されていて、個別のトレーニングと評価スクリプトがそれぞれ含まれる場合、グループ化を使用して、それらを大きな全体として表示します。
2. **複数のプロセス**: 複数の小さなプロセスを実験にグループ化する。
3. **K分割交差検証**: 異なるランダムシードを持つrunをまとめて、大規模な実験を確認します。こちらに [k分割交差検証をスイープとグループ化で実行する例](https://github.com/wandb/examples/tree/master/examples/wandb-sweeps/sweeps-cross-validation)があります。

グループ化の設定方法は3つあります。

### 1. スクリプト内でグループを設定する

wandb.init()にオプションとしてgroupとjob_typeを渡します。これにより、各実験の専用グループページが作成され、そこに個々のrunが含まれます。例えば, `wandb.init(group="experiment_1", job_type="eval")`

### 2. 環境変数でグループを設定する

`WANDB_RUN_GROUP` を使用して、環境変数としてrunのグループを指定します。詳細については、[**Environment Variables**](../track/environment-variables.md)**. Group**のドキュメントをご覧ください。 グループはプロジェクト内で一意であり、グループ内のすべてのrunで共有されるべきです。`wandb.util.generate_id()`を使って、すべてのプロセスで使用する一意の8文字の文字列を生成できます。例：`os.environ["WANDB_RUN_GROUP"] = "experiment-" + wandb.util.generate_id()`

### 3. UIでグループ化を切り替える
`wandb.config` を使ってバッチサイズや学習率をロギングする場合など、Webアプリで動的に任意の設定の列によってグループ化することができます。

## グループ化を用いた分散トレーニング

`wandb.init()` でグループ化を設定すると、デフォルトで UI で実行がグループ化されます。テーブルの上部にある**グループ**ボタンをクリックすることで、これを切り替えることができます。次に例として、グループ化を設定した [サンプルコード](http://wandb.me/grouping) から生成された [プロジェクト](https://wandb.ai/carey/group-demo?workspace=user-carey) を示します。サイドバーの「グループ」の行をクリックすることで、その実験に対する専用のグループページにアクセスできます。

![](/images/track/distributed_training_wgrouping_1.png)

上記のプロジェクトページから、左サイドバーの **グループ** をクリックして、[このような](https://wandb.ai/carey/group-demo/groups/exp\_5?workspace=user-carey) 専用ページにアクセスできます。

![](/images/track/distributed_training_wgrouping_2.png)

## UIでの動的なグループ化

任意の列で実行をグループ化できます。例えば、ハイパーパラメータによってグループ化することができます。その例を以下に示します。

- **サイドバー**：実行はエポック数でグループ化されています。
- **グラフ**：各線はグループごとの平均を表し、陰影は分散を示しています。この振る舞いは、グラフの設定で変更することができます。

![](/images/track/demo_grouping.png)

## グループ化の解除

グループ化ボタンをクリックしてグループのフィールドをいつでもクリアすることができます。これにより、テーブルとグラフをグループ化されていない状態に戻します。

![](/images/track/demo_no_grouping.png)

## グラフのグループ化設定

グラフの右上隅にある編集ボタンをクリックし、**詳細**タブを選択して、線と陰影を変更します。各グループの線について、平均、最小、最大値を選択できます。陰影については、陰影をオフにしたり、最小・最大値、標準偏差、標準誤差を表示できます。
![](/images/track/demo_grouping_options_for_line_plots.gif)

## よくある質問

### タグでrunsをグループ化できますか？

runには複数のタグがあるため、このフィールドでのグループ化はサポートしていません。runsの[`config`](../track/config.md)オブジェクトに値を追加し、そのconfigの値でグループ化することをお勧めします。これは[API](../track/config#update-config-files)を使って実行できます。