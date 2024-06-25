---
description: 大規模な Experiments にトレーニングと評価の Runs をグループ化する
displayed_sidebar: default
---


# Group Runs

<head>
  <title>Group W&B Runs</title>
</head>

個々のジョブを実験にグループ化するには、ユニークな**group**名を**wandb.init()**に渡します。

## ユースケース

1. **分散トレーニング:** 実験が異なるトレーニングと評価スクリプトに分割されている場合に、これらを大きな全体の一部として表示するためにグループ化を使用します。
2. **複数のプロセス:** 複数の小さなプロセスを1つの実験にまとめます。
3. **K-フォールド交差検証:** 異なるランダムシードを持つrunを1つの大きな実験としてまとめます。こちらにK-フォールド交差検証の[Sweepsとグルーピングの例](https://github.com/wandb/examples/tree/master/examples/wandb-sweeps/sweeps-cross-validation)があります。

グルーピングを設定する方法は三つあります:

### 1. スクリプトでgroupを設定

オプションのgroupとjob_typeをwandb.init()に渡します。これにより、個々のrunを含む各実験専用のグループページが作成されます。例えば: `wandb.init(group="experiment_1", job_type="eval")`

### 2. 環境変数でgroupを設定

環境変数としてrunのグループを指定するには、`WANDB_RUN_GROUP`を使用します。詳細は[**環境変数**](../track/environment-variables.md)のドキュメントをご確認ください。**Group**はプロジェクト内でユニークであり、グループ内のすべてのrunで共有される必要があります。すべてのプロセスで使用するユニークな8文字の文字列を生成するには`wandb.util.generate_id()`を使用できます。例えば `os.environ["WANDB_RUN_GROUP"] = "experiment-" + wandb.util.generate_id()`

### 3. UIでグルーピングをトグル

任意のconfigの列で動的にグループ化できます。例えば、`wandb.config`を使用してバッチサイズや学習率をログに記録すると、それらのハイパーパラメータによってwebアプリで動的にグループ化できます。

## グルーピングを使用した分散トレーニング

もし`wandb.init()`でグルーピングを設定した場合、UIではデフォルトでrunがグループ化されます。テーブル上部の**Group**ボタンをクリックしてこれをオン・オフできます。以下に、グルーピングを設定した場合の[プロジェクト例](https://wandb.ai/carey/group-demo?workspace=user-carey)と[サンプルコード](http://wandb.me/grouping)があります。サイドバーの各「Group」行をクリックすると、その実験専用のグループページに移動できます。

![](/images/track/distributed_training_wgrouping_1.png)

上記のプロジェクトページから、左のサイドバーで**Group**をクリックして、このような専用ページに移動できます: [この例](https://wandb.ai/carey/group-demo/groups/exp\_5?workspace=user-carey)

![](/images/track/distributed_training_wgrouping_2.png)

## UIでの動的なグルーピング

任意の列でrunをグループ化できます。例えばハイパーパラメータによってグループ化することができます。以下はその例です:

* **サイドバー**: エポック数でrunがグループ化されています。
* **グラフ**: 各ラインはグループの平均を表し、シェーディングは分散を示します。この振る舞いはグラフ設定で変更できます。

![](/images/track/demo_grouping.png)

## グルーピングの無効化

グルーピングボタンをクリックして、いつでもグループフィールドをクリアすると、テーブルとグラフがグループ化されていない状態に戻ります。

![](/images/track/demo_no_grouping.png)

## グラフ設定のグルーピング

グラフの右上の編集ボタンをクリックし、**高度な設定**タブを選択してラインとシェーディングを変更します。各グループのラインの平均値、最小値、最大値を選択できます。シェーディングについてはオフにすることもでき、最小値と最大値、標準偏差、標準誤差を表示することもできます。

![](/images/track/demo_grouping_options_for_line_plots.gif)

## よくある質問

### タグによってrunをグループ化できますか？

runには複数のタグを付けられるため、このフィールドでのグルーピングはサポートしていません。私たちのおすすめは、これらのrunの[`config`](../track/config.md)オブジェクトに値を追加し、このconfig値でグループ化することです。これは[API](../track/config.md#update-config-files)を使用して行うことができます。