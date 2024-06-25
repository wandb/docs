---
description: W&B App UIを使ってW&B Sweepsの結果を可視化します。
displayed_sidebar: default
---


# スイープ結果の可視化

<head>
  <title>W&B スイープの結果を可視化</title>
</head>

W&B App UIを使用して、W&B Sweepsの結果を可視化します。W&B App UIには[https://wandb.ai/home](https://wandb.ai/home)からアクセスします。W&B Sweepを初期化した際に指定したプロジェクトを選択します。プロジェクトの[ワークスペース](../app/pages/workspaces.md)にリダイレクトされます。左側のパネルで**スイープアイコン**（ホウキのアイコン）を選択します。[スイープUI](./visualize-sweep-results.md)から、リスト内のスイープ名を選択します。

デフォルトでは、W&BはW&B Sweepジョブの開始時に自動的にパラレル座標図、パラメータの重要性プロット、散布図を作成します。

![Sweep UIインターフェースに移動して自動生成されたプロットを表示するアニメーション。](/images/sweeps/navigation_sweeps_ui.gif)

パラレル座標図は、多数のハイパーパラメーターとモデルメトリクスの関係を一目で要約します。パラレル座標図の詳細については、[パラレル座標](../app/features/panels/parallel-coordinates.md)を参照してください。

![パラレル座標図の例。](/images/sweeps/example_parallel_coordiantes_plot.png)

左の散布図は、スイープ中に生成されたW&B Runsを比較します。散布図の詳細については、[散布図](../app/features/panels/scatter-plot.md)を参照してください。

右のパラメータの重要性プロットは、メトリクスの望ましい値と高い相関を持つ予測因子として最適だったハイパーパラメーターをリストアップします。パラメータの重要性プロットの詳細については、[パラメータの重要性](../app/features/panels/parameter-importance.md)を参照してください。

![散布図（左）とパラメータの重要性プロット（右）の例。](/images/sweeps/scatter_and_parameter_importance.png)

自動的に使用される従属値と独立値（x軸とy軸）を変更できます。各パネル内に**パネルを編集**という鉛筆アイコンがあります。**パネルを編集**を選択します。モーダルが表示され、その中でグラフの振る舞いを変更できます。

デフォルトのW&B可視化オプションの全てについての詳細は、[パネル](../app/features/panels/intro.md)を参照してください。W&B Sweepの一部ではないW&B Runsからプロットを作成する方法については、[データ可視化ドキュメント](../tables/intro.md)を参照してください。