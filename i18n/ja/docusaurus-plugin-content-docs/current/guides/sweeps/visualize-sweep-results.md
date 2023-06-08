---
description: Visualize the results of your Weights & Biases Sweeps with the W&B App UI.
displayed_sidebar: ja
---

# スイープ結果の視覚化

<head>
  <title>W&Bスイープの結果を視覚化する</title>
</head>
Weights & Biasesスイープの結果をW&B App UIで可視化しましょう。W&B App UIに[https://wandb.ai/home](https://wandb.ai/home) からアクセスしてください。W&Bスイープを初期化した際に指定したプロジェクトを選択します。プロジェクト[ワークスペース](https://docs.wandb.ai/ref/app/pages/workspaces)にリダイレクトされます。左パネルの**スイープアイコン**（ほうきアイコン）を選択してください。[スイープUI](./visualize-sweep-results.md)から、リストからスイープの名前を選択します。

デフォルトでは、Weights & BiasesはW&Bスイープジョブを開始すると、自動的に並行座標プロット、パラメータ重要度プロット、および散布図が作成されます。

![スイープUIインターフェースに移動し、自動生成されたプロットを表示する方法を示すアニメーション。](/images/sweeps/navigation_sweeps_ui.gif)

並行座標チャートは、多数のハイパーパラメーターとモデルメトリクスの関係を一目でまとめて表示します。並行座標プロットについての詳細は、[並行座標](../app/features/panels/parallel-coordinates)を参照してください。
![並行座標プロットの例](/images/sweeps/example_parallel_coordiantes_plot.png)

散布図（左）は、スイープ中に生成された W&B の Runs を比較します。散布図に関する詳細は、[散布図](../app/features/panels/scatter-plot.md)を参照してください。

パラメータ重要度プロット（右）は、メトリクスの望ましい値と高い相関があるハイパーパラメーターを最良の予測因子としてリストアップしています。パラメータ重要度プロットに関する詳細は、[パラメータ重要度](../app/features/panels/parameter-importance.md)を参照してください。

![散布図の例（左）とパラメータ重要度プロット（右）](/images/sweeps/scatter_and_parameter_importance.png)
自動的に使用される従属変数と独立変数（x軸とy軸）を変更することができます。各パネルには、**編集パネル**と呼ばれる鉛筆アイコンがあります。 **編集パネル**を選択してください。モーダルが表示されます。モーダル内で、グラフの振る舞いを変更することができます。

すべてのデフォルトのW&B可視化オプションについての詳細は、[Panels](../app/features/panels/intro.md)を参照してください。W&Bスイープの一部ではないW&B Runsからプロットを作成する方法についての情報は、[データ可視化ドキュメント](https://docs.wandb.ai/guides/data-vis)を参照してください。