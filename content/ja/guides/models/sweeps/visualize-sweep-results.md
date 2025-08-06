---
title: sweep の結果を可視化する
description: W&B App UI で W&B Sweeps の結果を可視化しましょう。
menu:
  default:
    identifier: visualize-sweep-results
    parent: sweeps
weight: 7
---

W&B Sweeps の結果は W&B App で可視化できます。[W&B App](https://wandb.ai/home) にアクセスし、Sweep を初期化した際に指定した Project を選択してください。対象の Project の [workspace]({{< relref "/guides/models/track/workspaces.md" >}}) にリダイレクトされます。左パネルの **Sweep アイコン**（ほうきのアイコン）を選択してください。Sweep UI から、リストの中から対象の Sweep 名を選択します。

デフォルトでは、W&B では Sweep ジョブを開始すると、自動的にパラレル座標プロット、パラメータの重要度プロット、散布図が作成されます。

{{< img src="/images/sweeps/navigation_sweeps_ui.gif" alt="Sweep UI navigation" >}}

パラレル座標チャートは、多数のハイパーパラメーターとモデルのメトリクス間の関係を一目で把握できます。パラレル座標プロットの詳細については、[パラレル座標]({{< relref "/guides/models/app/features/panels/parallel-coordinates.md" >}}) を参照してください。

{{< img src="/images/sweeps/example_parallel_coordiantes_plot.png" alt="Example parallel coordinates plot." >}}

散布図（左）は、Sweep 中に生成された W&B Runs を比較します。散布図についての詳細は、[散布図]({{< relref "/guides/models/app/features/panels/scatter-plot.md" >}}) を参照してください。

パラメータの重要度プロット（右）は、メトリクスの望ましい値と高い相関・予測力を持ったハイパーパラメーターをリストアップします。パラメータの重要度プロットについての詳細は、[パラメータの重要度]({{< relref "/guides/models/app/features/panels/parameter-importance.md" >}}) を参照してください。

{{< img src="/images/sweeps/scatter_and_parameter_importance.png" alt="Scatter plot and parameter importance" >}}

自動的に使用される従属変数および独立変数（x および y 軸）は変更可能です。各パネル内には **パネルを編集**（鉛筆アイコン）があります。**パネルを編集** を選択すると、モーダルが表示されます。そのモーダル内でグラフの振る舞いを変更できます。

W&B のすべてのデフォルト可視化オプションの詳細については、[パネル]({{< relref "/guides/models/app/features/panels/" >}}) を参照してください。W&B Sweep に含まれない W&B Runs からプロットを作成する方法については、[Data Visualization のドキュメント]({{< relref "/guides/models/tables/" >}}) をご覧ください。