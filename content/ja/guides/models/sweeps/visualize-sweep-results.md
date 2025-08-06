---
title: sweep 結果を可視化する
description: W&B App UI で W&B Sweeps の結果を可視化しましょう。
menu:
  default:
    identifier: ja-guides-models-sweeps-visualize-sweep-results
    parent: sweeps
weight: 7
---

W&B App を使って、W&B Sweeps の結果を可視化しましょう。[W&B App](https://wandb.ai/home) にアクセスします。Sweep を初期化した際に指定した Project を選択してください。Project の [workspace]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}}) にリダイレクトされます。左側のパネルから **Sweep アイコン**（ほうきアイコン）を選択します。Sweep UI で、リストから自分の Sweep 名を選んでください。

デフォルトで、W&B は W&B Sweep ジョブを開始すると、パラレル座標プロット、パラメータの重要度プロット、散布図を自動的に作成します。

{{< img src="/images/sweeps/navigation_sweeps_ui.gif" alt="Sweep UI navigation" >}}

パラレル座標チャートは、多数のハイパーパラメーターとモデルメトリクスの関係を一目でまとめて確認できます。パラレル座標プロットの詳細は [パラレル座標]({{< relref path="/guides/models/app/features/panels/parallel-coordinates.md" lang="ja" >}}) をご覧ください。

{{< img src="/images/sweeps/example_parallel_coordiantes_plot.png" alt="Example parallel coordinates plot." >}}

左側の散布図では、Sweep 実行中に生成された W&B Runs を比較できます。散布図について詳しくは [Scatter Plots]({{< relref path="/guides/models/app/features/panels/scatter-plot.md" lang="ja" >}}) をご覧ください。

右側のパラメータの重要度プロットには、目的とするメトリクスの望ましい値を最もよく予測し、高い相関があったハイパーパラメーターが表示されます。パラメータの重要度プロットの詳細は [パラメータの重要度]({{< relref path="/guides/models/app/features/panels/parameter-importance.md" lang="ja" >}}) をご覧ください。

{{< img src="/images/sweeps/scatter_and_parameter_importance.png" alt="Scatter plot and parameter importance" >}}

自動的に使われる従属・独立変数（x軸、y軸）は変更可能です。各パネル内に鉛筆のアイコン（**Edit panel**）があります。**Edit panel** を選択してください。モーダルが表示されますので、その中でグラフの振る舞いを変更できます。

W&B のデフォルトで利用できる可視化オプション全般については [Panels]({{< relref path="/guides/models/app/features/panels/" lang="ja" >}}) をご参照ください。W&B Sweep に属さない W&B Runs のプロット作成方法については [Data Visualization docs]({{< relref path="/guides/models/tables/" lang="ja" >}}) も参考にしてください。