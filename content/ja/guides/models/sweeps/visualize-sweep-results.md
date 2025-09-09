---
title: sweep の結果を可視化する
description: W&B App UI で W&B Sweeps の結果を可視化します。
menu:
  default:
    identifier: ja-guides-models-sweeps-visualize-sweep-results
    parent: sweeps
weight: 7
---

W&B App で W&B Sweeps の結果を可視化しましょう。[W&B App](https://wandb.ai/home) に移動します。Sweep を初期化したときに指定した Project を選択します。あなたの Project の [Workspace]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}}) にリダイレクトされます。左のパネルの **Sweep アイコン**（ほうきのアイコン）を選択します。Sweep の UI で、一覧から対象の Sweep 名を選びます。

デフォルトで、W&B は W&B Sweep ジョブを開始すると、平行座標プロット、パラメータの重要度プロット、散布図を自動的に作成します。

{{< img src="/images/sweeps/navigation_sweeps_ui.gif" alt="Sweep UI のナビゲーション" >}}

平行座標チャートは、多数のハイパーパラメーターとモデルのメトリクスの関係を一目で把握できるように要約します。平行座標プロットの詳細は、[平行座標]({{< relref path="/guides/models/app/features/panels/parallel-coordinates.md" lang="ja" >}}) を参照してください。

{{< img src="/images/sweeps/example_parallel_coordiantes_plot.png" alt="平行座標プロットの例。" >}}

散布図（左）は、Sweep 中に生成された W&B Runs を比較します。散布図の詳細は、[散布図]({{< relref path="/guides/models/app/features/panels/scatter-plot.md" lang="ja" >}}) を参照してください。

パラメータの重要度プロット（右）では、メトリクスの望ましい値を最もよく予測し、強く相関していたハイパーパラメーターが一覧表示されます。パラメータの重要度プロットの詳細は、[パラメータの重要度]({{< relref path="/guides/models/app/features/panels/parameter-importance.md" lang="ja" >}}) を参照してください。

{{< img src="/images/sweeps/scatter_and_parameter_importance.png" alt="散布図とパラメータの重要度" >}}

自動で使用される従属変数と独立変数（x 軸と y 軸）は変更できます。各パネルには **Edit panel** と呼ばれる鉛筆アイコンがあります。**Edit panel** を選択します。モーダルが表示されます。モーダル内で、グラフの振る舞いを変更できます。

すべてのデフォルトの W&B 可視化オプションについては、[パネル]({{< relref path="/guides/models/app/features/panels/" lang="ja" >}}) を参照してください。W&B Sweep の対象ではない W&B Runs からプロットを作成する方法は、[Data Visualization ドキュメント]({{< relref path="/guides/models/tables/" lang="ja" >}}) を参照してください。