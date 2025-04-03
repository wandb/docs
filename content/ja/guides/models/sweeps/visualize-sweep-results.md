---
title: Visualize sweep results
description: W&B App UIでW&B Sweepsの結果を可視化します。
menu:
  default:
    identifier: ja-guides-models-sweeps-visualize-sweep-results
    parent: sweeps
weight: 7
---

W&B App UI で W&B Sweeps の結果を可視化します。[https://wandb.ai/home](https://wandb.ai/home) にある W&B App UI に移動します。W&B Sweep を初期化する際に指定した project を選択します。project の [workspace]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}}) にリダイレクトされます。左側のパネルにある **Sweep icon** (ほうきのアイコン) を選択します。[Sweep UI]({{< relref path="./visualize-sweep-results.md" lang="ja" >}}) で、リストから Sweep の名前を選択します。

デフォルトでは、W&B は W&B Sweep ジョブを開始すると、自動的にパラレル座標図、パラメータのインポータンスプロット、および散布図を作成します。

{{< img src="/images/sweeps/navigation_sweeps_ui.gif" alt="Sweep UI インターフェースに移動し、自動生成されたプロットを表示する方法を示すアニメーション。" >}}

パラレル座標図は、多数のハイパーパラメータとモデルのメトリクスの関係を一目でまとめたものです。パラレル座標図の詳細については、[パラレル座標]({{< relref path="/guides/models/app/features/panels/parallel-coordinates.md" lang="ja" >}}) を参照してください。

{{< img src="/images/sweeps/example_parallel_coordiantes_plot.png" alt="パラレル座標図の例。" >}}

散布図 (左) は、Sweep 中に生成された W&B Runs を比較します。散布図の詳細については、[散布図]({{< relref path="/guides/models/app/features/panels/scatter-plot.md" lang="ja" >}}) を参照してください。

パラメータのインポータンスプロット (右) は、メトリクスの望ましい値の最良の予測因子であり、メトリクスの望ましい値と高度に相関するハイパーパラメータをリストします。パラメータのインポータンスプロットの詳細については、[パラメータの重要性]({{< relref path="/guides/models/app/features/panels/parameter-importance.md" lang="ja" >}}) を参照してください。

{{< img src="/images/sweeps/scatter_and_parameter_importance.png" alt="散布図の例 (左) とパラメータのインポータンスプロット (右)。" >}}

自動的に使用される従属値と独立値 (x 軸と y 軸) を変更できます。各 panel 内には、**Edit panel** という鉛筆アイコンがあります。**Edit panel** を選択します。モデルが表示されます。モーダル内で、グラフの振る舞いを変更できます。

すべてのデフォルトの W&B 可視化オプションの詳細については、[Panels]({{< relref path="/guides/models/app/features/panels/" lang="ja" >}}) を参照してください。W&B Sweep の一部ではない W&B Runs からプロットを作成する方法については、[Data Visualization のドキュメント]({{< relref path="/guides/models/tables/" lang="ja" >}}) を参照してください。
