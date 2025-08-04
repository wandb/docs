---
title: sweep 結果を可視化する
description: W&B App UI で W&B スイープの結果を可視化します。
menu:
  default:
    identifier: ja-guides-models-sweeps-visualize-sweep-results
    parent: sweeps
weight: 7
---

Visualize の結果を W&B Sweeps の W&B App UI で確認しましょう。[https://wandb.ai/home](https://wandb.ai/home) にアクセスして、W&B App UI に移動します。W&B Sweep を初期化した際に指定したプロジェクトを選択します。プロジェクトの[workspace]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}})にリダイレクトされます。左側のパネルから **Sweep アイコン** （ほうきのアイコン）を選択します。 [Sweep UI]({{< relref path="./visualize-sweep-results.md" lang="ja" >}}) で、リストから Sweep の名前を選択します。

デフォルトでは、W&B は W&B Sweep ジョブを開始すると、パラレル座標プロット、パラメータの重要度プロット、そして散布図を自動的に作成します。

{{< img src="/images/sweeps/navigation_sweeps_ui.gif" alt="Sweep UI インターフェースへ移動し、自動生成されたプロットを確認する方法を示すアニメーション。" >}}

パラレル座標チャートは、多数のハイパーパラメーターとモデルメトリクスの関係を一目で要約します。パラレル座標プロットの詳細については、[パラレル座標]({{< relref path="/guides/models/app/features/panels/parallel-coordinates.md" lang="ja" >}}) を参照してください。

{{< img src="/images/sweeps/example_parallel_coordiantes_plot.png" alt="パラレル座標プロットの例。" >}}

左の散布図は、Sweep の間に生成された W&B Runs を比較します。散布図の詳細については、[散布図]({{< relref path="/guides/models/app/features/panels/scatter-plot.md" lang="ja" >}}) を参照してください。

右のパラメータの重要度プロットは、メトリクスの望ましい値と高度に相関するハイパーパラメーターをリストアップします。パラメータの重要度プロットの詳細については、[パラメータの重要度]({{< relref path="/guides/models/app/features/panels/parameter-importance.md" lang="ja" >}}) を参照してください。

{{< img src="/images/sweeps/scatter_and_parameter_importance.png" alt="散布図（左）とパラメータの重要度プロット（右）の例。" >}}

自動で使用される従属と独立の値（x と y 軸）を変更できます。各パネル内には **Edit panel** という鉛筆のアイコンがあります。**Edit panel** を選択します。モデルが表示されます。そのモデル内で、グラフの振る舞いを変更することができます。

すべてのデフォルトの W&B 可視化オプションの詳細については、[パネル]({{< relref path="/guides/models/app/features/panels/" lang="ja" >}}) を参照してください。W&B Sweep の一部ではない W&B Runs からプロットを作成する方法については、[Data Visualization docs]({{< relref path="/guides/models/tables/" lang="ja" >}}) を参照してください。