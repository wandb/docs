---
title: Visualize sweep results
description: W&B スイープの結果を W&B App UI で視覚化する。
menu:
  default:
    identifier: ja-guides-models-sweeps-visualize-sweep-results
    parent: sweeps
weight: 7
---

W&B Sweeps の結果を W&B App UI で可視化します。 [https://wandb.ai/home](https://wandb.ai/home) の W&B App UI に移動します。W&B Sweep を初期化する際に指定したプロジェクトを選択します。プロジェクトの [workspace]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}}) にリダイレクトされます。左のパネルで **Sweep アイコン** （ほうきのアイコン）を選択します。[Sweep UI]({{< relref path="./visualize-sweep-results.md" lang="ja" >}}) からリストの中から自分の Sweep の名前を選びます。

デフォルトでは、W&B は W&B Sweep ジョブを開始すると、自動的にパラレル座標図、パラメータの重要性プロット、および散布図を作成します。

{{< img src="/images/sweeps/navigation_sweeps_ui.gif" alt="Sweep UI インターフェースに移動して自動生成されたプロットを表示する方法を示すアニメーション。" >}}

パラレル座標チャートは、多数のハイパーパラメータとモデルメトリクスの関係を一目で要約します。パラレル座標図の詳細については、[Parallel coordinates]({{< relref path="/guides/models/app/features/panels/parallel-coordinates.md" lang="ja" >}}) を参照してください。

{{< img src="/images/sweeps/example_parallel_coordiantes_plot.png" alt="パラレル座標図の例。" >}}

左側の散布図は、Sweep の間に生成された W&B Runs を比較します。散布図の詳細については、[Scatter Plots]({{< relref path="/guides/models/app/features/panels/scatter-plot.md" lang="ja" >}}) を参照してください。

右側のパラメータの重要性プロットは、メトリクスの望ましい値を予測するのに最も適しており、高度に相関しているハイパーパラメータをリストします。パラメータの重要性プロットの詳細については、[Parameter Importance]({{< relref path="/guides/models/app/features/panels/parameter-importance.md" lang="ja" >}}) を参照してください。

{{< img src="/images/sweeps/scatter_and_parameter_importance.png" alt="散布図（左）とパラメータの重要性プロット（右）の例。" >}}

自動的に使用される従属値と独立値（x 軸と y 軸）を変更することができます。各パネル内に **Edit panel** と呼ばれる鉛筆アイコンがあります。**Edit panel** を選択します。モーダルが表示され、その中でグラフの振る舞いを変更することができます。

すべてのデフォルトの W&B 可視化オプションの詳細については、[Panels]({{< relref path="/guides/models/app/features/panels/" lang="ja" >}}) を参照してください。W&B Sweep に含まれない W&B Runs からプロットを作成する方法については、[Data Visualization docs]({{< relref path="/guides/core/tables/" lang="ja" >}}) を参照してください。