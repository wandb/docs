---
title: Bar plots
description: メトリクス を可視化し、軸をカスタマイズし、カテゴリカルデータを棒グラフとして比較します。
menu:
  default:
    identifier: ja-guides-models-app-features-panels-bar-plot
    parent: panels
weight: 20
---

棒グラフは、カテゴリカル データを長方形の棒で表示し、垂直または水平にプロットできます。棒グラフは、すべてのログに記録された 値 の長さが1の場合、デフォルトで **wandb.log()** で表示されます。

{{< img src="/images/app_ui/bar_plot.png" alt="Plotting Box and horizontal Bar plots in W&B" >}}

チャートの 設定 で、表示する最大 Runs を制限したり、任意の設定で Runs をグループ化したり、ラベルの名前を変更したりできます。

{{< img src="/images/app_ui/bar_plot_custom.png" alt="" >}}

## 棒グラフのカスタマイズ

**Box** プロットまたは **Violin** プロットを作成して、多くの要約統計量を1つのチャートタイプにまとめることもできます。

1. Runs テーブルで Runs をグループ化します。
2. ワークスペース で [ パネル を追加 ] をクリックします。
3. 標準の [ 棒グラフ ] を追加し、プロットする指標を選択します。
4. [ グルーピング ] タブで、[ 箱ひげ図 ] または [ バイオリン ] などを選択して、これらのスタイルをプロットします。

{{< img src="/images/app_ui/bar_plots.gif" alt="Customize Bar Plots" >}}