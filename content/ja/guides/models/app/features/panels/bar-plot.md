---
title: Bar plots
description: メトリクスを可視化し、軸をカスタマイズして、カテゴリカルデータを棒グラフとして比較します。
menu:
  default:
    identifier: ja-guides-models-app-features-panels-bar-plot
    parent: panels
weight: 20
---

棒グラフは、カテゴリカルデータを長方形の棒で表示し、垂直または水平にプロットできます。棒グラフは、すべてのログに記録された値の長さが1の場合、デフォルトで **wandb.log()** で表示されます。

{{< img src="/images/app_ui/bar_plot.png" alt="Plotting Box and horizontal Bar plots in W&B" >}}

チャートの設定で、表示する最大 run 数を制限したり、configで run をグループ化したり、ラベルの名前を変更したりできます。

{{< img src="/images/app_ui/bar_plot_custom.png" alt="" >}}

## 棒グラフのカスタマイズ

また、**Box** プロットまたは **Violin** プロットを作成して、多くの要約統計量を1つのチャートタイプにまとめることもできます。

1. run テーブルで run をグループ化します。
2. ワークスペース で「パネルを追加」をクリックします。
3. 標準の「棒グラフ」を追加し、プロットするメトリックを選択します。
4. 「グループ化」タブで、「箱ひげ図」または「バイオリン」などを選択して、これらのスタイルをプロットします。

{{< img src="/images/app_ui/bar_plots.gif" alt="Customize Bar Plots" >}}
