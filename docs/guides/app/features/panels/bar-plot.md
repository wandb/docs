---
description: メトリクスを可視化し、軸をカスタマイズし、カテゴリカルデータをバーで比較します。
displayed_sidebar: default
---


# Bar Plot

バープロットは、カテゴリカルデータを長方形のバーで示すグラフで、垂直または水平にプロットできます。バープロットは、すべてのログされた値が長さ1である場合、 **wandb.log()** でデフォルトで表示されます。

![Plotting Box and horizontal Bar plots in W&B](/images/app_ui/bar_plot.png)

チャート設定でカスタマイズして表示する最大run数を制限したり、任意の設定でrunをグループ化したり、ラベルを変更したりできます。

![](/images/app_ui/bar_plot_custom.png)

### バープロットのカスタマイズ

**Box** や **Violin** プロットを作成して、多くの要約統計量を1つのチャートタイプにまとめることもできます。

1. Runsテーブルでrunをグループ化します。
2. ワークスペースで「Add panel」をクリックします。
3. 標準の「Bar Chart」を追加し、プロットするメトリクスを選択します。
4. 「Grouping」タブで、「box plot」や「Violin」などを選択して、いずれかのスタイルをプロットします。

![Customize Bar Plots](@site/static/images/app_ui/bar_plots.gif)