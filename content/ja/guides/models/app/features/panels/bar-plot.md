---
title: Bar plots
description: メトリクスを視覚化し、軸をカスタマイズし、カテゴリー データをバーとして比較する。
menu:
  default:
    identifier: ja-guides-models-app-features-panels-bar-plot
    parent: panels
weight: 20
---

バー プロットは、直方体のバーでカテゴリカルデータを表現し、垂直または水平にプロットできます。バー プロットは、ログされた値が長さ 1 の場合に **wandb.log()** デフォルトで表示されます。

{{< img src="/images/app_ui/bar_plot.png" alt="Plotting Box and horizontal Bar plots in W&B" >}}

チャート設定をカスタマイズして、表示する run の最大数を制限し、任意の設定で run をグループ化し、ラベルの名前を変更します。

{{< img src="/images/app_ui/bar_plot_custom.png" alt="" >}}

## バー プロットのカスタマイズ

**Box** または **Violin**プロットを作成して、多くの要約統計量を 1 つのチャートタイプに結合することもできます。**

1. runs テーブルを介して run をグループ化します。
2. ワークスペースで「Add panel」をクリックします。
3. 標準の「Bar Chart」を追加し、プロットするメトリクスを選択します。
4. 「Grouping」タブの下で「box plot」または「Violin」などを選択し、これらのスタイルのいずれかをプロットします。

{{< img src="/images/app_ui/bar_plots.gif" alt="Customize Bar Plots" >}}