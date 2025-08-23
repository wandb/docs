---
title: 棒グラフ
description: メトリクスの可視化、軸のカスタマイズ、カテゴリデータのバーによる比較ができます。
menu:
  default:
    identifier: ja-guides-models-app-features-panels-bar-plot
    parent: panels
weight: 20
---

棒グラフは、カテゴリカルデータを縦または横の長方形のバーで表示します。バーグラフは、すべてのログ値が長さ1の場合、`wandb.Run.log()`でデフォルト表示されます。

{{< img src="/images/app_ui/bar_plot.png" alt="W&Bでボックスプロットと横棒グラフをプロットする様子" >}}

チャート設定で最大表示 run 数を制限したり、任意の config で run をグループ化したり、ラベル名を変更できます。

{{< img src="/images/app_ui/bar_plot_custom.png" alt="カスタマイズされた棒グラフ" >}}

## 棒グラフのカスタマイズ

多数の要約統計量を 1 つのチャート形式にまとめるために、**Box** プロットや **Violin** プロットを作成することもできます。

1. runs テーブルで run をグループ化します。
2. ワークスペースの [Add panel]（パネル追加）をクリックします。
3. 標準の「Bar Chart」を追加し、プロットしたいメトリクスを選択します。
4. 「Grouping」タブで「box plot」や「Violin」などを選ぶと、それぞれのスタイルでプロットできます。

{{< img src="/images/app_ui/bar_plots.gif" alt="棒グラフのカスタマイズ" >}}