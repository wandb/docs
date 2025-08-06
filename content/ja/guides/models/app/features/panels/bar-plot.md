---
title: 棒グラフ
description: メトリクスを可視化し、軸をカスタマイズして、カテゴリカル データをバーで比較できます。
menu:
  default:
    identifier: bar-plot
    parent: panels
weight: 20
---

棒グラフは、カテゴリカルデータを縦または横の長方形のバーで表現します。全てのログ値が長さ1の場合、`wandb.Run.log()` でデフォルトで棒グラフが表示されます。

{{< img src="/images/app_ui/bar_plot.png" alt="W&B でのボックスおよび横棒グラフのプロット" >}}

チャート設定で最大表示 run 数を制限したり、run を任意の config でグループ化したり、ラベル名を変更したりしてカスタマイズできます。

{{< img src="/images/app_ui/bar_plot_custom.png" alt="カスタマイズされた棒グラフ" >}}

## 棒グラフのカスタマイズ

多くの要約統計情報を1つのチャートタイプで表示するために、**Box** や **Violin** プロットも作成できます。

1. runs テーブルで run をグループ化します。
2. ワークスペースで「パネルを追加」をクリックします。
3. 標準の「Bar Chart」を追加し、プロットしたいメトリクスを選択します。
4. 「グループ」タブで「box plot」や「Violin」などを選択して、それぞれのスタイルでプロットします。

{{< img src="/images/app_ui/bar_plots.gif" alt="棒グラフのカスタマイズ" >}}