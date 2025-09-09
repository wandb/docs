---
title: 棒グラフ
description: メトリクスを可視化し、軸をカスタマイズし、カテゴリカル データをバーで比較します。
menu:
  default:
    identifier: ja-guides-models-app-features-panels-bar-plot
    parent: panels
weight: 20
---

棒グラフは、縦または横に描画できる長方形のバーでカテゴリ データを表現します。すべてのログした値が長さ 1 の場合、`wandb.Run.log()` でデフォルトで棒グラフが表示されます。

{{< img src="/images/app_ui/bar_plot.png" alt="W&B で Box と 横向きの Bar プロットを描画" >}}

チャートの 設定 でカスタマイズできます。表示する runs の最大数を制限したり、任意の config で runs をグループ化したり、ラベル名を変更できます。

{{< img src="/images/app_ui/bar_plot_custom.png" alt="カスタマイズした棒グラフ" >}}

## 棒グラフをカスタマイズ

**Box** や **Violin** プロットを作成して、多くの要約統計を 1 つのチャート タイプにまとめることもできます**。**

1. runs テーブルで runs をグループ化します。
2. workspace で 'Add panel' をクリックします。
3. 標準の 'Bar Chart' を追加し、プロットする指標を選択します。
4. 'Grouping' タブで、'box plot' や 'Violin' などを選択して、これらのスタイルでプロットします。

{{< img src="/images/app_ui/bar_plots.gif" alt="棒グラフをカスタマイズ" >}}