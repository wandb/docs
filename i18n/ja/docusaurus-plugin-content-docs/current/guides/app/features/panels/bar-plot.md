---
description: Visualize metrics, customize axes, and compare categorical data as bars.
displayed_sidebar: ja
---

# 棒グラフ

棒グラフは、カテゴリデータを縦方向または横方向にプロットできる長方形の棒で表示します。すべてのログされた値の長さが1の場合、**wandb.log()** を使用して棒グラフがデフォルトで表示されます。

![W&Bで箱ひげ図と水平棒グラフをプロット](/images/app_ui/bar_plot.png)

チャート設定をカスタマイズして、表示する最大ランを制限し、任意の設定でランをグループ化し、ラベルの名前を変更します。

![](/images/app_ui/bar_plot_custom.png)

### 棒グラフのカスタマイズ

**Box** または **Violin** プロットを作成して、多くの要約統計を1つのチャートタイプに組み合わせることもできます。

1. ランテーブルでグループ化します。
2. ワークスペースで「パネルを追加」をクリックします。
3. 標準の「Bar Chart」を追加し、プロットするメトリックを選択します。
4. 「Grouping」タブの下で、「box plot」または「Violin」などを選択して、これらのスタイルのいずれかをプロットします。

![棒グラフのカスタマイズ](@site/static/images/app_ui/bar_plots.gif)