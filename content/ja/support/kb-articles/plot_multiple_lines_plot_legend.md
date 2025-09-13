---
title: 凡例付きで 1 つのプロットに複数の線を描画するにはどうすればよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-plot_multiple_lines_plot_legend
support:
- 実験
toc_hide: true
type: docs
url: /support/:filename
---

`wandb.plot.line_series()` で、複数の線を持つカスタムチャートを作成します。折れ線グラフを表示するには、[Project ページ]({{< relref path="/guides/models/track/project-page.md" lang="ja" >}}) に移動してください。凡例を追加するには、`wandb.plot.line_series()` の `keys` 引数を指定します。例:

```python

with wandb.init(project="my_project") as run:

    run.log(
        {
            "my_plot": wandb.plot.line_series(
                xs=x_data, ys=y_data, keys=["metric_A", "metric_B"]
            )
        }
    )
```

複数ラインのプロットに関する追加の詳細は、**Multi-line** タブ内の [こちら]({{< relref path="/guides/models/track/log/plots.md#basic-charts" lang="ja" >}}) を参照してください。