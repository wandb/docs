---
title: How do I plot multiple lines on a plot with a legend?
menu:
  support:
    identifier: ja-support-kb-articles-plot_multiple_lines_plot_legend
support:
- experiments
toc_hide: true
type: docs
url: /support/:filename
---

`wandb.plot.line_series()` で複数行のカスタムチャートを作成します。 [プロジェクトページ]({{< relref path="/guides/models/track/project-page.md" lang="ja" >}}) に移動して折れ線グラフを表示します。凡例を追加するには、 `wandb.plot.line_series()` に `keys` 引数を含めます。例：

```python
wandb.log(
    {
        "my_plot": wandb.plot.line_series(
            xs=x_data, ys=y_data, keys=["metric_A", "metric_B"]
        )
    }
)
```

複数行プロットに関するその他の詳細については、**Multi-line** タブの [こちら]({{< relref path="/guides/models/track/log/plots.md#basic-charts" lang="ja" >}}) を参照してください。
