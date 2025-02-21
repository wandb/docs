---
title: How do I plot multiple lines on a plot with a legend?
menu:
  support:
    identifier: ja-support-plot_multiple_lines_plot_legend
tags:
- experiments
toc_hide: true
type: docs
---

`wandb.plot.line_series()` を使用して、マルチラインのカスタムチャートを作成します。ラインチャートを表示するには、[プロジェクト ページ]({{< relref path="/guides/models/track/project-page.md" lang="ja" >}}) に移動します。凡例を追加するには、`wandb.plot.line_series()` に `keys` 引数を含めます。例えば:

```python
wandb.log(
    {
        "my_plot": wandb.plot.line_series(
            xs=x_data, ys=y_data, keys=["metric_A", "metric_B"]
        )
    }
)
```

マルチラインプロットに関する追加の詳細は、**Multi-line** タブの [こちら]({{< relref path="/guides/models/track/log/plots.md#basic-charts" lang="ja" >}}) をご参照ください。