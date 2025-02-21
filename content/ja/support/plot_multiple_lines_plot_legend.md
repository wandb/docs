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

`wandb.plot.line_series()` で複数行のカスタムチャートを作成します。 [プロジェクトページ]({{< relref path="/guides/models/track/project-page.md" lang="ja" >}}) に移動して、折れ線グラフを表示します。凡例を追加するには、`wandb.plot.line_series()` に `keys` 引数を含めます。例：

```python
wandb.log(
    {
        "my_plot": wandb.plot.line_series(
            xs=x_data, ys=y_data, keys=["metric_A", "metric_B"] # metric_A と metric_B
        )
    }
)
```

複数行プロットに関する追加の詳細は、 [こちら]({{< relref path="/guides/models/track/log/plots.md#basic-charts" lang="ja" >}}) の **Multi-line** タブにあります。
