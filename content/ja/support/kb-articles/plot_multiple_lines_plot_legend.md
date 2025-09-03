---
title: プロットで凡例付きの複数の線を描画するにはどうすればよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-plot_multiple_lines_plot_legend
support:
  - experiments
toc_hide: true
type: docs
url: /ja/support/:filename
---
`wandb.plot.line_series()` を使って複数行のカスタムチャートを作成します。 ラインチャートを表示するには、[プロジェクトページ]({{< relref path="/guides/models/track/project-page.md" lang="ja" >}}) に移動します。凡例を追加するには、`wandb.plot.line_series()` に `keys` 引数を含めます。例えば：

```python
wandb.log(
    {
        "my_plot": wandb.plot.line_series(
            xs=x_data, ys=y_data, keys=["metric_A", "metric_B"]
        )
    }
)
```

**Multi-line** タブの下にある複数行のプロットに関する追加の詳細は [こちら]({{< relref path="/guides/models/track/log/plots.md#basic-charts" lang="ja" >}}) を参照してください。