---
title: 凡例付きのプロットで複数の線を描画するにはどうすればよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-plot_multiple_lines_plot_legend
support:
- 実験
toc_hide: true
type: docs
url: /support/:filename
---

`wandb.plot.line_series()` を使って、複数行のカスタムチャートを作成できます。折れ線グラフを見るには [project page]({{< relref path="/guides/models/track/project-page.md" lang="ja" >}}) に移動してください。凡例を追加するには、`wandb.plot.line_series()` の中で `keys` 引数を指定します。例えば、次のように記述します。

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

複数ラインのプロットについてさらに詳しく知りたい方は、**Multi-line** タブ内の [こちら]({{< relref path="/guides/models/track/log/plots.md#basic-charts" lang="ja" >}}) をご参照ください。