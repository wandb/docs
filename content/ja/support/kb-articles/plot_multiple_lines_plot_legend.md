---
title: 凡例付きで複数の線をプロットするにはどうすればいいですか？
url: /support/:filename
toc_hide: true
type: docs
support:
- 実験管理
---

`wandb.plot.line_series()` を使って、複数行のカスタムチャートを作成できます。ラインチャートを見るには [project page]({{< relref "/guides/models/track/project-page.md" >}}) に移動してください。凡例を追加したい場合は、`wandb.plot.line_series()` の `keys` 引数を利用します。例えば、次のように実装します。

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

複数行プロットに関する追加の詳細は、[こちら]({{< relref "/guides/models/track/log/plots.md#basic-charts" >}})の **Multi-line** タブを参照してください。