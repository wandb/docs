---
title: histogram()
data_type_classification: function
menu:
  reference:
    identifier: ja-ref-python-sdk-custom-charts-Histogram
object_type: python_sdk_custom_charts
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/plot/histogram.py >}}




### <kbd>関数</kbd> `histogram`

```python
histogram(
    table: 'wandb.Table',
    value: 'str',
    title: 'str' = '',
    split_table: 'bool' = False
) → CustomChart
```

W&B Table からヒストグラムチャートを作成します。



**引数:**

 - `table`:  ヒストグラム用のデータが入った W&B Table。 
 - `value`:  ビンの軸（x軸）のラベル。 
 - `title`:  ヒストグラムのタイトル。 
 - `split_table`:  Table を W&B UI 内で個別セクションに分けて表示するかどうか。`True` の場合、テーブルは「Custom Chart Tables」というセクションに表示されます。デフォルトは `False` です。



**返り値:**

 - `CustomChart`:  W&B にログできるカスタムチャートオブジェクト。チャートをログするには、`wandb.log()` に渡してください。



**例:**

```python
import math
import random
import wandb

# ランダムなデータを生成
data = [[i, random.random() + math.sin(i / 10)] for i in range(100)]

# W&B Table を作成
table = wandb.Table(
    data=data,
    columns=["step", "height"],
)

# ヒストグラムプロットを作成
histogram = wandb.plot.histogram(
    table,
    value="height",
    title="My Histogram",
)

# ヒストグラムプロットを W&B にログ
with wandb.init(...) as run:
    run.log({"histogram-plot1": histogram})
```