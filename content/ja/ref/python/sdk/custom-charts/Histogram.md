---
title: histogram()
object_type: python_sdk_custom_charts
data_type_classification: function
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/plot/histogram.py >}}




### <kbd>function</kbd> `histogram`

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
 
 - `table`:  ヒストグラム用のデータが含まれる W&B Table。 
 - `value`:  ビン軸（x軸）のラベル。 
 - `title`:  ヒストグラムのタイトル。 
 - `split_table`:  テーブルを W&B UI 内で別セクションに分割するかどうか。`True` の場合、「Custom Chart Tables」という名前のセクションに表示されます。デフォルトは `False` です。



**戻り値:**
 
 - `CustomChart`:  W&B にログ可能なカスタムチャートオブジェクト。チャートをログするには、`wandb.log()` に渡してください。



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
