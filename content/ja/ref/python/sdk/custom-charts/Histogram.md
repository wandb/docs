---
title: histogram()
data_type_classification: function
menu:
  reference:
    identifier: ja-ref-python-sdk-custom-charts-Histogram
object_type: python_sdk_custom_charts
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

W&B テーブルからヒストグラム チャートを作成します。 

**引数:**
 
 - `table`: ヒストグラム用のデータを含む W&B テーブル。 
 - `value`: ビンの軸（x 軸）のラベル。 
 - `title`: ヒストグラム プロットのタイトル。 
 - `split_table`: W&B UI でテーブルを別セクションに分割するかどうか。`True` の場合、テーブルは "Custom Chart Tables" というセクションに表示されます。デフォルトは `False`。 

**戻り値:**
 
 - `CustomChart`: W&B にログできるカスタム チャート オブジェクト。チャートをログするには、`wandb.log()` に渡します。 

**例:**
 
```python
import math
import random
import wandb

# ランダムなデータを生成
data = [[i, random.random() + math.sin(i / 10)] for i in range(100)]

# W&B テーブルを作成
table = wandb.Table(
    data=data,
    columns=["step", "height"],
)

# ヒストグラム プロットを作成
histogram = wandb.plot.histogram(
    table,
    value="height",
    title="My Histogram",
)

# ヒストグラム プロットを W&B にログ
with wandb.init(...) as run:
    run.log({"histogram-plot1": histogram})
```