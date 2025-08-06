---
title: bar()
object_type: python_sdk_custom_charts
data_type_classification: function
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/plot/bar.py >}}




### <kbd>function</kbd> `bar`

```python
bar(
    table: 'wandb.Table',
    label: 'str',
    value: 'str',
    title: 'str' = '',
    split_table: 'bool' = False
) → CustomChart
```

wandb.Table のデータから棒グラフ（バーチャート）を作成します。



**引数:**
 
 - `table`:  棒グラフのデータを含むテーブル。
 - `label`:  各バーのラベルとして使うカラム名。
 - `value`:  各バーの値として使うカラム名。
 - `title`:  棒グラフのタイトル。
 - `split_table`:  テーブルを W&B UI 内の別のセクションに分割して表示するかどうか。`True` にすると、「Custom Chart Tables」というセクションに表示されます。デフォルトは `False` です。



**戻り値:**
 
 - `CustomChart`:  W&B にログ可能なカスタムチャートオブジェクト。このチャートをログするには `wandb.log()` に渡してください。



**例:**
 

```python
import random
import wandb

# テーブル用のランダムデータを生成
data = [
    ["car", random.uniform(0, 1)],
    ["bus", random.uniform(0, 1)],
    ["road", random.uniform(0, 1)],
    ["person", random.uniform(0, 1)],
]

# データからテーブルを作成
table = wandb.Table(data=data, columns=["class", "accuracy"])

# W&B の run を初期化しバープロットをログする
with wandb.init(project="bar_chart") as run:
    # テーブルからバープロットを作成
    bar_plot = wandb.plot.bar(
         table=table,
         label="class",
         value="accuracy",
         title="Object Classification Accuracy",
    )

    # バーチャートを W&B にログする
    run.log({"bar_plot": bar_plot})
```