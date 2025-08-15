---
title: bar()
data_type_classification: function
menu:
  reference:
    identifier: ja-ref-python-sdk-custom-charts-bar
object_type: python_sdk_custom_charts
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

wandb.Table のデータから棒グラフを作成します。



**引数:**
 
 - `table`:  棒グラフ用のデータが入ったテーブル。 
 - `label`:  各バーのラベルに使うカラム名。 
 - `value`:  各バーの値に使うカラム名。 
 - `title`:  棒グラフのタイトル。 
 - `split_table`:  テーブルを W&B UI で別セクションに分けて表示するかどうか。`True` の場合、「Custom Chart Tables」というセクションにテーブルが表示されます。デフォルトは `False` です。



**戻り値:**
 
 - `CustomChart`:  W&B にログできるカスタムチャートオブジェクト。チャートをログするには `wandb.log()` に渡してください。



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

# W&B run を初期化し、棒グラフをログする
with wandb.init(project="bar_chart") as run:
    # テーブルから棒グラフを作成
    bar_plot = wandb.plot.bar(
         table=table,
         label="class",
         value="accuracy",
         title="Object Classification Accuracy",
    )

    # 棒グラフを W&B にログ
    run.log({"bar_plot": bar_plot})
```