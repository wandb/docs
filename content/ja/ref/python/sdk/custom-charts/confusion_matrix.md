---
title: confusion_matrix()
data_type_classification: function
menu:
  reference:
    identifier: ja-ref-python-sdk-custom-charts-confusion_matrix
object_type: python_sdk_custom_charts
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/plot/confusion_matrix.py >}}




### <kbd>関数</kbd> `confusion_matrix`

```python
confusion_matrix(
    probs: 'Sequence[Sequence[float]] | None' = None,
    y_true: 'Sequence[T] | None' = None,
    preds: 'Sequence[T] | None' = None,
    class_names: 'Sequence[str] | None' = None,
    title: 'str' = 'Confusion Matrix Curve',
    split_table: 'bool' = False
) → CustomChart
```

確率または予測のシーケンスから混同行列を構築します。 



**Args:**
 
 - `probs`: 各クラスに対する予測確率のシーケンス。シーケンスの形状は (N, K) で、N はサンプル数、K はクラス数です。指定した場合は `preds` は指定しないでください。 
 - `y_true`: 正解ラベルのシーケンス。 
 - `preds`: 予測クラスラベルのシーケンス。指定した場合は `probs` は指定しないでください。 
 - `class_names`: クラス名のシーケンス。未指定の場合、クラス名は "Class_1"、"Class_2" のように定義されます。 
 - `title`: 混同行列チャートのタイトル。 
 - `split_table`: W&B UI でテーブルを別セクションに分割するかどうか。`True` の場合、テーブルは "Custom Chart Tables" というセクションに表示されます。デフォルトは `False`。 



**Returns:**
 
 - `CustomChart`: W&B にログできるカスタムチャートのオブジェクト。チャートをログするには `wandb.log()` に渡してください。 



**Raises:**
 
 - `ValueError`: `probs` と `preds` の両方を指定した場合、または予測と正解ラベルの数が一致しない場合。ユニークな予測クラス数がクラス名の数を超える場合、またはユニークな正解ラベル数がクラス名の数を超える場合。 
 - `wandb.Error`: numpy がインストールされていない場合。 



**Examples:**
野生動物の分類で、ランダムな確率を使って混同行列をログする: 

```python
import numpy as np
import wandb

# 野生動物のクラス名を定義
wildlife_class_names = ["Lion", "Tiger", "Elephant", "Zebra"]

# ランダムな正解ラベルを生成（10 サンプル、0〜3）
wildlife_y_true = np.random.randint(0, 4, size=10)

# 各クラスのランダムな確率を生成（10 サンプル × 4 クラス）
wildlife_probs = np.random.rand(10, 4)
wildlife_probs = np.exp(wildlife_probs) / np.sum(
    np.exp(wildlife_probs),
    axis=1,
    keepdims=True,
)

# W&B の run を初期化して混同行列をログする
with wandb.init(project="wildlife_classification") as run:
    confusion_matrix = wandb.plot.confusion_matrix(
         probs=wildlife_probs,
         y_true=wildlife_y_true,
         class_names=wildlife_class_names,
         title="Wildlife Classification Confusion Matrix",
    )
    run.log({"wildlife_confusion_matrix": confusion_matrix})
``` 

この例では、ランダムな確率を用いて混同行列を生成しています。 

シミュレートした モデル の予測で正解率 85% の混同行列をログする: 

```python
import numpy as np
import wandb

# 野生動物のクラス名を定義
wildlife_class_names = ["Lion", "Tiger", "Elephant", "Zebra"]

# 動物画像 200 枚の正解ラベルをシミュレート（不均衡な分布）
wildlife_y_true = np.random.choice(
    [0, 1, 2, 3],
    size=200,
    p=[0.2, 0.3, 0.25, 0.25],
)

# 正解率 85% の モデル 予測をシミュレート
wildlife_preds = [
    y_t
    if np.random.rand() < 0.85
    else np.random.choice([x for x in range(4) if x != y_t])
    for y_t in wildlife_y_true
]

# W&B の run を初期化して混同行列をログする
with wandb.init(project="wildlife_classification") as run:
    confusion_matrix = wandb.plot.confusion_matrix(
         preds=wildlife_preds,
         y_true=wildlife_y_true,
         class_names=wildlife_class_names,
         title="Simulated Wildlife Classification Confusion Matrix",
    )
    run.log({"wildlife_confusion_matrix": confusion_matrix})
``` 

この例では、正解率 85% の予測をシミュレートして混同行列を生成しています。