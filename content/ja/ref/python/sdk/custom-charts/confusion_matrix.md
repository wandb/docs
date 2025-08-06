---
title: confusion_matrix()
data_type_classification: function
menu:
  reference:
    identifier: ja-ref-python-sdk-custom-charts-confusion_matrix
object_type: python_sdk_custom_charts
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/plot/confusion_matrix.py >}}




### <kbd>function</kbd> `confusion_matrix`

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

確率や予測のシーケンスから混同行列を作成します。



**引数:**

 - `probs`:  各クラスの予測確率のシーケンス。シーケンスの形状は (N, K) で、N はサンプル数、K はクラス数です。こちらを指定する場合、`preds` は指定しないでください。
 - `y_true`:  正解ラベルのシーケンス。
 - `preds`:  予測クラスラベルのシーケンス。こちらを指定する場合、`probs` は指定しないでください。
 - `class_names`:  クラス名のシーケンス。指定しない場合は "Class_1"、"Class_2" のように自動で定義されます。
 - `title`:  混同行列チャートのタイトル。
 - `split_table`:  テーブルを W&B UI で個別セクション（"Custom Chart Tables"）に分割して表示するかどうか。`True` の場合、テーブルは専用セクションに表示されます。デフォルトは `False` です。



**戻り値:**

 - `CustomChart`:  W&B へログ可能なカスタムチャートオブジェクトです。チャートをログするには `wandb.log()` に渡してください。



**例外:**

 - `ValueError`:  `probs` と `preds` の両方が与えられている場合、または予測数と正解ラベル数が異なる場合発生します。さらに、一意な予測クラスや一意な正解ラベルの数が class_names の数を超える場合も発生します。
 - `wandb.Error`:  numpy がインストールされていない場合に発生します。



**使用例:**
ワイルドライフ分類でランダムな確率から混同行列をログする例:

```python
import numpy as np
import wandb

# ワイルドライフのクラス名を定義
wildlife_class_names = ["Lion", "Tiger", "Elephant", "Zebra"]

# ランダムな正解ラベルを生成（10サンプル 0〜3）
wildlife_y_true = np.random.randint(0, 4, size=10)

# 各クラスへのランダムな確率を生成（10サンプル x 4クラス）
wildlife_probs = np.random.rand(10, 4)
wildlife_probs = np.exp(wildlife_probs) / np.sum(
    np.exp(wildlife_probs),
    axis=1,
    keepdims=True,
)

# W&B run を初期化し、混同行列をログ
with wandb.init(project="wildlife_classification") as run:
    confusion_matrix = wandb.plot.confusion_matrix(
         probs=wildlife_probs,
         y_true=wildlife_y_true,
         class_names=wildlife_class_names,
         title="Wildlife Classification Confusion Matrix",
    )
    run.log({"wildlife_confusion_matrix": confusion_matrix})
```

この例では、ランダムな確率から混同行列を生成しています。

85%の精度でシミュレーションしたモデル予測を使った混同行列のログ例:

```python
import numpy as np
import wandb

# ワイルドライフのクラス名を定義
wildlife_class_names = ["Lion", "Tiger", "Elephant", "Zebra"]

# 200枚の動物画像（不均衡な分布）の正解ラベルをシミュレート
wildlife_y_true = np.random.choice(
    [0, 1, 2, 3],
    size=200,
    p=[0.2, 0.3, 0.25, 0.25],
)

# 85%の精度でモデルの予測をシミュレート
wildlife_preds = [
    y_t
    if np.random.rand() < 0.85
    else np.random.choice([x for x in range(4) if x != y_t])
    for y_t in wildlife_y_true
]

# W&B run を初期化し、混同行列をログ
with wandb.init(project="wildlife_classification") as run:
    confusion_matrix = wandb.plot.confusion_matrix(
         preds=wildlife_preds,
         y_true=wildlife_y_true,
         class_names=wildlife_class_names,
         title="Simulated Wildlife Classification Confusion Matrix",
    )
    run.log({"wildlife_confusion_matrix": confusion_matrix})
```

こちらの例では、85%の精度で予測をシミュレートし、混同行列を生成しています。