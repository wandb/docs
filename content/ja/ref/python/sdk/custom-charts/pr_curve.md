---
title: 'pr_curve()

  '
data_type_classification: function
menu:
  reference:
    identifier: ja-ref-python-sdk-custom-charts-pr_curve
object_type: python_sdk_custom_charts
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/plot/pr_curve.py >}}




### <kbd>function</kbd> `pr_curve`

```python
pr_curve(
    y_true: 'Iterable[T] | None' = None,
    y_probas: 'Iterable[numbers.Number] | None' = None,
    labels: 'list[str] | None' = None,
    classes_to_plot: 'list[T] | None' = None,
    interp_size: 'int' = 21,
    title: 'str' = 'Precision-Recall Curve',
    split_table: 'bool' = False
) → CustomChart
```

Precision-Recall（PR）曲線を作成します。

PR曲線は、特に不均衡なデータセットで分類器を評価する際に役立ちます。PR曲線下の面積（AUC）が大きいほど、高い適合率（偽陽性率が低い）と高い再現率（偽陰性率が低い）の両方を示します。PR曲線は、さまざまな閾値レベルでの偽陽性と偽陰性のバランスについての洞察を提供し、モデルのパフォーマンス評価を助けます。



**引数:**
 
 - `y_true`:  真の2値ラベル。形状は (`num_samples`,) で指定してください。
 - `y_probas`:  各クラスの予測スコアや確率。確率推定値、信頼スコア、または閾値なしの決定値などが指定可能です。形状は (`num_samples`, `num_classes`) です。
 - `labels`:  `y_true` の数値を置き換えてプロットの解釈をしやすくするための、任意のクラス名リスト。例: `labels = ['dog', 'cat', 'owl']` とすると、0 が 'dog'、1 が 'cat'、2 が 'owl' としてプロットされます。指定しない場合は `y_true` の数値が使用されます。
 - `classes_to_plot`:  プロットに含める `y_true` からのユニークなクラス値の任意のリスト。指定しない場合は、`y_true` に含まれるすべてのユニーククラスがプロットされます。
 - `interp_size`:  再現率を補間する点の数。再現率は [0, 1] 範囲で `interp_size` 個の均等な点に固定され、それに従って適合率も補間されます。
 - `title`:  プロットのタイトル。デフォルトは "Precision-Recall Curve" です。
 - `split_table`:  テーブルを W&B UI 内の別セクションに分割表示するかどうか。`True` の場合、「Custom Chart Tables」というセクションでテーブルが表示されます。デフォルトは `False` です。



**返り値:**
 
 - `CustomChart`:  W&B にログできるカスタムチャート オブジェクトです。チャートをログするには、 `wandb.log()` に渡してください。



**例外:**
 
 - `wandb.Error`:  NumPy、pandas、または scikit-learn がインストールされていない場合に発生します。





**例:**
 

```python
import wandb

# スパム検出（二値分類）の例
y_true = [0, 1, 1, 0, 1]  # 0 = スパムでない, 1 = スパム
y_probas = [
    [0.9, 0.1],  # 最初のサンプルの予測確率（スパムでない）
    [0.2, 0.8],  # 2番目のサンプル（スパム）、以下同様
    [0.1, 0.9],
    [0.8, 0.2],
    [0.3, 0.7],
]

labels = ["not spam", "spam"]  # 可読性のための任意のクラス名

with wandb.init(project="spam-detection") as run:
    pr_curve = wandb.plot.pr_curve(
         y_true=y_true,
         y_probas=y_probas,
         labels=labels,
         title="Spam Detection 向け PR 曲線",
    )
    run.log({"pr-curve": pr_curve})
```