---
title: pr_curve()
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

PR 曲線を作成します。

PR 曲線は、不均衡なデータセットで分類器を評価する際に特に有用です。PR 曲線下の面積が大きいほど、適合率が高い（偽陽性率が低い）かつ再現率も高い（偽陰性率が低い）ことを意味します。さまざまな閾値レベルにおける偽陽性と偽陰性のバランスに関する洞察を与え、モデルの性能評価に役立ちます。



**Args:**
 
 - `y_true`:  真の 2 値ラベル。形状は (`num_samples`,)。 
 - `y_probas`:  各クラスに対する予測スコアまたは確率。確率推定、信頼度スコア、またはしきい値を設けない決定関数の値を指定できます。形状は (`num_samples`, `num_classes`)。 
 - `labels`:  プロットを解釈しやすくするために、`y_true` の数値を置き換えるクラス名の任意リスト。たとえば、`labels = ['dog', 'cat', 'owl']` とすると、プロット内で 0 が 'dog'、1 が 'cat'、2 が 'owl' に置き換えられます。指定しない場合は `y_true` の数値がそのまま使われます。 
 - `classes_to_plot`:  プロットに含める `y_true` のユニークなクラス値の任意リスト。指定しない場合は、`y_true` に存在するすべてのユニーククラスがプロットされます。 
 - `interp_size`:  再現率を補間する点の数。[0, 1] の範囲に一様に分布する `interp_size` 個の点に再現率を固定し、それに合わせて適合率を補間します。 
 - `title`:  プロットのタイトル。デフォルトは "Precision-Recall Curve"。 
 - `split_table`:  W&B UI でテーブルを別セクションに分けるかどうか。`True` の場合、"Custom Chart Tables" という名前のセクションに表示されます。デフォルトは `False`。 



**Returns:**
 
 - `CustomChart`:  W&B にログできるカスタムチャートのオブジェクト。チャートをログするには、`wandb.log()` に渡します。 



**Raises:**
 
 - `wandb.Error`:  NumPy、pandas、scikit-learn がインストールされていない場合。 





**Example:**
 

```python
import wandb

# スパム検出の例（2 値分類）
y_true = [0, 1, 1, 0, 1]  # 0 = 非スパム、1 = スパム
y_probas = [
    [0.9, 0.1],  # 1 件目のサンプルに対する予測確率（非スパム）
    [0.2, 0.8],  # 2 件目（スパム）、以降も同様
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
         title="Precision-Recall Curve for Spam Detection",
    )
    run.log({"pr-curve": pr_curve})
```