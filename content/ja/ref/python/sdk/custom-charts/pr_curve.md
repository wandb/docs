---
title: 'pr_curve()

  '
object_type: python_sdk_custom_charts
data_type_classification: function
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

Precision-Recall (PR) 曲線を作成します。

Precision-Recall 曲線は、不均衡なデータセットで分類器を評価する際に特に有効です。PR 曲線下の面積（AUC）が高いほど、高い精度（誤検出の割合が低い）と高い再現率（見逃しの割合が低い）の両方を意味します。この曲線は、さまざまな閾値における誤検出（False positive）と見逃し（False negative）のバランスを可視化し、モデルのパフォーマンス評価に役立ちます。



**引数:**
 
 - `y_true`:  正解ラベル（二値ラベル）。形状は (`num_samples`,) です。
 - `y_probas`:  各クラスに対する予測スコアまたは確率。確率推定値、信頼スコア、またはしきい値なしの判定値が指定できます。形状は (`num_samples`, `num_classes`) です。
 - `labels`:  `y_true` の数値ラベルをプロット上でわかりやすく表示するための、クラス名のリスト（省略可）。たとえば、`labels = ['dog', 'cat', 'owl']` とすると、プロットでは 0 が 'dog'、1 が 'cat'、2 が 'owl' に置き換えられます。省略時は `y_true` の数値がそのまま使われます。
 - `classes_to_plot`:  プロットに含める `y_true` の一意なクラス値のリスト（省略可）。指定しない場合、`y_true` に含まれるすべてのクラスが表示されます。
 - `interp_size`:  再現率を補間するポイント数。再現率を [0, 1] 区間で `interp_size` 個の等間隔な点に揃え、それに合わせて精度を補間します。
 - `title`:  プロットのタイトル。デフォルトは "Precision-Recall Curve" です。
 - `split_table`:  テーブルを W&B UI 上の独立したセクション「Custom Chart Tables」として分割表示するかどうか。`True` の場合、分割されます。デフォルトは `False` です。



**戻り値:**
 
 - `CustomChart`:  W&B にログできるカスタムチャートオブジェクト。チャートをログするには `wandb.log()` に渡します。



**例外:**
 
 - `wandb.Error`:  NumPy、pandas、または scikit-learn がインストールされていない場合に発生します。





**例:**
 

```python
import wandb

# 迷惑メール検出（2クラス分類）の例
y_true = [0, 1, 1, 0, 1]  # 0 = 迷惑メールでない, 1 = 迷惑メール
y_probas = [
    [0.9, 0.1],  # 最初のサンプルの予測確率（迷惑メールでない確率0.9）
    [0.2, 0.8],  # 2番目のサンプル（迷惑メール）など
    [0.1, 0.9],
    [0.8, 0.2],
    [0.3, 0.7],
]

labels = ["迷惑メールでない", "迷惑メール"]  # 見やすさのためのクラス名

with wandb.init(project="spam-detection") as run:
    pr_curve = wandb.plot.pr_curve(
         y_true=y_true,
         y_probas=y_probas,
         labels=labels,
         title="迷惑メール検出の Precision-Recall Curve",
    )
    run.log({"pr-curve": pr_curve})
```