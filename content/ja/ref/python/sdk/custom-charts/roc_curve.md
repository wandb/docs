---
title: roc_curve()
data_type_classification: function
menu:
  reference:
    identifier: ja-ref-python-sdk-custom-charts-roc_curve
object_type: python_sdk_custom_charts
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/plot/roc_curve.py >}}




### <kbd>関数</kbd> `roc_curve`

```python
roc_curve(
    y_true: 'Sequence[numbers.Number]',
    y_probas: 'Sequence[Sequence[float]] | None' = None,
    labels: 'list[str] | None' = None,
    classes_to_plot: 'list[numbers.Number] | None' = None,
    title: 'str' = 'ROC Curve',
    split_table: 'bool' = False
) → CustomChart
```

ROC（Receiver Operating Characteristic）曲線のチャートを作成します。 



**引数:**
 
 - `y_true`:  目的変数の真のクラスラベル（正解）。形状は (num_samples,) である必要があります。 
 - `y_probas`:  各クラスに対する予測確率または判定スコア。形状は (num_samples, num_classes) である必要があります。 
 - `labels`:   `y_true` のクラスインデックスに対応する人間が読みやすいラベル。たとえば、 `labels=['dog', 'cat']` の場合、プロットではクラス 0 は 'dog'、クラス 1 は 'cat' と表示されます。None の場合、 `y_true` の生のクラスインデックスが使用されます。デフォルトは None です。 
 - `classes_to_plot`:  ROC 曲線に含める一意なクラスラベルのサブセット。None の場合、 `y_true` にあるすべてのクラスがプロットされます。デフォルトは None です。 
 - `title`:  ROC 曲線プロットのタイトル。デフォルトは "ROC Curve" です。 
 - `split_table`:  W&B UI でテーブルを別セクションに分割して表示するかどうか。 `True` の場合、テーブルは "Custom Chart Tables" というセクションに表示されます。デフォルトは `False` です。 



**戻り値:**
 
 - `CustomChart`:  W&B にログできるカスタムチャートのオブジェクト。グラフをログするには、 `wandb.log()` に渡してください。 



**例外:**
 
 - `wandb.Error`:  numpy、pandas、または scikit-learn が見つからない場合。 



**例:**
 ```python
import numpy as np
import wandb

# 3 つの疾患を対象とした医療診断の分類問題をシミュレート
n_samples = 200
n_classes = 3

# 正解ラベル: 各サンプルに "Diabetes", "Hypertension", "Heart Disease" のいずれかを割り当てる
disease_labels = ["Diabetes", "Hypertension", "Heart Disease"]
# 0: Diabetes、1: Hypertension、2: Heart Disease
y_true = np.random.choice([0, 1, 2], size=n_samples)

# 予測確率: 各サンプルで合計が 1 になるようにシミュレート
y_probas = np.random.dirichlet(np.ones(n_classes), size=n_samples)

# プロットするクラスを指定（3 疾患すべてをプロット）
classes_to_plot = [0, 1, 2]

# W&B の run を初期化し、疾患分類の ROC 曲線プロットをログする
with wandb.init(project="medical_diagnosis") as run:
    roc_plot = wandb.plot.roc_curve(
         y_true=y_true,
         y_probas=y_probas,
         labels=disease_labels,
         classes_to_plot=classes_to_plot,
         title="ROC Curve for Disease Classification",
    )
    run.log({"roc-curve": roc_plot})
```